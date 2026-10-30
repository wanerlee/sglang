"""Reusable invocation and exact-restore machinery for accelerator bindings.

No device is initialized by importing this module. Native stream/event factories
and weight reloads are supplied by the binding; none of these helpers certifies
an accelerator graph or creates another expert-weight allocation.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, fields
from threading import RLock, get_ident
from weakref import ref

from ..transfer import Completion, JoinedCompletion, TransferTicket
from .config import Phase
from .interfaces import ForwardCompletion, RuntimeCapabilities
from .resources import Allocation, AllocationLedger, ResourceKind, tensor_allocations

logger = logging.getLogger(__name__)


class ExecutionContext:
    """Thread-safe keepalives and fresh completion events for one submission.

    The runtime failure path must drain
    every device stream, including partially submitted native calls.
    On a failed drain the context deliberately keeps all resources alive.
    Capture contexts remain pinned by their graph bucket, not by replay tickets.
    """

    def __init__(
        self,
        phase,
        *,
        compute_stream,
        copy_stream,
        record_completion,
        capturing=False,
    ):
        if not isinstance(phase, Phase):
            raise ValueError("Unsupported execution phase")
        if phase is Phase.PREFILL and capturing:
            raise ValueError("Prefill cannot be captured")
        self.invocation_id = object()
        self.phase = phase
        self.capturing = capturing
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self._record = record_completion
        self._lock = RLock()
        self._resources = []
        self._resource_ids = set()
        self._temporary_resources = []
        self._submission_thread = get_ident()
        self._dependencies = {}
        self._frontiers = {}
        self._completion_streams = {}
        self._disposed = False
        self._closed = False
        self._completion = None

    def retain(self, resource):
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already closed")
            if (
                not isinstance(resource, (TransferTicket, Completion))
                and self._temporary_resources
                and get_ident() == self._submission_thread
            ):
                self._temporary_resources[-1].append(resource)
            elif id(resource) not in self._resource_ids:
                self._resource_ids.add(id(resource))
                self._resources.append(resource)

    def retain_completion(self, dependency):
        """Pin submission resources and register their independent completion."""
        self.retain(dependency)
        return self.register_completion(dependency)

    def register_completion(self, dependency, *, stream=None):
        """Track dependencies without implicitly retaining arbitrary resources.

        Only the recorder may supply stream: it establishes actual submission
        order. Unknown transport/communication completions remain independent.
        Nested joins are flattened and deduplicated by completion identity.
        """
        completion = (
            dependency.completion
            if isinstance(dependency, TransferTicket)
            else dependency
        )
        if not isinstance(completion, Completion):
            raise TypeError("Expected a recorded completion")
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already closed")
            if stream is not None:
                self._frontiers[id(stream)] = completion
                self._completion_streams[id(completion)] = id(stream)
            pending = [completion]
            while pending:
                item = pending.pop()
                if type(item) is JoinedCompletion:
                    pending.extend(item.completions)
                else:
                    self._dependencies[id(item)] = item
        return dependency

    def wait_for(self, dependency, stream):
        """Skip only leaves already enqueued by our recorder on this stream.

        Unknown transport and auxiliary-reader events retain their real waits.
        No ordering is inferred from an event name, tensor device or graph key.
        """
        completion = (
            dependency.completion
            if isinstance(dependency, TransferTicket)
            else dependency
        )
        leaves = (
            completion.completions
            if type(completion) is JoinedCompletion
            else (completion,)
        )
        for item in leaves:
            if self._completion_streams.get(id(item)) != id(stream):
                item.wait_on(stream)

    @contextmanager
    def temporary_scope(self):
        """Release submission-thread layer temporaries in prefill and decode.

        Only use after joining off-stream readers onto compute_stream. Ordinary
        tensor allocation/free then follows the same allocator rules as resident
        inference; graph pools keep addresses reserved without pinning every
        intermediate Tensor. Owners/tickets keep registrations and events alive.
        Background submitters bypass these scopes, retaining their resources for
        the invocation even while the submission thread has an open scope.
        """
        if get_ident() != self._submission_thread:
            raise RuntimeError("Only the submission thread may open temporary scopes")
        resources = []
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already closed")
            self._temporary_resources.append(resources)
        try:
            yield
        except BaseException:
            # Partial submission may have no usable event. Do not release these
            # resources until the runtime drains every device/communication stream.
            with self._lock:
                self._resources.extend(resources)
            raise
        finally:
            with self._lock:
                self._temporary_resources.pop()

    def record_compute_completion(self):
        return self._record_on(self.compute_stream)

    def _record_on(self, stream):
        ticket = self._record(stream)
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Completion factory must return TransferTicket")
        self.retain(ticket)
        self.register_completion(ticket, stream=stream)
        return ticket

    def finish(self, *, output_ready=None, residency_ready=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already finished")
            if self._temporary_resources:
                raise RuntimeError("Close temporary scopes before finishing invocation")
            # Replay requires fresh events. When neither milestone was supplied,
            # both describe the same compute frontier and share one record.
            output_ready = output_ready or self.record_compute_completion()
            residency_ready = residency_ready or output_ready
            for ticket in (output_ready, residency_ready):
                if not isinstance(ticket, TransferTicket):
                    raise TypeError("Forward completion requires TransferTicket")
                self.retain(ticket)
                self.register_completion(ticket)
            copy_done = self._record_on(self.copy_stream)
            if self.capturing:
                # CaptureEnd must see the auxiliary stream's LAST task, including
                # its final event, joined back onto the capture origin.
                copy_done.completion.wait_on(self.compute_stream)
            dependencies = {
                key: value
                for key, value in self._dependencies.items()
                if key not in self._completion_streams
            }
            dependencies.update(
                (id(value), value) for value in self._frontiers.values()
            )
            idle = TransferTicket(
                JoinedCompletion(dependencies.values()), (self._resources,)
            )
            self._completion = ForwardCompletion(
                output_ready, residency_ready, idle, self._resources, ref(self)
            )
            self._closed = True
            return self._completion

    def dispose_completed(self, *, graph_destroyed=False):
        """Retire after existing completion proof; no query or device wait.

        The owner calls this only after done()/wait succeeds. Captured contexts
        additionally require successful graph destruction. Failed submissions use
        dispose_after_drain because they may not have a usable completion ticket.
        """
        with self._lock:
            if self._disposed:
                return
            if not self._closed or self._completion is None:
                raise RuntimeError("Finish the invocation before releasing resources")
            if self.capturing and not graph_destroyed:
                raise RuntimeError(
                    "Destroy the captured graph before releasing resources"
                )
            self._dispose()

    def dispose_after_drain(self, *, graph_destroyed=False):
        """Internal teardown path after submissions stop and every stream drains."""
        with self._lock:
            if self._disposed:
                return
            if self.capturing and not graph_destroyed:
                raise RuntimeError(
                    "Destroy the captured graph before releasing resources"
                )
            self._closed = True
            self._dispose()

    def _dispose(self):
        self._resources.clear()
        self._resource_ids.clear()
        self._dependencies.clear()
        self._frontiers.clear()
        self._completion_streams.clear()
        self._completion = None
        self._record = None
        # These optional bindings otherwise retain the entire serving batch or
        # captured inputs through external references to an already retired ctx.
        self.__dict__.pop("forward_batch", None)
        self.__dict__.pop("num_token_non_padded", None)
        self._disposed = True


@dataclass
class HotSnapshot:
    """Only persistent residency belongs to the cross-phase snapshot."""

    key_to_slot: object
    slot_to_key: object
    slot_valid: object
    slot_generation: object


class DecodeStateSnapshot:
    """Preallocated metadata snapshot with weights reloaded into original slots.

    reload_weights(saved_residency, context) must load every saved valid slot
    from the immutable host master and return its complete bundle ticket. It
    must not publish live maps. This helper restores maps/policy only after the
    reload dependency; a failure leaves the owner poisoned for explicit teardown.
    """

    def __init__(self, residency, policy, policy_state, *, error_status):
        import torch

        self.residency = residency
        self.policy = policy
        self.policy_state = policy_state
        if (
            error_status.shape != ()
            or error_status.dtype != torch.int32
            or error_status.device != residency.key_to_slot.device
        ):
            raise ValueError("Snapshot needs the shared scalar device error_status")
        self.error_status = error_status
        self.saved = HotSnapshot(
            **{
                field.name: torch.empty_like(getattr(residency, field.name))
                for field in fields(HotSnapshot)
            }
        )
        self.saved_policy = policy.allocate_state(
            residency.key_to_slot.numel(), residency.key_to_slot.device
        )
        self._snapshot = None

    def save(self, context):
        if context.capturing:
            raise RuntimeError("Mode snapshots are outside graph capture")
        if self._snapshot is not None:
            raise RuntimeError("Decode snapshot already exists")
        context.retain(self)
        # Mode changes already drain the owner. Observe device faults at this
        # eager boundary, never from dynamic decode planning or graph replay.
        context.record_compute_completion().completion.synchronize()
        self._require_healthy()
        self._copy_residency(self.residency, self.saved)
        self.policy.copy_state(self.policy_state, self.saved_policy)
        ticket = context.record_compute_completion()
        self._snapshot = TransferTicket(ticket.completion, (ticket, self))
        context.retain(self._snapshot)
        return self._snapshot

    def restore(self, context, reload_weights):
        if context.capturing or self._snapshot is None:
            raise RuntimeError("Restore requires an eager saved decode snapshot")
        context.retain(self)
        # Establish on both streams: restore lowering may inspect saved maps
        # on compute before scheduling weight loads on copy.
        self._snapshot.completion.wait_on(context.compute_stream)
        self._snapshot.completion.wait_on(context.copy_stream)
        copied = reload_weights(self.saved, context)
        if not isinstance(copied, TransferTicket):
            raise TypeError("Snapshot reload must return TransferTicket")
        context.retain(copied)
        # Native submission success does not imply a valid copy: device lowering
        # can latch a fault and mask every write. A transition is eager and can
        # wait here; never publish saved valid maps for an unsuccessful reload.
        copied.completion.synchronize()
        self._require_healthy()
        copied.completion.wait_on(context.compute_stream)
        self._copy_residency(self.saved, self.residency)
        self.policy.copy_state(self.saved_policy, self.policy_state)
        self.residency.execution_valid.zero_()
        self.residency.step_access_count.zero_()
        # update_enabled belongs to the current invocation/capture, not history.
        published = context.record_compute_completion()
        ticket = TransferTicket(
            JoinedCompletion((copied.completion, published.completion)),
            (self, self._snapshot, copied, published),
        )
        context.retain(ticket)
        self._snapshot = None
        return ticket

    def _require_healthy(self):
        if self.error_status.item() != 0:
            raise RuntimeError("Device residency/transfer fault during mode transition")

    @staticmethod
    def _copy_residency(source, destination):
        for field in fields(HotSnapshot):
            getattr(destination, field.name).copy_(getattr(source, field.name))

    def allocations(self):
        """Include both saved residency and policy in the owner's metadata budget."""
        tensors = [
            ("residency." + field.name, getattr(self.saved, field.name))
            for field in fields(self.saved)
        ]
        # Cache policies can expose custom allocation reporting; the bundled
        # TensorLFUPolicy is a dataclass of tensors.
        if hasattr(self.saved_policy, "allocations"):
            policy_allocations = tuple(self.saved_policy.allocations())
        else:
            policy_allocations = ()
            tensors.extend(
                ("policy." + field.name, getattr(self.saved_policy, field.name))
                for field in fields(self.saved_policy)
            )
        return policy_allocations + tensor_allocations("snapshot.", tensors)


class BoundMoERuntime:
    """Shared invocation, transition and worker teardown for device bindings.

    A binding supplies streams/events, model hooks, storage and programs. Phase
    and pending-forward ownership remain with the single MoEOffloadOwner; this
    class does not duplicate that state machine.
    """

    capabilities = RuntimeCapabilities(True, True, True)

    def __init__(
        self,
        *,
        device_runtime,
        residency,
        residency_ops,
        cache_updates,
        snapshot,
        reload_weights,
        layer_programs,
        resources,
        expert_nbytes,
        storage,
        cache_stats=None,
        diagnostics=None,
        pipelines=(),
        release=None,
    ):
        self.device_runtime = device_runtime
        self.residency = residency
        self.ops = residency_ops
        self.cache_updates = cache_updates
        self.graph_maintenance_interval = getattr(
            cache_updates.policy, "graph_maintenance_interval", None
        )
        # Shared across graph buckets and preserved through prefill/capture.
        # Only successful, globally active decode submissions advance this clock.
        self._decode_step_mod = 0
        self._capture_maintenance = True
        self.cache_stats = cache_stats
        self.diagnostics = diagnostics
        self.snapshot = snapshot
        self.reload_weights = reload_weights
        self.layer_programs = dict(layer_programs)
        if not self.layer_programs:
            raise ValueError("At least one offloaded layer must be bound")
        self.resources = tuple(resources)
        self.expert_nbytes = expert_nbytes
        self.storage = tuple(storage)
        self.pipelines = tuple(pipelines)
        self._release = release
        self._model_binding = None
        self._context = None
        self._capture = None
        self._failed_contexts = []
        self._closed = False
        self._closing = False
        self._installed = False
        self._owner = None
        self._static_allocated = None
        self._captured_workspace = 0
        self._cache_initialized = False
        self.residency.update_enabled.fill_(True)

    def retire_graph_resources(self, contexts):
        """Forget only contexts disposed by their own successful graph cleanup.

        A runtime may serve several graph backends. Resetting one backend is no
        proof that another backend's failed capture can release its resources.
        Unknown failed contexts remain pinned until their owner resets or closes.
        """
        retired = {
            id(context)
            for context in contexts
            if context.capturing and context._disposed
        }
        self._failed_contexts = [
            context for context in self._failed_contexts if id(context) not in retired
        ]

    def _new_context(self, phase, *, stream=None, capturing=False):
        context = ExecutionContext(
            phase,
            compute_stream=stream or self.device_runtime.current_stream(),
            copy_stream=self.device_runtime.copy_stream,
            record_completion=self.device_runtime.record,
            capturing=capturing,
        )
        context.started_layers = set()
        return context

    def _require_open(self):
        if self._closed or self._closing:
            raise RuntimeError("MoE runtime is closed or closing")

    @property
    def context(self):
        # Closing rejects new submissions, but existing workers must still pin
        # partial native submissions until stop/join and device drain finish.
        if self._closed:
            raise RuntimeError("MoE runtime is closed")
        if self._context is None:
            raise RuntimeError("MoE execution requires a forward or capture scope")
        return self._context

    def _transition(self, operation):
        self._require_open()
        if self._context is not None:
            raise RuntimeError("Phase transition cannot overlap a forward")
        # The owner drained prior work before entering this transition.
        context = self._new_context(Phase.DECODE)
        self._context = context
        try:
            with self.device_runtime.stream_scope(context.compute_stream):
                ticket = operation(context)
                completion = context.finish(residency_ready=ticket)
            completion.resources_idle.completion.synchronize()
            context.dispose_completed()
        except BaseException:
            self._failed_contexts.append(context)
            raise
        finally:
            self._context = None

    def snapshot_decode_state(self):
        return self._transition(self.snapshot.save)

    def borrow_hot_arena(self):
        # Physical aliases are constructed before capture; no allocation/rebind.
        self.residency.execution_valid.zero_()

    def restore_decode_state(self):
        self._transition(
            lambda context: self.snapshot.restore(context, self.reload_weights)
        )

    def synchronize(self):
        contexts = list(self._failed_contexts)
        if self._context is not None:
            contexts.append(self._context)
        abort_error = None
        for context in {id(item): item for item in contexts}.values():
            if context.phase is not Phase.PREFILL or context._closed:
                continue
            for pipeline in self.pipelines:
                try:
                    ticket = pipeline.abort(context)
                    context.retain(ticket)
                except BaseException as error:
                    # A failed event record can follow a successful worker join.
                    # Still stop the other submitters and attempt a device drain;
                    # retain all resources until a later abort succeeds as well.
                    if abort_error is None:
                        abort_error = error
        self.device_runtime.synchronize()
        if abort_error is not None:
            raise abort_error

    def close(self):
        if self._owner is not None:
            self._owner.close()
            return
        if self._closed:
            return
        if self._context is not None or self._capture is not None:
            raise RuntimeError("Finish active host submissions before closing runtime")
        self._closing = True
        self.synchronize()
        self.close_after_drain()

    def close_after_drain(self):
        """Release after the owner/close path stopped submissions and drained."""
        if self._closed:
            return
        if self._context is not None or self._capture is not None:
            raise RuntimeError("Finish active host submissions before closing runtime")
        if self._owner is not None and self._owner._graphs:
            raise RuntimeError("Destroy graph buckets before closing runtime")
        self._closing = True
        for pipeline in self.pipelines:
            close = getattr(pipeline, "close", None)
            if close is not None:
                close()
        if self.diagnostics is not None:
            try:
                self.diagnostics.close()
            except Exception:
                logger.warning(
                    "MoE offload batch logger teardown failed", exc_info=True
                )
        self._remove_hooks()
        if self._release is not None:
            self._release()
        for context in self._failed_contexts:
            context.dispose_after_drain(graph_destroyed=True)
        self._failed_contexts.clear()
        if self.snapshot is not None:
            self.snapshot._snapshot = None
        self.layer_programs.clear()
        self.resources = self.storage = self.pipelines = ()
        self.snapshot = self.cache_updates = self.ops = self.residency = None
        self.cache_stats = self.diagnostics = None
        self.reload_weights = self._release = None
        self._closed = True

    def initialize_hot_cache(self, keys):
        """Materialize offline Top-K through the production transfer/publication.

        No synthetic decode step runs here: online usage and the periodic clock
        remain zero. The same fault latch and complete-bundle fence used during
        serving protect initial publication before any graph can capture banks.
        """
        import torch

        self._require_open()
        if self._installed or self._cache_initialized:
            raise RuntimeError("Initialize the hot cache exactly once before binding")
        keys = tuple(keys)
        if len(keys) != self.residency.slot_to_key.numel():
            raise ValueError("Offline Top-K must fill the configured hot capacity")
        if len(set(keys)) != len(keys) or any(
            type(key) is not int or not 0 <= key < self.residency.key_to_slot.numel()
            for key in keys
        ):
            raise ValueError("Offline Top-K must contain distinct catalog keys")

        def initialize(context):
            targets = self.cache_updates.target_keys
            source = torch.tensor(keys, dtype=torch.int64, device=targets.device)
            context.retain(source)
            targets.copy_(source)
            return self.cache_updates.access.apply_update(targets, context)

        self._transition(initialize)
        if self.ops.error_status.item() != 0:
            raise RuntimeError(
                "Device residency/transfer fault during cache initialization"
            )
        self._cache_initialized = True
        logger.info(
            "MoE offload initial cache ready: slots=%s layers=%s",
            len(keys),
            len(self.layer_programs),
        )

    def install_model_hooks(self, offloader, model_runner):
        self._require_open()
        if self._installed:
            raise RuntimeError("Model hooks are already installed")
        self._owner = offloader.owner
        try:
            self._model_binding = offloader.adapter.bind_entries(
                self, offloader._layers
            )
            self._runner = model_runner
            model_runner.capture_head_hooks.append(self._capture_head)
            model_runner.capture_tail_hooks.append(self._capture_tail)
            self._installed = True
        except BaseException:
            self._remove_hooks()
            raise

    def _enter_layer(self, layer_id):
        self._require_emission()
        context = self.context
        if layer_id not in context.started_layers:
            self.emit_layer_start(layer_id, context)
            context.started_layers.add(layer_id)

    def _require_emission(self):
        if self.context.phase is Phase.DECODE and self._capture is None:
            raise RuntimeError("MoE offload decode must use a captured full graph")

    def emit_layer_start(self, layer_id, context):
        if context.phase is Phase.PREFILL:
            if context.capturing:
                raise RuntimeError("Prefill layer-start cannot run during capture")
            program = self.layer_programs[layer_id].prefill
            program.layer_start(layer_id, context)

    def run_layer(self, layer_id, hidden_states, routing, *, pre_quant_input=None):
        self._require_emission()
        if pre_quant_input is not None:
            raise ValueError("Prequantized routed side streams are not supported")
        context = self.context
        # Eager preprocessing may replace the local-padding Tensor after
        # forward_scope entry. Read the finalized batch binding at actual D.
        context.num_token_non_padded = context.forward_batch.moe_num_token_non_padded()
        programs = self.layer_programs[layer_id]
        program = (
            programs.prefill if context.phase is Phase.PREFILL else programs.decode
        )
        # Dispatch/combine join native communication onto compute_stream. This
        # matches resident inference: release receive tensors/handles after C so
        # the graph allocator can reuse them for the next layer. Events and copy
        # descriptors remain pinned by the invocation; failed scopes keep tensors.
        with context.temporary_scope():
            context.retain(routing)
            return program.run(layer_id, hidden_states, routing, context)

    @staticmethod
    def _phase(forward_batch, *, has_global_work, current_phase):
        mode = forward_batch.forward_mode
        local_prefill = (
            not mode.is_decode_or_idle() and mode.is_extend_without_speculative()
        )
        if not (mode.is_decode_or_idle() or local_prefill):
            raise ValueError("MoE offload accepts decode/idle or ordinary extend only")
        if not has_global_work:
            # Padded slots are not work. An entirely idle group must not restore
            # decode weights merely because its local mode is IDLE.
            return current_phase
        # Attention-DP ranks may be IDLE or DECODE while peers run prefill.
        # This scheduler flag is shared even without two-batch overlap; the
        # local Attention mode remains unchanged. Decide before owner transitions
        # and before eager preprocessing can relabel/pad the local batch.
        if local_prefill or getattr(forward_batch, "is_extend_in_batch", False):
            return Phase.PREFILL
        return Phase.DECODE

    @staticmethod
    def _has_global_work(forward_batch):
        for name in ("original_global_num_tokens_cpu", "global_num_tokens_cpu"):
            counts = getattr(forward_batch, name, None)
            if counts is not None:
                return any(count > 0 for count in counts)
        count = getattr(forward_batch, "global_num_token_non_padded_cpu", None)
        if count is not None:
            return count > 0
        return forward_batch.input_ids.numel() > 0

    @contextmanager
    def forward_scope(self, owner, forward_batch):
        self._require_open()
        if self._context is not None or self._capture is not None:
            raise RuntimeError("Concurrent forwards/capture are not supported")
        has_global_work = self._has_global_work(forward_batch)
        phase = self._phase(
            forward_batch,
            has_global_work=has_global_work,
            current_phase=owner.phase,
        )
        stream = self.device_runtime.current_stream()
        token = owner.begin_forward(phase, stream)
        context = self._new_context(phase, stream=stream)
        context.forward_sequence = token.sequence
        self._context = context
        logger.debug(
            "MoE offload forward.begin: rank=%s seq=%s phase=%s mode=%s "
            "extend_in_batch=%s global_work=%s bs=%s tokens=%s",
            getattr(getattr(owner, "config", None), "ep_rank", None),
            token.sequence,
            phase.value,
            getattr(forward_batch.forward_mode, "name", None),
            getattr(forward_batch, "is_extend_in_batch", False),
            has_global_work,
            getattr(forward_batch, "batch_size", 0),
            getattr(getattr(forward_batch, "input_ids", None), "numel", lambda: 0)(),
        )
        try:
            # The graph consumes this fixed-address flag on replay. Active
            # decode peers advance even on locally idle ranks; prefill and an
            # entirely idle group never advance the decode cache policy.
            with self.device_runtime.stream_scope(stream):
                self.residency.update_enabled.fill_(
                    has_global_work and phase is Phase.DECODE
                )
            context.forward_batch = forward_batch
            context.num_token_non_padded = None
            active_decode = has_global_work and phase is Phase.DECODE
            interval = self.graph_maintenance_interval
            context.cache_maintenance = (
                active_decode and (self._decode_step_mod + 1) % interval == 0
                if interval is not None
                else True
            )
            context.cache_graph_replayed = False
            context.retain(forward_batch)
            yield context
            if phase is Phase.PREFILL:
                for pipeline in self.pipelines:
                    finish = getattr(pipeline, "finish_forward", None)
                    if finish is not None:
                        finish(context)
            if (
                interval is not None
                and active_decode
                and not context.cache_graph_replayed
            ):
                raise RuntimeError(
                    "Periodic decode requires exactly one full graph replay"
                )
            # Replay does not run Python hooks. Its tail was emitted in capture;
            # record fresh events only, without updating the policy twice.
            record = None
            if self.diagnostics is not None:
                try:
                    runner = getattr(
                        getattr(self, "_runner", None), "decode_cuda_graph_runner", None
                    )
                    record = self.diagnostics.snapshot(
                        phase,
                        token.sequence,
                        forward_batch,
                        bucket=(
                            getattr(runner, "bs", None)
                            if phase is Phase.DECODE
                            else None
                        ),
                    )
                    context.retain(record)
                except Exception:
                    logger.warning(
                        "MoE offload batch snapshot unavailable", exc_info=True
                    )
            completion = context.finish()
            owner.finish_forward(token, completion)
            if interval is not None and active_decode:
                self._decode_step_mod = (self._decode_step_mod + 1) % interval
            if self.diagnostics is not None:
                try:
                    self.diagnostics.publish(record, completion.output_ready.completion)
                except Exception:
                    logger.warning(
                        "MoE offload batch log publication failed", exc_info=True
                    )
            logger.debug(
                "MoE offload forward.submitted: seq=%s phase=%s",
                token.sequence,
                phase.value,
            )
        except BaseException:
            logger.exception(
                "MoE offload forward.failed: seq=%s phase=%s",
                token.sequence,
                phase.value,
            )
            self._failed_contexts.append(context)
            owner.fail_forward(token)
            # Stop workers before removing their active retention context. Any
            # failed drain leaves all contexts pinned for an explicit close retry.
            self.synchronize()
            raise
        finally:
            self._context = None

    def claim_decode_graph(self):
        """Choose from known batch metadata; never read a device policy scalar."""
        context = self.context
        if self._capture is not None or context.capturing:
            raise RuntimeError("Serving graph selection is outside capture")
        if context.cache_graph_replayed:
            raise RuntimeError("A decode batch may replay only one offload graph")
        context.cache_graph_replayed = True
        return context.cache_maintenance

    @contextmanager
    def capture_maintenance(self, enabled):
        """Select a static full-graph tail before warmup and capture."""
        if self._capture is None or self._context is not None:
            raise RuntimeError("Select maintenance inside an idle capture session")
        previous = self._capture_maintenance
        self._capture_maintenance = enabled
        try:
            yield
        finally:
            self._capture_maintenance = previous

    def _capture_head(self, graph_runner, forward_batch, num_tokens):
        if self._capture is None or self._context is not None:
            raise RuntimeError("Invalid offload capture invocation")
        stream = self.device_runtime.current_stream()
        context = self._new_context(Phase.DECODE, stream=stream, capturing=True)
        context.forward_batch = forward_batch
        context.num_token_non_padded = None
        context.retain(forward_batch)
        self._capture.append(context)
        self._context = context
        self.ops.begin_step(context)

    def _capture_tail(self, graph_runner, output, forward_batch, num_tokens):
        if self._capture is None or self._context is None:
            raise RuntimeError("Capture tail requires its matching head")
        tail = self.cache_updates.emit_step_end(
            self.context, maintenance=self._capture_maintenance
        )
        self.context.finish(residency_ready=tail)
        self._context = None

    @contextmanager
    def capture_scope(self, owner, graph_runner, stream):
        self._require_open()
        if self._capture is not None or self._context is not None:
            raise RuntimeError("Nested MoE graph capture")
        self.synchronize()
        self.residency.update_enabled.fill_(False)
        self.synchronize()  # flag must be visible before any warmup stream
        if self._static_allocated is None:
            # KV pools and graph input buffers are allocated after model hooks;
            # take the baseline only when the first capture session starts.
            self._static_allocated = self.device_runtime.allocated()
        self.device_runtime.reset_peak()
        retained = []
        self._capture = retained
        try:
            yield retained
            if self._context is not None:
                raise RuntimeError("Capture model invocation did not finish")
            self.synchronize()
            if self.ops.error_status.item() != 0:
                raise RuntimeError("Device residency fault during graph warmup")
            self._account_capture_workspace(owner)
        except BaseException:
            self._failed_contexts.extend(retained)
            raise
        finally:
            self._capture = None
            self._context = None
            # Outside the captured graph: replay observes the live device flag.
            self.residency.update_enabled.fill_(True)
            self.synchronize()

    def after_graph_warmup(self):
        self._require_open()
        if self._capture is None or self._context is not None or not self._capture:
            raise RuntimeError("Warmup retirement requires a finished invocation")
        # capture_one invokes its reset hook after one eager forward, before the
        # next warmup/capture. Older entries may belong to captured graph buckets.
        context = self._capture[-1]
        self.synchronize()  # Includes native EP streams and partially queued work.
        context.dispose_after_drain(graph_destroyed=True)
        self._capture.pop()

    def _account_capture_workspace(self, owner):
        if self._static_allocated is None:
            self._static_allocated = self.device_runtime.allocated()
        used = max(0, self.device_runtime.peak() - self._static_allocated)
        self._captured_workspace = max(used, self._captured_workspace)
        dynamic = Allocation(
            "npu.capture_and_temporary_workspace",
            ResourceKind.WORKSPACE,
            self._captured_workspace,
            ("capture_workspace", id(self)),
            self,
        )
        # Includes whole-model capture growth conservatively; no expert payload
        # can hide in an uncharged graph-pool allocation.
        allocations = self.resources + (dynamic,)
        ledger = AllocationLedger(owner.allocation_ledger.limits, allocations)
        owner.allocation_ledger = ledger

    def _remove_hooks(self):
        if self._model_binding is not None:
            self._model_binding.close()
            self._model_binding = None
        runner = getattr(self, "_runner", None)
        if runner is not None:
            for hooks, callback in (
                (runner.capture_head_hooks, self._capture_head),
                (runner.capture_tail_hooks, self._capture_tail),
            ):
                if callback in hooks:
                    hooks.remove(callback)
            self._runner = None
