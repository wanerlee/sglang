"""Extension points for model integration, residency and accelerator execution.

Host lifecycle methods are called outside capture. emit_* and program.run emit
device operations during capture and execute eagerly outside it. Python is NOT
called again on graph replay. Runtime bindings must own all captured addresses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ..transfer import Completion, TransferTicket
from .config import Phase
from .state import ResidencyState


@dataclass(frozen=True)
class LocalRoutes:
    """Post-dispatch assignments; expert IDs are local to this rank/layer."""

    expert_ids: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class RuntimeCapabilities:
    full_decode_graph: bool = False
    dynamic_expert_selection: bool = False
    canonical_residency: bool = False

    def require_decode(self):
        if not all(
            (
                self.full_decode_graph,
                self.dynamic_expert_selection,
                self.canonical_residency,
            )
        ):
            raise NotImplementedError(
                "MoE offload requires full decode graph, dynamic expert selection "
                "and public residency state; eager fallback is not supported."
            )


@dataclass(frozen=True)
class ForwardCompletion:
    output_ready: TransferTicket
    residency_ready: TransferTicket
    resources_idle: TransferTicket
    # idle joins copy/compute/communication/prefetch.


class ExecutionContext(ABC):
    """Runtime-owned invocation/capture context; phase/capturing are host constants.

    emit_* submits eager operations or emits captured nodes. Python is not rerun
    on replay. Prefill may start host work only when capturing is false.
    """

    phase: Phase
    capturing: bool
    compute_stream: object
    copy_stream: object

    @abstractmethod
    def retain(self, resource):
        """Pin before submission, through failure drain and graph leases.

        TransferTicket dependencies also join resources_idle. A partial submission
        without a ticket is drained by the runtime's stream/worker failure path.
        Capture-time tickets are not host proof of a subsequent replay finishing.
        Eager implementations must support calls from the prefill submitter thread.
        """

    @abstractmethod
    def record_compute_completion(self) -> TransferTicket:
        """Record the current compute frontier, without host synchronization."""


class ExpertFormat(ABC):
    """Extend the existing paged_experts format concepts for native backend layout.

    A bundle includes every expert weight, scale and format-dependent component.
    Repacking must be lossless, bounded by explicit workspace, and complete before
    freezing graph addresses. No implicit FP4 -> FP8/BF16 expansion is permitted.
    """

    @abstractmethod
    def describe(self, module):
        """Return checkpoint/runtime components, layout and physical bundle bytes."""

    @abstractmethod
    def finalize_host_store(self, module, store, workspace):
        """Finish bounded post-load repacking; return immutable pinned storage."""


class CachePolicy(ABC):
    def finish_step(self, state):
        """Finish policy accounting after selection (e.g. periodic LFU decay)."""

    @abstractmethod
    def allocate_state(self, num_experts, device):
        pass

    @abstractmethod
    def update_statistics(self, state, step_counts, update_enabled):
        """Device op; counts cover valid local token-expert executions only."""

    @abstractmethod
    def select_hot_set(self, state, key_to_slot, target_keys):
        """Write at most K keys, -1 padded; do not assign slots or move weights."""

    @abstractmethod
    def copy_state(self, source, destination):
        """Copy to/from preallocated snapshot metadata without rebinding tensors."""


class ResidencyOps(ABC):
    """Device lowering; maps, counters and plans have fixed addresses/capacity."""

    @abstractmethod
    def resolve_execution(self, local_demand, context):
        """Return ExecutionReservation; misses do not imply hot admission.

        Accept ExecutionRequest; include plan-ready, old target readers and source
        readiness in copy_dependencies. Keep source readers pinned through copy.
        """

    @abstractmethod
    def publish_execution(self, reservation, weights_ready, context) -> TransferTicket:
        """Wait for the whole bundle, publish temporary bindings, return readiness.

        Do not change the hot set. The returned ticket covers metadata publication
        as well as all streams used to establish execution validity.
        """

    @abstractmethod
    def finish_execution(self, reservation, read_done, context):
        """Retire temporary/source leases after the final reader; never on return."""

    @abstractmethod
    def prepare_update(self, target_keys, context):
        """Return UpdatePlan, retaining survivors and reserving protected victims.

        Include plan-ready and all old reader dependencies. Honor the runtime's
        device update_enabled guard during capture/warmup without reading it on CPU.
        """

    @abstractmethod
    def commit_update(self, update_plan, weights_ready, context) -> TransferTicket:
        """Wait, publish hot key/slot/generation, return publication completion."""


class WholeMoEProgram(ABC):
    def layer_start(self, layer_id, context):
        """Optional eager prefill hook. Decode does not prefetch weights."""

    @abstractmethod
    def run(self, layer, hidden_states, routing, context):
        """Full routed-expert execution; shared experts stay outside this program."""


class LocalExpertProgram(ABC):
    @abstractmethod
    def run(self, layer, dispatch_output, wave, context):
        """Prepare fixed bank bindings, wait for readiness, run F, record readers."""


class MoEDispatcher(ABC):
    """Common D/C boundary; preserve EP identity and original routing weights.

    The runtime retains the dispatch handle and all communication buffers until
    their device consumers finish, including work submitted before an exception.
    A Python return from combine does not establish device completion.
    """

    @abstractmethod
    def dispatch(self, hidden_states, routing, context):
        pass

    def dispatch_completion(self, dispatched, context) -> Completion:
        """Cover actual receive readiness, including communication streams.

        Required by prefill feedback. Python dispatch returning is not completion.
        Static schedules retain compatibility without implementing this method.
        """
        raise NotImplementedError(
            "Dispatch completion is required for prefill feedback"
        )

    @abstractmethod
    def combine(self, local_output, context):
        pass

    @abstractmethod
    def finish(self, accumulator, context):
        """Apply any common tail/reduction exactly once."""


class WaveDispatcher(MoEDispatcher):
    """Repeated DFC additionally needs route masking and result accumulation."""

    @abstractmethod
    def mask_routes(self, routing, wave, context):
        pass

    @abstractmethod
    def accumulator(self, hidden_states, context):
        """Return a zeroed, budgeted output buffer."""

    @abstractmethod
    def accumulate(self, accumulator, partial, context):
        pass


class DispatchLayoutAdapter(ABC):
    """Slice a single full dispatch and assemble its original combine layout.

    These methods emit device operations; Python is not rerun during replay.
    Use device counts/masks with fixed-capacity storage, never host decisions
    from .item()/.tolist(). Compact and padded layouts need backend adapters.
    """

    @abstractmethod
    def bind(self, dispatch_output, context):
        """Return a per-invocation layout over runtime-owned, budgeted buffers.

        Retain the full dispatch metadata, activation scales and combine handle.
        Initialize output/padding as required, including entirely empty demand.
        The runtime keeps buffers alive through resources_idle and graph leases;
        releasing this Python layout must not release in-flight device storage.
        """

    @abstractmethod
    def select_wave(self, layout, wave, context):
        """Return LocalExpertProgram input for [local_start, local_end).

        Preserve scales, counts and inverse row mapping; bank lanes are local to
        F and must not change EP IDs. Support empty waves and the partial tail.
        """

    @abstractmethod
    def write_wave(self, layout, wave, local_output, context):
        """Place F output at its original receive rows, without top-k weighting.

        Establish dependencies before reusing per-wave input/output scratch.
        This assembles the full combine input; it is not token accumulation.
        """

    @abstractmethod
    def combine_input(self, layout, context):
        """Join all wave writes; return full output with original D metadata.

        Preserve the backend's weighting convention without applying weights a
        second time. Keep layout storage live until combine finishes reading it.
        """


class BoundMoERuntime(ABC):
    """One rank/model binding, exclusively owned by MoEOffloadOwner.

    resources enumerates unique physical allocations (including policy sort
    workspace, snapshots, copy plans, graph workspaces and any staging weights).
    Prefill views can alias the hot arena; aliases are not new allocations.
    """

    capabilities: RuntimeCapabilities
    resources: tuple
    expert_nbytes: int
    residency: ResidencyState
    prefill_program: WholeMoEProgram
    decode_program: WholeMoEProgram
    cache_updates = None  # Optional bound CacheUpdateCoordinator.

    def eager_step_scope(self, forward_batch):
        """Optional step emission; capture uses separately installed hooks."""
        return nullcontext()

    @abstractmethod
    def install_model_hooks(self, offloader, model_runner):
        """Bind layer-start/router hooks and capture_head_hooks/capture_tail_hooks.

        Every graph bucket pins owner with a graph lease. Release that lease only
        after destroying the graph. Reject unsupported runtime features before
        any collective: EPLB, speculative decoding, weight updates and overlap.
        Warmup/capture use a device update_enabled=0; real replay sets it to 1.
        """

    @abstractmethod
    def forward_scope(self, owner, forward_batch):
        """Host context around eager forward OR replay; serialize with idle fence.

        Must finish_forward with fresh completion events recorded after submission.
        On exceptions call fail_forward; never reuse capture-time host tickets as
        completion evidence for a later replay. Reject unsupported ForwardModes.
        """

    @abstractmethod
    def emit_step_begin(self, context):
        """Clear step accesses and establish fixed per-forward device state."""

    def emit_layer_start(self, layer_id, context):
        """Start eager prefill at layer entry; decode emits no weight prefetch.

        Bind resident layers to a no-op program or skip this hook for those layers.
        """
        if context.phase is Phase.PREFILL:
            if context.capturing:
                raise RuntimeError("Prefill layer-start cannot run during capture")
            self.prefill_program.layer_start(layer_id, context)
        elif context.phase is not Phase.DECODE:
            raise ValueError("Unsupported execution phase")

    @abstractmethod
    def emit_after_router(self, layer_id, routing, context):
        """Emit routing bookkeeping; complete local demand is known after D."""

    def emit_step_end(self, context):
        """Decode: policy update/select, materialize new hot set, then commit.

        A PeriodicLFUPolicy binding calls finish_step after select_hot_set and
        before preparing transfers. It must not gate valid steps on whether this
        EP rank has local routes. Disable both policy updates and commits during
        warmup/capture; selection alone does not publish residency.
        Prefill: no hotness updates. Whole-model step counts must ignore padding.
        All background work must join the resources_idle completion.
        """
        if context.phase is Phase.PREFILL:
            if context.capturing:
                raise RuntimeError("Prefill step-end cannot run during capture")
            return None
        if context.phase is not Phase.DECODE:
            raise ValueError("Unsupported execution phase")
        if self.cache_updates is None:
            raise NotImplementedError("Bind cache_updates or implement emit_step_end")
        return self.cache_updates.emit_step_end(context)

    @abstractmethod
    def snapshot_decode_state(self):
        """Return TransferTicket for exact slot/policy metadata snapshot, no weights."""

    @abstractmethod
    def borrow_hot_arena(self):
        """After snapshot completes, expose budgeted views to prefill programs."""

    @abstractmethod
    def restore_decode_state(self):
        """Reload original slots from host; then restore mappings and policy."""

    @abstractmethod
    def synchronize(self):
        """Drain every backend stream, including work from a failed submission."""

    @abstractmethod
    def close(self):
        """Release resources after synchronization and destruction of all graphs."""


class MoEBackend(ABC):
    @property
    def forbid_copy_engine_usage(self):
        """Preserve V2's copy-engine coordination for ordinary CPU weight copies.

        A device-initiated backend may override this after validating contention.
        """
        return True

    @abstractmethod
    def prepare_layer(self, module, spec):
        """During model construction, establish host checkpoint loading targets.

        Keep global/local logical expert counts unchanged. Do not first allocate
        all routed weights on device and hope post_init will make them fit.
        """

    @abstractmethod
    def finalize(self, layers, catalog, config, policy) -> BoundMoERuntime:
        """After quantization post-load: freeze formats/addresses, bind programs.

        For PeriodicLFUPolicy, call initial_keys(config.hot_slots(catalog),
        model_id) on the host, load the complete bundles and publish their
        mappings before serving. Allocate zeroed policy state independently of
        offline frequencies, and wire finish_step into the bound runtime.
        """

    @abstractmethod
    def close_unbound(self):
        """Drain and release any partially built resources after a load failure."""
