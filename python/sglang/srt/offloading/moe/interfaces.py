"""Extension points for model integration, residency and accelerator execution.

Host lifecycle methods are called outside capture. emit_* and program.run emit
device operations during capture and execute eagerly outside it. Python is NOT
called again on graph replay. Runtime bindings must own all captured addresses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

from ..transfer import TransferTicket
from .state import ResidencyState


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
        """Produce a bounded bank load plan; misses do not imply hot admission."""

    @abstractmethod
    def prepare_update(self, target_keys, context):
        """Retain surviving slots; wait for readers before reserving victims."""

    @abstractmethod
    def commit_update(self, update_plan, weights_ready, context):
        """Publish key/slot/generation only after the complete bundle is ready."""


class WholeMoEProgram(ABC):
    @abstractmethod
    def run(self, layer, hidden_states, routing, context):
        """Full routed-expert execution; shared experts stay outside this program."""


class LocalExpertProgram(ABC):
    @abstractmethod
    def run(self, layer, dispatch_output, wave, context):
        """Prepare fixed bank bindings, wait for readiness, run F, record readers."""


class WaveDispatcher(ABC):
    """Adapter to existing Dispatcher/CombineInput without changing EP identity."""

    @abstractmethod
    def mask_routes(self, routing, wave, context):
        pass

    @abstractmethod
    def dispatch(self, hidden_states, routing, context):
        pass

    @abstractmethod
    def combine(self, local_output, context):
        pass

    @abstractmethod
    def accumulator(self, hidden_states, context):
        """Return a zeroed, budgeted output buffer."""

    @abstractmethod
    def accumulate(self, accumulator, partial, context):
        pass

    @abstractmethod
    def finish(self, accumulator, context):
        """Apply any common tail/reduction exactly once."""


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

    @abstractmethod
    def emit_layer_start(self, layer_id, context):
        """Prefill starts all-local-expert prefetch before the router."""

    @abstractmethod
    def emit_after_router(self, layer_id, routing, context):
        """Emit routing bookkeeping; complete local demand is known after D."""

    @abstractmethod
    def emit_step_end(self, context):
        """Decode: policy update/select, materialize new hot set, then commit.

        A PeriodicLFUPolicy binding calls finish_step after select_hot_set and
        before preparing transfers. It must not gate valid steps on whether this
        EP rank has local routes. Disable both policy updates and commits during
        warmup/capture; selection alone does not publish residency.
        Prefill: no hotness updates. Whole-model step counts must ignore padding.
        All background work must join the resources_idle completion.
        """

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
