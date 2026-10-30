"""Extension points for model integration, residency and accelerator execution.

Host lifecycle methods are called outside capture. emit_* and program.run emit
device operations during capture and execute eagerly outside it. Python is NOT
called again on graph replay. Runtime bindings must own all captured addresses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import BoundMoERuntime

from ..transfer import TransferTicket


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
    resources: list = field(default_factory=list, repr=False, compare=False)
    context_ref: object = field(default=None, repr=False, compare=False)

    def dispose_completed(self):
        """Drop keepalives after the caller has proved completion; never wait.

        Owner retirement and graph teardown supply the proof. The same mutable
        resource bag is referenced by resources_idle, so an externally held
        completion does not retain a second copy after retirement.
        """
        context = self.context_ref() if self.context_ref is not None else None
        if context is not None:
            context.dispose_completed()
        else:
            self.resources.clear()


class CachePolicy(ABC):
    def initial_keys(self, capacity, model_id):
        """Host-only optional warm set; None keeps an explicitly cold start.

        Return logical keys only. The physical backend must materialize complete
        bundles and publish residency before serving or graph capture.
        """
        return None

    def finish_step(self, state):
        """Finish policy accounting after selection (e.g. periodic LFU decay)."""

    @abstractmethod
    def allocate_state(self, num_experts, device):
        pass

    @abstractmethod
    def observe(self, state, step_counts, update_enabled, due_out):
        """Accumulate valid local assignments and write the device update gate.

        due_out is a caller-owned fixed-address bool scalar, distinct from policy
        state. It remains valid through finish_step and transaction planning.
        Disabled steps must write False and leave statistics/clocks unchanged.
        The runtime never interprets a policy's state or configuration fields.
        """

    @abstractmethod
    def select_hot_set(self, state, key_to_slot, target_keys):
        """Write distinct logical keys, -1 padded; positions are not physical slots.

        Selection may run on every captured step. The explicit observe gate
        controls publication, independently of any internal selection mask.
        """

    @abstractmethod
    def copy_state(self, source, destination):
        """Copy to/from preallocated snapshot metadata without rebinding tensors."""


class ResidencyOps(ABC):
    """Device lowering; maps, counters and plans have fixed addresses/capacity."""

    @abstractmethod
    def resolve_execution(self, local_demand, context):
        """Return MaterializationPlan; misses do not imply hot admission.

        Accept LayerExecutionRequest; include plan-ready, old target readers and source
        readiness in copy_dependencies. Decode plan readiness also covers the
        preceding compute frontier: even different banks must not prefetch F.
        Keep source readers pinned through copy.
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
    def prepare_update(self, target_keys, context, *, update_enabled=None):
        """Return MaterializationPlan, retaining survivors and reserving protected victims.

        Include plan-ready and all old reader dependencies. Honor the runtime's
        device update_enabled guard during capture/warmup without reading it on CPU.
        An explicit policy gate is combined with that guard and latched into the
        transaction. None uses the lifecycle guard alone for initialization.
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


class MoEDispatcher(ABC):
    """Common D/C boundary; preserve EP identity and original routing weights.

    The runtime retains the dispatch handle and all communication buffers until
    their device consumers finish, including work submitted before an exception.
    A Python return from combine does not establish device completion.
    """

    @abstractmethod
    def dispatch(self, hidden_states, routing, context):
        pass

    @abstractmethod
    def combine(self, local_output, context):
        pass

    @abstractmethod
    def finish(self, accumulator, context):
        """Apply any common tail/reduction exactly once."""


class DispatchLayoutAdapter(ABC):
    """Slice a single full dispatch and assemble its original combine layout.

    During graph capture these methods emit device operations; Python is not
    rerun during replay. Use device counts/masks with fixed-capacity storage,
    never host decisions from .item()/.tolist(). Eager prefill may slice compact
    receive rows using host counts already supplied by its dispatcher, without
    synchronizing device metadata. Compact and padded layouts need backend adapters.
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
    def select_layer(self, layout, context):
        """Return the full expert-grouped input, masking fixed decode padding."""

    @abstractmethod
    def write_layer(self, layout, local_output, context):
        """Place full-layer F output in the original receive layout."""

    @abstractmethod
    def combine_input(self, layout, context):
        """Finish the full-layer output; return full output with original D metadata.

        Preserve the backend's weighting convention without applying weights a
        second time. Keep layout storage live until combine finishes reading it.
        """


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
