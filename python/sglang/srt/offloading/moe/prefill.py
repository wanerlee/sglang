"""Eager prefill pipeline contract; accelerator workers and allocation stay bound.

This interface does not implement an ACL worker or claim transfer overlap. It
isolates submission/sealing/bank scheduling from the shared D/F/C programs.
"""

from abc import ABC, abstractmethod

from ..transfer import TransferTicket


class PrefillWeightPipeline(ABC):
    """Runtime-owned sessions over borrowed A/B banks; a is a view inside A.

    Implementations key sessions by invocation and layer, retain in-flight plans,
    and never allocate hidden expert storage. All methods are eager-only.
    """

    def __init__(self, access):
        self.access = access

    @abstractmethod
    def start(self, layer_id, context):
        """Start bounded ACL batches of whole experts into A at layer entry.

        Pin session state before starting a worker. Set device/context explicitly;
        permit at most Q unfinished microbatches and at most K_cap experts.
        """

    @abstractmethod
    def validate_session(self, layer, context):
        """Reject missing/stale/failed sessions before dispatching a collective."""

    @abstractmethod
    def seal(self, layer, dispatch_ready, context):
        """Return this invocation's tuple of ExpertWave, covering all local experts.

        Observe recorded dispatch readiness, stop NEW submissions, freeze K after
        whole microbatches. Already submitted copies must complete. Record ready0
        and hand off stream submission ownership; do not mutate shared waves.
        Join microbatch leases with access.combine_prefill into one large-group
        lease; retain exactly the loaded expert range and validate its bindings.
        """

    @abstractmethod
    def acquire(self, layer, wave, context):
        """Return current WeightLease and schedule next wave before current F.

        Use access.stage with HOST_ONLY requests. First serve A, then B/a. Resolve
        previous-generation reader dependencies before every overwrite. Remember
        returned leases so access.release_after supplies their reader completions.
        """

    @abstractmethod
    def finish(self, layer, context) -> TransferTicket:
        """Close submission and return all layer worker/copy/reader dependencies.

        No device synchronization is needed in the normal path. Do not free or
        reuse sessions until their ticket completes; runtime joins resources_idle.
        """

    @abstractmethod
    def abort(self, context) -> TransferTicket:
        """Stop/join host submitters and fence partial work, including failed start.

        Idempotent for the invocation. Keep all resources pinned if draining fails;
        the runtime failure path must then synchronize its streams before release.
        """
