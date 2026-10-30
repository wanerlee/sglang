"""One-layer lookahead over two whole-layer banks, with no host worker/polling."""

from dataclasses import dataclass, field

from ..transfer import ImmediateCompletion, TransferTicket
from .execution import require_prefill


@dataclass
class _Session:
    invocation_id: object
    current: int = 0
    leases: dict = field(default_factory=dict)
    started: bool = False


class LayerPrefetchPipeline:
    # The pipeline installs a forward-entry dependency and per-bank reader
    # dependencies. A global layer-entry copy barrier would defeat lookahead.
    manages_layer_dependencies = True

    def __init__(
        self, access, layer_ids, *, make_request, record_copy_completion, profile=None
    ):
        self.access = access
        self.layer_ids = tuple(layer_ids)
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("Layer prefetch requires a unique ordered layer list")
        self.make_request = make_request
        self.record_copy_completion = record_copy_completion
        self.profile = profile
        self._session = None
        self._closed = self._failed = False

    def _scope(self, name, layer, bank, context):
        from contextlib import nullcontext

        return (
            self.profile(name, layer, bank, context)
            if self.profile is not None
            else nullcontext()
        )

    def _current(self, layer, context):
        require_prefill(context)
        s = self._session
        if (
            s is None
            or s.invocation_id is not context.invocation_id
            or self.layer_ids[s.current] != layer
        ):
            raise RuntimeError("Consume prefetched layers once in model order")
        return s

    def _submit(self, s, index, context, previous_reader=None):
        layer, bank = self.layer_ids[index], index % 2
        if index in s.leases:
            raise RuntimeError("Layer weights were already submitted")
        if previous_reader is not None:
            with self._scope("bank_reuse_wait", layer, bank, context):
                context.wait_for(previous_reader, context.copy_stream)
        with self._scope("layer_load_submit", layer, bank, context):
            request = self.make_request(layer, bank, context)
            if request.layer_id != layer or request.bank_id != bank:
                raise ValueError("Layer load request has a different bank or layer")
            s.leases[index] = self.access.stage(request, context)

    def start(self, layer, context):
        require_prefill(context)
        if self._closed or self._failed:
            raise RuntimeError("Layer prefetch pipeline is closed or failed")
        if self._session is None:
            if layer != self.layer_ids[0]:
                raise RuntimeError("Start prefill at the first offloaded layer")
            s = self._session = _Session(context.invocation_id)
            context.retain(s)
            # Includes owner.begin_forward's resources_idle join and any phase
            # transition. Install exactly once, before either bank is written.
            context.wait_for(context.record_compute_completion(), context.copy_stream)
            self._submit(s, 0, context)
            if len(self.layer_ids) > 1:
                self._submit(s, 1, context)
        s = self._current(layer, context)
        if s.started:
            raise RuntimeError("Layer prefill entry was already consumed")
        s.started = True

    def acquire(self, layer, context):
        s = self._current(layer, context)
        if not s.started or s.leases[s.current].read_done is not None:
            raise RuntimeError("Layer has not started or its weights were retired")
        return s.leases[s.current]

    def retire(self, layer, lease, read_done, context):
        s = self._current(layer, context)
        if lease is not s.leases[s.current]:
            raise ValueError("Foreign layer weight lease")
        self.access.release_after(lease, read_done, context)
        following = s.current + 2
        if following < len(self.layer_ids):
            # The record is already submitted, never wait for an unrecorded
            # event. Copies are ordered behind next-layer loading on one stream.
            self._submit(s, following, context, previous_reader=read_done)

    def finish_layer(self, layer, context):
        s = self._current(layer, context)
        if not s.started or s.leases[s.current].read_done is None:
            raise RuntimeError("Finish the layer only after its final weight reader")
        s.current += 1
        s.started = False
        if s.current == len(self.layer_ids):
            self._session = None

    def finish_forward(self, context):
        require_prefill(context)
        if self._session is not None:
            raise RuntimeError("Prefill ended before every offloaded layer completed")

    def abort(self, context):
        require_prefill(context)
        s = self._session
        if s is None:
            return TransferTicket(ImmediateCompletion())
        if s.invocation_id is not context.invocation_id:
            raise RuntimeError("Cannot abort another prefill invocation")
        self._failed = True
        # Host submission is synchronous. Fence even a partially failed native
        # call; its allocation owners were retained before the call began.
        ticket = self.record_copy_completion(context)
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Copy fence must be a TransferTicket")
        context.retain_completion(ticket)
        context.retain(s)
        self._session = None
        return ticket

    def close(self):
        if self._session is not None:
            raise RuntimeError("Finish or abort prefill before closing the pipeline")
        self._closed = True
