"""Fixed-capacity tensor residency planning, independent of weight transport.

All persistent plans and arithmetic scratch are allocated at binding time. The
operations below never read device values in Python. Capture/operator support is
still a hardware-runtime obligation; this class alone does not certify a graph.
The runtime must execute these operations on ``context.compute_stream``, serialize
forwards and call ``begin_step`` once at the beginning of the complete graph.
"""

from dataclasses import dataclass

import torch

from ..transfer import CopySourceKind, DeviceCopyPlan, HostCopyPlan, TransferTicket
from .config import Phase
from .execution import MaterializationPlan, require_prefill
from .interfaces import ResidencyOps
from .layer_execution import BoundExecutionLayer, LayerExecutionRequest
from .resources import tensor_allocations


@dataclass
class _Reservation:
    owner: object
    invocation_id: object
    buffers: object
    prefill: bool = False
    published: bool = False
    retired: bool = False


class TensorResidencyOps(ResidencyOps):
    """Canonical hit/miss plans and transactional, survivor-preserving hot updates.

    local_demand is a rank-local int32/int64 vector of post-dispatch token-expert
    counts, one entry per local expert, contributed once per layer invocation.
    Temporary execution never admits experts into the hot pool.

    prepare_prefill(request, context), when supplied, returns an eager
    MaterializationPlan with a HostCopyPlan and protected bank bindings. It owns
    full-layer bank bindings and old-reader dependencies. Prefill never changes the
    decode residency metadata or statistics.
    """

    def __init__(
        self,
        state,
        catalog,
        *,
        update_bindings,
        error_status=None,
        prepare_prefill=None,
        cache_stats=None,
        execution_layers=(),
    ):
        self.state, self.catalog = state, catalog
        self.update_bindings = update_bindings
        self.prepare_prefill = prepare_prefill
        self.cache_stats = cache_stats
        self._owned = {}
        self._token = object()
        self._readers = {}
        self._active_execution = {}
        self._pending_update = None
        self.device = state.key_to_slot.device
        self.num_experts = catalog.num_experts
        self.hot_slots = state.slot_to_key.numel()
        self.execution_slots = state.execution_valid.numel()
        self._validate_state()
        self.error_status = (
            self._tensor("error_status", (), torch.int32)
            if error_status is None
            else error_status
        )
        self._validate_tensor(self.error_status, (), torch.int32, "error_status")
        self._healthy = self._tensor("healthy", (), torch.bool)
        self._error_code = self._tensor("error_code", (), torch.int32)
        self._bad = self._tensor("bad", (), torch.bool)
        self._invalid = self._tensor("constant.invalid", (), fill=-1)
        self._zero = self._tensor("constant.none", (), fill=int(CopySourceKind.NONE))
        self._host = self._tensor("constant.host", (), fill=int(CopySourceKind.HOST))
        self._hot = self._tensor("constant.hot", (), fill=int(CopySourceKind.HOT_SLOT))
        self._key_sentinel = self._tensor(
            "constant.key_sentinel", (), fill=self.num_experts
        )
        self._slot_sentinel = self._tensor(
            "constant.slot_sentinel", (), fill=self.hot_slots
        )
        self._banks = {}
        self._layers = {layer.layer_id: layer for layer in catalog.offloaded_layers}
        self._layer_sources = {
            layer.layer_id: self._allocate_layer_sources(layer)
            for layer in catalog.offloaded_layers
        }
        self._check_bindings(update_bindings)
        self._execution_layers = {}
        for bound in execution_layers:
            self._bind_layer(bound)
        banks = tuple(self._banks.values())
        for index, (start, size) in enumerate(banks):
            for other_start, other_size in banks[index + 1 :]:
                if max(start, other_start) < min(
                    start + size, other_start + other_size
                ):
                    raise ValueError("Distinct execution banks cannot overlap")
        self._tail = self._allocate_tail()

    def _tensor(self, name, shape, dtype=torch.int64, fill=0):
        if name in self._owned:
            raise ValueError("Duplicate residency metadata binding: " + name)
        tensor = torch.full(shape, fill, dtype=dtype, device=self.device)
        self._owned[name] = tensor
        return tensor

    def _validate_tensor(self, tensor, shape, dtype, name):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype
            or tensor.device != self.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Invalid fixed tensor binding: {name}")

    def _validate_state(self):
        if self.num_experts <= 0 or self.hot_slots > self.num_experts:
            raise ValueError("Invalid residency capacities")
        for name, shape, dtype in (
            ("key_to_slot", (self.num_experts,), torch.int64),
            ("slot_to_key", (self.hot_slots,), torch.int64),
            ("slot_valid", (self.hot_slots,), torch.bool),
            ("slot_generation", (self.hot_slots,), torch.int64),
            ("execution_valid", (self.execution_slots,), torch.bool),
            ("step_access_count", (self.num_experts,), torch.int64),
            ("update_enabled", (), torch.bool),
        ):
            self._validate_tensor(getattr(self.state, name), shape, dtype, name)

    def _check_bindings(self, bindings):
        status = getattr(bindings, "error_status", self.error_status)
        if status is not self.error_status:
            raise ValueError("Transport and residency must share error_status")
        generations = getattr(bindings, "hot_generations", self.state.slot_generation)
        if generations is not self.state.slot_generation:
            raise ValueError("Transport must read canonical slot generations")

    def _plan(self, prefix, capacity):
        return DeviceCopyPlan(
            self._tensor(prefix + ".source_kind", (capacity,)),
            self._tensor(prefix + ".source_index", (capacity,), fill=-1),
            self._tensor(prefix + ".destination_slot", (capacity,)),
            self._tensor(prefix + ".valid", (capacity,), torch.bool),
            self._tensor(prefix + ".expected_generation", (capacity,), fill=-1),
            self._tensor(prefix + ".count", (), torch.int32),
            capacity,
        )

    def _bind_layer(self, bound):
        if not isinstance(bound, BoundExecutionLayer):
            raise TypeError("execution_layers requires BoundExecutionLayer records")
        layer = self._layers.get(bound.layer_id)
        if layer is None or bound.bank_id < 0:
            raise ValueError("Invalid bound execution layer")
        size = layer.local_experts
        if not 0 <= bound.execution_start <= self.execution_slots - size:
            raise ValueError("Execution bank cannot hold the complete local layer")
        if bound.layer_id in self._execution_layers:
            raise ValueError("Bind each layer to exactly one execution schedule")
        old = self._banks.get(bound.bank_id)
        if old is not None and old[0] != bound.execution_start:
            raise ValueError("One execution bank must keep its physical offset")
        self._banks[bound.bank_id] = (
            bound.execution_start,
            max(size, old[1] if old else 0),
        )
        self._check_bindings(bound.bindings)
        buf = self._layer_sources[bound.layer_id]
        buf.update(bound=bound, bank_id=bound.bank_id)
        buf["published_valid"] = self._tensor(
            f"layer.{bound.layer_id}.published_valid", (size,), torch.bool
        )
        torch.arange(
            bound.execution_start,
            bound.execution_start + size,
            out=buf["plan"].destination_slot,
        )
        self._execution_layers[bound.layer_id] = buf

    def _allocate_layer_sources(self, layer):
        prefix = f"layer.{layer.layer_id}"
        size = layer.local_experts
        buf = {"layer": layer, "plan": self._plan(prefix, size)}
        for name in ("keys", "slots", "safe_slots", "slot_keys", "counts", "frequency"):
            buf[name] = self._tensor(prefix + "." + name, (size,))
        for name in ("needed", "hit", "flag", "flag2"):
            buf[name] = self._tensor(prefix + "." + name, (size,), torch.bool)
        first = self.catalog.key(layer.layer_id, 0)
        torch.arange(first, first + size, out=buf["keys"])
        return buf

    def _allocate_tail(self):
        h = self.hot_slots
        buf = {"plan": self._plan("tail.plan", h)}
        for name in (
            "safe_target",
            "marks",
            "slots",
            "safe_slots",
            "slot_keys",
            "old_keys",
            "new_rank",
            "rank_index",
            "new_values",
            "free_rank",
            "safe_rank",
            "proposed",
            "generation",
            "inverse_keys",
            "inverse_values",
        ):
            buf[name] = self._tensor("tail." + name, (h,))
        for name in (
            "valid",
            "flag",
            "flag2",
            "survivor_target",
            "new",
            "survivor_slot",
            "free",
            "changed",
            "proposed_valid",
        ):
            buf[name] = self._tensor("tail." + name, (h,), torch.bool)
        buf["membership"] = self._tensor("tail.membership", (self.num_experts + 1,))
        buf["duplicate"] = self._tensor(
            "tail.duplicate", (self.num_experts,), torch.bool
        )
        buf["inverse"] = self._tensor("tail.inverse", (self.num_experts + 1,))
        buf["new_keys"] = self._tensor("tail.new_keys", (h + 1,))
        buf["new_count"] = self._tensor("tail.new_count", ())
        buf["enabled"] = self._tensor("tail.enabled", (), torch.bool)
        torch.arange(h, out=buf["plan"].destination_slot)
        return buf

    def allocations(self):
        """Owned scratch only; external state/bindings are reported separately."""
        return tensor_allocations("tensor_residency.", self._owned.items())

    def _latch(self, bad):
        self._error_code.copy_(bad)
        self.error_status.bitwise_or_(self._error_code)
        torch.eq(self.error_status, 0, out=self._healthy)

    def _fault_if_any(self, mask, enabled=None):
        torch.any(mask, out=self._bad)
        if enabled is not None:
            torch.logical_and(self._bad, enabled, out=self._bad)
        self._latch(self._bad)

    def begin_step(self, context):
        if context.phase is Phase.PREFILL:
            require_prefill(context)
            return
        if context.phase is not Phase.DECODE:
            raise ValueError("Unsupported phase")
        if self._active_execution or self._pending_update is not None:
            raise RuntimeError("Retire the previous step's reservations first")
        context.retain(self)
        self.state.step_access_count.zero_()
        if self.cache_stats is not None:
            self.cache_stats.clear()
        # Cross-forward reuse is protected by the owner's resources_idle join.
        self._readers.clear()

    def _validate_demand(self, demand, layer):
        if (
            not isinstance(demand, torch.Tensor)
            or demand.shape != (layer.local_experts,)
            or demand.dtype not in (torch.int32, torch.int64)
            or demand.device != self.device
        ):
            raise ValueError(
                "Post-dispatch demand must contain one count per local expert"
            )

    def resolve_execution(self, request, context):
        if not isinstance(request, LayerExecutionRequest):
            raise TypeError("Expected a whole-layer execution request")
        if request.phase is not context.phase:
            raise ValueError("Execution request phase does not match context")
        context.retain((self, request))
        if request.phase is Phase.PREFILL:
            require_prefill(context)
            if self.prepare_prefill is None:
                raise NotImplementedError(
                    "Bind the eager host-bank planner for prefill"
                )
            reservation = self.prepare_prefill(request, context)
            if not isinstance(reservation, MaterializationPlan) or not isinstance(
                reservation.copy_plan, HostCopyPlan
            ):
                raise TypeError(
                    "Prefill planner must return a host MaterializationPlan"
                )
            context.retain(reservation)
            token = _Reservation(
                self._token, context.invocation_id, reservation, prefill=True
            )
            return MaterializationPlan(
                reservation.copy_plan,
                reservation.bindings,
                reservation.copy_dependencies,
                (reservation,),
                token,
            )
        if request.phase is not Phase.DECODE:
            raise ValueError("Decode requires device planning")
        return self._resolve_layer_execution(request, context)

    def _resolve_layer_execution(self, request, context):
        if self._active_execution or self._pending_update:
            raise RuntimeError("Retire prior readers/plans before executing a layer")
        buf = self._execution_layers.get(request.layer_id)
        if buf is None or buf["bank_id"] != request.bank_id:
            raise ValueError("Bind the complete decode layer before capture")
        self._validate_demand(request.local_demand, buf["layer"])
        reader = self._readers.get(request.bank_id)
        if reader is not None:
            context.wait_for(reader, context.compute_stream)
        self._prepare_sources(buf, request.local_demand, request.layer_id, 0)
        bound, plan = buf["bound"], buf["plan"]
        self.state.execution_valid[
            bound.execution_start : bound.execution_start + plan.capacity
        ].zero_()
        ready = context.record_compute_completion()
        token = _Reservation(self._token, context.invocation_id, buf)
        self._active_execution[request.bank_id] = token
        return MaterializationPlan(
            plan, bound.bindings, (ready.completion,), (self, buf, ready), token
        )

    def _prepare_sources(self, buf, demand, layer_id, start):
        plan = buf["plan"]
        buf["counts"].copy_(demand)
        buf["counts"].clamp_(min=0, max=torch.iinfo(torch.int64).max // 4)
        torch.gt(buf["counts"], 0, out=buf["needed"])
        torch.index_select(self.state.key_to_slot, 0, buf["keys"], out=buf["slots"])
        torch.ge(buf["slots"], 0, out=buf["hit"])
        if self.hot_slots:
            torch.ge(buf["slots"], self.hot_slots, out=buf["flag"])
            torch.lt(buf["slots"], -1, out=buf["flag2"])
            torch.logical_or(buf["flag"], buf["flag2"], out=buf["flag"])
            self._fault_if_any(buf["flag"])
            torch.clamp(buf["slots"], 0, self.hot_slots - 1, out=buf["safe_slots"])
            torch.index_select(
                self.state.slot_valid, 0, buf["safe_slots"], out=buf["flag"]
            )
            torch.index_select(
                self.state.slot_to_key, 0, buf["safe_slots"], out=buf["slot_keys"]
            )
            torch.eq(buf["slot_keys"], buf["keys"], out=buf["flag2"])
            torch.logical_and(buf["flag"], buf["flag2"], out=buf["flag"])
            torch.logical_not(buf["flag"], out=buf["flag2"])
            torch.logical_and(buf["hit"], buf["flag2"], out=buf["flag2"])
            self._fault_if_any(buf["flag2"])
            torch.logical_and(buf["hit"], buf["flag"], out=buf["hit"])
            torch.index_select(
                self.state.slot_generation,
                0,
                buf["safe_slots"],
                out=plan.expected_generation,
            )
        else:
            torch.ne(buf["slots"], -1, out=buf["flag"])
            self._fault_if_any(buf["flag"])
            buf["hit"].zero_()
            plan.expected_generation.fill_(-1)
        torch.logical_and(buf["needed"], self._healthy, out=plan.valid)
        torch.where(buf["hit"], buf["slots"], buf["keys"], out=plan.source_index)
        torch.where(buf["hit"], self._hot, self._host, out=plan.source_kind)
        torch.where(plan.valid, plan.source_kind, self._zero, out=plan.source_kind)
        torch.where(
            buf["hit"],
            plan.expected_generation,
            self._invalid,
            out=plan.expected_generation,
        )
        torch.sum(plan.valid, dim=0, dtype=torch.int32, out=plan.count)
        if self.cache_stats is not None:
            self.cache_stats.record_counts(
                buf["counts"], buf["hit"], self.state.update_enabled & self._healthy
            )
        key_start = self.catalog.key(layer_id, start)
        counts = self.state.step_access_count[key_start : key_start + plan.capacity]
        torch.add(counts, buf["counts"], out=buf["frequency"])
        torch.clamp(buf["frequency"], max=torch.iinfo(torch.int64).max // 4, out=counts)

    def _check(self, reservation, context):
        token = reservation.state
        if (
            not isinstance(token, _Reservation)
            or token.owner is not self._token
            or token.invocation_id is not context.invocation_id
        ):
            raise ValueError("Foreign residency reservation or invocation")
        return token

    def publish_execution(self, reservation, weights_ready, context):
        token = self._check(reservation, context)
        if token.published or token.retired:
            raise RuntimeError("Execution reservation already published or retired")
        if not isinstance(weights_ready, TransferTicket):
            raise TypeError("Publication requires a complete bundle ticket")
        context.retain((reservation, weights_ready))
        if token.prefill:
            # Publish only host-static bindings. Waiting on compute here would
            # serialize Attention/Router behind the prefetch. The
            # local program joins this ticket immediately before its FFN.
            token.published = True
            ticket = TransferTicket(
                weights_ready.completion, (weights_ready, reservation)
            )
            context.retain(ticket)
            return ticket
        weights_ready.completion.wait_on(context.compute_stream)
        buf, plan = token.buffers, reservation.copy_plan
        torch.eq(self.error_status, 0, out=self._healthy)
        torch.logical_and(plan.valid, self._healthy, out=buf["published_valid"])
        first = buf["bound"].execution_start
        self.state.execution_valid[first : first + plan.capacity].copy_(
            buf["published_valid"]
        )
        token.published = True
        ticket = context.record_compute_completion()
        context.retain(ticket)
        return ticket

    def finish_execution(self, reservation, read_done, context):
        token = self._check(reservation, context)
        if not token.published or token.retired:
            raise RuntimeError("Execution must be published and retired exactly once")
        if not isinstance(read_done, TransferTicket):
            raise TypeError("Reader completion must be a TransferTicket")
        context.retain((reservation, read_done))
        if not token.prefill:
            self._readers[token.buffers["bank_id"]] = read_done
            del self._active_execution[token.buffers["bank_id"]]
        token.retired = True

    def _inverse(self, keys, valid, buf):
        torch.where(valid, keys, self._key_sentinel, out=buf["inverse_keys"])
        buf["inverse_keys"].clamp_(0, self.num_experts)
        torch.add(buf["plan"].destination_slot, 1, out=buf["inverse_values"])
        torch.where(valid, buf["inverse_values"], self._zero, out=buf["inverse_values"])
        buf["inverse"].zero_()
        buf["inverse"].scatter_add_(0, buf["inverse_keys"], buf["inverse_values"])
        buf["inverse"].sub_(1)

    def prepare_update(self, target_keys, context, *, update_enabled=None):
        if context.phase is not Phase.DECODE:
            raise RuntimeError("Hot updates are decode-only")
        self._validate_tensor(
            target_keys, (self.hot_slots,), torch.int64, "target_keys"
        )
        if self._pending_update is not None:
            raise RuntimeError("Finish the previous update before reserving another")
        if self._active_execution:
            raise RuntimeError("Retire execution readers before updating hot weights")
        # A reader may finish on an auxiliary stream. Join it before touching
        # residency and record the resulting frontier once in this transaction.
        for reader in self._readers.values():
            reader.completion.wait_on(context.compute_stream)
        context.retain((self, target_keys))
        b, s, h = self._tail, self.state, self.hot_slots
        p = b["plan"]
        if update_enabled is None:
            b["enabled"].copy_(s.update_enabled)
        else:
            self._validate_tensor(update_enabled, (), torch.bool, "update_enabled")
            torch.logical_and(s.update_enabled, update_enabled, out=b["enabled"])
        if h:
            # A stale invalid slot may keep its old key, but every valid slot
            # must have exactly one consistent inverse mapping.
            torch.ge(s.slot_to_key, 0, out=b["valid"])
            torch.lt(s.slot_to_key, self.num_experts, out=b["flag"])
            torch.logical_and(b["valid"], b["flag"], out=b["valid"])
            torch.logical_not(b["valid"], out=b["flag"])
            torch.logical_and(b["flag"], s.slot_valid, out=b["flag"])
            self._fault_if_any(b["flag"])
            torch.logical_and(b["valid"], s.slot_valid, out=b["valid"])
            torch.where(
                b["valid"], s.slot_to_key, self._key_sentinel, out=b["inverse_keys"]
            )
            b["marks"].copy_(b["valid"])
            b["membership"].zero_()
            b["membership"].scatter_add_(0, b["inverse_keys"], b["marks"])
            torch.gt(b["membership"][: self.num_experts], 1, out=b["duplicate"])
            self._fault_if_any(b["duplicate"])
            self._inverse(s.slot_to_key, b["valid"], b)
            torch.ne(
                b["inverse"][: self.num_experts], s.key_to_slot, out=b["duplicate"]
            )
            self._fault_if_any(b["duplicate"])
            torch.ge(target_keys, 0, out=b["valid"])
            torch.lt(target_keys, self.num_experts, out=b["flag"])
            torch.logical_and(b["valid"], b["flag"], out=b["valid"])
            torch.lt(target_keys, -1, out=b["flag"])
            torch.ge(target_keys, self.num_experts, out=b["flag2"])
            torch.logical_or(b["flag"], b["flag2"], out=b["flag"])
            self._fault_if_any(b["flag"], b["enabled"])
            torch.where(
                b["valid"], target_keys, self._key_sentinel, out=b["safe_target"]
            )
            b["marks"].copy_(b["valid"])
            b["membership"].zero_()
            b["membership"].scatter_add_(0, b["safe_target"], b["marks"])
            torch.gt(b["membership"][: self.num_experts], 1, out=b["duplicate"])
            self._fault_if_any(b["duplicate"], b["enabled"])
            torch.logical_and(b["enabled"], self._healthy, out=b["enabled"])
            torch.clamp(target_keys, 0, self.num_experts - 1, out=b["safe_target"])
            torch.index_select(s.key_to_slot, 0, b["safe_target"], out=b["slots"])
            torch.clamp(b["slots"], 0, h - 1, out=b["safe_slots"])
            torch.index_select(
                s.slot_valid, 0, b["safe_slots"], out=b["survivor_target"]
            )
            torch.ge(b["slots"], 0, out=b["flag"])
            torch.logical_and(b["survivor_target"], b["flag"], out=b["survivor_target"])
            torch.index_select(s.slot_to_key, 0, b["safe_slots"], out=b["slot_keys"])
            torch.eq(b["slot_keys"], target_keys, out=b["flag"])
            torch.logical_and(b["survivor_target"], b["flag"], out=b["survivor_target"])
            torch.logical_not(b["survivor_target"], out=b["new"])
            torch.logical_and(b["new"], b["valid"], out=b["new"])
            b["marks"].copy_(b["new"])
            torch.cumsum(b["marks"], 0, out=b["new_rank"])
            b["new_rank"].sub_(1)
            torch.sum(b["marks"], dim=0, out=b["new_count"])
            torch.where(
                b["new"], b["new_rank"], self._slot_sentinel, out=b["rank_index"]
            )
            torch.add(target_keys, 1, out=b["new_values"])
            torch.where(b["new"], b["new_values"], self._zero, out=b["new_values"])
            b["new_keys"].zero_()
            b["new_keys"].scatter_add_(0, b["rank_index"], b["new_values"])
            b["new_keys"].sub_(1)
            torch.clamp(s.slot_to_key, 0, self.num_experts, out=b["old_keys"])
            torch.index_select(b["membership"], 0, b["old_keys"], out=b["marks"])
            torch.gt(b["marks"], 0, out=b["survivor_slot"])
            torch.logical_and(b["survivor_slot"], s.slot_valid, out=b["survivor_slot"])
            torch.ge(s.slot_to_key, 0, out=b["flag"])
            torch.logical_and(b["survivor_slot"], b["flag"], out=b["survivor_slot"])
            torch.logical_not(b["survivor_slot"], out=b["free"])
            b["marks"].copy_(b["free"])
            torch.cumsum(b["marks"], 0, out=b["free_rank"])
            b["free_rank"].sub_(1)
            torch.clamp(b["free_rank"], 0, h, out=b["safe_rank"])
            torch.index_select(b["new_keys"], 0, b["safe_rank"], out=b["proposed"])
            torch.lt(b["free_rank"], b["new_count"], out=b["flag"])
            torch.where(b["flag"], b["proposed"], self._invalid, out=b["proposed"])
            torch.where(
                b["survivor_slot"], s.slot_to_key, b["proposed"], out=b["proposed"]
            )
            torch.where(b["enabled"], b["proposed"], s.slot_to_key, out=b["proposed"])
            torch.ge(b["proposed"], 0, out=b["proposed_valid"])
            torch.logical_not(s.slot_valid, out=b["flag"])
            torch.logical_and(b["flag"], b["proposed_valid"], out=b["flag"])
            torch.ne(b["proposed"], s.slot_to_key, out=b["changed"])
            torch.logical_or(b["changed"], b["flag"], out=b["changed"])
            torch.logical_and(b["changed"], b["enabled"], out=b["changed"])
            b["marks"].copy_(b["changed"])
            torch.add(s.slot_generation, b["marks"], out=b["generation"])
            torch.logical_and(b["changed"], b["proposed_valid"], out=p.valid)
            p.source_index.copy_(b["proposed"])
            torch.where(p.valid, self._host, self._zero, out=p.source_kind)
            p.expected_generation.copy_(b["generation"])
            torch.sum(p.valid, dim=0, dtype=torch.int32, out=p.count)
            # Victims cease being sources before writes begin. New keys are only
            # visible after complete-bundle copy AND metadata publication.
            torch.logical_not(b["changed"], out=b["flag"])
            torch.logical_and(s.slot_valid, b["flag"], out=b["survivor_slot"])
            self._inverse(s.slot_to_key, b["survivor_slot"], b)
            torch.where(
                b["enabled"],
                b["inverse"][: self.num_experts],
                s.key_to_slot,
                out=s.key_to_slot,
            )
            s.slot_valid.copy_(b["survivor_slot"])
            s.slot_generation.copy_(b["generation"])
        else:
            p.count.zero_()
        ready = context.record_compute_completion()
        context.retain(ready)
        token = _Reservation(self._token, context.invocation_id, b)
        self._pending_update = token
        return MaterializationPlan(
            p,
            self.update_bindings,
            (ready.completion,),
            (self, ready),
            token,
        )

    def commit_update(self, update_plan, weights_ready, context):
        token = self._check(update_plan, context)
        if token is not self._pending_update or token.published:
            raise RuntimeError("Hot update must commit exactly once")
        if not isinstance(weights_ready, TransferTicket):
            raise TypeError("Hot publication requires a complete bundle ticket")
        context.retain((update_plan, weights_ready))
        weights_ready.completion.wait_on(context.compute_stream)
        b, s = self._tail, self.state
        if self.hot_slots:
            torch.ne(s.slot_generation, b["generation"], out=b["flag"])
            torch.logical_and(b["flag"], b["enabled"], out=b["flag"])
            self._fault_if_any(b["flag"])
            torch.logical_and(b["enabled"], self._healthy, out=b["enabled"])
            self._inverse(b["proposed"], b["proposed_valid"], b)
            torch.where(b["enabled"], b["proposed"], s.slot_to_key, out=s.slot_to_key)
            torch.where(
                b["enabled"], b["proposed_valid"], s.slot_valid, out=s.slot_valid
            )
            torch.where(
                b["enabled"],
                b["inverse"][: self.num_experts],
                s.key_to_slot,
                out=s.key_to_slot,
            )
        token.published = True
        self._pending_update = None
        ticket = context.record_compute_completion()
        context.retain(ticket)
        return ticket
