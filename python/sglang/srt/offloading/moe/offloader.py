"""BaseOffloader facade. Backend injection is deliberate until M0 is validated."""

from ..base import BaseOffloader
from .config import ExpertCatalog, ExpertLayerSpec
from .owner import MoEOffloadOwner
from .periodic_lfu import PeriodicLFUPolicy
from .policy import TensorLFUPolicy


class DeepSeekV4Adapter:
    def describe(self, layer):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not getattr(mlp, "is_deepseek_v4", False):
            raise ValueError("Initial MoE offload adapter requires DeepSeek V4 layers")
        experts = mlp.experts
        if experts.moe_tp_size != 1:
            raise ValueError("MoE TP is not supported")
        if getattr(mlp, "is_nextn", False):
            raise ValueError("Speculative/MTP experts are not supported")
        if getattr(experts, "num_fused_shared_experts", 0):
            raise ValueError("Keep shared experts separate and resident")
        spec = ExpertLayerSpec(
            layer_id=mlp.layer_id,
            global_experts=experts._num_global_routed,
            ep_size=experts.moe_ep_size,
            ep_rank=experts.moe_ep_rank,
            resident=mlp.layer_id < 3 or getattr(mlp, "is_hash", False),
        )
        if experts._num_local_routed != spec.local_experts:
            raise ValueError("Elastic/remapped EP ownership is not supported")
        return spec


class MoEOffloaderV2(BaseOffloader):
    """Shares V2's framework contract, not its whole-module rotation policy.

    No default accelerator backend is installed. prepare_layer must arrange
    bounded host loading; finalize must supply a full-graph-capable binding.
    """

    def __init__(
        self, config, backend, *, adapter=None, policy=None, periodic_lfu_config=None
    ):
        if policy is not None and periodic_lfu_config is not None:
            raise ValueError("Supply either policy or periodic_lfu_config, not both")
        self.config = config
        self.backend = backend
        self.adapter = adapter or DeepSeekV4Adapter()
        self.policy = policy or TensorLFUPolicy()
        self.periodic_lfu_config = periodic_lfu_config
        self.owner = None
        self._runtime = None
        self._layers = None
        self._catalog = None
        self._failed = False
        self._installed = False
        self._closed = False

    def wrap_modules(
        self,
        all_modules_generator,
        submodule_accessor=None,
        whitelist_param_names_creator=None,
    ):
        if self._layers is not None or self._failed or self._closed:
            raise RuntimeError("MoE modules may only be registered once")
        if submodule_accessor is not None or whitelist_param_names_creator is not None:
            raise ValueError("MoE registration uses expert formats, not V2 whitelists")
        modules, specs, offloaded = [], [], []
        try:
            for module in all_modules_generator:
                spec = self.adapter.describe(module)
                if (spec.ep_size, spec.ep_rank) != (
                    self.config.ep_size,
                    self.config.ep_rank,
                ):
                    raise ValueError("Model and offloader EP topology disagree")
                if any(existing.layer_id == spec.layer_id for existing in specs):
                    raise ValueError("Duplicate expert layer")
                specs.append(spec)
                if not spec.resident:
                    self.backend.prepare_layer(module, spec)
                    offloaded.append((module, spec))
                modules.append(module)
            self._catalog = ExpertCatalog(tuple(specs))
            if not self._catalog.num_experts:
                raise ValueError("No eligible routed experts to offload")
            self._layers = tuple(offloaded)
        except BaseException:
            self._failed = True
            raise
        return modules

    def post_init(self):
        if (
            self.owner is not None
            or self._layers is None
            or self._failed
            or self._closed
        ):
            raise RuntimeError("Register modules before finalizing exactly once")
        runtime = None
        try:
            if self.periodic_lfu_config is not None:
                # Layer registration determines rank-local keys. Bind the policy
                # only now, before the backend allocates state or captures ops.
                self.policy = PeriodicLFUPolicy(self.periodic_lfu_config, self._catalog)
            runtime = self.backend.finalize(
                self._layers, self._catalog, self.config, self.policy
            )
            self._runtime = runtime
            self.owner = MoEOffloadOwner(self.config, self._catalog, runtime)
        except BaseException:
            self._failed = True
            if runtime is not None:
                runtime.synchronize()
                runtime.close()
                self._runtime = None
            raise

    def bind_model_runner(self, model_runner):
        if self.owner is None or self._installed or self._failed or self._closed:
            raise RuntimeError("Finalize before binding model hooks exactly once")
        try:
            # Existing weight-update entry points already honor this model-owned
            # guard. The backend must additionally reject transaction preparation
            # paths before they can rebind/repack captured weights.
            model = model_runner.model
            if getattr(model, "_derived_weight_cache_error", None) is None:
                model._derived_weight_cache_error = (
                    "Online weight updates are not supported with MoE offload; "
                    "pinned host weights and captured expert bindings are immutable."
                )
            self.owner.runtime.install_model_hooks(self, model_runner)
            self._installed = True
        except BaseException:
            self._failed = True
            raise

    def forward_scope(self, forward_batch):
        if not self._installed or self._failed or self._closed:
            raise RuntimeError("MoE runtime/model hooks have not been bound")
        return self.owner.runtime.forward_scope(self.owner, forward_batch)

    def eager_step_scope(self, forward_batch):
        return self.owner.runtime.eager_step_scope(forward_batch)

    @property
    def forbid_copy_engine_usage(self):
        return self.backend.forbid_copy_engine_usage

    def close(self):
        if self._closed:
            return
        if self.owner is not None:
            self.owner.close()
        elif self._runtime is not None:
            self._runtime.synchronize()
            self._runtime.close()
        else:
            self.backend.close_unbound()
        self._runtime = None
        self._closed = True
