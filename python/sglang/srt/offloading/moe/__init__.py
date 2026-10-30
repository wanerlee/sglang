"""Opt-in MoE offloading framework; accelerator execution is backend supplied."""

from .cache_metadata import CacheRouteStats
from .config import ExpertCatalog, ExpertLayerSpec, MoEOffloadConfig, Phase
from .execution import (
    CacheUpdateCoordinator,
    ExpertWeightAccess,
    LocalExecution,
    MaterializationPlan,
    WeightLease,
)
from .formats import BundleComponent, BundleSchema, ExpertFormat
from .layer_execution import BoundExecutionLayer, LayerExecutionRequest
from .layer_prefetch import LayerPrefetchPipeline
from .layer_programs import LayerExpertProgram
from .models import DeepSeekV4Adapter, ModelAdapter
from .offloader import MoEOffloaderV2
from .owner import MoEOffloadOwner
from .periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy, PeriodicLFUState
from .policy import TensorLFUPolicy
from .residency import TensorResidencyOps
from .runtime import BoundMoERuntime, DecodeStateSnapshot, ExecutionContext
from .storage import HostMemoryCapability

__all__ = [
    "BoundExecutionLayer",
    "LayerExecutionRequest",
    "LayerPrefetchPipeline",
    "LayerExpertProgram",
    "BoundMoERuntime",
    "BundleComponent",
    "BundleSchema",
    "CacheRouteStats",
    "CacheUpdateCoordinator",
    "DecodeStateSnapshot",
    "DeepSeekV4Adapter",
    "ExecutionContext",
    "MaterializationPlan",
    "ExpertCatalog",
    "ExpertLayerSpec",
    "ExpertFormat",
    "ExpertWeightAccess",
    "LocalExecution",
    "MoEOffloadConfig",
    "ModelAdapter",
    "MoEOffloadOwner",
    "MoEOffloaderV2",
    "PeriodicLFUConfig",
    "PeriodicLFUPolicy",
    "PeriodicLFUState",
    "Phase",
    "HostMemoryCapability",
    "TensorLFUPolicy",
    "TensorResidencyOps",
    "WeightLease",
]
