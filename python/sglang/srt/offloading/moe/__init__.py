"""Opt-in MoE offloading framework; accelerator execution is backend supplied."""

from .cache_metadata import CacheMetadataAdapter, cache_update_enabled
from .cache_residency import CopySource, TensorResidencyOps
from .cache_runtime import (
    CacheExecutionContext,
    CacheRuntime,
    LocalRoutes,
)
from .config import ExpertCatalog, ExpertLayerSpec, MoEOffloadConfig, Phase
from .execution import (
    CacheUpdateCoordinator,
    ExecutionRequest,
    ExecutionReservation,
    ExpertWeightAccess,
    LocalExecution,
    SourcePolicy,
    UpdatePlan,
    WeightLease,
)
from .interfaces import ExecutionContext
from .offloader import MoEOffloaderV2
from .owner import MoEOffloadOwner
from .periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy, PeriodicLFUState
from .policy import TensorLFUPolicy
from .prefill import PrefillWeightPipeline
from .programs import (
    ComposedDFCProgram,
    DecodeLocalExpertProgram,
    DispatchOnceProgram,
    PrefillDispatchOnceProgram,
    PrefillLocalExpertProgram,
)

__all__ = [
    "CacheExecutionContext",
    "CacheMetadataAdapter",
    "CacheRuntime",
    "CacheUpdateCoordinator",
    "ComposedDFCProgram",
    "CopySource",
    "DecodeLocalExpertProgram",
    "DispatchOnceProgram",
    "ExecutionContext",
    "ExecutionRequest",
    "ExecutionReservation",
    "ExpertCatalog",
    "ExpertLayerSpec",
    "ExpertWeightAccess",
    "LocalExecution",
    "LocalRoutes",
    "MoEOffloadConfig",
    "MoEOffloadOwner",
    "MoEOffloaderV2",
    "PeriodicLFUConfig",
    "PeriodicLFUPolicy",
    "PeriodicLFUState",
    "Phase",
    "PrefillDispatchOnceProgram",
    "PrefillLocalExpertProgram",
    "PrefillWeightPipeline",
    "SourcePolicy",
    "TensorLFUPolicy",
    "TensorResidencyOps",
    "UpdatePlan",
    "WeightLease",
    "cache_update_enabled",
]
