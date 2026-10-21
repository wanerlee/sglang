"""Opt-in MoE offloading framework; accelerator execution is backend supplied."""

from .cache_metadata import CacheMetadataAdapter, CacheSetDelta, cache_update_enabled
from .cache_residency import CopySource, TensorResidencyOps
from .cache_runtime import (
    CacheExecutionContext,
    CacheLocalExpertProgram,
    CacheRuntime,
    LocalRoutes,
)
from .config import ExpertCatalog, ExpertLayerSpec, MoEOffloadConfig, Phase
from .offloader import MoEOffloaderV2
from .owner import MoEOffloadOwner
from .periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy, PeriodicLFUState
from .policy import TensorLFUPolicy

__all__ = [
    "CacheExecutionContext",
    "CacheLocalExpertProgram",
    "CacheMetadataAdapter",
    "CacheRuntime",
    "CacheSetDelta",
    "CopySource",
    "ExpertCatalog",
    "ExpertLayerSpec",
    "LocalRoutes",
    "MoEOffloadConfig",
    "MoEOffloaderV2",
    "MoEOffloadOwner",
    "Phase",
    "PeriodicLFUConfig",
    "PeriodicLFUPolicy",
    "PeriodicLFUState",
    "TensorLFUPolicy",
    "TensorResidencyOps",
    "cache_update_enabled",
]
