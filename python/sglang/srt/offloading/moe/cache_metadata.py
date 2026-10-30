"""Fixed-address assignment counters shared by cache policies and diagnostics."""

from dataclasses import dataclass, fields

from .resources import tensor_allocations


@dataclass
class CacheRouteStats:
    """Valid token-expert assignments, not unique experts or transferred bytes."""

    accesses: object
    hits: object
    misses: object

    @classmethod
    def allocate(cls, device):
        import torch

        return cls(
            *(torch.zeros((), dtype=torch.int64, device=device) for _ in range(3))
        )

    def clear(self):
        for field in fields(self):
            getattr(self, field.name).zero_()

    def record_counts(self, counts, hit, enabled):
        """Count native post-dispatch demand without expanding it into routes.

        Each layer contributes once. Counts include repeated assignments;
        masks and the fixed-address enabled flag exclude padding and capture.
        """
        import torch

        accesses = counts.sum(dtype=torch.int64)
        hits = torch.where(hit, counts, 0).sum(dtype=torch.int64)
        self.accesses.add_(torch.where(enabled, accesses, 0))
        self.hits.add_(torch.where(enabled, hits, 0))
        self.misses.add_(torch.where(enabled, accesses - hits, 0))

    def allocations(self):
        return tensor_allocations(
            "cache_stats.",
            ((field.name, getattr(self, field.name)) for field in fields(self)),
        )
