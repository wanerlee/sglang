"""NPU metadata for packed decode, physical Hot slots, and whole-layer prefill."""

import triton
import triton.language as tl


@triton.jit
def prepare_packed_copies(
    Counts,
    Error,
    Enabled,
    Access,
    Sources,
    Groups,
    OutSources,
    EXPERTS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    START: tl.constexpr,
    COMPONENT_COUNT: tl.constexpr,
    SKIP_OFFSETS: tl.constexpr,
    CUMULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPONENT_MAP: tl.constexpr = 0,
    COMPONENT_BITS: tl.constexpr = 0,
):
    # Exactly one program owns the fault latch and this layer's access counters.
    e = tl.arange(0, BLOCK)
    raw = tl.load(Counts + e, e < EXPERTS, 0)
    count = raw
    if CUMULATIVE:
        previous = tl.load(Counts + e - 1, (e > 0) & (e < EXPERTS), 0)
        count = raw - previous
    bad = tl.sum(((count < 0) & (e < EXPERTS)).to(tl.int32), 0) > 0
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    # group_pack_copy expects per-expert counts, never cumulative boundaries.
    # Its packed output becomes GMM's count vector after the copy completes.
    tl.store(Groups + e, tl.where(error == 0, count, 0), e < EXPERTS)
    enabled = tl.load(Enabled) != 0
    accesses = tl.load(Access + START + e, e < EXPERTS, 0)
    tl.store(
        Access + START + e,
        accesses + tl.where(enabled & (error == 0), count, 0),
        e < EXPERTS,
    )
    for position in tl.static_range(COMPONENT_COUNT):
        # Triton-Ascend 3.2.2 cannot subscript a tuple wrapped in constexpr.
        # Scalar shifts fold at compile time, without extra device loads.
        if COMPONENT_BITS > 0:
            component = (COMPONENT_MAP >> (position * COMPONENT_BITS)) & (
                (1 << COMPONENT_BITS) - 1
            )
        else:
            component = (position // 2) * 3 + position % 2 if SKIP_OFFSETS else position
        source = tl.load(
            Sources + component * SOURCE_STRIDE + START + e, e < EXPERTS, 0
        )
        tl.store(OutSources + position * EXPERTS + e, source, e < EXPERTS)


@triton.jit
def _store_slot_count_pairs(
    Groups, physical, count, needed, EXPERTS: tl.constexpr, BLOCK: tl.constexpr
):
    position = tl.cumsum(needed.to(tl.int32), 0) - 1
    pair = tl.arange(0, 2 * BLOCK)
    match = (pair[:, None] // 2 == position[None, :]) & needed[None, :]
    values = tl.where(
        pair[:, None] % 2 == 0,
        physical[None, :].to(tl.float32),
        count[None, :].to(tl.float32),
    )
    packed = tl.sum(tl.where(match, values, 0), 1).to(tl.int64)
    # Clear stale entries in the mutable prefix. The rest stays zero.
    tl.store(Groups + pair, packed, pair < 2 * EXPERTS)


@triton.jit
def prepare_prefill_counts(
    Counts,
    Error,
    Enabled,
    Access,
    OutCounts,
    EXPERTS: tl.constexpr,
    CUMULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
    TRACK_USAGE: tl.constexpr = True,
    Groups=None,
    ResidentBase=0,
    ColdBase=0,
    RESIDENT_EXPERTS: tl.constexpr = 0,
):
    """Normalize logical counts and optionally select physical weight slots."""
    e = tl.arange(0, BLOCK)
    inside = e < EXPERTS
    count = tl.load(Counts + e, inside, 0)
    if CUMULATIVE:
        previous = tl.load(Counts + e - 1, (e > 0) & inside, 0)
        count = count - previous
    bad = tl.sum(((count < 0) & inside).to(tl.int32), 0) > 0
    if RESIDENT_EXPERTS:
        # Slot/count pairs use the float32 reduction shared with decode.
        bad = bad | (tl.sum((count > 16777216).to(tl.int32), 0) > 0)
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    count = tl.where(error == 0, count, 0)
    tl.store(OutCounts + e, count, inside)
    if RESIDENT_EXPERTS:
        physical = tl.where(
            e < RESIDENT_EXPERTS, ResidentBase + e, ColdBase + e - RESIDENT_EXPERTS
        )
        _store_slot_count_pairs(
            Groups, physical, count, inside & (count > 0), EXPERTS, BLOCK
        )
    if TRACK_USAGE:
        accesses = tl.load(Access + e, inside, 0)
        enabled = tl.load(Enabled) != 0
        tl.store(Access + e, accesses + tl.where(enabled, count, 0), inside)


@triton.jit
def prepare_fixed_copies(
    Counts,
    Error,
    Enabled,
    Access,
    Sources,
    Lengths,
    OutCounts,
    OutSources,
    OutLengths,
    EXPERTS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    START: tl.constexpr,
    COMPONENT_COUNT: tl.constexpr,
    SKIP_OFFSETS: tl.constexpr,
    CUMULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECT_HOT: tl.constexpr = False,
    HotSlots=None,
    HOT_CAPACITY: tl.constexpr = 0,
    Groups=None,
    ExecutionKeys=None,
    KEY_START: tl.constexpr = 0,
    COMPONENT_MAP: tl.constexpr = 0,
    COMPONENT_BITS: tl.constexpr = 0,
):
    # Exactly one program owns the fault latch and this layer's access counters.
    e = tl.arange(0, BLOCK)
    raw = tl.load(Counts + e, e < EXPERTS, 0)
    count = raw
    if CUMULATIVE:
        previous = tl.load(Counts + e - 1, (e > 0) & (e < EXPERTS), 0)
        count = raw - previous
    bad = tl.sum(((count < 0) & (e < EXPERTS)).to(tl.int32), 0) > 0
    if DIRECT_HOT:
        # Counts are packed through float32 below; reject inexact integers.
        bad = bad | (tl.sum((count > 16777216).to(tl.int32), 0) > 0)
        hot_slot = tl.load(HotSlots + KEY_START + e, e < EXPERTS, -1)
        bad = bad | (
            tl.sum(((hot_slot < -1) | (hot_slot >= HOT_CAPACITY)).to(tl.int32), 0) > 0
        )
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    tl.store(OutCounts + e, tl.where(error == 0, raw, 0), e < EXPERTS)
    enabled = tl.load(Enabled) != 0
    accesses = tl.load(Access + START + e, e < EXPERTS, 0)
    tl.store(
        Access + START + e,
        accesses + tl.where(enabled & (error == 0), count, 0),
        e < EXPERTS,
    )
    needed = (count > 0) & (error == 0)
    copy_needed = needed
    if DIRECT_HOT:
        hit = hot_slot >= 0
        physical = tl.where(hit, EXPERTS + hot_slot, e)
        _store_slot_count_pairs(Groups, physical, count, needed, EXPERTS, BLOCK)
        previous = tl.load(ExecutionKeys + e, e < EXPERTS, -1)
        copy_needed = needed & ~hit & (previous != KEY_START + e)
        # Hot reads must leave the Cold slot's identity untouched. Both weight
        # copy phases finish before another layer can reuse this identity.
        tl.store(ExecutionKeys + e, KEY_START + e, (e < EXPERTS) & copy_needed)
    for position in tl.static_range(COMPONENT_COUNT):
        # Triton-Ascend 3.2.2 cannot subscript a tuple wrapped in constexpr.
        # Scalar shifts fold at compile time, without extra device loads.
        if COMPONENT_BITS > 0:
            component = (COMPONENT_MAP >> (position * COMPONENT_BITS)) & (
                (1 << COMPONENT_BITS) - 1
            )
        else:
            component = (position // 2) * 3 + position % 2 if SKIP_OFFSETS else position
        source = tl.load(
            Sources + component * SOURCE_STRIDE + START + e, e < EXPERTS, 0
        )
        length = tl.load(Lengths + component * EXPERTS + e, e < EXPERTS, 0)
        tl.store(OutSources + position * EXPERTS + e, source, e < EXPERTS)
        tl.store(
            OutLengths + position * EXPERTS + e,
            tl.where(copy_needed, length, 0),
            e < EXPERTS,
        )
