"""CPU event/context doubles and tensor-operation replay for MoE unit tests."""

from contextlib import nullcontext

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.transfer import Completion, TransferTicket


class Fence(Completion):
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def wait_on(self, stream):
        self.calls.append(("wait", self.name, stream))

    def synchronize(self):
        raise AssertionError("No host synchronization in layer planning")

    def done(self):
        raise AssertionError("No completion polling in layer planning")


class Context:
    phase = Phase.DECODE
    capturing = True
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self):
        self.invocation_id = object()
        self.calls, self.retained = [], []

    def retain(self, value):
        self.retained.append(value)

    retain_completion = retain

    def record_compute_completion(self):
        name = "compute-" + str(len(self.calls))
        self.calls.append(("record", name))
        return TransferTicket(Fence(self.calls, name))

    def copied(self):
        return TransferTicket(Fence(self.calls, "copy-done"))

    def wait_for(self, ticket, stream):
        ticket.completion.wait_on(stream)

    def temporary_scope(self):
        return nullcontext()


def load_hot(state, keys):
    state.key_to_slot.fill_(-1)
    state.slot_to_key.copy_(torch.tensor(keys, dtype=torch.int64))
    state.slot_valid.copy_(state.slot_to_key >= 0)
    for slot, key in enumerate(keys):
        if key >= 0:
            state.key_to_slot[key] = slot
    state.slot_generation.add_(1)


class TensorTrace(TorchDispatchMode):
    """Replay recorded CPU tensor operations without rerunning planner Python.

    This checks dynamic data dependencies, not an NPU graph/event implementation.
    """

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        self.calls.append((func, args, kwargs, output))
        return output

    def replay(self):
        values = {}

        def remap(value):
            return (
                values.get(id(value), value)
                if isinstance(value, torch.Tensor)
                else value
            )

        for func, args, kwargs, original in self.calls:
            result = func(*tree_map(remap, args), **tree_map(remap, kwargs))

            def save(old, new):
                if isinstance(old, torch.Tensor):
                    values[id(old)] = new

            tree_map(save, original, result)
        return values
