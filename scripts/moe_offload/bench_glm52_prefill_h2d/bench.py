#!/usr/bin/env python3
"""Synthetic whole-stage H2D using the deployed prefill offloader's exact API.

Only config.json is read. No model, checkpoint tensors, HCCL group or FFN runs.
NPU imports are delayed so --dry-run and CPU unit tests work on a laptop.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback
from types import SimpleNamespace

from bench_spec import build_spec, summarize_samples


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-config', required=True, help='config.json or model directory; no weights read')
    p.add_argument('--sglang-root', default=os.getenv('SGLANG_ROOT'), help='matching SGLang checkout; otherwise use installed package')
    p.add_argument('--pp-partition', default=os.getenv('SGLANG_PP_LAYER_PARTITION', '18,20,24,16'))
    p.add_argument('--pp-rank', type=int, default=3, help='zero-based PP stage index')
    p.add_argument('--ep-size', type=int, default=4)
    p.add_argument('--devices', default='0,1,2,3', help='visible logical NPU IDs; subset allowed, at most ep-size')
    p.add_argument('--source-layers', type=int, help='default: distinct sources for every routed layer; smaller means source reuse')
    p.add_argument('--host-pool-gib', type=int, default=56, help='MemFabric physical reservation PER worker, matching serving launcher')
    p.add_argument('--native-extension-dir', default='/tmp/sglang-glm52-prefill-pp4-cp4-runtime/native_acl')
    p.add_argument('--build-native', action='store_true', help='explicitly build/rebuild the production ACL shim, then run')
    p.add_argument('--build-only', action='store_true', help='build production ACL shim and exit')
    p.add_argument('--cann-home', help='CANN SDK root for native build')
    p.add_argument('--scenarios', choices=('all', 'solo', 'solo,all'), default='solo,all')
    p.add_argument('--passes', type=int, default=1, help='complete stage traversals per measurement')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--projection-timing', action='store_true', help='extra W13/W2 timing events; changes instrumentation overhead')
    p.add_argument('--verify', choices=('sample', 'full', 'none'), default='sample', help='untimed verification of every source layer and final banks')
    p.add_argument('--timeout', type=float, default=900, help='timeout per initialization/measurement/cleanup phase')
    p.add_argument('--output', help='new report directory (must not already exist)')
    p.add_argument('--dry-run', action='store_true', help='standard-library-only capacity plan, no NPU/import/build')
    return p


def setup_path(root):
    if root:
        path = Path(root).expanduser().resolve() / 'python'
        if not (path / 'sglang/srt/offloading/moe/npu/native_acl.py').is_file():
            raise ValueError(f'Not the expected SGLang checkout: {path}')
        sys.path.insert(0, str(path))


def source_fingerprint(module):
    base = Path(module.__file__).resolve().parent
    names = ('native_acl.py', 'native/acl_batch.cpp', 'native/acl_batch.h',
             'prefill.py', 'format.py', 'w8a8_format.py', 'storage.py')
    return {name: {'path': str(base / name), 'sha256': hashlib.sha256((base / name).read_bytes()).hexdigest()}
            for name in names}


def byte_pattern(source_index, expert, component_index):
    # Reproducible byte payload, deliberately not a numerical FFN input.
    return 1 + (source_index * 67 + expert * 7 + component_index * 31) % 251


def verification_ranges(nbytes, full=False, block=8 * 1024**2):
    if full:
        return [(start, min(block, nbytes - start)) for start in range(0, nbytes, block)]
    size = min(64, nbytes)
    return [(start, size) for start in sorted({0, (nbytes - size) // 2, nbytes - size})]


def process_placement(pool):
    result = {'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    status = Path('/proc/self/status')
    if status.is_file():
        result['allowed'] = [s for s in status.read_text().splitlines() if s.startswith(('Cpus_allowed_list:', 'Mems_allowed_list:'))]
    # Locate all VMAs overlapping the registered slab; NUMA data is diagnostic,
    # not a claim that CPU affinity determines physical memory placement.
    maps, numa = Path('/proc/self/maps'), Path('/proc/self/numa_maps')
    if maps.is_file() and numa.is_file():
        lo, hi = pool.capability.host_address, pool.capability.host_address + pool.nbytes
        starts = set()
        for line in maps.read_text().splitlines():
            a, b = [int(x, 16) for x in line.split()[0].split('-')]
            if a < hi and lo < b:
                starts.add(a)
        result['source_numa_maps'] = [s for s in numa.read_text().splitlines() if int(s.split()[0], 16) in starts]
    return result


class Runtime:
    def __init__(self, device_id, spec, options, notify):
        self.device_id, self.spec, self.options, self.notify = device_id, spec, options, notify
        self.pool = None
        self.ops = None
        self.keepalive, self.sources, self.plans = [], [], []
        self.bank_owner = self.banks = None

    def open(self):
        setup_path(self.options['sglang_root'])
        import torch
        import torch_npu
        from sglang.srt.offloading.moe.npu import native_acl
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps, get_native_acl
        from sglang.srt.offloading.moe.npu.w8a8_format import NpuV4W8A8Format
        from sglang.srt.offloading.moe.npu.storage import MemFabricHostPool
        from sglang.srt.offloading.moe.storage import HostMemoryCapability
        from sglang.srt.offloading.moe.resources import ResourceKind
        from sglang.srt.offloading.moe.npu.prefill import _projection_copy_plans

        self.torch = torch
        torch.set_num_threads(1)
        torch.npu.set_device(self.device_id)
        if os.getenv('SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT') == '1':
            raise ValueError('Unset SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT: this benchmark requires production NZ weights')
        # Match SGLang init_npu_backend's format/compile settings without
        # importing the model server or globally patching CUDA APIs.
        torch.npu.config.allow_internal_format = True
        torch.npu.set_compile_mode(jit_compile=False)
        self.device = torch.device(f'npu:{self.device_id}')
        self.native = get_native_acl(self.options['native_extension_dir'])
        self.notify('probing production NZ layout')
        self.fmt = NpuV4W8A8Format(self.spec['hidden_size'], self.spec['intermediate_size'],
                                  self.device, native_acl=self.native)
        experts = self.spec['local_experts']
        host_bytes = self.fmt.host_nbytes(experts) * self.spec['source_layer_count']
        if host_bytes > self.spec['host_pool_bytes']:
            raise ValueError('Probed physical Host layout exceeds --host-pool-gib')
        self.bank_owner = self.fmt.allocate_bank(2 * experts, ResourceKind.EXECUTION_WEIGHTS, 'h2d_bench')
        self.banks = (self.bank_owner.view(0, experts), self.bank_owner.view(experts, experts))
        self.stream = torch.npu.Stream(device=self.device)
        self.notify(f'allocating/registering Host pool: {host_bytes / 1024**3:.3f} GiB payload')
        self.pool = MemFabricHostPool(host_bytes, self.device_id,
                                     physical_bytes=self.spec['host_pool_bytes'], native_acl=self.native)
        for source_index in range(self.spec['source_layer_count']):
            caps = {}
            for ci, (name, layout) in enumerate(self.fmt.layouts.items()):
                allocation = self.pool.allocate(layout.host_nbytes * experts, alignment=layout.alignment)
                rows = allocation.tensor.view(experts, layout.host_nbytes)
                for expert in range(experts):
                    rows[expert].fill_(byte_pattern(source_index, expert, ci))
                base = allocation.tensor.data_ptr()
                caps[name] = tuple(HostMemoryCapability(
                    base + expert * layout.host_nbytes, layout.nbytes, True,
                    allocation.capability.device_pointer(base + expert * layout.host_nbytes, layout.nbytes),
                    self.pool) for expert in range(experts))
            self.sources.append(SimpleNamespace(component_capabilities=caps))
            self.notify(f'initialized source layer {source_index + 1}/{self.spec["source_layer_count"]}')
        self.ops = AclHostTransferOps(self.native, self.keepalive.append)
        for index, _ in enumerate(self.spec['offloaded_layer_ids']):
            pair = _projection_copy_plans(self.sources[index % len(self.sources)], self.banks[index % 2], self.fmt.layouts)
            self.plans.append(pair)
            for plan in pair:
                self.ops.prepare(plan, self.device)
        torch.npu.synchronize(self.device)
        self.actual_bytes_per_layer = sum(x.nbytes for x in self.fmt.layouts.values()) * experts
        self.sample_bytes = self.actual_bytes_per_layer * len(self.plans) * self.options['passes']
        layout_info = [{
            'name': name, 'dtype': str(layout.dtype), 'runtime_shape': list(layout.runtime_shape),
            'nbytes_per_expert': layout.nbytes, 'host_stride_per_expert': layout.host_nbytes,
        } for name, layout in self.fmt.layouts.items()]
        self.metadata = {
            'device': self.device_id, 'pid': os.getpid(), 'torch': torch.__version__,
            'torch_npu': torch_npu.__version__, 'native_api': dict(self.native.api_info()),
            'sources': source_fingerprint(native_acl), 'layout': layout_info,
            'actual_bytes_per_layer': self.actual_bytes_per_layer, 'sample_bytes': self.sample_bytes,
            'host_slab_bytes': self.pool.nbytes, 'host_physical_reservation_bytes': self.pool.physical_bytes,
            'two_bank_bytes': 2 * self.actual_bytes_per_layer,
            'native_extension': str(getattr(self.native, '__file__', 'unknown')),
            # torch_npu 2.10 exposes a setter-only config attribute. Record the
            # requested option; the format probe above verifies actual NZ layout.
            'allow_internal_format_requested': True,
            'environment': {k: os.getenv(k) for k in (
                'TASK_QUEUE_ENABLE', 'ASCEND_LAUNCH_BLOCKING', 'ASCEND_RT_VISIBLE_DEVICES',
                'PYTORCH_NPU_ALLOC_CONF', 'ASCEND_HOME_PATH', 'MEMFABRIC_HYBRID_EXTEND_LIB_PATH')},
            'placement': process_placement(self.pool),
        }
        self.notify('initialization complete')

    def copy_layer(self, index):
        for plan in self.plans[index]:
            self.ops.materialize_into(plan, self.banks[index % 2], self.stream)

    def verify_bank(self, index):
        full = self.options['verify'] == 'full'
        source_index, bank = index % len(self.sources), self.banks[index % 2]
        # Use raw D2H to inspect NZ physical bytes, not a format-converting .cpu().
        scratch = self.torch.empty(8 * 1024**2 if full else 64, dtype=self.torch.uint8, device='cpu')
        checked = 0
        for ci, (name, layout) in enumerate(self.fmt.layouts.items()):
            for expert, region in enumerate(bank.component_regions[name]):
                expected = byte_pattern(source_index, expert, ci)
                for offset, size in verification_ranges(layout.nbytes, full):
                    self.native.memcpy_sync(scratch.data_ptr(), scratch.numel(), region.address + offset,
                                            size, 'device_to_host', self.device_id)
                    if not bool(scratch[:size].eq(expected).all()):
                        raise RuntimeError(f'Byte verification failed layer_index={index} expert={expert} component={name} offset={offset}')
                    checked += size
        return checked

    def verify_sources(self):
        if self.options['verify'] == 'none':
            return 0
        checked = 0
        # Verify each layer before either bank is reused. This is outside timing.
        for index in range(len(self.plans)):
            self.copy_layer(index)
            self.stream.synchronize()
            self.keepalive.clear()
            checked += self.verify_bank(index)
        self.notify(f'untimed {self.options["verify"]} validation passed: {checked} bytes checked')
        return checked

    def measure(self):
        torch = self.torch
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        timed = []
        # Allocate instrumentation before timing; only record events in the window.
        if self.options['projection_timing']:
            for pass_id in range(self.options['passes']):
                for index in range(len(self.plans)):
                    timed.append((pass_id, index, tuple(torch.npu.Event(enable_timing=True) for _ in range(3))))
        h0 = time.perf_counter_ns()
        start.record(self.stream)
        for pass_id in range(self.options['passes']):
            for index, (w13, w2) in enumerate(self.plans):
                events = timed[pass_id * len(self.plans) + index][2] if timed else None
                if events:
                    events[0].record(self.stream)
                self.ops.materialize_into(w13, self.banks[index % 2], self.stream)
                if events:
                    events[1].record(self.stream)
                self.ops.materialize_into(w2, self.banks[index % 2], self.stream)
                if events:
                    events[2].record(self.stream)
        end.record(self.stream)
        h1 = time.perf_counter_ns()
        self.stream.synchronize()
        h2 = time.perf_counter_ns()
        result = {'device': self.device_id, 'bytes': self.sample_bytes, 'host_start_ns': h0,
                  'host_submit_end_ns': h1, 'host_end_ns': h2, 'event_ms': start.elapsed_time(end)}
        details = []
        for pass_id, index, events in timed:
            for group, a, b, plan in (('w13', events[0], events[1], self.plans[index][0]),
                                      ('w2', events[1], events[2], self.plans[index][1])):
                ms = a.elapsed_time(b)
                nbytes = sum(r.nbytes for r in plan.regions)
                details.append({'device': self.device_id, 'pass': pass_id,
                                'layer': self.spec['offloaded_layer_ids'][index], 'projection': group,
                                'bytes': nbytes, 'event_ms': ms, 'gbps': nbytes / (ms * 1e6) if ms > 0 else None})
        self.keepalive.clear()
        return result, details

    def verify_final(self):
        if self.options['verify'] == 'none':
            return 0
        return sum(self.verify_bank(i) for i in range(max(0, len(self.plans) - 2), len(self.plans)))

    def close(self):
        if hasattr(self, 'torch'):
            # If this fails, do not unregister/free memory still used by the device.
            self.torch.npu.synchronize(self.device_id)
        self.keepalive.clear()
        if self.ops:
            self.ops.clear_cache()
        self.plans.clear()
        self.sources.clear()
        self.bank_owner = self.banks = None
        if self.pool:
            self.pool.close()
            self.pool = None


def worker(device, spec, options, scenarios, barrier, messages):
    runtime = None
    try:
        runtime = Runtime(device, spec, options, lambda s: messages.put({'kind': 'progress', 'device': device, 'text': s}))
        runtime.open()
        checked = runtime.verify_sources()
        messages.put({'kind': 'ready', 'device': device, 'metadata': runtime.metadata, 'verified_bytes': checked})
        for name, active in scenarios:
            for iteration in range(-options['warmup'], options['iterations']):
                barrier.wait(timeout=options['timeout'])
                if device in active:
                    row, details = runtime.measure()
                    if iteration >= 0:
                        messages.put({'kind': 'sample', 'row': dict(row, scenario=name, iteration=iteration),
                                      'details': [dict(d, scenario=name, iteration=iteration) for d in details]})
                barrier.wait(timeout=options['timeout'])
        checked += runtime.verify_final()
        runtime.close()
        runtime = None
        messages.put({'kind': 'closed', 'device': device, 'verified_bytes': checked})
    except BaseException:
        messages.put({'kind': 'error', 'device': device, 'traceback': traceback.format_exc()})
        barrier.abort()
        if runtime is not None:
            try:
                runtime.close()
            except BaseException:
                messages.put({'kind': 'error', 'device': device, 'traceback': 'Cleanup/drain failed; allocation was not deliberately freed.\n' + traceback.format_exc()})
        raise


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(options, spec, devices, scenarios, output):
    context = mp.get_context('spawn')
    messages = context.Queue()
    barrier = context.Barrier(len(devices) + 1)
    processes = [context.Process(target=worker, args=(d, spec, options, scenarios, barrier, messages)) for d in devices]
    samples, details, metadata, closed = [], [], [], []
    inbox = []
    live_log = (output / 'events.jsonl').open('w')

    def receive(kind, count):
        found = []
        deadline = time.monotonic() + options['timeout']
        while len(found) < count:
            match = next((i for i, msg in enumerate(inbox) if msg['kind'] == kind), None)
            if match is not None:
                found.append(inbox.pop(match))
                continue
            try:
                msg = messages.get(timeout=min(1, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                dead = [p for p in processes if p.exitcode not in (None, 0)]
                if dead or time.monotonic() >= deadline:
                    raise RuntimeError(f'Worker failure or timeout while awaiting {kind}; inspect events.jsonl and stderr')
                continue
            live_log.write(json.dumps(msg) + '\n')
            live_log.flush()
            if msg['kind'] == 'error':
                raise RuntimeError(f'Device {msg["device"]}:\n{msg["traceback"]}')
            if msg['kind'] == 'progress':
                print(f'[device {msg["device"]}] {msg["text"]}', flush=True)
                continue
            if msg['kind'] == kind:
                found.append(msg)
            else:
                # Queue ordering is per producer, not across processes. A fast
                # inactive rank may finish cleanup before another rank's sample
                # is delivered by its feeder thread.
                inbox.append(msg)
        return found

    try:
        for process in processes:
            process.start()
        metadata = [m['metadata'] for m in receive('ready', len(devices))]
        if {m['device'] for m in metadata} != set(devices):
            raise RuntimeError('Missing or duplicated device initialization')
        (output / 'devices.json').write_text(json.dumps(metadata, indent=2))
        for name, active in scenarios:
            for iteration in range(-options['warmup'], options['iterations']):
                print(f'{name} {"warmup" if iteration < 0 else "sample"} {iteration}', flush=True)
                barrier.wait(timeout=options['timeout'])
                barrier.wait(timeout=options['timeout'])
                if iteration >= 0:
                    batch = receive('sample', len(active))
                    if ({m['row']['device'] for m in batch} != set(active)
                            or any(m['row']['scenario'] != name or m['row']['iteration'] != iteration for m in batch)):
                        raise RuntimeError('Incomplete or mismatched measurement round')
                    for msg in batch:
                        samples.append(msg['row'])
                        details.extend(msg['details'])
                    write_csv(output / 'samples.csv', samples)
        closed = receive('closed', len(devices))
        if {m['device'] for m in closed} != set(devices):
            raise RuntimeError('Missing device cleanup acknowledgment')
        for process in processes:
            process.join(timeout=5)
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('A worker did not exit cleanly')
        report = summarize_samples(samples)
        for name in ('summary', 'ranks', 'iterations'):
            write_csv(output / f'{name}.csv', report[name])
        write_csv(output / 'projections.csv', details)
        result = {'complete': True, 'spec': spec, 'options': options, 'devices': metadata,
                  'verification': closed, 'samples': samples, 'projection_samples': details, **report}
        (output / 'results.json').write_text(json.dumps(result, indent=2))
        for row in report['summary']:
            print(f'{row["scenario"]}: total={row["median_total_gbps"]:.2f} GB/s '
                  f'window={row["median_window_ms"]:.3f} ms '
                  f'CV={row["window_cv_pct"]:.2f}% skew_max={row["max_start_skew_ms"]:.3f} ms')
        print(f'COMPLETE: {output}', flush=True)
    except BaseException:
        barrier.abort()
        # Preserve queued device errors even when the barrier reports failure first.
        while True:
            try:
                msg = messages.get(timeout=0.1)
            except queue.Empty:
                break
            live_log.write(json.dumps(msg) + '\n')
            if msg.get('kind') == 'error':
                print(msg['traceback'], file=sys.stderr, flush=True)
        live_log.flush()
        write_csv(output / 'samples.csv', samples)
        (output / 'failure.json').write_text(json.dumps({'complete': False, 'error': traceback.format_exc(),
                                                        'spec': spec, 'options': options, 'devices': metadata}, indent=2))
        raise
    finally:
        # Let error handlers drain/free normally before terminating any stuck
        # worker. The grace budget is shared across all workers.
        grace_end = time.monotonic() + 5
        for process in processes:
            if process.pid is not None:
                process.join(timeout=max(0, grace_end - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
        live_log.close()


def main(argv=None):
    args = parser().parse_args(argv)
    devices = [int(x) for x in args.devices.split(',')]
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0 or len(devices) > args.ep_size:
        raise ValueError('Use distinct nonnegative logical devices, at most --ep-size workers')
    if args.passes <= 0 or args.iterations <= 0 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Invalid passes/iterations/warmup/timeout')
    spec = build_spec(args.model_config, args.pp_partition, args.pp_rank, args.ep_size,
                      source_layers=args.source_layers, host_pool_gib=args.host_pool_gib)
    spec['devices'] = devices
    spec['node_host_reservation_bytes'] = spec['host_pool_bytes'] * len(devices)
    spec['benchmark_boundary'] = 'copy-only; no GMM, CP/EP/PP communication or consumer waits; two banks on one FIFO copy stream'
    print(json.dumps(spec, indent=2), flush=True)
    if args.dry_run:
        return
    if os.getenv('ASCEND_LAUNCH_BLOCKING') == '1':
        raise ValueError('Unset ASCEND_LAUNCH_BLOCKING (or set 0) for the asynchronous copy benchmark')
    os.environ.setdefault('TASK_QUEUE_ENABLE', '1')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    options = vars(args).copy()
    options['native_extension_dir'] = str(Path(args.native_extension_dir).expanduser().resolve())
    if args.sglang_root:
        options['sglang_root'] = str(Path(args.sglang_root).expanduser().resolve())
    if args.build_native or args.build_only:
        setup_path(options['sglang_root'])
        from sglang.srt.offloading.moe.npu.native_acl import build_native_acl
        module = build_native_acl(options['native_extension_dir'], cann_home=args.cann_home, verbose=True)
        print(dict(module.api_info()), flush=True)
        if args.build_only:
            return
    output = Path(args.output or (Path(__file__).resolve().parent / 'results' / time.strftime('%Y%m%d_%H%M%S'))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps({'spec': spec, 'options': options}, indent=2))
    scenarios = [(f'solo:{d}', [d]) for d in devices] if 'solo' in args.scenarios else []
    if 'all' in args.scenarios:
        scenarios.append(('all', devices))
    run(options, spec, devices, scenarios, output)


if __name__ == '__main__':
    main()
