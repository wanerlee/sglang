"""Exercise real spawn/queue/barrier orchestration with no NPU imports."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import bench


def fake_worker(device, spec, options, scenarios, barrier, messages):
    messages.put({'kind': 'ready', 'device': device, 'metadata': {'device': device}})
    last_message = None
    for name, active in scenarios:
        for iteration in range(-options['warmup'], options['iterations']):
            barrier.wait(timeout=options['timeout'])
            if device in active and iteration >= 0:
                start = time.perf_counter_ns()
                message = {'kind': 'sample', 'row': {
                    'device': device, 'scenario': name, 'iteration': iteration,
                    'bytes': 4096, 'host_start_ns': start,
                    'host_submit_end_ns': start + 1000,
                    'host_end_ns': start + 2000, 'event_ms': 0.002,
                }, 'details': []}
                if name == scenarios[-1][0] and iteration == options['iterations'] - 1:
                    last_message = message
                else:
                    messages.put(message)
            barrier.wait(timeout=options['timeout'])
    # Deliberately invert completion delivery to stress the coordinator inbox.
    messages.put({'kind': 'closed', 'device': device, 'verified_bytes': 0})
    if last_message:
        messages.put(last_message)


def failed_worker(device, spec, options, scenarios, barrier, messages):
    messages.put({'kind': 'error', 'device': device, 'traceback': 'synthetic allocation failure'})


class CoordinatorTests(unittest.TestCase):
    def test_spawn_common_window_and_out_of_order_closed(self):
        options = {'timeout': 10, 'warmup': 1, 'iterations': 2}
        scenarios = [('solo:0', [0]), ('all', [0, 1])]
        with tempfile.TemporaryDirectory() as folder, patch.object(bench, 'worker', fake_worker):
            output = Path(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                bench.run(options, {}, [0, 1], scenarios, output)
            result = json.loads((output / 'results.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['samples']), 6)
            self.assertEqual({row['device'] for row in result['verification']}, {0, 1})
            self.assertEqual({row['scenario'] for row in result['summary']}, {'solo:0', 'all'})
            self.assertTrue((output / 'summary.csv').is_file())

    def test_worker_failure_never_produces_complete_report(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(bench, 'worker', failed_worker):
            output = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'synthetic allocation failure'):
                bench.run({'timeout': 10, 'warmup': 0, 'iterations': 1}, {}, [0], [('all', [0])], output)
            self.assertFalse((output / 'results.json').exists())
            self.assertFalse(json.loads((output / 'failure.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
