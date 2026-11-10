"""Run initialization against a torch_npu-style setter-only config object."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from bench import Runtime


class RuntimeInitTests(unittest.TestCase):
    def test_initialization_does_not_read_setter_only_npu_config(self):
        option_writes = []

        class WriteOnlyConfig:
            # Like torch_npu.npu.npu_config._npuConfig, forward the option to
            # the runtime without storing an instance attribute or a getter.
            def __setattr__(self, name, value):
                option_writes.append((name, value))

        config = WriteOnlyConfig()
        npu = SimpleNamespace(
            config=config, set_device=Mock(), set_compile_mode=Mock(),
            Stream=Mock(return_value=object()), synchronize=Mock(),
        )
        device = SimpleNamespace(type='npu', index=0)
        native = SimpleNamespace(api_info=lambda: {'queue_ordering': 'torch_npu_op_queue'},
                                 __file__='fake_native.so')
        layout = SimpleNamespace(host_nbytes=128, nbytes=64, alignment=64,
                                 dtype='int8', runtime_shape=(1, 8, 8))
        bank_owner = SimpleNamespace(view=Mock(side_effect=[object(), object()]))
        fmt = SimpleNamespace(layouts={'w13_weight': layout},
                              host_nbytes=lambda count: 128 * count,
                              allocate_bank=Mock(return_value=bank_owner))
        tensor = MagicMock()
        tensor.data_ptr.return_value = 0x10000
        capability = SimpleNamespace(device_pointer=lambda addr, size: addr + 0x100000)
        pool = SimpleNamespace(
            allocate=Mock(return_value=SimpleNamespace(tensor=tensor, capability=capability)),
            nbytes=256, physical_bytes=4096,
        )
        ops = SimpleNamespace(prepare=Mock())
        modules = {}

        def module(name, **attrs):
            parts = name.split('.')
            for end in range(1, len(parts) + 1):
                key = '.'.join(parts[:end])
                if key not in modules:
                    obj = modules[key] = ModuleType(key)
                    obj.__path__ = []
                    if end > 1:
                        setattr(modules['.'.join(parts[:end - 1])], parts[end - 1], obj)
            modules[name].__dict__.update(attrs)

        module('torch', __version__='fake', npu=npu, set_num_threads=Mock(), device=lambda _: device)
        module('torch_npu', __version__='fake')
        prefix = 'sglang.srt.offloading.moe'
        module(prefix + '.npu.native_acl', get_native_acl=Mock(return_value=native),
               AclHostTransferOps=Mock(return_value=ops))
        module(prefix + '.npu.w8a8_format', NpuV4W8A8Format=Mock(return_value=fmt))
        module(prefix + '.npu.storage', MemFabricHostPool=Mock(return_value=pool))
        module(prefix + '.storage', HostMemoryCapability=Mock())
        module(prefix + '.resources', ResourceKind=SimpleNamespace(EXECUTION_WEIGHTS='execution'))
        module(prefix + '.npu.prefill', _projection_copy_plans=Mock(return_value=(object(), object())))

        runtime = Runtime(0, {
            'hidden_size': 64, 'intermediate_size': 64, 'local_experts': 2,
            'source_layer_count': 1, 'host_pool_bytes': 4096, 'offloaded_layer_ids': [62],
        }, {'sglang_root': None, 'native_extension_dir': 'unused', 'passes': 1}, Mock())
        with patch.dict(sys.modules, modules), \
                patch.dict('os.environ', {'SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT': '0'}), \
                patch('bench.source_fingerprint', return_value={}), \
                patch('bench.process_placement', return_value={}):
            runtime.open()

        self.assertEqual(option_writes, [('allow_internal_format', True)])
        npu.set_compile_mode.assert_called_once_with(jit_compile=False)
        self.assertTrue(runtime.metadata['allow_internal_format_requested'])
        self.assertNotIn('allow_internal_format', runtime.metadata)
        self.assertEqual(runtime.metadata['sample_bytes'], 128)
        self.assertEqual(ops.prepare.call_count, 2)
        with self.assertRaises(AttributeError):
            _ = config.allow_internal_format


if __name__ == '__main__':
    unittest.main()
