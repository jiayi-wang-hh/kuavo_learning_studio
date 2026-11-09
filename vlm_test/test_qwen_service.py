import base64
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
from vlm_test.qwen_service import RemoteQwenAdapter


class ServiceTests(unittest.TestCase):
    def test_monitor_ready_with_remote_adapter_without_local_model(self):
        from vlm_test.passive_vlm_monitor import adapter_ready_metadata
        with patch('vlm_test.qwen_service.urlopen', return_value=BytesIO(
                json.dumps(dict(model_path='/tmp/qwen', precision='nf4')).encode())):
            adapter = RemoteQwenAdapter('http://127.0.0.1:5560', '/tmp/qwen', 'nf4')
        self.assertFalse(hasattr(adapter, 'model'))
        metadata = adapter_ready_metadata(adapter)
        self.assertEqual(metadata['backend'], 'persistent_qwen_service')
        self.assertFalse(metadata['model_loaded_in_monitor'])

    def test_remote_frame_and_raw_output_preserved(self):
        metadata = dict(model_path=str(Path('/tmp/qwen').resolve()), precision='nf4')
        replies = [BytesIO(json.dumps(metadata).encode()), BytesIO(json.dumps(dict(raw='invalid output', timing={})).encode())]
        checks = []
        with patch('vlm_test.qwen_service.urlopen', side_effect=replies) as rpc:
            client = RemoteQwenAdapter('http://127.0.0.1:5560', '/tmp/qwen', 'nf4')
            raw, _ = client.generate_raw(Image.new('RGB', (3, 2), (10, 20, 30)), 'test', before_generate=lambda: checks.append(True))
            body = json.loads(rpc.call_args.args[0].data)
        self.assertEqual(raw, 'invalid output')  # strict parsing remains the monitor's responsibility
        self.assertEqual(len(checks), 1)
        with Image.open(BytesIO(base64.b64decode(body['image']))) as image:
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(body['prompt'], 'test')

    def test_mismatched_model_rejected(self):
        with patch('vlm_test.qwen_service.urlopen', return_value=BytesIO(b'{}')):
            with self.assertRaises(RuntimeError):
                RemoteQwenAdapter('http://127.0.0.1:5560', '/tmp/qwen', 'nf4')


if __name__ == '__main__':
    unittest.main()
