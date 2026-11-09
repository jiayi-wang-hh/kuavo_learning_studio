"""Local persistent Qwen service; no ROS or robot interfaces."""
from __future__ import annotations
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


class RemoteQwenAdapter:
    def __init__(self, url, model_path, precision):
        self.url = url.rstrip('/')
        with urlopen(self.url + '/metadata', timeout=5) as response:
            metadata = json.load(response)
        if metadata != {'model_path': str(Path(model_path).resolve()), 'precision': precision}:
            raise RuntimeError(f'Qwen service configuration mismatch: {metadata}')

    def generate_raw(self, image, prompt, *, max_new_tokens=24, before_generate=None):
        buffer = BytesIO()
        image.save(buffer, format='PNG', compress_level=0)
        payload = json.dumps(dict(image=base64.b64encode(buffer.getvalue()).decode(), prompt=prompt,
                                  max_new_tokens=max_new_tokens, deadline_mono=time.monotonic() + .5)).encode()
        if before_generate:
            before_generate()
        request = Request(self.url + '/generate', data=payload, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        return result['raw'], result['timing']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('/media/data/jiayi/hf_model/Qwen3.5-9B'))
    parser.add_argument('--precision', choices=['nf4', 'bf16'], default='nf4')
    parser.add_argument('--port', type=int, default=5560)
    args = parser.parse_args()
    lock = threading.Lock()
    metadata = dict(model_path=str(args.model_path.resolve()), precision=args.precision)
    adapter = None

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, metadata) if self.path == '/metadata' else self.reply(404, {'error': 'unknown endpoint'})

        def do_POST(self):
            if self.path != '/generate':
                self.reply(404, {'error': 'unknown endpoint'})
                return
            if not lock.acquire(blocking=False):
                self.reply(409, {'error': 'busy; frames are never queued'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                deadline = float(body['deadline_mono'])
                def fresh():
                    if not time.monotonic() <= deadline <= time.monotonic() + 1:
                        raise ValueError('Request expired before generation')
                fresh()
                from PIL import Image
                image = Image.open(BytesIO(base64.b64decode(body['image']))).convert('RGB')
                raw, timing = adapter.generate_raw(image, body['prompt'], max_new_tokens=24, before_generate=fresh)
                self.reply(200, dict(raw=raw, timing=timing))
            except Exception as exc:
                self.reply(500, dict(error=f'{type(exc).__name__}: {exc}'))
            finally:
                lock.release()

        def log_message(self, *_):
            pass

    # Bind before loading, but do not accept requests until the model is ready.
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    try:
        from vlm_test.passive_vlm_monitor import load_adapter
        print('Loading Qwen once...', flush=True)
        adapter = load_adapter(args.model_path, args.precision)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f'Qwen ready: http://127.0.0.1:{args.port}\nEnter = 保持加载；q / quit / exit = 关闭并释放显存；Ctrl+C 也可关闭。', flush=True)
        try:
            while True:
                try:
                    command = input('Qwen [Enter=保持 / q=关闭]> ').strip().lower()
                except EOFError:
                    print('终端输入已关闭；服务继续保持，使用 Ctrl+C 或 SIGTERM 关闭。', flush=True)
                    threading.Event().wait()
                    break
                if command in ['q', 'quit', 'exit']:
                    break
                print('Qwen 保持加载，可重复运行 Demo。', flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
    finally:
        server.server_close()
    print('Qwen 服务已关闭，进程退出后释放显存。', flush=True)


if __name__ == '__main__':
    main()
