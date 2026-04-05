import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests


@dataclass
class RouterArgs:
    host: str = "127.0.0.1"
    port: int = 30000
    balance_abs_threshold: int = 0
    prometheus_port: Optional[int] = None
    prometheus_host: str = "127.0.0.1"
    log_level: str = "warn"


class _RouterState:
    def __init__(self):
        self._lock = threading.Lock()
        self._workers = []
        self._next_index = 0

    def add_worker(self, url: str) -> list[str]:
        with self._lock:
            if url not in self._workers:
                self._workers.append(url)
            return list(self._workers)

    def remove_worker(self, url: str) -> list[str]:
        with self._lock:
            self._workers = [worker for worker in self._workers if worker != url]
            if self._workers:
                self._next_index %= len(self._workers)
            else:
                self._next_index = 0
            return list(self._workers)

    def list_workers(self) -> list[str]:
        with self._lock:
            return list(self._workers)

    def next_worker(self) -> Optional[str]:
        with self._lock:
            if not self._workers:
                return None
            worker = self._workers[self._next_index % len(self._workers)]
            self._next_index = (self._next_index + 1) % len(self._workers)
            return worker


class _RouterServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls, router_state: _RouterState):
        super().__init__(server_address, handler_cls)
        self.router_state = router_state


class _RouterHandler(BaseHTTPRequestHandler):
    server_version = "SimpleSGLangRouter/0.1"

    @property
    def router_state(self) -> _RouterState:
        return self.server.router_state

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _write_json(self, payload, status_code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_response(self, response):
        body = response.content
        self.send_response(response.status_code)
        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/list_workers":
            self._write_json({"urls": self.router_state.list_workers()})
            return

        if parsed.path == "/health":
            self._write_json({"status": "ok"})
            return

        self._write_json({"error": f"Unknown endpoint: {parsed.path}"}, status_code=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/add_worker":
            url = query.get("url", [None])[0]
            if not url:
                self._write_json({"error": "Missing worker url"}, status_code=400)
                return
            self._write_json({"urls": self.router_state.add_worker(url)})
            return

        if parsed.path == "/remove_worker":
            url = query.get("url", [None])[0]
            if not url:
                self._write_json({"error": "Missing worker url"}, status_code=400)
                return
            self._write_json({"urls": self.router_state.remove_worker(url)})
            return

        if parsed.path == "/generate":
            worker_url = self.router_state.next_worker()
            if worker_url is None:
                self._write_json({"error": "No registered SGLang workers"}, status_code=503)
                return

            try:
                payload = self._read_json_body()
                response = requests.post(f"{worker_url}/generate", json=payload)
                self._write_response(response)
            except Exception as exc:
                self._write_json({"error": str(exc)}, status_code=502)
            return

        self._write_json({"error": f"Unknown endpoint: {parsed.path}"}, status_code=404)

    def log_message(self, format, *args):
        return


def launch_simple_router(args):
    router_state = _RouterState()
    server = _RouterServer((args.host, args.port), _RouterHandler, router_state)
    print(f"Simple SGLang router launched at {args.host}:{args.port}", flush=True)
    server.serve_forever()
