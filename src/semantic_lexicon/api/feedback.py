# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Lightweight HTTP API for streaming intent feedback."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Optional

from ..analysis import RewardComponents

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..intent import IntentClassifier


@dataclass
class FeedbackEvent:
    """Payload captured from the feedback API."""

    prompt: str
    selected_intent: str
    optimal_intent: str
    feedback: float
    timestamp: float
    reward_components: RewardComponents


class FeedbackService:
    """Apply incoming feedback to an intent classifier."""

    def __init__(self, classifier: IntentClassifier) -> None:
        self._classifier = classifier
        self._lock = threading.Lock()
        self._history: list[FeedbackEvent] = []

    @property
    def history(self) -> list[FeedbackEvent]:
        """Return a copy of processed events."""

        with self._lock:
            return list(self._history)

    def submit(
        self,
        prompt: str,
        selected_intent: str,
        optimal_intent: str,
        feedback: float,
    ) -> dict[str, object]:
        """Update classifier weights and return the new composite reward weights."""

        with self._lock:
            components = self._classifier.reward_components(
                prompt,
                selected_intent,
                optimal_intent,
                feedback,
            )
            weights = self._classifier.register_feedback(
                prompt,
                selected_intent,
                optimal_intent,
                feedback,
            )
            event = FeedbackEvent(
                prompt=prompt,
                selected_intent=selected_intent,
                optimal_intent=optimal_intent,
                feedback=feedback,
                timestamp=time.time(),
                reward_components=components,
            )
            self._history.append(event)
        return {"weights": weights.tolist(), "components": asdict(components)}


class _FeedbackHandler(BaseHTTPRequestHandler):
    """HTTP handler used by :class:`FeedbackAPI`."""

    service: FeedbackService

    def _set_headers(self, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - quiet server
        return

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.rstrip("/") != "/feedback":
            self._set_headers(HTTPStatus.NOT_FOUND)
            self.wfile.write(b"{}")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length)
        try:
            data = json.loads(payload.decode("utf8"))
            prompt = str(data["prompt"])
            selected = str(data["selected_intent"])
            optimal = str(data["optimal_intent"])
            feedback = float(data.get("feedback", 0.9))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover - guarded
            self._set_headers(HTTPStatus.BAD_REQUEST)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf8"))
            return
        result = self.service.submit(prompt, selected, optimal, feedback)
        self._set_headers()
        self.wfile.write(json.dumps(result).encode("utf8"))


class FeedbackAPI:
    """Background server that exposes the feedback submission endpoint."""

    def __init__(
        self,
        service: FeedbackService,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self._service = service
        self._host = host
        self._port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> tuple[str, int]:
        return self._host, self._port

    def _build_server(self) -> ThreadingHTTPServer:
        handler = _FeedbackHandler
        handler.service = self._service
        server = ThreadingHTTPServer(self.address, handler)
        server.daemon_threads = True
        return server

    def start(self, background: bool = True) -> None:
        """Start serving feedback requests."""

        if self._server is not None:
            return
        self._server = self._build_server()
        if background:
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()
            self._thread = thread
        else:  # pragma: no cover - manual execution path
            self._server.serve_forever()

    def stop(self) -> None:
        """Stop the running server."""

        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
