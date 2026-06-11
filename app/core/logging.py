"""Structured logging setup.

Provides JSON and human-readable formatters, a filter that stamps every record
with the current request id, and a `configure_logging` entry point. Modules log
structured fields by passing `extra={...}`; in JSON mode those fields become
top-level keys, so latency, token usage, and tool-call traces are queryable
rather than buried in a message string.
"""

import json
import logging
from contextvars import ContextVar

# Holds the id of the in-flight request so any log line can be correlated to it.
# Defaults to "-" for logs emitted outside a request (startup, shutdown).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# LogRecord attributes that are not user-supplied "extra" fields.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp the record with the request id and keep it."""
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON, including any `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the record to a JSON string."""
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str, fmt: str = "json") -> None:
    """Configure the root logger with structured output."""
    if fmt == "text":
        formatter: logging.Formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s %(message)s"
        )
    else:
        formatter = JsonFormatter()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    root.addHandler(handler)

    # The httpx client logs a line per request; our own per-call logs already
    # cover that with richer fields, so keep its output to warnings and above.
    logging.getLogger("httpx").setLevel(logging.WARNING)
