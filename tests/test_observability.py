"""Tests for structured logging and request-id correlation."""

import json
import logging

from app.core.logging import JsonFormatter, RequestIdFilter, request_id_var


def test_json_formatter_includes_request_id_and_extras():
    token = request_id_var.set("req-123")
    try:
        record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", (), None)
        RequestIdFilter().filter(record)
        record.event = "cohere.chat"
        record.latency_ms = 12.5

        data = json.loads(JsonFormatter().format(record))

        assert data["message"] == "hello"
        assert data["level"] == "INFO"
        assert data["logger"] == "app.test"
        assert data["request_id"] == "req-123"
        assert data["event"] == "cohere.chat"
        assert data["latency_ms"] == 12.5
    finally:
        request_id_var.reset(token)


def test_response_includes_generated_request_id_header(client):
    response = client.get("/health")
    assert response.headers.get("x-request-id")


def test_request_id_is_echoed_when_supplied(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-xyz"})
    assert response.headers["x-request-id"] == "trace-xyz"
