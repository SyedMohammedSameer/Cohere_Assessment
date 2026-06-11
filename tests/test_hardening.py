"""Integration tests for the production-hardening features.

Covers API-key authentication, per-owner isolation of history, rate limiting,
the catch-all error handler, and context windowing. Each uses `client_factory`
to enable the relevant setting on an isolated app and database.
"""


def test_auth_required_when_keys_configured(client_factory):
    with client_factory(API_KEYS="secret1,secret2") as client:
        denied = client.get("/history")
        assert denied.status_code == 401
        assert denied.json()["error_code"] == "unauthorized"

        allowed = client.get("/history", headers={"X-API-Key": "secret1"})
        assert allowed.status_code == 200


def test_auth_rejects_invalid_key(client_factory):
    with client_factory(API_KEYS="secret1") as client:
        response = client.get("/history", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401


def test_owner_isolation_across_keys(client_factory):
    with client_factory(API_KEYS="key-a,key-b") as client:
        created = client.post("/chat", json={"query": "hi"}, headers={"X-API-Key": "key-a"})
        conversation_id = created.json()["conversation_id"]

        # key-b cannot read key-a's conversation and sees none of its own.
        cross = client.get(f"/conversations/{conversation_id}", headers={"X-API-Key": "key-b"})
        assert cross.status_code == 404
        assert client.get("/history", headers={"X-API-Key": "key-b"}).json()["total"] == 0

        # key-a sees its own conversation.
        assert client.get("/history", headers={"X-API-Key": "key-a"}).json()["total"] == 1


def test_rate_limit_returns_429(client_factory):
    with client_factory(RATE_LIMIT_PER_MINUTE="2") as client:
        assert client.get("/history").status_code == 200
        assert client.get("/history").status_code == 200
        third = client.get("/history")
        assert third.status_code == 429
        assert third.json()["error_code"] == "rate_limited"


def test_health_is_exempt_from_rate_limit(client_factory):
    with client_factory(RATE_LIMIT_PER_MINUTE="1") as client:
        for _ in range(5):
            assert client.get("/health").status_code == 200


def test_unhandled_error_returns_envelope(client_factory):
    class Boom:
        async def answer(self, query, history=None):
            raise RuntimeError("kaboom")

    with client_factory(raise_server_exceptions=False) as client:
        client.app.state.chat_service = Boom()
        response = client.post("/chat", json={"query": "hi"})
        assert response.status_code == 500
        assert response.json() == {
            "error_code": "internal_error",
            "detail": "An unexpected error occurred.",
        }


def test_history_replay_is_windowed(client_factory, fake_chat_service):
    with client_factory(MAX_HISTORY_MESSAGES="2") as client:
        cid = client.post("/chat", json={"query": "q1"}).json()["conversation_id"]
        client.post("/chat", json={"query": "q2", "conversation_id": cid})
        client.post("/chat", json={"query": "q3", "conversation_id": cid})

        # By the third turn there are four stored messages, but only the last
        # two are replayed to the model.
        replayed = fake_chat_service.calls[-1]["history"]
        assert len(replayed) == 2


def test_windowed_history_starts_with_user_turn(client_factory, fake_chat_service):
    # An odd window can slice mid-turn; the replayed history must still begin
    # with a user message rather than an assistant one.
    with client_factory(MAX_HISTORY_MESSAGES="3") as client:
        cid = client.post("/chat", json={"query": "q1"}).json()["conversation_id"]
        client.post("/chat", json={"query": "q2", "conversation_id": cid})
        client.post("/chat", json={"query": "q3", "conversation_id": cid})

        replayed = fake_chat_service.calls[-1]["history"]
        assert replayed
        assert replayed[0].role == "user"
