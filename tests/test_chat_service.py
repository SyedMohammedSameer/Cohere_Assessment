"""Unit tests for the tool-calling orchestration loop."""

from app.clients.cohere import ChatMessage, ToolCall
from app.clients.wikipedia import WikiArticle
from app.core.exceptions import WikipediaError
from app.schemas.streaming import SourcesChunk, TokenChunk, ToolCallStatus
from app.services.chat import AnswerDoneEvent, ChatService
from tests.fakes import ScriptedCohereClient, StubWikipediaClient, make_chat_result, make_tool_call

ARTICLE = WikiArticle(
    id=440,
    title="Buzz Aldrin",
    snippet="second person to walk on the Moon",
    url="https://en.wikipedia.org/?curid=440",
)


def make_service(results, *, articles=None, error=None, max_iterations=5):
    """Build a ChatService over scripted Cohere results and a stub Wikipedia."""
    cohere = ScriptedCohereClient(results)
    wiki = StubWikipediaClient(articles=articles, error=error)
    return ChatService(cohere, wiki, max_iterations=max_iterations), cohere, wiki


async def test_direct_answer_without_tool_call():
    service, cohere, wiki = make_service(
        [make_chat_result("Hello there.", input_tokens=5, output_tokens=3)]
    )
    result = await service.answer("Say hello")

    assert result.text == "Hello there."
    assert result.tool_invocations == 0
    assert result.sources == []
    assert result.latency_ms is not None
    assert wiki.queries == []  # tool never invoked


async def test_tool_loop_grounds_answer_and_sums_usage():
    service, cohere, wiki = make_service(
        [
            make_chat_result(
                tool_calls=[make_tool_call("second person moon")],
                finish_reason="TOOL_CALL",
                input_tokens=10,
                output_tokens=2,
            ),
            make_chat_result("Buzz Aldrin.", input_tokens=30, output_tokens=4),
        ],
        articles=[ARTICLE],
    )
    result = await service.answer("Who walked second on the moon?")

    assert result.text == "Buzz Aldrin."
    assert result.tool_invocations == 1
    assert result.input_tokens == 40 and result.output_tokens == 6
    assert len(result.sources) == 1 and result.sources[0].title == "Buzz Aldrin"
    assert wiki.queries == ["second person moon"]

    # The second model call must carry the tool result after the assistant turn.
    second_messages = cohere.calls[1]["messages"]
    assert [m.role for m in second_messages] == ["system", "user", "assistant", "tool"]
    assert second_messages[-1].content[0]["document"]["data"]["title"] == "Buzz Aldrin"


async def test_history_is_replayed_before_the_new_turn():
    service, cohere, _ = make_service([make_chat_result("Answer.")])
    history = [
        ChatMessage(role="user", content="prior question"),
        ChatMessage(role="assistant", content="prior answer"),
    ]
    await service.answer("follow up", history=history)

    sent = cohere.calls[0]["messages"]
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1].content == "prior question"
    assert sent[-1].content == "follow up"


async def test_graceful_degradation_when_wikipedia_fails():
    service, cohere, wiki = make_service(
        [
            make_chat_result(tool_calls=[make_tool_call("x")], finish_reason="TOOL_CALL"),
            make_chat_result("Answered without grounding."),
        ],
        error=WikipediaError("down"),
    )
    result = await service.answer("q")

    assert result.text == "Answered without grounding."
    assert result.sources == []
    # The model is told the tool failed, rather than the request erroring.
    tool_message = cohere.calls[1]["messages"][-1]
    assert tool_message.content[0]["document"]["data"]["error"]


async def test_safety_bound_forces_tool_free_final_call():
    looping = [
        make_chat_result(tool_calls=[make_tool_call("x")], finish_reason="TOOL_CALL")
        for _ in range(3)
    ]
    looping.append(make_chat_result("Forced final answer."))
    service, cohere, _ = make_service(looping, articles=[ARTICLE], max_iterations=3)

    result = await service.answer("q")

    assert result.text == "Forced final answer."
    # Final call is made with tools disabled to force an answer.
    assert cohere.calls[-1]["tools"] is None


async def test_unknown_tool_name_is_reported_not_executed():
    service, cohere, wiki = make_service(
        [
            make_chat_result(
                tool_calls=[ToolCall(id="c1", name="unknown_tool", arguments="{}")],
                finish_reason="TOOL_CALL",
            ),
            make_chat_result("Recovered."),
        ]
    )
    result = await service.answer("q")

    assert result.text == "Recovered."
    assert wiki.queries == []  # the unknown tool is never dispatched to Wikipedia
    assert cohere.calls[1]["messages"][-1].content[0]["document"]["data"]["error"]


async def test_answer_stream_grounds_and_streams_tokens():
    service, cohere, wiki = make_service(
        [
            make_chat_result(
                tool_calls=[make_tool_call("second person moon")],
                finish_reason="TOOL_CALL",
                input_tokens=10,
                output_tokens=2,
            ),
            make_chat_result("Buzz Aldrin.", input_tokens=30, output_tokens=4),
        ],
        articles=[ARTICLE],
    )
    events = [event async for event in service.answer_stream("Who walked second?")]

    tool_calls = [e for e in events if isinstance(e, ToolCallStatus)]
    sources = [e for e in events if isinstance(e, SourcesChunk)]
    tokens = "".join(e.text for e in events if isinstance(e, TokenChunk))
    done = [e for e in events if isinstance(e, AnswerDoneEvent)]

    assert tool_calls[0].query == "second person moon"
    assert sources[-1].sources[0]["title"] == "Buzz Aldrin"
    assert tokens == "Buzz Aldrin."
    assert done[0].result.text == "Buzz Aldrin."
    assert done[0].result.input_tokens == 40 and done[0].result.output_tokens == 6
    assert wiki.queries == ["second person moon"]


async def test_invalid_tool_arguments_are_reported():
    service, cohere, wiki = make_service(
        [
            make_chat_result(
                tool_calls=[ToolCall(id="c1", name="search_wikipedia", arguments="not json")],
                finish_reason="TOOL_CALL",
            ),
            make_chat_result("Recovered."),
        ]
    )
    result = await service.answer("q")

    assert result.text == "Recovered."
    assert wiki.queries == []
    assert cohere.calls[1]["messages"][-1].content[0]["document"]["data"]["error"]
