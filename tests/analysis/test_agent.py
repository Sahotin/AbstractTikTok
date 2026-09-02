from __future__ import annotations

import json

import httpx
import pytest

from analysis.agent import (
    AgentToolRegistry,
    AgentRequest,
    OpenAICompatibleToolCallingProvider,
    ProviderToolCall,
    ProviderTurn,
    PublicOpinionAgentService,
    ToolDefinition,
    ToolCallingProvider,
)
from analysis.llm import LLMConfig
from analysis.llm.base import StructuredOutputError
from analysis.embeddings import FakeEmbeddingProvider
from api.main import app
from api.routers.analysis import AgentRuntime, get_agent_runtime


class _FakeToolProvider(ToolCallingProvider):
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    @property
    def provider_name(self):
        return "fake"

    @property
    def model(self):
        return "fake-agent-v1"

    async def complete(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        return self.turns.pop(0)


class _FakeTools:
    def __init__(self):
        self.calls = []

    def definitions(self):
        return [ToolDefinition(
            name="get_sentiment_distribution",
            description="Get sentiment.",
            parameters={"type": "object", "properties": {}},
        )]

    async def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return {"analysis_coverage": 0.8, "negative": 3, "positive": 7}


def _tool_turn(call_id="call-1"):
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_sentiment_distribution", "arguments": "{}"},
    }
    return ProviderTurn(
        tool_calls=[ProviderToolCall(
            id=call_id, name="get_sentiment_distribution", arguments={}
        )],
        assistant_message={"role": "assistant", "tool_calls": [raw]},
    )


def _answer_turn(text="负面评论占 30%，当前分析覆盖率为 80%。"):
    return ProviderTurn(
        content=text,
        assistant_message={"role": "assistant", "content": text},
    )


@pytest.mark.asyncio
async def test_agent_calls_tool_then_synthesizes() -> None:
    provider = _FakeToolProvider([_tool_turn(), _answer_turn()])
    tools = _FakeTools()
    result = await PublicOpinionAgentService(provider, tools).run(
        AgentRequest(question="负面情绪怎么样？")
    )

    assert result.finish_reason == "completed"
    assert result.steps == 2
    assert result.tool_calls[0].tool_name == "get_sentiment_distribution"
    assert result.tool_calls[0].success is True
    assert tools.calls == [("get_sentiment_distribution", {})]
    second_messages = provider.calls[1][0]
    assert any(message["role"] == "tool" for message in second_messages)


@pytest.mark.asyncio
async def test_agent_topic_total_does_not_replace_collection_scope() -> None:
    class ScopedTools(_FakeTools):
        def definitions(self):
            return [
                ToolDefinition(name="get_sentiment_distribution", description="scope", parameters={"type": "object"}),
                ToolDefinition(name="get_topics", description="topics", parameters={"type": "object"}),
            ]

        async def execute(self, name, arguments):
            self.calls.append((name, arguments))
            if name == "get_sentiment_distribution":
                return {
                    "total_comments": 131,
                    "eligible_comments": 104,
                    "excluded_comments": 27,
                    "analyzed_comments": 104,
                    "ai_analysis_coverage": 1.0,
                }
            return {
                "total": 5,
                "items": [{
                    "topic_name": "资源获取",
                    "percentage": 12.5,
                    "representative_comments": [{"comment_id": None, "text": "资源在哪里获取", "similarity": 0.9}],
                }],
            }

    def turn(call_id, name):
        return ProviderTurn(
            tool_calls=[ProviderToolCall(id=call_id, name=name, arguments={})],
            assistant_message={"role": "assistant", "tool_calls": []},
        )

    provider = _FakeToolProvider([
        turn("scope", "get_sentiment_distribution"),
        turn("topics", "get_topics"),
        _answer_turn("主题已确认。"),
    ])
    result = await PublicOpinionAgentService(provider, ScopedTools()).run(
        AgentRequest(question="大家主要在讨论什么？")
    )

    assert result.data_scope.total_comments == 131
    assert result.data_scope.eligible_comments == 104
    assert result.data_scope.excluded_comments == 27
    assert result.data_scope.analyzed_comments == 104
    assert len(result.evidence) == 1


@pytest.mark.asyncio
async def test_agent_blocks_duplicate_call_and_forces_final_answer() -> None:
    provider = _FakeToolProvider([
        _tool_turn("call-1"),
        _tool_turn("call-2"),
        _answer_turn("基于已有结果作答。"),
    ])
    tools = _FakeTools()
    result = await PublicOpinionAgentService(provider, tools).run(
        AgentRequest(question="重复查询", max_steps=5)
    )

    assert result.finish_reason == "duplicate_call"
    assert len(tools.calls) == 1
    assert result.tool_calls[-1].success is False
    assert provider.calls[-1][1] == []


@pytest.mark.asyncio
async def test_openai_tool_provider_parses_calls_and_final_content() -> None:
    responses = [
        {"choices": [{"message": {"content": None, "tool_calls": [{
            "id": "abc",
            "type": "function",
            "function": {"name": "get_topics", "arguments": json.dumps({"limit": 3})},
        }]}}]},
        {"choices": [{"message": {"content": "最终结论"}}]},
    ]

    async def handler(request: httpx.Request):
        payload = json.loads(request.content)
        if responses[0]["choices"][0]["message"].get("tool_calls"):
            assert payload["tool_choice"] == "auto"
            assert payload["tools"][0]["function"]["name"] == "get_topics"
        return httpx.Response(200, json=responses.pop(0))

    provider = OpenAICompatibleToolCallingProvider(
        LLMConfig(model="agent-model", base_url="https://llm.invalid/v1", api_key="secret"),
        transport=httpx.MockTransport(handler),
    )
    definition = ToolDefinition(
        name="get_topics", description="topics", parameters={"type": "object"}
    )
    first = await provider.complete([{"role": "user", "content": "test"}], [definition])
    second = await provider.complete([{"role": "user", "content": "test"}], [])

    assert first.tool_calls[0].arguments == {"limit": 3}
    assert second.content == "最终结论"


@pytest.mark.asyncio
async def test_agent_api_executes_real_read_only_tool(
    phase1b_context, semantic_repository
) -> None:
    provider = _FakeToolProvider([
        ProviderTurn(
            tool_calls=[ProviderToolCall(
                id="stats-1", name="get_comment_statistics", arguments={}
            )],
            assistant_message={
                "role": "assistant",
                "tool_calls": [{
                    "id": "stats-1", "type": "function",
                    "function": {"name": "get_comment_statistics", "arguments": "{}"},
                }],
            },
        ),
        _answer_turn("当前范围共有 2 条评论。"),
    ])
    runtime = AgentRuntime(
        analysis_repository=phase1b_context.repository,
        semantic_repository=semantic_repository,
        llm_provider=provider,
        embedding_provider=FakeEmbeddingProvider(dimension=2),
        llm_config=LLMConfig(),
    )

    async def override_runtime():
        yield runtime

    app.dependency_overrides[get_agent_runtime] = override_runtime
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/analysis/agent/query", json={
                "question": "有多少评论？",
                "run_id": phase1b_context.run_id,
            })
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("基于当前采集的 2 条评论")
    assert payload["tool_calls"][0]["tool_name"] == "get_comment_statistics"
    assert payload["tool_calls"][0]["result_summary"]
    assert payload["tool_calls"][0]["latency_ms"] >= 0
    assert payload["data_scope"]["total_comments"] == 2


@pytest.mark.asyncio
async def test_agent_tool_rejects_semantic_uuid_as_collection_run(
    phase1b_context, semantic_repository
) -> None:
    semantic_id = phase1b_context.run_id
    request = AgentRequest(question="统计", semantic_run_id=semantic_id)
    registry = AgentToolRegistry(
        phase1b_context.repository,
        semantic_repository,
        None,
        request,
    )
    with pytest.raises(ValueError, match="CollectionRun UUID"):
        await registry.execute("get_comment_statistics", {"run_id": semantic_id})


@pytest.mark.asyncio
async def test_agent_api_returns_a_complete_failure_payload(
    phase1b_context, semantic_repository
) -> None:
    class FailingProvider(_FakeToolProvider):
        async def complete(self, messages, tools):
            raise StructuredOutputError("mocked invalid structured output")

    runtime = AgentRuntime(
        analysis_repository=phase1b_context.repository,
        semantic_repository=semantic_repository,
        llm_provider=FailingProvider([]),
        embedding_provider=FakeEmbeddingProvider(dimension=2),
        llm_config=LLMConfig(max_retries=0),
    )

    async def override_runtime():
        yield runtime

    app.dependency_overrides[get_agent_runtime] = override_runtime
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/analysis/agent/query", json={"question": "测试失败响应"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {
        "status": "failed",
        "error_type": "StructuredOutputError",
        "message": "mocked invalid structured output",
        "partial_tool_calls": [],
    }
