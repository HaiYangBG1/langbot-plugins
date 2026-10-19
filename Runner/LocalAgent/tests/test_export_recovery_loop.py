"""Attachment recovery stays inside the original loop and run controls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langbot_plugin.api.entities.builtin.provider.message import FunctionCall, Message, MessageChunk, ToolCall

from components.runner.default import RunDeadline, RunInterruptChecker, _iterate_with_run_controls
from pkg.agent_core import AgentLoop, AgentLoopEventType, AgentLoopHooks, LangBotModelAdapter, LangBotToolExecutor
from pkg.box import LocalAgentBox
from tests.test_runner import make_context


class ScriptedAPI:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.histories = []
        self.call_tool = AsyncMock(return_value={"value": "repaired"})

    def next_turn(self, messages):
        self.histories.append([m.model_copy(deep=True) for m in messages])
        return next(self.turns)

    async def invoke_llm_with_usage(self, messages, **kwargs):
        return {"message": self.next_turn(messages), "usage": {"total_tokens": 5}}

    async def invoke_llm_stream_events(self, messages, **kwargs):
        turn = self.next_turn(messages)
        yield MessageChunk(role="assistant", content=turn.content, tool_calls=turn.tool_calls, is_final=True)
        yield {"usage": {"total_tokens": 5}}


def tool_turn(identifier):
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id=identifier,
                type="function",
                function=FunctionCall(name="exec", arguments="{}"),
            )
        ],
    )


def setup_loop(streaming, *, exports=None, turns=None, budget=10, hooks=None):
    api = ScriptedAPI(
        turns
        or [
            Message(role="assistant", content="Draft."),
            tool_turn("repair"),
            Message(role="assistant", content="Done."),
        ]
    )
    ctx = SimpleNamespace(
        export_box_files=AsyncMock(
            side_effect=exports
            or [RuntimeError("Sandbox outbox exceeds the file-count limit"), [SimpleNamespace(id="file")]]
        ),
        delivery=SimpleNamespace(automatic_reply=True),
    )
    box = LocalAgentBox(ctx, None)
    box.binding = object()
    loop = AgentLoop(
        model_adapter=LangBotModelAdapter(api),
        tool_executor=LangBotToolExecutor(api, {"exec"}),
        model_ids=["model"],
        messages=[Message(role="user", content="Create an attachment")],
        tools=[],
        streaming=streaming,
        max_tool_iterations=budget,
        before_complete=box.before_complete,
        hooks=hooks,
    )
    return loop, api, box, ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_repair_preserves_context_usage_and_legal_messages_with_one_completion(streaming):
    loop, api, box, ctx = setup_loop(streaming)
    events = [event async for event in loop.run()]
    assert sum(e.type == AgentLoopEventType.AGENT_END for e in events) == 1
    assert events[-1].usage["total_tokens"] == 15
    assert events[-1].usage["model_calls"] == 3
    feedback = api.histories[1][-1]
    assert feedback.role == "user" and "repair 1/2" in feedback.content
    assert [m.role for m in api.histories[2]] == ["user", "assistant", "user", "assistant", "tool"]
    assert api.histories[2][-1].tool_call_id == "repair"
    assert box.file_ids == ["file"] and ctx.export_box_files.await_count == 2
    if streaming:
        deltas = [e.chunk.content for e in events if e.type == AgentLoopEventType.MESSAGE_UPDATE]
        assert deltas[0] == "Draft." and deltas[-1] == "Draft.Done."


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_exhaustion_never_emits_success_or_starts_fourth_model_turn(streaming):
    failure = RuntimeError("Sandbox outbox exceeds the file-count limit")
    loop, api, box, ctx = setup_loop(
        streaming, exports=[failure] * 3, turns=[Message(role="assistant", content="Done")] * 3
    )
    events = []
    with pytest.raises(RuntimeError, match="two repair opportunities"):
        async for event in loop.run():
            events.append(event)
    assert len(api.histories) == ctx.export_box_files.await_count == 3
    assert not any(e.type == AgentLoopEventType.AGENT_END for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_repair_cannot_reset_original_tool_budget(streaming):
    failure = RuntimeError("Sandbox outbox exceeds the file-count limit")
    loop, api, box, ctx = setup_loop(
        streaming,
        exports=[failure] * 2,
        turns=[tool_turn("original"), Message(role="assistant", content="Draft"), tool_turn("repair")],
        budget=1,
    )
    events = []
    with pytest.raises(RuntimeError, match="remaining run budget"):
        async for event in loop.run():
            events.append(event)
    api.call_tool.assert_awaited_once()
    assert len(api.histories) == 3
    assert not any(e.type == AgentLoopEventType.AGENT_END for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_stop_hook_cannot_bypass_final_export(streaming):
    hooks = AgentLoopHooks()
    hooks.should_stop_after_turn = AsyncMock(return_value=True)
    loop, api, box, ctx = setup_loop(streaming, hooks=hooks)
    events = [event async for event in loop.run()]
    assert events[-1].type == AgentLoopEventType.RUN_FAILED
    assert len(api.histories) == ctx.export_box_files.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["timeout", "cancel"])
async def test_original_run_control_interrupts_export_after_repair_feedback(control):
    loop, api, box, ctx = setup_loop(False)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def export():
        if ctx.export_box_files.await_count == 1:
            raise RuntimeError("Sandbox outbox exceeds the file-count limit")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    ctx.export_box_files.side_effect = export
    checker = RunInterruptChecker(None, make_context())
    deadline = RunDeadline(0.1) if control == "timeout" else None
    events = []

    async def consume():
        async for event in _iterate_with_run_controls(loop.run(), checker, deadline):
            events.append(event)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    if control == "cancel":
        task.cancel()
    with pytest.raises(asyncio.TimeoutError if control == "timeout" else asyncio.CancelledError):
        await task
    assert cancelled.is_set() and ctx.export_box_files.await_count == 2
    assert box._repair_attempts == 1
    assert not any(e.type == AgentLoopEventType.AGENT_END for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exhausted", "timeout", "cancelled", "permission"])
async def test_runner_terminal_failure_keeps_usage_and_never_completes(monkeypatch, failure):
    from unittest.mock import MagicMock

    from langbot_plugin.api.entities.builtin.runner.context import RunnerContext
    from langbot_plugin.api.entities.builtin.runner.resources import AgentResources, ModelResource

    from components.runner.default import DefaultRunner
    from tests.test_runner import FakeRunnerAPIProxy

    api = FakeRunnerAPIProxy(models=[ModelResource(model_id="model")])
    api.invoke_llm_with_usage = AsyncMock(
        return_value={
            "message": Message(role="assistant", content="Draft"),
            "usage": {"total_tokens": 5},
        }
    )
    ctx = make_context(
        config={"model": {"primary": "model", "fallbacks": []}},
        resources=AgentResources(models=[ModelResource(model_id="model")]),
        delivery_supports_streaming=False,
    )
    ctx.context.available_apis.run_get = failure == "cancelled"
    exports = 0
    prepared = 0

    async def prepare(self):
        nonlocal prepared
        prepared += 1
        self.binding = object()

    async def export(self):
        nonlocal exports
        exports += 1
        if failure == "permission":
            raise PermissionError("binding no longer authorized")
        if failure in {"timeout", "cancelled"} and exports == 2:
            if failure == "cancelled":
                api.run_get.return_value = {"run_id": ctx.run_id, "status": "cancelled"}
            await asyncio.sleep(10)
        raise RuntimeError("Sandbox outbox exceeds the file-count limit")

    runner = DefaultRunner()
    runner.bind_runtime(plugin_runtime_handler=MagicMock(), plugin_identity="langbot/local-agent")
    monkeypatch.setattr(runner, "get_run_api", lambda context: api)
    monkeypatch.setattr(LocalAgentBox, "needed", lambda self: True)
    monkeypatch.setattr(LocalAgentBox, "prepare", prepare)
    monkeypatch.setattr(RunnerContext, "export_box_files", export)
    if failure == "timeout":
        monkeypatch.setattr("components.runner.default.get_run_timeout_seconds", lambda config: 0.1)
    results = [result async for result in runner.run(ctx)]
    types = [getattr(result.type, "value", result.type) for result in results]
    assert types.count("run.failed") == 1
    assert "run.completed" not in types and "message.completed" not in types
    expected = {"exhausted": 3, "timeout": 2, "cancelled": 2, "permission": 1}[failure]
    assert prepared == 1 and exports == expected
    assert results[-1].usage.total_tokens == expected * 5
    if failure in {"timeout", "cancelled"}:
        assert results[-1].data["code"] == ("runner.timeout" if failure == "timeout" else "cancelled")
