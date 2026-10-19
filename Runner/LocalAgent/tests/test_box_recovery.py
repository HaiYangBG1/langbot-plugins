import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pkg.box import LocalAgentBox


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", [True, False])
async def test_export_repair_then_success(automatic):
    ctx = SimpleNamespace(
        export_box_files=AsyncMock(
            side_effect=[
                RuntimeError("Sandbox outbox exceeds the file-count limit"),
                [SimpleNamespace(id="file")],
            ]
        ),
        reply_files=AsyncMock(),
        delivery=SimpleNamespace(automatic_reply=automatic),
    )
    box = LocalAgentBox(ctx, None)
    box.binding = object()
    assert "repair 1/2" in await box.before_complete()
    assert await box.before_complete() is None
    assert box.file_ids == (["file"] if automatic else [])
    assert ctx.export_box_files.await_count == 2
    assert ctx.reply_files.await_count == (0 if automatic else 1)


@pytest.mark.asyncio
async def test_repair_is_bounded():
    ctx = SimpleNamespace(
        export_box_files=AsyncMock(side_effect=RuntimeError("Output files exceed the per-run byte limit"))
    )
    box = LocalAgentBox(ctx, None)
    box.binding = object()
    assert "repair 1/2" in await box.before_complete()
    assert "repair 2/2" in await box.before_complete()
    with pytest.raises(RuntimeError):
        await box.before_complete()
    assert ctx.export_box_files.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [PermissionError("denied"), TimeoutError(), asyncio.CancelledError(), RuntimeError("stale Workspace placement")],
)
async def test_unrepairable_error_propagates(error):
    ctx = SimpleNamespace(export_box_files=AsyncMock(side_effect=error))
    box = LocalAgentBox(ctx, None)
    box.binding = object()
    with pytest.raises(type(error)):
        await box.before_complete()
    assert box._repair_attempts == 0


def detailed_error(code="box.export.limit_exceeded", **details):
    from langbot_plugin.api.entities.builtin.runner.errors import AgentAPIError

    return AgentAPIError(
        code=code,
        message="Attachment export failed: Sandbox outbox exceeds the file-count limit",
        retryable=True,
        details={
            "version": 1,
            "phase": "export",
            "limit_type": "file_count",
            "limit": 20,
            "actual": 21,
            "actual_is_lower_bound": True,
            "remaining": None,
            "relative_paths": ["nested/answer.txt"],
            "files_retained": True,
            "suggested_action": "Consolidate the requested files.",
            **details,
        },
    )


@pytest.mark.parametrize("relay", [False, True])
def test_structured_and_sdk_text_relay_preserve_actionable_feedback(relay):
    from langbot_plugin.api.entities.builtin.runner.errors import AgentAPIException

    from pkg.box import export_failure, repair_feedback

    error = detailed_error()
    transported = (
        RuntimeError("ActionCallError: ActionCallError: LANGBOT_EXPORT_V1:" + error.model_dump_json())
        if relay
        else AgentAPIException(error)
    )
    feedback = repair_feedback(export_failure(transported))
    for fact in ('"limit": 20', '"actual": 21', "nested/answer.txt", '"files_retained": true', "Consolidate"):
        assert fact in feedback


@pytest.mark.parametrize(
    "details",
    [
        {"limit_type": []},
        {"limit": True},
        {"actual": 20},
        {"remaining": -1},
        {"relative_paths": ["../secret"]},
        {"files_retained": False},
        {"version": 2},
    ],
)
def test_incomplete_or_invalid_quota_details_are_not_repairable(details):
    from pkg.box import repair_feedback

    assert repair_feedback(detailed_error(**details)) is None


@pytest.mark.parametrize(
    "message",
    [
        "Invalid JSON",
        "failed with a file-count limit in user input",
        "prefix LANGBOT_EXPORT_V1:{}",
        "LANGBOT_EXPORT_V1:{broken",
    ],
)
def test_unknown_failure_text_is_never_guessed_as_quota_or_truncation(message):
    from pkg.box import export_failure

    assert export_failure(RuntimeError(message)) is None


def test_explicit_transport_truncation_allows_unknown_actual_size():
    from pkg.box import repair_feedback

    assert (
        repair_feedback(
            detailed_error(
                code="box.export.transport_truncated",
                limit_type="transport_bytes",
                limit=1048576,
                actual=None,
            )
        )
        is not None
    )
    assert repair_feedback(detailed_error(code="box.export.invalid_response")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", [False, True])
async def test_finish_after_accepted_export_never_exports_or_delivers_again(automatic):
    ctx = SimpleNamespace(
        export_box_files=AsyncMock(return_value=[SimpleNamespace(id="accepted")]),
        reply_files=AsyncMock(),
        delivery=SimpleNamespace(automatic_reply=automatic),
    )
    box = LocalAgentBox(ctx, None)
    box.binding = object()
    assert await box.before_complete() is None
    await box.finish()
    await box.finish()
    ctx.export_box_files.assert_awaited_once()
    assert ctx.reply_files.await_count == (0 if automatic else 1)
    assert box.file_ids == (["accepted"] if automatic else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("platform down"), asyncio.CancelledError(), TimeoutError(), PermissionError()]
)
async def test_platform_failure_never_reenters_generation_or_duplicates_delivery(failure):
    from pkg.box import AttachmentDeliveryError

    ctx = SimpleNamespace(
        export_box_files=AsyncMock(return_value=[SimpleNamespace(id="accepted")]),
        reply_files=AsyncMock(side_effect=failure),
        delivery=SimpleNamespace(automatic_reply=False),
    )
    box = LocalAgentBox(ctx, None)
    box.binding = object()
    expected = AttachmentDeliveryError if type(failure) is RuntimeError else type(failure)
    for _ in range(2):
        with pytest.raises(expected):
            await box.before_complete()
    assert box._repair_attempts == 0 and box.file_ids == ["accepted"]
    ctx.export_box_files.assert_awaited_once()
    ctx.reply_files.assert_awaited_once()
