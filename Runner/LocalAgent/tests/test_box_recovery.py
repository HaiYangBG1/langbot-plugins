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
