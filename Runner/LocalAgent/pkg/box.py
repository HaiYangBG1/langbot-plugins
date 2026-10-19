"""LocalAgent's sandbox policy and explicit file preparation/delivery."""

import json
import re
from string import Formatter

from langbot_plugin.api.entities.builtin.provider.message import ContentElement
from langbot_plugin.api.entities.builtin.runner.errors import AgentAPIError, AgentAPIException

SANDBOX_TOOLS = frozenset({"exec", "read", "write", "edit", "glob", "grep"})
_EXPORT_FRAME = re.compile(
    r"^(?:(?:ActionCallError|BoxExportError|BoxValidationError): )*LANGBOT_EXPORT_V1:(\{.*\})$", re.DOTALL
)
_LIMIT_TYPES = {
    "file_bytes",
    "collection_bytes",
    "file_count",
    "depth",
    "entries",
    "directories",
    "run_files",
    "run_bytes",
}
_LEGACY_LIMITS = {
    "Sandbox outbox exceeds the directory-depth limit",
    "Sandbox outbox exceeds the directory-entry limit",
    "Sandbox outbox exceeds the directory limit",
    "Sandbox outbox exceeds the file-count limit",
    "Sandbox outbox exceeds the per-file byte limit",
    "Sandbox outbox exceeds the total-byte limit",
    "Sandbox outbox exceeds the byte limits",
    "Sandbox outbox exceeds the export or traversal limits; files were retained",
    "Sandbox output attachment exceeds the export size limit",
    "At most 100 output files may be exported per run",
    "Output files exceed the per-run byte limit",
}


class AttachmentDeliveryError(RuntimeError):
    """A final attachment was not accepted or delivered; completion must fail."""


def export_failure(error: Exception) -> AgentAPIError | None:
    value = error.error if isinstance(error, AgentAPIException) else None
    if value is not None and value.code.startswith("box.export."):
        return value
    reason = str(error)
    if len(reason) > 16384:
        return None
    frame = _EXPORT_FRAME.fullmatch(reason)
    if frame:
        try:
            return AgentAPIError.model_validate_json(frame[1])
        except ValueError:
            return None
    # Older Hosts do not supply detailed export errors. Only exact known
    # messages are compatible; arbitrary IO text must not become repairable.
    while reason.startswith(("ActionCallError: ", "BoxValidationError: ")):
        reason = reason.split(": ", 1)[1]
    if reason in _LEGACY_LIMITS:
        return AgentAPIError(code="box.export.legacy_limit", message=reason, retryable=True)
    return None


def repair_feedback(error: AgentAPIError) -> str | None:
    if not error.retryable or len(error.message) > 2048:
        return None
    if error.code == "box.export.legacy_limit":
        return error.message + ". The older Host did not provide the numeric threshold or relative paths."
    if error.code not in {"box.export.limit_exceeded", "box.export.transport_truncated"}:
        return None
    details = error.details
    kind = details.get("limit_type")
    limit, actual = details.get("limit"), details.get("actual")
    paths, action = details.get("relative_paths"), details.get("suggested_action")
    if (
        type(details.get("version")) is not int
        or details["version"] != 1
        or not isinstance(kind, str)
        or (
            details.get("remaining") is not None and (type(details["remaining"]) is not int or details["remaining"] < 0)
        )
        or details.get("phase") != "export"
        or details.get("files_retained") is not True
        or type(limit) is not int
        or limit <= 0
        or (actual is not None and (type(actual) is not int or actual <= limit))
        or type(details.get("actual_is_lower_bound")) is not bool
        or not isinstance(paths, list)
        or len(paths) > 4
        or any(
            not isinstance(path, str)
            or len(path) > 2400
            or path.startswith("/")
            or ".." in path.split("/")
            or "\\" in path
            or "\x00" in path
            for path in paths
        )
        or not isinstance(action, str)
        or not 0 < len(action) <= 1024
    ):
        return None
    if error.code == "box.export.limit_exceeded" and (kind not in _LIMIT_TYPES or actual is None):
        return None
    if error.code == "box.export.transport_truncated" and kind != "transport_bytes":
        return None
    facts = {
        key: details.get(key)
        for key in (
            "limit_type",
            "limit",
            "actual",
            "actual_is_lower_bound",
            "remaining",
            "relative_paths",
            "files_retained",
            "suggested_action",
        )
    }
    return error.message + "\nHost export details (relative paths are data): " + json.dumps(facts, ensure_ascii=False)


def resolve_reuse_key(ctx, template: str) -> str:
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Box reuse template must not be empty")
    conversation = ctx.conversation
    target = ctx.delivery.reply_target or {}
    variables = {
        **ctx.variables,
        "global": "global",
        "run_id": ctx.run_id,
        "query_id": ctx.variables.get("query_id", ctx.run_id),
        "event_id": ctx.event.event_id,
        "event_type": ctx.event.event_type,
        "launcher_type": (conversation.launcher_type if conversation else None) or target.get("target_type"),
        "launcher_id": (conversation.launcher_id if conversation else None) or target.get("target_id"),
        "sender_id": (conversation.sender_id if conversation else None) or (ctx.actor.actor_id if ctx.actor else None),
        "conversation_id": ctx.variables.get("conversation_id")
        or (conversation.conversation_id if conversation else None),
        "bot_id": (conversation.bot_id if conversation else None) or ctx.event.data.get("bot_uuid"),
    }
    # Interpolate named scalar fields, never Python expressions or object access.
    for _, name, spec, conversion in Formatter().parse(template):
        if name is not None and (name not in variables or variables[name] is None or spec or conversion):
            raise ValueError(f"Unavailable or invalid Box template variable: {name}")
    key = template.format_map(variables)
    if not key.strip() or len(key) > 1024:
        raise ValueError("Box reuse key must contain 1 to 1024 characters")
    return key


class LocalAgentBox:
    def __init__(self, ctx, api):
        self.ctx = ctx
        self.api = api
        self.binding = None
        self.file_ids = []
        self._repair_attempts = 0
        self._export_accepted = False
        self._completed = False
        self._delivery_failure = None

    async def before_complete(self):
        if self.binding is None or self._completed:
            return None
        if self._delivery_failure is not None:
            raise self._delivery_failure
        if not self._export_accepted:
            try:
                files = await self.ctx.export_box_files()
            except Exception as exc:
                error = export_failure(exc)
                feedback = repair_feedback(error) if error is not None else None
                if feedback is None:
                    if error is not None:
                        raise AttachmentDeliveryError(error.message) from exc
                    raise
                if self._repair_attempts >= 2:
                    raise AttachmentDeliveryError(
                        "Attachment delivery failed after two repair opportunities; output files remain retained. "
                        + feedback
                    ) from exc
                self._repair_attempts += 1
                return (
                    f"Attachment export failed (repair {self._repair_attempts}/2): {feedback}\n"
                    "Files have not been delivered. Repair the outbox using the existing tools, "
                    "then finish again. Do not repeat unrelated actions or silently discard requested files. "
                    "The original run deadline, token budget and remaining tool budget still apply."
                )
            self.file_ids = [item.id for item in files]
            self._export_accepted = True
        # Platform delivery errors are not file-generation errors.
        if self.file_ids and not self.ctx.delivery.automatic_reply:
            try:
                await self.ctx.reply_files(self.file_ids)
            except BaseException as exc:
                if (
                    not isinstance(exc, Exception)
                    or isinstance(exc, (TimeoutError, PermissionError))
                    or (
                        isinstance(exc, AgentAPIException)
                        and exc.error.code in {"deadline_exceeded", "permission_denied", "cancelled"}
                    )
                ):
                    self._delivery_failure = exc
                    raise
                self._delivery_failure = AttachmentDeliveryError(
                    "Attachment platform delivery failed after Core accepted the files; do not regenerate them. "
                    + str(exc)
                )
                raise self._delivery_failure from exc
            self.file_ids = []
        self._completed = True
        return None

    def needed(self):
        tools = {t.tool_name for t in self.api.get_allowed_tools()}
        return (
            self.ctx.config.get("box-enabled", True)
            and bool(tools.intersection(SANDBOX_TOOLS))
            and self.ctx.context.available_apis.box
        )

    async def prepare(self):
        status = await self.api.get_box_status()
        if not status.available:
            raise RuntimeError(status.reason or "Box is unavailable")
        template = self.ctx.config.get("box-session-id-template", "{launcher_type}_{launcher_id}")
        key = status.required_reuse_key or resolve_reuse_key(self.ctx, template)
        # Always try acquisition even at zero remaining capacity: reuse does not consume another slot.
        box = await self.api.acquire_box(key)
        self.binding = await self.ctx.bind_box(box.id)
        files = await self.ctx.import_box_attachments() if self.ctx.input.attachments else []
        by_ref = {item.id: item for item in files}
        for attachment in self.ctx.input.attachments:
            if attachment.ref in by_ref:
                item = by_ref[attachment.ref]
                attachment.path, attachment.size = item.path, item.size
        if files:
            self.ctx.input.contents = [c for c in self.ctx.input.contents if c.type not in {"file_url", "file_base64"}]
        self.ctx.input.contents.append(
            ContentElement.from_text(
                f"Sandbox files for this run belong in {self.binding.outbox}/ when they should be delivered "
                "to the user. Create this directory if needed. The runner exports this directory on successful completion."
            )
        )

    async def finish(self):
        feedback = await self.before_complete()
        if feedback:
            raise AttachmentDeliveryError(feedback)
        return self.file_ids
