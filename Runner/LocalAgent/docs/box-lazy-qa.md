# Lazy Box preparation (LANG-16)

With LangBot v4.11.0 (`502b38130f9878b405498b93e4203593573eb19b`), SDK 0.7.11,
LocalAgent 0.2.1 and a tool-capable model, a plain keyword-triggered answer created a Docker
sandbox even when the user explicitly requested no tools. The model returned `QA_OK` and
no tool ran, yet the session count increased from zero to one.

The Runner now prepares Box before assembly only when attachments need importing. For plain
input it wraps the authorized tool executor and acquires/binds on the first sandbox tool call.
A lock makes parallel tools share one completed preparation. Tool results include `sandbox_outbox`
so the model can discover the delivery path after lazy preparation; successful completion still
exports files. Non-sandbox tools and plain answers do not acquire Box.

Real isolated Core + Plugin RT + Box + Vite + loopback OneBot + deepseek-chat validation:

| Case | Before | After |
| --- | --- | --- |
| Unmentioned group chat | 0 sandboxes after Core response-gate fix | 0 |
| Keyword, plain `QA_OK` answer | 1 | 0 |
| `exec` runs `printf LANG16_TOOL_OK` | Reuses 1 | Creates 1, exit 0 |

Before evidence run: `baseline-20261017-154144` (marketplace package).
After: `baseline-20261017-192121` (local package built from this worktree).
All messages terminated successfully, query queue/cache returned to zero, Core/RT/Box/Web stayed
healthy, and the recorded task container/processes were cleaned. No real QQ traffic was sent.
The before marketplace artifact and repository HEAD have unrelated source differences; the eager
Box policy itself is identical. This is a behavior reproduction, not a throughput benchmark.

254 LocalAgent tests passed against SDK 0.7.11, including real loopback SDK scope tests for
attachment import/export, denied acquisition, concurrency and automatic/explicit file replies.
`tests/test_shared_release.py` is excluded here because it hard-codes SDK 0.7.4 admission;
this report does not claim that separate release-environment gate passed.
