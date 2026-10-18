# Lazy Box integration check

Based on upstream 31a05080b58f2cda8b1acecee668b2cda5503c4b.
The LANG-16 fix f33146a0 is still absent upstream. Reapplied the minimal lazy
acquisition wrapper: attachments prepare eagerly, sandbox tools prepare on first
use, parallel tools share one preparation lock. Non-sandbox tools delegate directly.

Before: 2 failed / 32 passed in Box and shared-runtime tests.
After: 270 passed in the LocalAgent suite, excluding test_shared_release.py,
whose historical SDK pin does not match this integration target. This is not a
plugin release-gate pass. Actual sandbox execution requires an isolated Docker
daemon or VM; shared-daemon execution is not permitted for this QA task.

Live integration used an isolated OrbStack machine with a distinct Docker daemon,
not container naming on the shared daemon. With actual Core, RT, LocalAgent and
Box plus a synthetic OpenAI streaming peer, session counts were 0 after unrelated
input, 0 after a plain answer and 1 after exec; the peer observed real container
stdout LANG17_TOOL_OK. This is protocol/sandbox validation, not a real model test.
The paired Core branch also removes legacy implicit outbox scanning; explicit
Runner output attachments remain the delivery contract.
