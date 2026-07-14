# Security Policy

## Reporting a vulnerability

Please report security issues privately to **security@ambertrace.ai** rather than
opening a public issue. Include steps to reproduce and the affected version.
We'll acknowledge receipt and keep you updated on the fix.

## Handling API keys

`ambertrace-rlvr` reads your AmberTrace API key from the environment
(`AMBERTRACE_API_KEY`) only — it is never hardcoded, logged, or written to run
reports (reports redact key-like fields). Use a **scoped, platform-only** key for
training jobs, never a full-account key, and keep it out of version control
(`.env` is gitignored).
