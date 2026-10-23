# Security Policy

## Supported versions

TopoGeoML is a pre-1.0 research toolkit. Only the latest published `0.0.x`
release receives security fixes; older releases do not. Until `v1.0`, APIs and
internals may change without notice (see [`LIMITATIONS.md`](LIMITATIONS.md)).

| Version  | Supported          |
| -------- | ------------------ |
| latest `0.0.x` | :white_check_mark: |
| older `0.0.x`  | :x:                |

## Reporting a vulnerability

Please report security issues privately — do **not** open a public issue for an
unpatched vulnerability.

- **Preferred:** GitHub private vulnerability reporting on this repository
  (the **Security** tab → **Report a vulnerability**).
- **Email:** santiago@topologica.ai

Please include enough detail to reproduce (affected version, environment, and a
minimal proof of concept where possible).

### What to expect

- **Acknowledgement:** within 5 business days.
- **Assessment & next steps:** within 10 business days, including whether the
  report is accepted and an expected fix timeline, or why it was declined.
- **Disclosure:** coordinated. We will agree on a disclosure window with the
  reporter and credit reporters who wish to be named.

## FileLock compatibility constraint (2026-10-23)

The optional PyTorch environments resolved FileLock 3.32.3, inside the affected range for CVE-2026-112940. The `torch` and `bench` extras now require 3.20.3, which predates the affected SQLite locking feature. This is a compatibility constraint, not a vulnerability suppression. Revisit it when upstream PyTorch permits the patched >=4.2.1 line. No reachability of the vulnerable code path from TopoGeoML was established.

Advisory: https://github.com/tox-dev/filelock/security/advisories/GHSA-v4x8-3wr5-3wq8
