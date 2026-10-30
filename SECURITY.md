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

## Optional filelock dependency floor (2026-10-29)

The optional `torch` and `bench` environments previously resolved vulnerable
`filelock` versions. In addition to the SQLite-backed read-write lock issue
CVE-2026-112940 (introduced in 3.20.4), the older `UnixFileLock` issue
CVE-2026-112909 also affects 3.20.3 and earlier releases. The known patched
release floor is **4.2.1**. Both optional extras now require
`filelock>=4.2.1` on supported Python 3.11+. We do not pin an artificial
upper bound or suppress any vulnerability advisory.

The full-dependency coverage job installs the real optional dependency closure
and runs `pip-audit --local`; benchmark smoke jobs install the separate
`bench` extra. Such checks provide evidence for the resolved, tested
configuration, not a universal claim that every consumer's environment is safe.
No vulnerable-lock exploit path through TopoGeoML code has been demonstrated.
Users should keep lock directories private from untrusted local accounts.

Advisories:
- https://github.com/tox-dev/filelock/security/advisories/GHSA-v4x8-3wr5-3wq8
- https://github.com/tox-dev/filelock/security/advisories/GHSA-69hj-f68j-2879
