# Vexx 2.1 Security Model

The code execution subsystem runs candidate code in an isolated subprocess
(`python -I -S`), never in the Vexx host process.

Guarantees provided:
- Out-of-process execution.
- Minimal environment (no API keys/secrets forwarded).
- Temporary working directory, cleaned on every exit path.
- Wall timeout and POSIX rlimits (CPU, address space, nofile, fsize, core).
- AST policy + import allow-list at both AST and runtime import layers.
- Honest `limits_enforced` / `network_isolated` reporting.

Not a complete OS security boundary: restricted builtins and AST checks alone
are insufficient for truly hostile code. Network isolation depends on the host
OS/container; `network_isolated` is reported `False` when it cannot be enforced.
There is never a silent fallback to in-process `exec`.