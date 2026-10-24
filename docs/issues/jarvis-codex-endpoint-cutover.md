# jarvis endpoint upgrade prerequisite

jarvis remains pinned to provider-runtime commit
`98ea2f2a8646bf42aec943596f88135512f612bd` (`jarvis/pyproject.toml:23`). its
`src/jarvis/kernel.py:88` accepts `Mapping[str, Path]`; `collector.py:275` supplies
the configured path to `ArchiveHome`. upgrading to the explicit endpoint union
without migrating these owners will fail runtime/archive validation. existing
pins are unaffected. migrate runtime configuration and native archive home
construction to `CodexUnixEndpoint`/`CodexWebSocketEndpoint` before changing its
pin; verify native generation and archive reads through the new public types.
