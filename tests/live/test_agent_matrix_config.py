from __future__ import annotations

from pathlib import Path

import pytest

from tests.live.agent_matrix import MatrixSelectionError, parse_codex_endpoint


def test_codex_live_endpoint_is_an_explicit_absolute_socket_path() -> None:
    assert parse_codex_endpoint("/run/codex-shared-personal/app-server.sock") == Path(
        "/run/codex-shared-personal/app-server.sock"
    )
    for raw in (None, "", "relative/app-server.sock"):
        with pytest.raises(MatrixSelectionError, match="absolute"):
            parse_codex_endpoint(raw)
