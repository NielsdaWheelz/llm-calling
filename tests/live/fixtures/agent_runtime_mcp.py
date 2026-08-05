"""Tiny no-network MCP stdio server used by the opt-in live agent matrix."""

from __future__ import annotations

import json
import sys


def _result(identifier: object, result: object) -> None:
    sys.stdout.write(
        json.dumps(
            {"jsonrpc": "2.0", "id": identifier, "result": result},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stdout.flush()


def _error(identifier: object, code: int, message: str) -> None:
    sys.stdout.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": identifier,
                "error": {"code": code, "message": message},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stdout.flush()


def _handle(message: object) -> None:
    if not isinstance(message, dict):
        return
    identifier = message.get("id")
    method = message.get("method")
    if identifier is None or not isinstance(method, str):
        return
    if method == "initialize":
        params = message.get("params")
        protocol_version = (
            params.get("protocolVersion") if isinstance(params, dict) else "2025-06-18"
        )
        _result(
            identifier,
            {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "provider-runtime-live-certifier", "version": "1"},
            },
        )
    elif method == "ping":
        _result(identifier, {})
    elif method == "tools/list":
        _result(
            identifier,
            {
                "tools": [
                    {
                        "name": "live_probe",
                        "description": "Return a fixed live-certification marker.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ]
            },
        )
    elif method == "tools/call":
        _result(
            identifier,
            {"content": [{"type": "text", "text": "AGENT_RUNTIME_MCP_LIVE_OK"}]},
        )
    else:
        _error(identifier, -32601, "method not supported by live certifier")


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        _handle(message)


if __name__ == "__main__":
    main()
