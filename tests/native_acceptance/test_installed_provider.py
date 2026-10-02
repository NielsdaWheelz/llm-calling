"""Temporary actual native/model proof; echo callbacks are not research receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from provider_runtime.agent_runtime import (
    AgentAccepted,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentTerminal,
    AgentToolCall,
    AgentToolReply,
    AgentTurnControls,
    CodexCatalogSessionRequest,
    CodexNativeOptions,
    CredentialRef,
    JsonSchemaAgentOutput,
    NativeTerminalEvidence,
    NewSession,
    PermissionPolicy,
    TextContent,
    TurnRequest,
    decode_agent_output,
    terminal_to_json,
)
from provider_runtime.types import CanonicalTool, thaw_json_value


@pytest.mark.skipif(
    "NATIVE_PROVIDER_RECEIPT" not in os.environ, reason="explicit isolated live host"
)
@pytest.mark.parametrize("declared_tools", (False, True))
async def test_actual_personal_selected_model_structured_callback(declared_tools: bool) -> None:
    receipt_path = Path(os.environ["NATIVE_PROVIDER_RECEIPT"])
    host = json.loads(receipt_path.read_text())
    auth = CredentialRef("local_account", "codex-personal")
    output = JsonSchemaAgentOutput(
        name="qualification",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "string", "const": "owned live callback result"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    async with AgentRuntime(
        AgentRuntimeConfig(
            state_root_base=Path(host["worker_state"]),
            codex_endpoints={"codex-personal": Path(host["socket"])},
        )
    ) as runtime:
        catalog = await runtime.model_catalog(backend="codex", transport="sdk", auth=auth)
        row = next(model for model in catalog.models if model.key == "gpt-6-luna")
        assert any(reasoning.key == "xhigh" for reasoning in row.reasoning)
        owner = await runtime.open_session(
            CodexCatalogSessionRequest(
                auth=auth,
                open=NewSession(),
                cwd=host["cwd"],
                policy=PermissionPolicy(allowed_tools=("*",)),
                model_key=row.key,
                reasoning="xhigh",
                agent_definition_revision=catalog.definition_revision,
                row_fingerprint=row.row_fingerprint,
                native=CodexNativeOptions(builtin_tools="disabled", web_search=False),
                output=output,
                tools=(
                    CanonicalTool(
                        name="probe__echo",
                        description="return the exact supplied text for native callback qualification",
                        parameters={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    ),
                )
                if declared_tools
                else (),
            )
        )
        turn = runtime.prepare_turn(
            owner,
            TurnRequest(
                input=(
                    TextContent(
                        'call probe__echo with {"text":"owned live callback result"}. '
                        'then return the tool result as {"answer":"owned live callback result"}.'
                        if declared_tools
                        else 'return {"answer":"owned live callback result"} without using tools.'
                    ),
                ),
                timeout_seconds=300,
            ),
            attempt_id=f"installed-personal-callback-{uuid4()}",
            input_id=f"installed-personal-callback-input-{uuid4()}",
            controls=AgentTurnControls(rpc_seconds=30, pending_calls=4, pending_call_bytes=32768),
        )
        result = await turn.submit()
        assert isinstance(result, AgentAccepted)
        calls = 0
        terminal: AgentTerminal | None = None
        async for event in turn.events():
            if isinstance(event, AgentToolCall):
                assert event.name == "probe__echo"
                assert event.arguments == {"text": "owned live callback result"}
                calls += 1
                await turn.reply(
                    event, AgentToolReply(text="owned live callback result", success=True)
                )
            if isinstance(event, AgentTerminal):
                terminal = event
        assert calls == int(declared_tools)
        assert terminal is not None
        assert terminal.status == "succeeded"
        assert isinstance(terminal.evidence, NativeTerminalEvidence)
        assert thaw_json_value(decode_agent_output(output, terminal)) == {
            "answer": "owned live callback result"
        }
        qualified = {
            "kind": "actual_native_echo_callback" if declared_tools else "actual_native_no_tools",
            "actual_research": False,
            "native_version": host["native_version"],
            "credential_profile": auth.profile_key,
            "selected_model": row.key,
            "selected_dispatch_model": row.dispatch_model,
            "selected_reasoning": "xhigh",
            "submitted_request": thaw_json_value(turn.submitted_request),
            "terminal": terminal_to_json(terminal),
            "calls": calls,
        }
        receipt_path.with_name(
            "native-callback.json" if declared_tools else "native-no-tools.json"
        ).write_text(json.dumps(qualified, indent=2) + "\n")
        closed = await turn.close()
        assert closed.local_closed
