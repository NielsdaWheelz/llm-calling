"""Canonical public codec for bounded provider continuation artifacts."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import cast

from provider_runtime.errors import InvalidRequest
from provider_runtime.registry import _ModelRow, _resolve_target, _row_fingerprint
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    CanonicalTool,
    ContinuationArtifact,
    ContinueGeneration,
    GenerateIntent,
    ImageBlock,
    Present,
    PromptBlock,
    PromptMessage,
    ProviderTarget,
    ResponsePayload,
    StrictJsonOutput,
    SystemMessage,
    TextContent,
    TextOutput,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    canonical_json_bytes,
    freeze_json_object,
    thaw_json_value,
)

_SCHEMA_VERSION = "provider-continuation.v2"
_FIELDS = frozenset({"schema_version", "target", "codec_id", "opaque_payload"})
_TARGET_FIELDS = frozenset({"provider", "model"})
_MAX_ENCODED_BYTES = 16 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


def encode_continuation(artifact: ContinuationArtifact) -> bytes:
    """Encode one artifact into its canonical, target-bound JSON envelope."""
    if not isinstance(artifact, ContinuationArtifact):
        raise InvalidRequest(message="encode_continuation requires ContinuationArtifact")
    if len(canonical_json_bytes(artifact.opaque_payload)) > _MAX_PAYLOAD_BYTES:
        raise InvalidRequest(message="continuation payload exceeds 16 MiB")
    envelope = freeze_json_object(
        {
            "schema_version": _SCHEMA_VERSION,
            "target": {
                "provider": artifact.target.provider,
                "model": artifact.target.model,
            },
            "codec_id": artifact.codec_id,
            "opaque_payload": artifact.opaque_payload,
        },
        context="continuation envelope",
    )
    encoded = canonical_json_bytes(envelope)
    if len(encoded) > _MAX_ENCODED_BYTES:
        raise InvalidRequest(message="encoded continuation exceeds 16 MiB")
    return encoded


def decode_continuation(
    encoded: bytes,
    target: ProviderTarget,
    codec_id: str,
) -> ContinuationArtifact:
    """Decode canonical bytes and require the caller's exact target and codec."""
    if not isinstance(encoded, bytes):
        raise InvalidRequest(message="continuation encoding must be bytes")
    if len(encoded) > _MAX_ENCODED_BYTES:
        raise InvalidRequest(message="encoded continuation exceeds 16 MiB")
    if not isinstance(target, ProviderTarget):
        raise InvalidRequest(message="continuation target must be ProviderTarget")
    if type(codec_id) is not str or not codec_id:
        raise InvalidRequest(message="continuation codec_id must be a non-empty string")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise InvalidRequest(message="continuation encoding is not valid JSON") from None
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise InvalidRequest(message="continuation envelope has an invalid field set")
    if value.get("schema_version") != _SCHEMA_VERSION:
        raise InvalidRequest(message="continuation schema version is unsupported")
    encoded_target = value.get("target")
    if not isinstance(encoded_target, Mapping) or set(encoded_target) != _TARGET_FIELDS:
        raise InvalidRequest(message="continuation target is malformed")
    if (
        encoded_target.get("provider") != target.provider
        or encoded_target.get("model") != target.model
    ):
        raise InvalidRequest(message="continuation target does not match the requested target")
    if value.get("codec_id") != codec_id:
        raise InvalidRequest(message="continuation codec does not match the requested codec")
    payload = value.get("opaque_payload")
    if not isinstance(payload, Mapping):
        raise InvalidRequest(message="continuation opaque_payload must be a JSON object")
    try:
        artifact = ContinuationArtifact(
            target=target,
            codec_id=codec_id,
            opaque_payload=payload,
        )
    except (TypeError, ValueError):
        raise InvalidRequest(
            message="continuation payload is outside the bounded JSON domain"
        ) from None
    if encode_continuation(artifact) != encoded:
        raise InvalidRequest(message="continuation encoding is not canonical")
    return artifact


# A continuation is one frozen request plus a flat sequence of native assistant
# turns and their ordered tool results. Native turns are never reconstructed
# from display text, and an earlier outer artifact is never nested in a newer one.

_GENERATION_SCHEMA = "provider-generation.v2"


def _first_state(row: _ModelRow, intent: GenerateIntent) -> dict[str, object]:
    messages: list[dict[str, object]] = []
    for message in intent.messages:
        if isinstance(message, SystemMessage):
            messages.append({"role": "system", "blocks": [block.text for block in message.blocks]})
        elif isinstance(message, UserMessage):
            blocks: list[dict[str, object]] = []
            for block in message.blocks:
                if isinstance(block, PromptBlock):
                    blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    blocks.append(
                        {
                            "type": "image",
                            "media_type": block.media_type,
                            "data": base64.b64encode(block.data).decode("ascii"),
                        }
                    )
            messages.append({"role": "user", "blocks": blocks})
        else:
            raise InvalidRequest(message="initial generation accepts only system and user messages")
    if isinstance(intent.output, TextOutput):
        output: dict[str, object] = {"type": "text"}
    else:
        output = {
            "type": "strict_json",
            "name": intent.output.name,
            "schema": dict(intent.output.schema),
        }
    return {
        "schema": _GENERATION_SCHEMA,
        "row_fingerprint": _row_fingerprint(row),
        "first": {
            "messages": messages,
            "max_output_tokens": intent.max_output_tokens,
            "reasoning": intent.reasoning,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                }
                for tool in intent.tools
            ],
            "tool_choice": intent.tool_choice,
            "output": output,
            "provider_options": dict(intent.provider_options),
        },
        "turns": [],
    }


def preflight_generation(row: _ModelRow, intent: GenerateIntent) -> dict[str, object]:
    """Reject a tool-capable prefix that cannot fit its required artifact before IO."""
    state = _first_state(row, intent)
    if not intent.tools or intent.tool_choice == "none":
        return state
    try:
        encode_continuation(ContinuationArtifact(intent.target, row.continuation_codec, state))
    except (TypeError, ValueError, InvalidRequest) as error:
        raise InvalidRequest(
            message=f"generation prefix cannot fit a continuation: {error}"
        ) from None
    return state


def resume_generation(
    row: _ModelRow, request: ContinueGeneration
) -> tuple[GenerateIntent, dict[str, object]]:
    artifact = request.continuation
    encode_continuation(artifact)
    if (
        artifact.target != ProviderTarget(row.provider, row.model_id)
        or artifact.codec_id != row.continuation_codec
    ):
        raise InvalidRequest(message="continuation target or codec does not match current model")
    payload = thaw_json_value(artifact.opaque_payload)
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "row_fingerprint",
        "first",
        "turns",
    }:
        raise InvalidRequest(message="continuation has an invalid generation envelope")
    if payload["schema"] != _GENERATION_SCHEMA or payload["row_fingerprint"] != _row_fingerprint(
        row
    ):
        raise InvalidRequest(message="continuation does not match the current model definition")
    first = payload["first"]
    turns = payload["turns"]
    if not isinstance(first, dict) or not isinstance(turns, list) or not turns:
        raise InvalidRequest(message="continuation has no valid pending turn")
    last = turns[-1]
    if (
        not isinstance(last, dict)
        or set(last) != {"text", "calls", "native", "results"}
        or last["results"] is not None
    ):
        raise InvalidRequest(message="continuation is not pending tool results")
    calls = last["calls"]
    if (
        not isinstance(calls, list)
        or not calls
        or not all(isinstance(call, dict) for call in calls)
    ):
        raise InvalidRequest(message="continuation has no pending tool calls")
    expected = [call.get("id") for call in calls]
    actual = [result.call_id for result in request.tool_results]
    if len(set(expected)) != len(expected) or actual != expected:
        raise InvalidRequest(message="tool results must match pending call IDs in exact order")
    last["results"] = [
        {"call_id": result.call_id, "output": result.output, "is_error": result.is_error}
        for result in request.tool_results
    ]
    encode_continuation(ContinuationArtifact(artifact.target, artifact.codec_id, payload))
    try:
        original = cast("dict[str, object]", first)
        source_messages = cast("list[dict[str, object]]", original["messages"])
        messages: list[PromptMessage] = []
        for entry in source_messages:
            if entry["role"] == "system":
                messages.append(
                    SystemMessage(
                        blocks=tuple(
                            PromptBlock(text=text) for text in cast("list[str]", entry["blocks"])
                        )
                    )
                )
            elif entry["role"] == "user":
                blocks = []
                for block in cast("list[dict[str, str]]", entry["blocks"]):
                    if block["type"] == "text":
                        blocks.append(PromptBlock(text=block["text"]))
                    elif block["type"] == "image":
                        blocks.append(
                            ImageBlock(
                                media_type=block["media_type"],
                                data=base64.b64decode(block["data"], validate=True),
                            )
                        )
                    else:
                        raise ValueError("unknown prompt block")
                messages.append(UserMessage(blocks=tuple(blocks)))
            else:
                raise ValueError("unknown prompt role")
        for turn in turns:
            native = cast("dict[str, object]", turn["native"])
            calls = tuple(
                ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
                for call in turn["calls"]
            )
            messages.append(
                AssistantMessage(
                    text=turn["text"],
                    tool_calls=calls,
                    continuation=Present(
                        ContinuationArtifact(artifact.target, row.continuation_codec, native)
                    ),
                )
            )
            messages.extend(ToolResultMessage(**result) for result in turn["results"])
        output_data = cast("dict[str, object]", original["output"])
        if output_data["type"] == "text":
            output = TextOutput()
        elif output_data["type"] == "strict_json":
            output = StrictJsonOutput(
                name=cast("str", output_data["name"]),
                schema=cast("dict[str, object]", output_data["schema"]),
            )
        else:
            raise ValueError("unknown output mode")
        choice = original["tool_choice"]
        if choice not in ("auto", "none"):
            raise ValueError("unknown tool choice")
        intent = GenerateIntent(
            target=artifact.target,
            messages=tuple(messages),
            max_output_tokens=cast("int", original["max_output_tokens"]),
            reasoning=cast("str", original["reasoning"]),
            tools=tuple(
                CanonicalTool(
                    name=cast("str", tool["name"]),
                    description=cast("str", tool["description"]),
                    parameters=cast("dict[str, object]", tool["parameters"]),
                )
                for tool in cast("list[dict[str, object]]", original["tools"])
            ),
            tool_choice=("none" if choice == "none" else "auto"),
            output=output,
            provider_options=cast("dict[str, object]", original["provider_options"]),
        )
        return intent, payload
    except (KeyError, TypeError, ValueError, IndexError):
        raise InvalidRequest(message="continuation payload is malformed") from None


def seal_generation(
    row: _ModelRow,
    intent: GenerateIntent,
    state: dict[str, object],
    response: ResponsePayload,
) -> ResponsePayload:
    """Append one complete native assistant turn only when tools remain pending."""
    content = response.content
    if not isinstance(content, TextContent) or not content.tool_calls:
        return ResponsePayload(content=content, continuation=Absent())
    if not isinstance(response.continuation, Present):
        raise InvalidRequest(message="provider omitted native continuation for tool-bearing turn")
    native = response.continuation.value
    if native.target != intent.target or native.codec_id != row.continuation_codec:
        raise InvalidRequest(message="provider emitted a mismatched native continuation")
    turns = cast("list[object]", state["turns"])
    turns.append(
        {
            "text": content.text,
            "calls": [
                {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                for call in content.tool_calls
            ],
            "native": thaw_json_value(native.opaque_payload),
            "results": None,
        }
    )
    artifact = ContinuationArtifact(intent.target, row.continuation_codec, state)
    try:
        encode_continuation(artifact)
    except InvalidRequest as error:
        raise ValueError(str(error)) from None
    return ResponsePayload(content=content, continuation=Present(artifact))


def pending_tool_calls(artifact: ContinuationArtifact) -> tuple[ToolCall, ...]:
    """Read exact pending calls from a complete, current catalog continuation."""
    row = _resolve_target(artifact.target)
    encode_continuation(artifact)
    if artifact.codec_id != row.continuation_codec:
        raise InvalidRequest(message="continuation codec does not match current model")
    payload = thaw_json_value(artifact.opaque_payload)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "row_fingerprint", "first", "turns"}
        or payload["schema"] != _GENERATION_SCHEMA
        or payload["row_fingerprint"] != _row_fingerprint(row)
    ):
        raise InvalidRequest(message="continuation does not match the current model definition")
    turns = payload["turns"]
    if not isinstance(turns, list) or not turns or not isinstance(turns[-1], dict):
        raise InvalidRequest(message="continuation has no pending tool calls")
    last = turns[-1]
    if last.get("results") is not None or not isinstance(last.get("calls"), list):
        raise InvalidRequest(message="continuation is not pending tool results")
    try:
        calls = tuple(
            ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
            for call in last["calls"]
        )
    except (KeyError, TypeError, ValueError):
        raise InvalidRequest(message="continuation pending tool calls are malformed") from None
    if not calls or len({call.id for call in calls}) != len(calls):
        raise InvalidRequest(message="continuation pending tool call ids are invalid")
    resume_generation(
        row,
        ContinueGeneration(
            continuation=artifact,
            tool_results=tuple(
                ToolResultMessage(call_id=call.id, output="", is_error=False) for call in calls
            ),
        ),
    )
    return calls
