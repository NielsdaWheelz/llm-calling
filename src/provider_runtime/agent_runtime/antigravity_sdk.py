"""One native Antigravity turn inside the isolated paid host."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from secrets import token_hex
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from provider_runtime.types import (
    Absent,
    CanonicalTool,
    Presence,
    Present,
    TokenUsage,
    canonical_json_bytes,
)

from ._limits import (
    _MAX_EVENT_COUNT,
    _MAX_EVENT_TEXT_BYTES,
    _MAX_MESSAGE_ITEMS,
    _MAX_TURN_OUTPUT_BYTES,
    OutputLimitExceeded,
    bounded_payload_size,
)
from ._structured_output import OutputSchemaMismatch, freeze_structured_output
from .codex_app_server import NativeRequestFit
from .errors import InvalidAgentRequest, ProtocolDefect, SessionUnavailable
from .events import (
    AgentEvent,
    AgentFailure,
    AgentFailureCause,
    AgentTerminal,
    AgentTerminalStatus,
    AgentToolUse,
    StockToolObservation,
)
from .neon_config import AGY_SDK_VERSION, NEON_MAX_OUTPUT_TOKENS, NEON_MODEL
from .turn import (
    AgentAccepted,
    AgentAttempt,
    AgentCloseResult,
    AgentControlReceipt,
    AgentInputRecorded,
    AgentMessage,
    AgentNotSubmitted,
    AgentSubmission,
    AgentToolCall,
    AgentToolReply,
    AgentTurnControls,
    AgentTurnRef,
    AgentUncertain,
    LocalStopEvidence,
    NativeTerminalEvidence,
    RawAgentOutput,
)
from .types import (
    AgentOutputSpec,
    ContentPart,
    JsonObject,
    JsonSchemaAgentOutput,
    TextAgentOutput,
    TextContent,
    TurnRequest,
    freeze_json_object,
    freeze_json_value,
    thaw_json_value,
)

if TYPE_CHECKING:
    import google.antigravity.conversation.conversation as sdk_conversation  # pyright: ignore[reportMissingImports]
    from google.antigravity import types as sdk  # pyright: ignore[reportMissingImports]

_BINDING_FIELD = "jarvis_binding"
# The pinned SDK rewrites these before protobuf encoding. Reject loss of exact input.
_SANITIZED_BY_SDK = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff]")
_WORKSPACE_TOOLS = (
    "list_directory",
    "search_directory",
    "find_file",
    "view_file",
    "create_file",
    "edit_file",
    "run_command",
    "invoke_subagent",
)


def antigravity_native_request_fit(
    text: str,
    *,
    output: AgentOutputSpec,
    input_id: str,
    operation: Literal["start", "steer"] = "start",
) -> NativeRequestFit:
    """Measure the pinned SDK user-input envelope against the shared host bounds."""
    if operation != "start":
        raise InvalidAgentRequest("Antigravity accepts follow-ups in a fresh turn")
    if (not text.strip() and text != " ") or _SANITIZED_BY_SDK.search(text):
        raise InvalidAgentRequest("Antigravity would rewrite the prepared text input")
    # Output schema configures the session and input_id belongs to the host journal.
    # Neither is part of the SDK input event. Local wire fit and model context differ.
    wire = json.dumps({"userInput": {"parts": [{"text": text}]}}, ensure_ascii=True, indent=2)
    return NativeRequestFit(len(text), len(wire))


@dataclass(frozen=True, slots=True)
class AntigravitySdkOptions:
    endpoint_url: str
    attempt_bearer: str = field(repr=False)
    cwd: Path
    state_dir: Path
    system_instructions: str
    workspace_tools: bool
    output: AgentOutputSpec = TextAgentOutput()


@dataclass(frozen=True, slots=True)
class _CallbackBinding:
    step_id: str
    tool: str
    arguments: JsonObject
    size: int


def _usage(reported: sdk.UsageMetadata) -> Presence[TokenUsage]:
    if reported.prompt_token_count is None or reported.candidates_token_count is None:
        return Absent()
    reasoning = reported.thoughts_token_count
    total = reported.total_token_count
    cached = reported.cached_content_token_count
    return Present(
        TokenUsage.from_components(
            input_tokens=reported.prompt_token_count,
            output_tokens=reported.candidates_token_count + (reasoning or 0),
            total_tokens=Absent() if total is None else Present(total),
            reasoning_tokens=Absent() if reasoning is None else Present(reasoning),
            cache_read_input_tokens=Absent() if cached is None else Present(cached),
            cache_write_input_tokens=Absent(),
        )
    )


class AntigravitySdkTurn:
    def __init__(
        self,
        *,
        turn: AgentTurnRef,
        attempt: AgentAttempt,
        submitted_request: JsonObject,
        input_id: str,
        request: TurnRequest,
        controls: AgentTurnControls,
        options: AntigravitySdkOptions,
        tools: tuple[CanonicalTool, ...] = (),
    ) -> None:
        if len(request.input) != 1 or not isinstance(request.input[0], TextContent):
            raise InvalidAgentRequest("Antigravity requires one exact text input")
        if turn.session_ref.backend != "antigravity":
            raise InvalidAgentRequest("Antigravity requires its prepared session")
        if not antigravity_native_request_fit(
            request.input[0].text,
            output=options.output,
            input_id=input_id,
        ).fits:
            raise InvalidAgentRequest("Antigravity prepared input exceeds the local wire bounds")
        self._turn = turn
        self._root = turn.session_ref.native_session_id
        self._attempt = attempt
        self._submitted_request = submitted_request
        self._input_id = input_id
        self._text = request.input[0].text
        self._timeout = request.timeout_seconds
        self._controls = controls
        self._options = options
        self._tools = tools
        self._submission: AgentSubmission | None = None
        self._accepted: asyncio.Future[AgentSubmission] | None = None
        self._terminal: AgentTerminal | None = None
        self._reported_usage: Presence[TokenUsage] = Absent()
        self._runner: asyncio.Task[None] | None = None
        self._conversation: sdk_conversation.Conversation | None = None
        self._events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._event_count = 0
        self._event_bytes = 0
        self._consumed = False
        self._revoked = False
        self._interrupted = False
        self._close: AgentCloseResult | None = None
        self._bindings: dict[str, _CallbackBinding] = {}
        self._proposals: dict[str, tuple[str, bytes]] = {}
        self._calls: dict[str, tuple[AgentToolCall, asyncio.Future[AgentToolReply], int]] = {}
        self._callback_defect = False

    @property
    def attempt(self) -> AgentAttempt:
        return self._attempt

    @property
    def submitted_request(self) -> JsonObject:
        return self._submitted_request

    @property
    def submission(self) -> AgentSubmission | None:
        return self._submission

    @property
    def terminal(self) -> AgentTerminal | None:
        return self._terminal

    async def submit(self) -> AgentSubmission:
        if self._runner is not None or self._revoked:
            raise SessionUnavailable("Antigravity turn is no longer available for submission")
        self._accepted = asyncio.get_running_loop().create_future()
        self._runner = asyncio.create_task(self._run(self._accepted))
        return await asyncio.shield(self._accepted)

    async def check_submission(self, *, for_effect: bool = False) -> None:
        if not isinstance(self._submission, AgentAccepted):
            raise SessionUnavailable("Antigravity has not recorded the exact input")
        if for_effect and (self._revoked or self._terminal is not None):
            raise SessionUnavailable("Antigravity turn has no live effect authority")

    async def events(self) -> AsyncIterator[AgentEvent]:
        if self._consumed:
            raise ProtocolDefect("Antigravity events already have a consumer")
        self._consumed = True
        while (event := await self._events.get()) is not None:
            yield event

    async def reply(self, call: AgentToolCall, result: AgentToolReply) -> None:
        await self.check_submission(for_effect=True)
        pending = self._calls.get(call.reply_token)
        if pending is None or pending[0] != call or pending[1].done():
            raise ProtocolDefect("Antigravity tool reply did not match a pending call")
        pending[1].set_result(result)

    async def steer(self, *, input_id: str, input: tuple[ContentPart, ...]) -> AgentControlReceipt:
        return AgentControlReceipt(
            "steer",
            str(uuid4()),
            self._turn,
            "not_sent",
            input_id,
            "Antigravity handles follow-ups in a fresh turn",
        )

    def revoke(self) -> None:
        self._revoked = True
        self._bindings.clear()
        for _, future, _ in self._calls.values():
            future.cancel()

    async def interrupt(self) -> AgentControlReceipt:
        self.revoke()
        conversation = self._conversation
        disposition = "not_sent"
        if conversation is not None and self._terminal is None and not self._interrupted:
            self._interrupted = True
            disposition = "unknown"
            try:
                async with asyncio.timeout(self._controls.rpc_seconds):
                    await conversation.cancel()
            except (TimeoutError, OSError, RuntimeError):
                pass
        return AgentControlReceipt(
            "interrupt",
            str(uuid4()),
            self._turn,
            disposition,
            None,
            "Antigravity stop requires a terminal observation",
        )

    async def close(self) -> AgentCloseResult:
        if self._close is not None:
            return self._close
        self.revoke()
        runner = self._runner
        if runner is None:
            self._events.put_nowait(None)
        elif not runner.done():
            await self.interrupt()
            try:
                async with asyncio.timeout(self._controls.rpc_seconds):
                    await asyncio.shield(runner)
            except TimeoutError:
                runner.cancel()
                try:
                    async with asyncio.timeout(self._controls.rpc_seconds):
                        await asyncio.shield(runner)
                except TimeoutError:
                    return AgentCloseResult(False, ("Antigravity runtime did not close",))
        self._close = AgentCloseResult(True, ())
        return self._close

    def _emit(self, event: AgentEvent, size: int = 0) -> None:
        self._event_count += 1
        self._event_bytes += size
        if self._event_count > _MAX_EVENT_COUNT or self._event_bytes > _MAX_TURN_OUTPUT_BYTES:
            raise OutputLimitExceeded(_MAX_TURN_OUTPUT_BYTES, unit="Antigravity output")
        self._events.put_nowait(event)

    async def _run(self, accepted: asyncio.Future[AgentSubmission]) -> None:
        try:
            if version("google-antigravity") != AGY_SDK_VERSION:
                return
            await self._run_sdk(accepted)
        except asyncio.CancelledError:
            if self._callback_defect:
                self._finish_local(
                    "failed", "backend_failed", "Antigravity callback identity conflicted"
                )
            else:
                self._finish_local("cancelled", None, "Antigravity local runtime was stopped")
        except Exception:
            self._finish_local("failed", "backend_failed", "Antigravity runtime failed")
        finally:
            self._conversation = None
            self.revoke()
            if self._submission is None:
                self._submission = AgentNotSubmitted(
                    self._attempt, "Antigravity runtime unavailable"
                )
            if not accepted.done():
                accepted.set_result(self._submission)
            if self._terminal is not None:
                self._events.put_nowait(self._terminal)
            self._events.put_nowait(None)

    def _finish_local(
        self, status: AgentTerminalStatus, cause: AgentFailureCause | None, reason: str
    ) -> None:
        if isinstance(self._submission, AgentAccepted | AgentUncertain):
            self._terminal = AgentTerminal(
                status,
                AgentFailure(cause) if cause else None,
                "",
                self._turn.session_ref,
                LocalStopEvidence(self._submission, reason),
                usage=self._reported_usage,
            )

    async def _run_sdk(self, accepted: asyncio.Future[AgentSubmission]) -> None:
        import google.antigravity as agy  # pyright: ignore[reportMissingImports]
        from google.antigravity import types  # pyright: ignore[reportMissingImports]
        from google.antigravity.connections import local  # pyright: ignore[reportMissingImports]
        from google.antigravity.hooks import hooks, policy  # pyright: ignore[reportMissingImports]
        from google.antigravity.tools import tool_runner  # pyright: ignore[reportMissingImports]

        denial = "Antigravity tool authority was not established."
        options = self._options
        tool_names = {tool.name for tool in self._tools}
        workspace = _WORKSPACE_TOOLS if options.workspace_tools else ()
        observations: dict[str, str] = {}
        schema = (
            thaw_json_value(options.output.schema)
            if isinstance(options.output, JsonSchemaAgentOutput)
            else None
        )

        @hooks.pre_tool_call_decide
        async def authorize(_context: hooks.HookContext, call: types.ToolCall) -> types.HookResult:
            name = str(call.name)
            if not isinstance(await asyncio.shield(accepted), AgentAccepted) or self._revoked:
                return types.HookResult(allow=False, message=denial)
            if call.server_name:
                return types.HookResult(allow=False, message=denial)
            if name == "finish" and schema is not None:
                return types.HookResult(allow=True)
            step_id = call.step_id or ""
            trajectory, _, sequence = step_id.partition(":")
            if (
                not trajectory
                or len(step_id) > 256
                or not sequence.isascii()
                or not sequence.isdecimal()
            ):
                return types.HookResult(allow=False, message=denial)
            if name in workspace:
                if step_id in observations:
                    return types.HookResult(allow=False, message=denial)
                size = bounded_payload_size(
                    call.args, _MAX_EVENT_TEXT_BYTES, max_items=_MAX_MESSAGE_ITEMS
                )
                self._emit(
                    AgentToolUse(
                        step_id,
                        name,
                        "started",
                        freeze_json_object(call.args),
                        stock=StockToolObservation(),
                    ),
                    size,
                )
                observations[step_id] = name
                return types.HookResult(allow=True)
            if name not in tool_names or trajectory != self._root:
                return types.HookResult(allow=False, message=denial)
            try:
                bounded_payload_size(
                    call.args,
                    self._controls.pending_call_bytes,
                    max_items=_MAX_MESSAGE_ITEMS,
                )
                arguments = freeze_json_object(
                    {key: value for key, value in call.args.items() if key != _BINDING_FIELD}
                )
                encoded = canonical_json_bytes(arguments)
            except (InvalidAgentRequest, OutputLimitExceeded):
                return types.HookResult(allow=False, message=denial)
            if (
                len(self._bindings) + len(self._calls) >= self._controls.pending_calls
                or len(encoded)
                + sum(item.size for item in self._bindings.values())
                + sum(item[2] for item in self._calls.values())
                > self._controls.pending_call_bytes
            ):
                return types.HookResult(allow=False, message=denial)
            proposal = name, sha256(encoded).digest()
            previous = self._proposals.get(step_id)
            if (previous is not None and previous != proposal) or (
                previous is None and len(self._proposals) >= _MAX_EVENT_COUNT
            ):
                self._callback_defect = True
                self.revoke()
                if self._runner is not None:
                    self._runner.cancel()
                return types.HookResult(allow=False, message=denial)
            self._proposals[step_id] = proposal
            proof = token_hex(32)
            self._bindings[proof] = _CallbackBinding(step_id, name, arguments, len(encoded))
            return types.HookResult(allow=True, modified_args={_BINDING_FIELD: proof})

        @hooks.post_tool_call
        async def observe(_context: hooks.HookContext, result: types.ToolResult) -> None:
            step_id = result.step_id
            name = str(result.name)
            if step_id is None or observations.get(step_id) != name or result.server_name:
                return
            value = result.error if result.error is not None else result.result
            size = bounded_payload_size(value, _MAX_EVENT_TEXT_BYTES, max_items=_MAX_MESSAGE_ITEMS)
            stock = StockToolObservation()
            if isinstance(value, local.RunCommandResult):
                # The SDK reports shell output text. It exposes no typed shell exit code.
                value = value.output
                stock = StockToolObservation(shell_output="reported")
            elif hasattr(value, "model_dump"):
                value = value.model_dump(mode="json")
            self._emit(
                AgentToolUse(
                    step_id,
                    name,
                    "completed",
                    freeze_json_value(value),
                    result.error is None,
                    stock,
                ),
                size,
            )
            del observations[step_id]

        def register(tool: CanonicalTool) -> tool_runner.ToolWithSchema:
            tool_schema = cast(
                dict[str, object], thaw_json_value(freeze_json_object(tool.parameters))
            )
            properties = tool_schema.get("properties")
            if (isinstance(properties, dict) and _BINDING_FIELD in properties) or tool.name in {
                *_WORKSPACE_TOOLS,
                "finish",
                "manage_task",
            }:
                raise InvalidAgentRequest("Antigravity tool declaration used a reserved name")

            async def invoke(jarvis_binding: object = None, **arguments: object) -> str:
                binding = (
                    self._bindings.pop(jarvis_binding, None)
                    if type(jarvis_binding) is str
                    else None
                )
                if (
                    self._revoked
                    or binding is None
                    or binding.tool != tool.name
                    or self._proposals[binding.step_id][1]
                    != sha256(canonical_json_bytes(freeze_json_object(arguments))).digest()
                ):
                    raise types.ToolExecutionError(denial, tool.name)
                call = AgentToolCall(
                    self._turn,
                    binding.step_id,
                    token_hex(32),
                    tool.name,
                    binding.arguments,
                )
                reply: asyncio.Future[AgentToolReply] = asyncio.get_running_loop().create_future()
                self._calls[call.reply_token] = (call, reply, binding.size)
                try:
                    self._emit(call, binding.size)
                    result = await reply
                finally:
                    del self._calls[call.reply_token]
                if not result.success:
                    raise types.ToolExecutionError(result.text, tool.name, step_id=binding.step_id)
                return result.text

            invoke.__name__ = tool.name
            invoke.__doc__ = tool.description
            return tool_runner.ToolWithSchema(invoke, tool_schema)

        status: AgentTerminalStatus = "failed"
        cause: AgentFailureCause | None = "backend_failed"
        public: AgentMessage | None = None
        public_index = -1
        latest_action = -1
        raw_output: RawAgentOutput | None = None
        with TemporaryDirectory(prefix="antigravity-", dir=options.state_dir) as state:
            endpoint = agy.GeminiAPIEndpoint(
                api_key=options.attempt_bearer,
                base_url=options.endpoint_url,
                options=agy.GeminiModelOptions(
                    thinking_level=agy.ThinkingLevel.HIGH,
                    service_tier=types.ServiceTier.STANDARD,
                    max_output_tokens=NEON_MAX_OUTPUT_TOKENS,
                ),
            )
            configuration = agy.LocalAgentConfig(
                system_instructions=options.system_instructions,
                model=agy.ModelTarget(name=NEON_MODEL, endpoint=endpoint),
                api_key=options.attempt_bearer,
                vertex=False,
                conversation_id=self._root,
                session_continuation_mode=types.SessionContinuationMode.CREATE_ONLY,
                workspaces=[str(options.cwd)],
                app_data_dir=str(Path(state) / "app"),
                save_dir=str(Path(state) / "sessions"),
                response_schema=schema,
                retry_config=agy.RetryConfig(
                    api_retry=agy.ModelAPIRetryConfig(max_retries=0),
                    model_output_retry=agy.ModelOutputRetryConfig(max_retries=0),
                ),
                capabilities=types.CapabilitiesConfig(
                    enable_subagents=options.workspace_tools,
                    enabled_tools=[types.BuiltinTools(name) for name in workspace],
                    max_subagent_depth=2 if options.workspace_tools else None,
                    run_command_config=types.RunCommandConfig(enable_sandbox=True)
                    if workspace
                    else None,
                ),
                policies=[policy.allow("run_command")] if workspace else [],
                tools=[register(tool) for tool in self._tools],
                hooks=[authorize, observe],
            )
            try:
                async with asyncio.timeout(self._timeout):
                    async with agy.Agent(configuration) as agent:
                        conversation = agent.conversation
                        if workspace and (
                            conversation.sandbox_status is None
                            or not conversation.sandbox_status.available
                        ):
                            return
                        if self._revoked:
                            return
                        self._conversation = conversation
                        self._submission = AgentUncertain(
                            self._attempt,
                            self._turn,
                            "Antigravity input delivery was unconfirmed",
                        )
                        try:
                            await conversation.send(self._text)
                            async for step in conversation.receive_steps():
                                if (
                                    step.trajectory_id != self._root
                                    or step.depth != 0
                                    or step.parent_trajectory_id
                                ):
                                    continue
                                if (
                                    step.source == types.StepSource.MODEL
                                    and step.type == types.StepType.TOOL_CALL
                                    and step.target == types.StepTarget.ENVIRONMENT
                                ):
                                    latest_action = max(latest_action, step.step_index)
                                if (
                                    step.source == types.StepSource.USER
                                    and step.status == types.StepStatus.DONE
                                ):
                                    if step.content != self._text or accepted.done() or not step.id:
                                        raise ProtocolDefect(
                                            "Antigravity changed the prepared root input"
                                        )
                                    self._emit(
                                        AgentInputRecorded(self._turn, self._input_id, step.id)
                                    )
                                    self._submission = AgentAccepted(self._attempt, self._turn)
                                    accepted.set_result(self._submission)
                                elif (
                                    step.source == types.StepSource.MODEL
                                    and step.type == types.StepType.TEXT_RESPONSE
                                    and step.target == types.StepTarget.USER
                                    and step.status == types.StepStatus.DONE
                                    and step.is_complete_response
                                    and step.step_index > public_index
                                ):
                                    if not isinstance(self._submission, AgentAccepted):
                                        raise ProtocolDefect(
                                            "Antigravity output preceded recorded delivery"
                                        )
                                    size = len(step.content.encode("utf-8"))
                                    if size > _MAX_EVENT_TEXT_BYTES:
                                        raise OutputLimitExceeded(
                                            _MAX_EVENT_TEXT_BYTES, unit="bytes"
                                        )
                                    if public is not None:
                                        self._emit(public, len(public.text.encode("utf-8")))
                                    public_index = step.step_index
                                    public = AgentMessage(
                                        self._turn, step.id, "commentary", step.content
                                    )
                                if public is not None and latest_action > public_index:
                                    self._emit(public, len(public.text.encode("utf-8")))
                                    public = None
                            if not conversation.is_idle:
                                raise ProtocolDefect("Antigravity stream ended with an active root")
                            if isinstance(schema, dict):
                                output = conversation.get_last_structured_output()
                                bounded_payload_size(
                                    output, _MAX_EVENT_TEXT_BYTES, max_items=_MAX_MESSAGE_ITEMS
                                )
                                raw_output = RawAgentOutput(freeze_structured_output(output))
                            if public is not None or raw_output is not None:
                                status, cause = "succeeded", None
                        except types.AntigravityCancelledError:
                            status, cause = "cancelled", None
                        except OutputSchemaMismatch:
                            status, cause = "failed", "output_schema_violation"
                        finally:
                            self._reported_usage = _usage(conversation.total_usage)
            except TimeoutError:
                self._finish_local("failed", "turn_timeout", "Antigravity turn deadline expired")
                return
            except OutputLimitExceeded:
                self._finish_local(
                    "failed", "output_limit_exceeded", "Antigravity output exceeded its bound"
                )
                return
        if (
            not isinstance(self._submission, AgentAccepted)
            or self._callback_defect
            or self._calls
            or (status == "succeeded" and observations)
            or (status == "failed" and cause == "backend_failed")
        ):
            self._finish_local("failed", "backend_failed", "Antigravity root did not settle safely")
            return
        final = ""
        if status == "succeeded" and public is not None:
            final = public.text
            self._emit(
                AgentMessage(self._turn, public.message_id, "final_answer", final),
                len(final.encode("utf-8")),
            )
        self._terminal = AgentTerminal(
            status,
            AgentFailure(cause) if cause else None,
            final,
            self._turn.session_ref,
            NativeTerminalEvidence(self._attempt, self._turn, "antigravity-sdk-turn.v1"),
            raw_structured_output=raw_output,
            usage=self._reported_usage,
        )
