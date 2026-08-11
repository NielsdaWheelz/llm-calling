"""Profile isolation, credential environment ownership, and native redaction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from provider_runtime.errors import sanitize_provider_text

from .errors import (
    CredentialUnavailable,
    InvalidAgentRequest,
    ProtocolDefect,
)
from .policy import PermissionPolicy
from .types import (
    Backend,
    CredentialRef,
    FrozenJsonDict,
    JsonObject,
    thaw_json_value,
)
from .types import (
    freeze_json_object as _freeze_json_object,
)
from .types import (
    freeze_json_value as _freeze_json_value,
)


def freeze_native_json_value(value: object, *, context: str = "native_payload"):
    try:
        return _freeze_json_value(value, context=context)
    except InvalidAgentRequest:
        raise ProtocolDefect(
            "provider emitted malformed JSON data",
            code="malformed_native_payload",
        ) from None


def freeze_native_json_object(
    value: Mapping[str, object], *, context: str = "native_payload"
) -> FrozenJsonDict:
    try:
        return _freeze_json_object(value, context=context)
    except InvalidAgentRequest:
        raise ProtocolDefect(
            "provider emitted malformed JSON data",
            code="malformed_native_payload",
        ) from None


_CREDENTIAL_ENVIRONMENT: dict[Backend, tuple[str, ...]] = {
    "codex": ("OPENAI_API_KEY",),
    "claude": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ),
}
# Provider API keys that authenticate no agent backend but are the operator secrets most
# likely to be exported in the shell that starts one. They are credential-class for exactly
# the reason the two backends' own names are: a child that receives one can spend it.
# `GOOGLE_API_KEY` and the rest of the `GOOGLE_`/`AWS_`/`AZURE_` families are already covered
# by `_AUTH_CONTROL_PREFIXES`; these five match no prefix and must be named.
_OTHER_PROVIDER_CREDENTIAL_ENVIRONMENT = frozenset(
    {
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    }
)
_ALL_CREDENTIAL_ENVIRONMENT = (
    frozenset(name for names in _CREDENTIAL_ENVIRONMENT.values() for name in names)
    | _OTHER_PROVIDER_CREDENTIAL_ENVIRONMENT
)
_AUTH_CONTROL_PREFIXES = (
    "ANTHROPIC_",
    "AWS_",
    "AZURE_",
    "CLAUDE_CODE_",
    "CLOUD_ML_",
    "CODEX_",
    "GOOGLE_",
    "OPENAI_",
)
_PROCESS_CONTROL_NAMES = frozenset(
    {
        "ALL_PROXY",
        "BASH_ENV",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "CURL_CA_BUNDLE",
        "ENV",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LANGUAGE",
        "LD_PRELOAD",
        "NODE_EXTRA_CA_CERTS",
        "NODE_OPTIONS",
        # Module search paths for the interpreters a child agent is or can start: Claude Code
        # is a Node program, and either backend's shell tool can run a script whose imports
        # these names redirect.
        "NODE_PATH",
        "NO_PROXY",
        "PATH",
        "PERL5LIB",
        "PYTHONHOME",
        "PYTHONPATH",
        "REQUESTS_CA_BUNDLE",
        "RUBYLIB",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "ZDOTDIR",
        # Both installed agents honour the lowercase proxy spellings as well.
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_PROCESS_CONTROL_PREFIXES = ("DYLD_", "LC_", "LD_", "PYTHON")
_STATE_ROOT_ENVIRONMENT: dict[Backend, str] = {
    "codex": "CODEX_HOME",
    "claude": "CLAUDE_CONFIG_DIR",
}
# The child's HOME is a runtime-owned directory *inside* the profile state root, never the
# state root itself: both agents treat their state root as a config directory they own and
# rewrite, so pointing HOME at it would let unrelated tooling write into provider config.
_CHILD_HOME_DIRECTORY = "home"
# A vetted absolute PATH, not `os.environ["PATH"]`: inheriting the operator's PATH would hand
# the sandboxed child every shim, version manager, and project-local `node_modules/.bin` on it.
# It must also be present rather than merely safe: with PATH unset a child `bash` falls back to
# its compiled-in default and Codex's `shell_environment_policy.inherit = "core"` has no PATH to
# inherit at all.
_CHILD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# One deterministic UTF-8 locale so native output the adapters parse never changes shape with
# the operator's locale, and the system temporary directory both sandboxes already allow.
_CHILD_LOCALE = "C.UTF-8"
_CHILD_TMPDIR = "/tmp"
_PROFILE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SENSITIVE_FIELD = re.compile(
    r"(?i)(?:authorization|api[_-]?key|token|secret|password|credential|cookie)"
)
# `token` is the one word in that vocabulary both backends also use for usage counters,
# which are passthrough data the contract preserves and carry no auth material: Claude
# reports `usage.input_tokens` / `cache_read_input_tokens` and per-model `model_usage`,
# Codex reports `thread/tokenUsage/updated` with `tokenUsage.total.{totalTokens,
# inputTokens,cachedInputTokens,cacheWriteInputTokens,outputTokens,reasoningOutputTokens}`.
# A key is exempted only when *every* one of its token words sits directly beside a
# counting word and no other credential word appears anywhere in it.
#
# Adjacency is what keeps the exemption tight, and it is deliberately tighter than
# "plural means a count": Codex's own `auth.json` stores OAuth material under a bare
# `tokens` key, so `tokens`, `token`, `access_token`, and even `cached_access_token` all
# stay redacted. An unrecognized counter name is a redacted counter — lost data, never a
# leaked secret.
_KEY_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_TOKEN_WORDS = frozenset({"token", "tokens"})
_TOKEN_COUNT_WORDS = frozenset(
    {
        "budget",
        "budgets",
        "count",
        "counts",
        "input",
        "limit",
        "limits",
        "max",
        "maximum",
        "min",
        "minimum",
        "num",
        "number",
        "output",
        "per",
        "remaining",
        "total",
        "totals",
        "usage",
        "used",
        "window",
        "windows",
    }
)


def _is_sensitive_payload_key(key: str) -> bool:
    """Whether a native-payload field name may carry credential material."""
    if _SENSITIVE_FIELD.search(key) is None:
        return False
    words = [word.lower() for word in _KEY_WORD.findall(key)]
    token_positions = [index for index, word in enumerate(words) if word in _TOKEN_WORDS]
    if not token_positions:
        return True
    # Whatever is left once the token words are removed must not itself be credential-ish,
    # so `api_key_token_count` is redacted on `api_key` even though its token word counts.
    remainder = "_".join(word for index, word in enumerate(words) if index not in token_positions)
    if _SENSITIVE_FIELD.search(remainder) is not None:
        return True
    return not all(
        any(
            neighbour in _TOKEN_COUNT_WORDS
            for neighbour in words[max(index - 1, 0) : index] + words[index + 1 : index + 2]
        )
        for index in token_positions
    )


def credential_environment_names(backend: Backend) -> tuple[str, ...]:
    try:
        return _CREDENTIAL_ENVIRONMENT[backend]
    except KeyError as error:
        raise InvalidAgentRequest(f"unknown agent backend {backend!r}") from error


def mcp_header_environment_name(backend: Backend, source: CredentialRef) -> str:
    """Return an opaque, stable child-environment alias for one MCP header source."""
    credential_environment_names(backend)
    if not isinstance(source, CredentialRef) or source.kind == "local_account":
        raise InvalidAgentRequest("MCP header aliases require a named credential source")
    if source.name is None:
        raise InvalidAgentRequest("MCP header credential source has no name")
    identity = "\0".join((backend, source.profile_key, source.kind, source.name))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"PROVIDER_RUNTIME_MCP_SECRET_{digest}"


def resolve_state_root(base: str | Path, backend: Backend, profile_key: str) -> Path:
    if backend not in _STATE_ROOT_ENVIRONMENT:
        raise InvalidAgentRequest(f"unknown agent backend {backend!r}")
    if type(profile_key) is not str or _PROFILE_KEY.fullmatch(profile_key) is None:
        raise InvalidAgentRequest("profile_key must be a safe named state scope")
    base_path = Path(base).resolve()
    state_root = base_path / backend / profile_key
    resolved = state_root.resolve()
    if resolved != state_root:
        raise InvalidAgentRequest("profile state roots must not contain symlink aliases")
    if not resolved.is_relative_to(base_path):
        raise InvalidAgentRequest("profile_key escapes the configured state-root base")
    return state_root


def state_root_environment_name(backend: Backend) -> str:
    try:
        return _STATE_ROOT_ENVIRONMENT[backend]
    except KeyError as error:
        raise InvalidAgentRequest(f"unknown agent backend {backend!r}") from error


def state_root_from_environment(backend: Backend, environment: Mapping[str, str]) -> Path:
    """Read back the profile state root the runtime put in one child environment.

    Both backends read it the same way and to the same standard: the value must be the
    absolute, already-resolved, existing directory `build_child_environment` wrote. Accepting
    an unresolved or absent path on one route and not the other would let the same profile
    mean two different state roots.
    """
    name = state_root_environment_name(backend)
    value = environment.get(name)
    if type(value) is not str or not value:
        raise CredentialUnavailable(f"agent environment has no {name}")
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or not path.is_dir():
        raise CredentialUnavailable(f"{name} must be an existing resolved absolute directory")
    return path


def child_home_directory(state_root: Path) -> Path:
    """The runtime-owned HOME handed to a child agent for one profile."""
    return state_root / _CHILD_HOME_DIRECTORY


def validate_policy_environment(backend: Backend, policy: PermissionPolicy) -> None:
    credential_environment_names(backend)
    forbidden = sorted(set(policy.environment) & _ALL_CREDENTIAL_ENVIRONMENT)
    if forbidden:
        raise InvalidAgentRequest(
            f"policy environment names credential-class variables: {forbidden}"
        )
    auth_controls = sorted(
        name
        for name in policy.environment
        if any(name.startswith(prefix) for prefix in _AUTH_CONTROL_PREFIXES)
    )
    if auth_controls:
        raise InvalidAgentRequest(
            "policy environment names authentication or provider selection variables: "
            f"{auth_controls}"
        )
    process_controls = sorted(
        name for name in policy.environment if is_process_control_environment_name(name)
    )
    if process_controls:
        raise InvalidAgentRequest(
            f"policy environment names process-control variables: {process_controls}"
        )


def is_process_control_environment_name(name: str) -> bool:
    """Names that steer how the child process loads code, resolves state, or reaches the network."""
    return name in _PROCESS_CONTROL_NAMES or any(
        name.startswith(prefix) for prefix in _PROCESS_CONTROL_PREFIXES
    )


def is_runtime_owned_environment_name(name: str) -> bool:
    """Every child-environment name auth.py owns: credentials, provider selection, and control.

    Both the policy allowlist and MCP credential destinations answer to this single predicate,
    so a name rejected on one path can never slip in through the other.
    """
    return (
        name in _ALL_CREDENTIAL_ENVIRONMENT
        or any(name.startswith(prefix) for prefix in _AUTH_CONTROL_PREFIXES)
        or is_process_control_environment_name(name)
    )


@dataclass(frozen=True, slots=True)
class AuthEnvironmentRequest:
    backend: Backend
    credential: CredentialRef
    inherited_environment: Mapping[str, str] = field(repr=False, compare=False)
    allowed_environment: tuple[str, ...]
    state_root: Path

    def __post_init__(self) -> None:
        if self.backend not in _STATE_ROOT_ENVIRONMENT:
            raise InvalidAgentRequest(f"unknown agent backend {self.backend!r}")
        if not isinstance(self.credential, CredentialRef):
            raise InvalidAgentRequest("AuthEnvironmentRequest.credential must be CredentialRef")
        if self.credential.kind != "local_account":
            # Subscription auth only. An API-key or secret-reference session credential is
            # rejected here as well as at every route, so no path can forward one.
            raise InvalidAgentRequest(
                "agent sessions accept only local_account subscription credentials"
            )
        if not isinstance(self.inherited_environment, Mapping) or any(
            type(key) is not str or type(value) is not str
            for key, value in self.inherited_environment.items()
        ):
            raise InvalidAgentRequest("inherited_environment must map strings to strings")
        if not isinstance(self.allowed_environment, tuple) or any(
            type(name) is not str or not name for name in self.allowed_environment
        ):
            raise InvalidAgentRequest("allowed_environment must be a tuple of names")
        if len(self.allowed_environment) != len(set(self.allowed_environment)):
            raise InvalidAgentRequest("allowed_environment must not contain duplicates")
        if not isinstance(self.state_root, Path) or not self.state_root.is_absolute():
            raise InvalidAgentRequest("state_root must be an absolute Path")


def build_child_environment(request: AuthEnvironmentRequest) -> dict[str, str]:
    """Build the complete child environment without mutating inherited state.

    The base map below is owned outright: every one of its names is a process-control name
    that `validate_policy_environment` already refuses in `policy.environment`, so a caller
    can neither set nor unset one. It is applied first so the caller's allowlist can only ever
    add names beside it, never over it. Credential-class names never pass the allowlist, so
    the child sees no API key regardless of the operator's environment.
    """
    validate_policy_environment(
        request.backend, PermissionPolicy(environment=request.allowed_environment)
    )
    state_root = request.state_root.resolve()
    child = {
        "PATH": _CHILD_PATH,
        "HOME": str(child_home_directory(state_root)),
        "LANG": _CHILD_LOCALE,
        "LC_ALL": _CHILD_LOCALE,
        "TMPDIR": _CHILD_TMPDIR,
    }
    child.update(
        {
            name: request.inherited_environment[name]
            for name in request.allowed_environment
            if name in request.inherited_environment and name not in _ALL_CREDENTIAL_ENVIRONMENT
        }
    )
    child[_STATE_ROOT_ENVIRONMENT[request.backend]] = str(state_root)
    return child


class _PayloadLimit(Exception):
    pass


def _redact_value(
    value: object,
    *,
    depth: int,
    max_depth: int,
    counter: list[int],
    max_items: int,
    max_string_length: int,
) -> object:
    if depth > max_depth:
        raise _PayloadLimit
    counter[0] += 1
    if counter[0] > max_items:
        raise _PayloadLimit
    if value is None or type(value) is bool:
        return value
    if type(value) is float:
        # `json.loads` accepts NaN/Infinity by default, so a backend frame can carry one.
        # JSON cannot represent it, which makes it the same class of malformedness as an
        # oversized integer: the payload is dropped whole, never raised past this boundary.
        if not math.isfinite(value):
            raise _PayloadLimit
        return value
    if type(value) is int:
        if not -(2**63) <= value <= 2**63 - 1:
            raise _PayloadLimit
        return value
    if type(value) is str:
        return sanitize_provider_text(value, limit=max_string_length)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise _PayloadLimit
            if _is_sensitive_payload_key(key):
                result[key] = "...redacted"
            else:
                result[key] = _redact_value(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    counter=counter,
                    max_items=max_items,
                    max_string_length=max_string_length,
                )
        return result
    if isinstance(value, tuple | list):
        return [
            _redact_value(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                counter=counter,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for child in value
        ]
    raise _PayloadLimit


def _dropped(reason: Literal["size_limit", "shape_limit"]) -> JsonObject:
    return FrozenJsonDict({"redaction": "payload_dropped", "reason": reason})


def redact_native_payload(
    payload: Mapping[str, object],
    *,
    max_bytes: int = 16_384,
    max_depth: int = 8,
    max_items: int = 512,
    max_string_length: int = 2_000,
) -> JsonObject:
    """Recursively scrub credential-shaped keys and bound every retained value.

    This is the one representation native frames may cross the public boundary
    in: no per-version field allowlists exist — every key is kept unless its
    name is credential-shaped, every string is sanitized and bounded, and a
    payload that exceeds its depth/item/byte bounds is dropped whole.
    """
    if not isinstance(payload, Mapping):
        return _dropped("shape_limit")
    try:
        redacted = _redact_value(
            payload,
            depth=0,
            max_depth=max_depth,
            counter=[0],
            max_items=max_items,
            max_string_length=max_string_length,
        )
        if not isinstance(redacted, Mapping):
            return _dropped("shape_limit")
        frozen = freeze_native_json_object(redacted, context="native_payload")
    except _PayloadLimit:
        return _dropped("shape_limit")
    try:
        encoded = json.dumps(
            thaw_json_value(frozen), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return _dropped("shape_limit")
    if len(encoded) > max_bytes:
        return _dropped("size_limit")
    return frozen
