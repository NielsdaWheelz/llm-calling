from __future__ import annotations

import json
from pathlib import Path

import pytest

from provider_runtime.agent_runtime.auth import (
    AuthEnvironmentRequest,
    build_child_environment,
    child_home_directory,
    credential_environment_names,
    is_runtime_owned_environment_name,
    mcp_header_environment_name,
    redact_native_payload,
    resolve_state_root,
    state_root_from_environment,
    validate_policy_environment,
)
from provider_runtime.agent_runtime.errors import (
    CredentialUnavailable,
    InvalidAgentRequest,
)
from provider_runtime.agent_runtime.policy import PermissionPolicy
from provider_runtime.agent_runtime.types import CredentialRef, thaw_json_value


def test_profile_state_roots_are_isolated_and_cannot_escape(tmp_path: Path) -> None:
    root = resolve_state_root(tmp_path, "codex", "personal")

    assert root == (tmp_path / "codex" / "personal").resolve()
    with pytest.raises(InvalidAgentRequest, match="profile_key"):
        resolve_state_root(tmp_path, "codex", "../shared")


def test_profile_state_roots_reject_symlink_aliases_and_escapes(tmp_path: Path) -> None:
    codex = tmp_path / "codex"
    codex.mkdir()
    actual = codex / "actual"
    actual.mkdir()
    (codex / "alias").symlink_to(actual, target_is_directory=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-agent-state"
    outside.mkdir()
    (codex / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidAgentRequest, match="symlink"):
        resolve_state_root(tmp_path, "codex", "alias")
    with pytest.raises(InvalidAgentRequest, match="symlink"):
        resolve_state_root(tmp_path, "codex", "escape")


def test_local_account_environment_scrubs_every_credential_class_variable(tmp_path: Path) -> None:
    inherited = {
        "TERM": "xterm-256color",
        "PATH": "/operator/shims:/usr/bin",
        "HOME": "/home/operator",
        "OPENAI_API_KEY": "sk-secret-secret-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-secret",
    }
    state_root = tmp_path / "codex" / "personal"
    child = build_child_environment(
        AuthEnvironmentRequest(
            backend="codex",
            credential=CredentialRef(kind="local_account", profile_key="personal"),
            inherited_environment=inherited,
            allowed_environment=("TERM",),
            state_root=state_root,
        )
    )

    assert child == {
        "CODEX_HOME": str(state_root.resolve()),
        "HOME": str(child_home_directory(state_root.resolve())),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/tmp",
        "TERM": "xterm-256color",
    }


@pytest.mark.parametrize("name", ("PATH", "HOME", "LANG", "LC_ALL", "LC_TIME", "TMPDIR"))
def test_the_runtime_owned_base_environment_is_never_caller_settable(
    tmp_path: Path, name: str
) -> None:
    """Every name `build_child_environment` supplies itself is refused in the policy allowlist.

    Without this the caller could hand the child a PATH of their own shims, a HOME outside the
    isolated profile, or a locale that changes the shape of the output the adapters parse.
    """
    assert is_runtime_owned_environment_name(name)
    with pytest.raises(InvalidAgentRequest, match="process-control"):
        build_child_environment(
            AuthEnvironmentRequest(
                backend="codex",
                credential=CredentialRef(kind="local_account", profile_key="personal"),
                inherited_environment={name: "caller-supplied"},
                allowed_environment=(name,),
                state_root=tmp_path / "codex" / "personal",
            )
        )


@pytest.mark.parametrize(
    "name",
    (
        "CLAUDE_CODE_USE_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
    ),
)
def test_policy_environment_rejects_alternate_auth_and_provider_selection(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(InvalidAgentRequest, match="authentication or provider selection"):
        build_child_environment(
            AuthEnvironmentRequest(
                backend="claude",
                credential=CredentialRef(kind="local_account", profile_key="personal"),
                inherited_environment={name: "must-not-cross"},
                allowed_environment=(name,),
                state_root=tmp_path / "claude" / "personal",
            )
        )


def test_named_session_credentials_are_rejected_before_any_forwarding(tmp_path: Path) -> None:
    """Subscription auth only: no path exists that forwards an API key into a session child."""
    for credential in (
        CredentialRef(kind="api_key_environment", profile_key="api", name="ANTHROPIC_API_KEY"),
        CredentialRef(
            kind="secret_reference", profile_key="vault", name="providers/openai/personal"
        ),
    ):
        with pytest.raises(InvalidAgentRequest, match="local_account"):
            build_child_environment(
                AuthEnvironmentRequest(
                    backend="claude",
                    credential=credential,
                    inherited_environment={"ANTHROPIC_API_KEY": "must-never-cross"},
                    allowed_environment=(),
                    state_root=tmp_path / "claude" / "api",
                )
            )


def test_mcp_header_environment_alias_is_stable_and_opaque() -> None:
    source = CredentialRef(
        kind="secret_reference",
        profile_key="work",
        name="vault/provider/token-name",
    )

    alias = mcp_header_environment_name("codex", source)

    assert alias == mcp_header_environment_name("codex", source)
    assert alias.startswith("PROVIDER_RUNTIME_MCP_SECRET_")
    assert "vault" not in alias
    assert "token" not in alias
    assert alias != mcp_header_environment_name("claude", source)
    with pytest.raises(InvalidAgentRequest, match="named credential"):
        mcp_header_environment_name(
            "codex", CredentialRef(kind="local_account", profile_key="personal")
        )


def test_policy_can_never_allow_credential_class_environment_names() -> None:
    assert "OPENAI_API_KEY" in credential_environment_names("codex")
    with pytest.raises(InvalidAgentRequest, match="credential-class"):
        validate_policy_environment(
            "codex", PermissionPolicy(environment=("LANG", "OPENAI_API_KEY"))
        )
    with pytest.raises(InvalidAgentRequest, match="credential-class"):
        validate_policy_environment("codex", PermissionPolicy(environment=("ANTHROPIC_API_KEY",)))


# Provider API keys that authenticate no agent backend but are the operator secrets most
# likely to be exported in the shell that starts one. They must be as uncopyable as the two
# backends' own auth names.
OTHER_PROVIDER_CREDENTIAL_NAMES = (
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
)


@pytest.mark.parametrize("name", OTHER_PROVIDER_CREDENTIAL_NAMES)
def test_no_provider_api_key_can_be_copied_into_an_agent_child(tmp_path: Path, name: str) -> None:
    assert is_runtime_owned_environment_name(name)
    with pytest.raises(InvalidAgentRequest, match="credential-class"):
        validate_policy_environment("codex", PermissionPolicy(environment=(name,)))
    with pytest.raises(InvalidAgentRequest, match="credential-class"):
        build_child_environment(
            AuthEnvironmentRequest(
                backend="codex",
                credential=CredentialRef(kind="local_account", profile_key="personal"),
                inherited_environment={name: "must-never-cross"},
                allowed_environment=(name,),
                state_root=tmp_path / "codex" / "personal",
            )
        )


def test_native_payloads_are_recursively_scrubbed_without_field_allowlists() -> None:
    payload = redact_native_payload(
        {
            "message": "safe",
            "authorization": "Bearer definitely-secret-token",
            "nested": {"token": "sk-secret-secret-secret"},
            "extra": "kept because redaction owns keys, not schemas",
        }
    )

    assert payload == {
        "message": "safe",
        "authorization": "...redacted",
        "nested": {"token": "...redacted"},
        "extra": "kept because redaction owns keys, not schemas",
    }
    assert "definitely-secret-token" not in json.dumps(thaw_json_value(payload))


# Every credential-bearing key shape the two shipped backends put in a native payload, plus
# the near misses that make the usage-counter exemption dangerous if it is written loosely.
# Sources: `_CREDENTIAL_ENVIRONMENT` in auth.py (the variables both agents authenticate
# with), Codex's `~/.codex/auth.json` (`OPENAI_API_KEY` beside a `tokens` object holding
# `id_token`/`access_token`/`refresh_token`), Claude Code's `system/init` `apiKeySource`,
# and the HTTP header names either backend can echo from an MCP server definition.
CREDENTIAL_KEYS = (
    "authorization",
    "Authorization",
    "api_key",
    "apiKey",
    "API-Key",
    "x-api-key",
    "APIKEY",
    "apiKeySource",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "token",
    "tokens",
    "id_token",
    "access_token",
    "refresh_token",
    "accessToken",
    "refreshToken",
    "authToken",
    "sessionToken",
    "tokenValue",
    "cached_access_token",
    "secret",
    "client_secret",
    "secretRef",
    "password",
    "credential",
    "credentials",
    "cookie",
    "Set-Cookie",
)
# Backend-reported usage, which the contract preserves as passthrough data. Claude's names
# come from `ResultMessage.usage` / `model_usage` (tests/fixtures/agent_runtime/claude);
# Codex's from a captured `thread/tokenUsage/updated` notification.
USAGE_COUNTER_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "ephemeral_5m_input_tokens",
    "tokenUsage",
    "totalTokens",
    "inputTokens",
    "cachedInputTokens",
    "cacheWriteInputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "max_tokens",
    "budget_tokens",
    "token_count",
    "tokens_remaining",
)


@pytest.mark.parametrize("key", CREDENTIAL_KEYS)
def test_every_credential_key_shape_is_redacted(key: str) -> None:
    """A false negative here leaks a secret, so the vocabulary is asserted key by key."""
    secret = "sk-live-definitely-secret-value"

    passthrough = redact_native_payload({"outer": {key: secret}})
    assert passthrough == {"outer": {key: "...redacted"}}, (
        f"{key!r} must be redacted wherever it appears"
    )
    assert secret not in json.dumps(thaw_json_value(passthrough))


@pytest.mark.parametrize("key", USAGE_COUNTER_KEYS)
def test_backend_reported_usage_counter_names_are_not_treated_as_credentials(key: str) -> None:
    assert redact_native_payload({"usage": {key: 140}}) == {"usage": {key: 140}}


def test_captured_backend_usage_payloads_survive_redaction_intact() -> None:
    """The exact usage shapes both backends emit reach the caller unchanged.

    Codex's block is a captured `thread/tokenUsage/updated` notification (codex-cli
    SDK notification); Claude's is the `ResultMessage` usage the SDK fixture replays. Redaction owns
    credentials, not counters: a scrubbed counter is silently wrong accounting data in the
    one event the contract says retains native data.
    """
    codex_params: dict[str, object] = {
        "threadId": "019fd041-e1c2-7d10-a64a-456a148c4165",
        "turnId": "019fd041-e383-78a2-8648-35c0996aaebc",
        "tokenUsage": {
            "total": {
                "totalTokens": 19707,
                "inputTokens": 19702,
                "cachedInputTokens": 9984,
                "cacheWriteInputTokens": 0,
                "outputTokens": 5,
                "reasoningOutputTokens": 0,
            },
            "modelContextWindow": 258400,
        },
    }
    assert redact_native_payload(codex_params) == codex_params

    claude_result: dict[str, object] = {
        "usage": {
            "input_tokens": 140,
            "output_tokens": 32,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 20,
        },
        "model_usage": {"native-model": {"input_tokens": 140, "output_tokens": 32}},
    }
    assert redact_native_payload(claude_result) == claude_result


def test_unknown_payloads_are_bounded_scrubbed_or_dropped_to_a_stub() -> None:
    scrubbed = redact_native_payload({"detail": "Bearer abcdefghijklmnopqrstuvwxyz"})
    assert scrubbed == {"detail": "Bearer ...redacted"}

    dropped = redact_native_payload({"detail": "x" * 100}, max_bytes=20)
    assert dropped == {"redaction": "payload_dropped", "reason": "size_limit"}

    huge_integer = redact_native_payload({"detail": 10**10000})
    assert huge_integer == {"redaction": "payload_dropped", "reason": "shape_limit"}


@pytest.mark.parametrize(
    "name",
    (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "CURL_CA_BUNDLE",
    ),
)
def test_proxy_and_tls_trust_variables_are_runtime_owned(tmp_path: Path, name: str) -> None:
    """Redirecting the agent's endpoint or its TLS trust store is credential redirection."""
    assert is_runtime_owned_environment_name(name)
    with pytest.raises(InvalidAgentRequest, match="process-control"):
        build_child_environment(
            AuthEnvironmentRequest(
                backend="claude",
                credential=CredentialRef(kind="local_account", profile_key="personal"),
                inherited_environment={name: "must-not-cross"},
                allowed_environment=(name,),
                state_root=tmp_path / "claude" / "personal",
            )
        )


@pytest.mark.parametrize(
    "name", ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AWS_SECRET_ACCESS_KEY", "CODEX_HOME")
)
def test_credential_auth_and_process_control_names_share_one_predicate(name: str) -> None:
    assert is_runtime_owned_environment_name(name)


@pytest.mark.parametrize("name", ("TERM", "MY_APP_SETTING", "COLUMNS"))
def test_ordinary_environment_names_stay_forwardable(name: str) -> None:
    assert not is_runtime_owned_environment_name(name)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_a_non_finite_native_number_drops_the_payload_like_any_other_malformed_value(
    value: float,
) -> None:
    """`json.loads` accepts NaN/Infinity, so a backend frame can carry one.

    Every other value redaction cannot represent — an oversized int, a non-string key, a
    non-JSON type — degrades to the bounded stub. A non-finite float is the same class of
    malformedness and must not be the one that kills the turn with a defect instead.
    """
    assert redact_native_payload({"score": value}) == {
        "redaction": "payload_dropped",
        "reason": "shape_limit",
    }


def test_state_root_read_back_is_strict_and_identical_for_every_backend(
    tmp_path: Path,
) -> None:
    """One profile must not mean two different state roots on the two shipped backends.

    This resolver is the single owner of that rule, so a `profile_key` can never name a
    directory on one lane and a different one — or nothing at all — on the other. It reads
    back only an existing, absolute, fully resolved directory: a symlinked, missing,
    relative, or empty value is a credential failure, never a silent redirect to the link's
    target.
    """
    real = tmp_path / "codex" / "personal"
    real.mkdir(parents=True)
    child = build_child_environment(
        AuthEnvironmentRequest(
            backend="codex",
            credential=CredentialRef(kind="local_account", profile_key="personal"),
            inherited_environment={},
            allowed_environment=(),
            state_root=real,
        )
    )
    assert state_root_from_environment("codex", child) == real

    link = tmp_path / "codex" / "alias"
    link.symlink_to(real, target_is_directory=True)
    missing = tmp_path / "codex" / "absent"
    for value in (str(link), str(missing), "relative/path", ""):
        with pytest.raises(CredentialUnavailable, match="CODEX_HOME"):
            state_root_from_environment("codex", {"CODEX_HOME": value})
    with pytest.raises(CredentialUnavailable, match="CLAUDE_CONFIG_DIR"):
        state_root_from_environment("claude", {})


def test_the_child_home_is_inside_the_profile_root_and_never_the_root_itself(
    tmp_path: Path,
) -> None:
    """Claude Code owns its config directory outright, so HOME must not be pointed at it."""
    state_root = tmp_path / "claude" / "personal"
    home = child_home_directory(state_root)

    assert home.parent == state_root
    assert home != state_root
