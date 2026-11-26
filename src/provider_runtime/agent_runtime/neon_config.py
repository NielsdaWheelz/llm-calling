"""The pinned stock contract for the direct Gemini Neon inference host."""

from __future__ import annotations

import asyncio
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from ._limits import _OPERATION_TIMEOUT_SECONDS
from .codex_app_server import CodexAppServerClient, CodexAppServerConfig
from .codex_containment import CODEX_CONTAINMENT_VERSION, _startup_flag
from .codex_endpoint import CodexEndpoint
from .errors import UnsupportedCapability

if TYPE_CHECKING:
    from .model_catalog import AgentModelCatalog

NEON_MODEL = "gemini-4-neon"
NEON_PROFILE = "google_neon"
NEON_PROVIDER = "jarvis_neon"
NEON_STOCK_BEARER_ENV = "CODEX_NEON_STOCK_AUTHORIZATION"
NEON_EFFORTS = ("low", "medium", "high", "xhigh")
NEON_CONTEXT_WINDOW = 1_048_576
NEON_MAX_OUTPUT_TOKENS = 65_536
NEON_AUTO_COMPACT_TOKENS = 900_000
NEON_TOOL_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "TERM")


def neon_shell_environment() -> dict[str, object]:
    return {
        "inherit": "core",
        "include_only": list(NEON_TOOL_ENVIRONMENT),
        "exclude": [],
        "set": {},
    }


def _catalog_bytes() -> bytes:
    source = files(__package__).joinpath("_codex_models_0_171_0.json").read_bytes()
    model = next(row for row in json.loads(source)["models"] if row["slug"] == "gpt-6.2-astra")
    model.update(
        slug=NEON_MODEL,
        display_name="Gemini 4 Neon",
        description="Google Gemini Developer API standard inference",
        default_reasoning_level="high",
        supported_reasoning_levels=[
            row for row in model["supported_reasoning_levels"] if row["effort"] in NEON_EFFORTS
        ],
        context_window=NEON_CONTEXT_WINDOW,
        max_context_window=NEON_CONTEXT_WINDOW,
        auto_compact_token_limit=NEON_AUTO_COMPACT_TOKENS,
        prefer_websockets=False,
        use_responses_lite=False,
        tool_mode="direct",
        apply_patch_tool_type=None,
        experimental_supported_tools=[],
        supports_search_tool=False,
        supports_reasoning_summaries=False,
        supports_reasoning_summary_parameter=False,
        additional_speed_tiers=[],
        service_tiers=[],
        available_in_plans=[],
    )
    return json.dumps({"models": [model]}, separators=(",", ":"), sort_keys=True).encode()


NEON_CATALOG_REVISION = f"codex-neon.v1:{hashlib.sha256(_catalog_bytes()).hexdigest()}"
_NEON_CATALOG_FILENAME = f"{NEON_CATALOG_REVISION.replace(':', '-')}.json"


def materialize_neon_catalog(directory: Path) -> Path:
    if not directory.is_absolute():
        raise ValueError("Codex host catalog directory must be absolute")
    path = directory / _NEON_CATALOG_FILENAME
    path.write_bytes(_catalog_bytes())
    path.chmod(0o600)
    return path


def _origin(origin: str) -> str:
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.port == 0
        or origin != f"http://127.0.0.1:{parsed.port}"
    ):
        raise ValueError("Neon adapter origin must be an explicit HTTP loopback origin")
    return origin


def _provider_options(origin: str) -> dict[str, object]:
    return {
        "name": "Jarvis Gemini Neon",
        "base_url": f"{_origin(origin)}/v1",
        "wire_api": "responses",
        "request_max_retries": 0,
        "stream_max_retries": 0,
        "supports_websockets": False,
        "requires_openai_auth": False,
        "supports_standalone_web_search": False,
        "env_http_headers": {"Authorization": NEON_STOCK_BEARER_ENV},
    }


_NO_PRE_SAMPLE_FEATURES = (
    "hooks",
    "plugin_hooks",
    "remote_plugin",
    "tool_suggest",
    "memories",
    "auth_elicitation",
    "api_key_model_discovery",
)


def neon_stock_overrides(origin: str) -> tuple[str, ...]:
    options = _provider_options(origin)
    options.pop("env_http_headers")
    return (
        f'model_provider="{NEON_PROVIDER}"',
        f'model="{NEON_MODEL}"',
        'model_reasoning_effort="high"',
        'web_search="disabled"',
        "allow_login_shell=false",
        'shell_environment_policy.inherit="core"',
        f"shell_environment_policy.include_only={json.dumps(NEON_TOOL_ENVIRONMENT)}",
        f'agents.default_subagent_model="{NEON_MODEL}"',
        'agents.default_subagent_reasoning_effort="high"',
        "features.multi_agent_v2=true",
        *(f"features.{feature}=false" for feature in _NO_PRE_SAMPLE_FEATURES),
        *(
            f"model_providers.{NEON_PROVIDER}.{key}={json.dumps(value)}"
            for key, value in options.items()
        ),
        f'model_providers.{NEON_PROVIDER}.env_http_headers={{ Authorization = "{NEON_STOCK_BEARER_ENV}" }}',
    )


def validate_neon_startup(snapshot: object, metadata: dict[str, object], origin: str) -> None:
    agent = metadata.get("userAgent")
    config = snapshot.get("config") if isinstance(snapshot, dict) else None
    if (
        not isinstance(agent, str)
        or (agent.split(" ", 1)[0].rpartition("/")[2] != CODEX_CONTAINMENT_VERSION)
        or not isinstance(config, dict)
    ):
        raise UnsupportedCapability("Neon requires the qualified Codex host version")
    path = _startup_flag(snapshot, "model_catalog_json")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or Path(path).name != _NEON_CATALOG_FILENAME
        or _startup_flag(snapshot, "model_provider") != NEON_PROVIDER
        or _startup_flag(snapshot, "model") != NEON_MODEL
        or _startup_flag(snapshot, "model_reasoning_effort") != "high"
        or config.get("web_search") != "disabled"
        or config.get("service_tier") not in (None, "default")
    ):
        raise UnsupportedCapability("Neon host catalog/model/tier startup policy is unqualified")
    providers = config.get("model_providers")
    provider = providers.get(NEON_PROVIDER) if isinstance(providers, dict) else None
    expected = _provider_options(origin)
    if (
        not isinstance(provider, dict)
        or any(
            (key in expected and (type(value) is not type(expected[key]) or value != expected[key]))
            or (key not in expected and value is not None)
            for key, value in provider.items()
        )
        or not expected.keys() <= provider.keys()
    ):
        raise UnsupportedCapability(
            "Neon host provider, authentication or retry policy is unqualified"
        )
    agents = config.get("agents")
    features = config.get("features")
    mcp = config.get("mcp_servers")
    shell = config.get("shell_environment_policy")
    if (
        not isinstance(agents, dict)
        or agents.get("default_subagent_model") != NEON_MODEL
        or agents.get("default_subagent_reasoning_effort") != "high"
        or not isinstance(features, dict)
        or features.get("multi_agent_v2") is not True
        or any(features.get(feature) is not False for feature in _NO_PRE_SAMPLE_FEATURES)
        or mcp != {}
        or config.get("allow_login_shell") is not False
        or not isinstance(shell, dict)
        or shell.get("inherit") != "core"
        or shell.get("include_only") != list(NEON_TOOL_ENVIRONMENT)
        or shell.get("set") not in (None, {})
    ):
        raise UnsupportedCapability("Neon host child or pre-sample activity policy is unqualified")


async def qualify_neon_host(endpoint: CodexEndpoint, *, origin: str) -> AgentModelCatalog:
    from .model_catalog import read_codex_model_catalog

    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
        async with CodexAppServerClient(
            CodexAppServerConfig(
                endpoint=endpoint, request_policy="observe_only", experimental_api=True
            )
        ) as client:
            validate_neon_startup(
                await client.request("config/read", {"includeLayers": False}),
                client.metadata,
                origin,
            )
            await require_neon_auth(client)
            return await read_codex_model_catalog(client, neon=True)


async def require_neon_auth(client: CodexAppServerClient) -> None:
    account = await client.request("account/read", {"refreshToken": False})
    if not isinstance(account, dict) or not (
        account.keys() >= {"account", "requiresOpenaiAuth"}
        and account["account"] is None
        and account["requiresOpenaiAuth"] is False
    ):
        raise UnsupportedCapability("Neon host must have no subscription or OpenAI credential")
