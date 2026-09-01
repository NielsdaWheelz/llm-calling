"""Authenticated Codex model discovery through the SDK's generic async RPC."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from provider_runtime.types import (
    Absent,
    Presence,
    Present,
    canonical_json_bytes,
    freeze_json_object,
)

from .errors import ProtocolDefect

AGENT_BACKEND_CONTRACT_REVISION = "provider-runtime.agent-model-catalog.v1"
_MAX_PAGES = 64

type AgentModelKey = str
type AgentReasoningKey = str
type NativeCatalogRevision = str


@dataclass(frozen=True, slots=True)
class AgentReasoningFacts:
    key: AgentReasoningKey
    label: str
    native_wire_value: str


@dataclass(frozen=True, slots=True)
class AgentUpgradeFacts:
    target_key: AgentModelKey


@dataclass(frozen=True, slots=True)
class UpgradeTargetUnresolved:
    model_key: AgentModelKey
    native_target: str
    kind: Literal["upgrade_target_unresolved"] = "upgrade_target_unresolved"


@dataclass(frozen=True, slots=True)
class UpgradeTargetAmbiguous:
    model_key: AgentModelKey
    native_target: str
    kind: Literal["upgrade_target_ambiguous"] = "upgrade_target_ambiguous"


@dataclass(frozen=True, slots=True)
class UpgradeSourceConflict:
    model_key: AgentModelKey
    native_targets: tuple[str, ...]
    kind: Literal["upgrade_source_conflict"] = "upgrade_source_conflict"


type AgentCatalogDiagnostic = (
    UpgradeTargetUnresolved | UpgradeTargetAmbiguous | UpgradeSourceConflict
)


@dataclass(frozen=True, slots=True)
class AgentModelFacts:
    key: AgentModelKey
    dispatch_model: str
    label: str
    source_context_window: Presence[int]
    source_max_output_tokens: Presence[int]
    input_modalities: tuple[Literal["text", "image"], ...]
    reasoning: tuple[AgentReasoningFacts, ...]
    source_default_reasoning: Presence[AgentReasoningKey]
    upgrade: Presence[AgentUpgradeFacts]
    retirement: Absent
    row_fingerprint: str


@dataclass(frozen=True, slots=True)
class AgentModelCatalog:
    backend_contract_revision: str
    definition_revision: str
    native_revision: Presence[NativeCatalogRevision]
    observed_at: datetime
    models: tuple[AgentModelFacts, ...]
    diagnostics: tuple[AgentCatalogDiagnostic, ...]


class GenericAsyncRpc(Protocol):
    def request(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        response_model: type[Any],
    ) -> Awaitable[object]: ...


@dataclass(frozen=True, slots=True)
class _ObservedModel:
    key: str
    dispatch_model: str
    label: str
    source_context_window: Presence[int]
    source_max_output_tokens: Presence[int]
    input_modalities: tuple[Literal["text", "image"], ...]
    reasoning: tuple[AgentReasoningFacts, ...]
    source_default_reasoning: Presence[str]
    native_upgrade_targets: tuple[str, ...]


async def read_codex_model_catalog(
    client: GenericAsyncRpc,
    response_model: type[Any],
    *,
    now: Callable[[], datetime] | None = None,
) -> AgentModelCatalog:
    """Read every visible page and normalize it without convenience APIs."""
    cursor: str | None = None
    requested_cursors: set[str] = set()
    rows: list[Mapping[str, object]] = []
    native_revisions: set[str] = set()
    for page_number in range(1, _MAX_PAGES + 1):
        response = await client.request(
            "model/list",
            {"includeHidden": False, "cursor": cursor},
            response_model=response_model,
        )
        payload = _object_mapping(response, "Codex model/list response")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ProtocolDefect("Codex model/list data was not an array")
        for item in data:
            row = _object_mapping(item, "Codex model/list row")
            hidden = row.get("hidden")
            if type(hidden) is not bool:
                raise ProtocolDefect("Codex model/list row had no boolean hidden fact")
            if not hidden:
                rows.append(row)
        revision = payload.get("revision")
        if revision is not None:
            if type(revision) is not str or not revision:
                raise ProtocolDefect("Codex model/list revision was malformed")
            native_revisions.add(revision)
        next_cursor = payload.get("nextCursor", payload.get("next_cursor"))
        if next_cursor is None:
            break
        if type(next_cursor) is not str or not next_cursor:
            raise ProtocolDefect("Codex model/list cursor was malformed")
        if next_cursor == cursor or next_cursor in requested_cursors:
            raise ProtocolDefect("Codex model/list repeated a pagination cursor")
        if page_number == _MAX_PAGES:
            raise ProtocolDefect("Codex model/list exceeded 64 pages")
        if cursor is not None:
            requested_cursors.add(cursor)
        cursor = next_cursor
    else:  # pragma: no cover - loop exits or the page-64 branch raises
        raise ProtocolDefect("Codex model/list pagination did not terminate")

    if len(native_revisions) > 1:
        raise ProtocolDefect("Codex model/list changed native revision during pagination")
    observed = tuple(_normalize_row(row) for row in rows)
    _validate_observed_rows(observed)
    facts, diagnostics = _resolve_upgrades(observed)
    observed_at = (now or (lambda: datetime.now(UTC)))()
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ProtocolDefect("AgentModelCatalog.observed_at must be timezone-aware")
    definition_revision = _hash(
        b"provider-runtime.agent-model-catalog.v1",
        {"row_fingerprints": tuple(row.row_fingerprint for row in facts)},
    )
    return AgentModelCatalog(
        backend_contract_revision=AGENT_BACKEND_CONTRACT_REVISION,
        definition_revision=definition_revision,
        native_revision=Present(next(iter(native_revisions))) if native_revisions else Absent(),
        observed_at=observed_at,
        models=facts,
        diagnostics=diagnostics,
    )


def _normalize_row(row: Mapping[str, object]) -> _ObservedModel:
    key = _required_string(row, "id")
    dispatch_model = _required_string(row, "model")
    label = _required_string(row, "displayName", fallback="display_name")
    modalities_value = row.get("inputModalities", row.get("input_modalities"))
    if not isinstance(modalities_value, list) or not modalities_value:
        raise ProtocolDefect(f"Codex model {key!r} has incomplete input modalities")
    modalities: list[Literal["text", "image"]] = []
    for item in modalities_value:
        value = _enum_string(item)
        if value not in ("text", "image"):
            raise ProtocolDefect(f"Codex model {key!r} has an unknown input modality")
        if value in modalities:
            raise ProtocolDefect(f"Codex model {key!r} has duplicate input modalities")
        modalities.append(value)

    reasoning_value = row.get("supportedReasoningEfforts", row.get("supported_reasoning_efforts"))
    if not isinstance(reasoning_value, list) or not reasoning_value:
        raise ProtocolDefect(f"Codex model {key!r} has incomplete reasoning facts")
    reasoning: list[AgentReasoningFacts] = []
    for item in reasoning_value:
        option = _object_mapping(item, f"Codex model {key!r} reasoning row")
        native = _required_enum_string(option, "reasoningEffort", fallback="reasoning_effort")
        label_value = _required_string(option, "description")
        reasoning.append(
            AgentReasoningFacts(key=native, label=label_value, native_wire_value=native)
        )
    if len({item.key for item in reasoning}) != len(reasoning):
        raise ProtocolDefect(f"Codex model {key!r} has duplicate reasoning keys")

    default_value = row.get("defaultReasoningEffort", row.get("default_reasoning_effort"))
    if default_value is None:
        source_default: Presence[str] = Absent()
    else:
        default = _enum_string(default_value)
        if not default or sum(item.key == default for item in reasoning) != 1:
            raise ProtocolDefect(f"Codex model {key!r} source default is not one reasoning row")
        source_default = Present(default)

    upgrades: list[str] = []
    direct_upgrade = row.get("upgrade")
    if direct_upgrade is not None:
        upgrades.append(_nonempty_string(direct_upgrade, f"Codex model {key!r} upgrade"))
    upgrade_info_value = row.get("upgradeInfo", row.get("upgrade_info"))
    if upgrade_info_value is not None:
        upgrade_info = _object_mapping(upgrade_info_value, f"Codex model {key!r} upgradeInfo")
        upgrades.append(_required_string(upgrade_info, "model"))

    return _ObservedModel(
        key=key,
        dispatch_model=dispatch_model,
        label=label,
        # The pinned public SDK's Model row exposes neither capacity. Do not
        # infer private cache/debug fields into the source contract; a future
        # SDK addition must be adopted here deliberately with conformance.
        source_context_window=Absent(),
        source_max_output_tokens=Absent(),
        input_modalities=tuple(modalities),
        reasoning=tuple(reasoning),
        source_default_reasoning=source_default,
        native_upgrade_targets=tuple(dict.fromkeys(upgrades)),
    )


def _validate_observed_rows(rows: tuple[_ObservedModel, ...]) -> None:
    if not rows:
        raise ProtocolDefect("Codex model/list returned no visible models")
    if len({row.key for row in rows}) != len(rows):
        raise ProtocolDefect("Codex model/list returned duplicate model ids")
    if len({row.dispatch_model for row in rows}) != len(rows):
        raise ProtocolDefect("Codex model/list returned duplicate dispatch models")


def _resolve_upgrades(
    rows: tuple[_ObservedModel, ...],
) -> tuple[tuple[AgentModelFacts, ...], tuple[AgentCatalogDiagnostic, ...]]:
    facts: list[AgentModelFacts] = []
    diagnostics: list[AgentCatalogDiagnostic] = []
    for row in rows:
        upgrade: Presence[AgentUpgradeFacts] = Absent()
        if len(row.native_upgrade_targets) > 1:
            diagnostics.append(
                UpgradeSourceConflict(
                    model_key=row.key,
                    native_targets=row.native_upgrade_targets,
                )
            )
        elif row.native_upgrade_targets:
            source_target = row.native_upgrade_targets[0]
            matches = tuple(
                candidate
                for candidate in rows
                if source_target in (candidate.key, candidate.dispatch_model)
            )
            if len(matches) == 1:
                upgrade = Present(AgentUpgradeFacts(target_key=matches[0].key))
            elif not matches:
                diagnostics.append(
                    UpgradeTargetUnresolved(model_key=row.key, native_target=source_target)
                )
            else:
                diagnostics.append(
                    UpgradeTargetAmbiguous(model_key=row.key, native_target=source_target)
                )
        fingerprint = _row_fingerprint(row, upgrade)
        facts.append(
            AgentModelFacts(
                key=row.key,
                dispatch_model=row.dispatch_model,
                label=row.label,
                source_context_window=row.source_context_window,
                source_max_output_tokens=row.source_max_output_tokens,
                input_modalities=row.input_modalities,
                reasoning=row.reasoning,
                source_default_reasoning=row.source_default_reasoning,
                upgrade=upgrade,
                retirement=Absent(),
                row_fingerprint=fingerprint,
            )
        )
    return tuple(facts), tuple(diagnostics)


def _row_fingerprint(
    row: _ObservedModel,
    upgrade: Presence[AgentUpgradeFacts],
) -> str:
    return _hash(
        b"provider-runtime.agent-model-row.v1",
        {
            "key": row.key,
            "dispatch_model": row.dispatch_model,
            "label": row.label,
            "source_context_window": _presence_json(row.source_context_window),
            "source_max_output_tokens": _presence_json(row.source_max_output_tokens),
            "input_modalities": row.input_modalities,
            "reasoning": tuple(
                {
                    "key": item.key,
                    "native_wire_value": item.native_wire_value,
                }
                for item in row.reasoning
            ),
            "source_default_reasoning": _presence_json(row.source_default_reasoning),
            "native_upgrade_targets": row.native_upgrade_targets,
            "upgrade": _presence_json(upgrade),
        },
    )


def _presence_json(value: Presence[object]) -> object:
    if isinstance(value, Present):
        child = value.value
        if isinstance(child, AgentUpgradeFacts):
            child = {"target_key": child.target_key}
        return {"kind": "present", "value": child}
    return {"kind": "absent"}


def _hash(domain: bytes, value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        domain + b"\0" + canonical_json_bytes(freeze_json_object(value))
    ).hexdigest()


def _object_mapping(value: object, context: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", by_alias=True, exclude_none=False)
        if isinstance(dumped, Mapping):
            return dumped
    raise ProtocolDefect(f"{context} was not an object")


def _required_string(value: Mapping[str, object], key: str, *, fallback: str | None = None) -> str:
    candidate = value.get(key)
    if candidate is None and fallback is not None:
        candidate = value.get(fallback)
    return _nonempty_string(candidate, key)


def _nonempty_string(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise ProtocolDefect(f"{context} was not a non-empty string")
    return value


def _enum_string(value: object) -> str:
    candidate = getattr(value, "value", value)
    return candidate if type(candidate) is str else ""


def _required_enum_string(
    value: Mapping[str, object], key: str, *, fallback: str | None = None
) -> str:
    candidate = value.get(key)
    if candidate is None and fallback is not None:
        candidate = value.get(fallback)
    normalized = _enum_string(candidate)
    if not normalized:
        raise ProtocolDefect(f"{key} was not a non-empty string")
    return normalized


__all__ = [
    "AGENT_BACKEND_CONTRACT_REVISION",
    "AgentCatalogDiagnostic",
    "AgentModelCatalog",
    "AgentModelFacts",
    "AgentModelKey",
    "AgentReasoningFacts",
    "AgentReasoningKey",
    "AgentUpgradeFacts",
    "NativeCatalogRevision",
    "UpgradeSourceConflict",
    "UpgradeTargetAmbiguous",
    "UpgradeTargetUnresolved",
    "read_codex_model_catalog",
]
