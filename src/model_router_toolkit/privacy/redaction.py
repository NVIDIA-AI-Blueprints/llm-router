"""Best-effort request text redaction for cloud-bound inference calls."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_PRESIDIO_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "DATE_TIME",
    "URL",
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
)


class TextRedactor(Protocol):
    """Minimal interface implemented by request redaction backends."""

    def redact_text(self, text: str) -> str:
        """Return ``text`` with sensitive spans replaced."""


class NoopRedactor:
    """Redactor that preserves text exactly."""

    def redact_text(self, text: str) -> str:
        return text


@dataclass(frozen=True)
class RedactionConfig:
    """Environment-driven redaction settings."""

    enabled: bool = False
    backend: str = "presidio"
    score_threshold: float = 0.35
    entities: tuple[str, ...] | None = DEFAULT_PRESIDIO_ENTITIES
    language: str = "en"
    fail_open: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> RedactionConfig:
        enabled_value = env.get("MODEL_ROUTER_REDACTION_ENABLED", "")
        legacy_value = env.get("MODEL_ROUTER_REDACTION", "")
        enabled = _parse_bool(enabled_value or legacy_value, default=False)
        backend = (env.get("MODEL_ROUTER_REDACTION_BACKEND") or "presidio").strip().lower()
        score_threshold = _parse_float(
            env.get("MODEL_ROUTER_REDACTION_SCORE_THRESHOLD"),
            default=0.35,
        )
        entities = (
            _parse_entities(env.get("MODEL_ROUTER_REDACTION_ENTITIES"))
            or DEFAULT_PRESIDIO_ENTITIES
        )
        language = (env.get("MODEL_ROUTER_REDACTION_LANGUAGE") or "en").strip() or "en"
        fail_open = _parse_bool(env.get("MODEL_ROUTER_REDACTION_FAIL_OPEN"), default=True)
        return cls(
            enabled=enabled,
            backend=backend,
            score_threshold=score_threshold,
            entities=entities,
            language=language,
            fail_open=fail_open,
        )


class PresidioRedactor:
    """Presidio-backed text redactor.

    Presidio is imported lazily so deployments that do not enable redaction do
    not need the optional privacy dependencies installed.
    """

    def __init__(
        self,
        *,
        score_threshold: float = 0.35,
        entities: tuple[str, ...] | None = None,
        language: str = "en",
    ):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError as exc:
            raise RuntimeError(
                "Presidio redaction requires optional privacy dependencies. "
                "Install model-router-toolkit[privacy] and a compatible spaCy model."
            ) from exc

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operator_config = OperatorConfig("replace", {"new_value": "<PII>"})
        self._score_threshold = score_threshold
        self._entities = list(entities or DEFAULT_PRESIDIO_ENTITIES)
        self._language = language

    def redact_text(self, text: str) -> str:
        if not text:
            return text
        results = self._analyzer.analyze(
            text=text,
            language=self._language,
            entities=self._entities,
            score_threshold=self._score_threshold,
        )
        if not results:
            return text
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={"DEFAULT": self._operator_config},
        )
        return anonymized.text


def create_text_redactor(config: RedactionConfig) -> TextRedactor:
    """Create the configured redaction backend."""

    if not config.enabled:
        return NoopRedactor()
    if config.backend == "presidio":
        return PresidioRedactor(
            score_threshold=config.score_threshold,
            entities=config.entities,
            language=config.language,
        )
    raise ValueError(f"Unsupported redaction backend: {config.backend}")


def redact_openai_chat_payload(payload: Any, redactor: TextRedactor) -> Any:
    """Redact text-bearing fields in an OpenAI-compatible chat payload.

    The traversal intentionally limits itself to request text content. It does
    not rewrite tools, function schemas, images, or arbitrary metadata.
    """

    redacted = copy.deepcopy(payload)
    if not isinstance(redacted, dict):
        return redacted

    messages = redacted.get("messages")
    if isinstance(messages, list):
        for message in messages:
            _redact_message_content(message, redactor)

    return redacted


def redact_json_request_body(body: bytes, redactor: TextRedactor) -> tuple[bytes, bool]:
    """Return a redacted JSON request body and whether it changed."""

    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return body, False

    redacted = redact_openai_chat_payload(payload, redactor)
    if redacted == payload:
        return body, False
    return json.dumps(redacted, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), True


def _redact_message_content(message: Any, redactor: TextRedactor) -> None:
    if not isinstance(message, dict):
        return

    content = message.get("content")
    if isinstance(content, str):
        message["content"] = redactor.redact_text(content)
        return

    if not isinstance(content, list):
        return

    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            part["text"] = redactor.redact_text(part["text"])


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "presidio"}:
        return True
    if normalized in {"0", "false", "no", "off", "none", "noop"}:
        return False
    logger.warning("Unrecognized redaction boolean %r; using default %s", value, default)
    return default


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Invalid redaction score threshold %r; using %.2f", value, default)
        return default
    return max(0.0, min(1.0, parsed))


def _parse_entities(value: str | None) -> tuple[str, ...] | None:
    if value is None or value.strip() == "":
        return None
    entities = tuple(entity.strip() for entity in value.split(",") if entity.strip())
    return entities or None
