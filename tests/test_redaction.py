import json

from model_router_toolkit.privacy.redaction import (
    NoopRedactor,
    RedactionConfig,
    redact_json_request_body,
    redact_openai_chat_payload,
)


class RecordingRedactor:
    def __init__(self):
        self.seen: list[str] = []

    def redact_text(self, text: str) -> str:
        self.seen.append(text)
        return f"<redacted:{len(self.seen)}>"


def test_redaction_config_is_disabled_by_default():
    config = RedactionConfig.from_env({})

    assert config.enabled is False
    assert config.backend == "presidio"
    assert config.score_threshold == 0.35
    assert config.fail_open is True


def test_redaction_config_uses_limited_default_entity_scope():
    config = RedactionConfig.from_env(
        {
            "MODEL_ROUTER_REDACTION_ENABLED": "true",
        }
    )

    assert config.entities == (
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "DATE_TIME",
        "URL",
        "PERSON",
        "LOCATION",
        "ORGANIZATION",
    )


def test_redaction_config_parses_enabled_presidio_settings():
    config = RedactionConfig.from_env(
        {
            "MODEL_ROUTER_REDACTION_ENABLED": "true",
            "MODEL_ROUTER_REDACTION_BACKEND": "presidio",
            "MODEL_ROUTER_REDACTION_SCORE_THRESHOLD": "0.5",
            "MODEL_ROUTER_REDACTION_ENTITIES": "EMAIL_ADDRESS,PHONE_NUMBER",
            "MODEL_ROUTER_REDACTION_FAIL_OPEN": "false",
        }
    )

    assert config.enabled is True
    assert config.backend == "presidio"
    assert config.score_threshold == 0.5
    assert config.entities == ("EMAIL_ADDRESS", "PHONE_NUMBER")
    assert config.fail_open is False


def test_redact_openai_chat_payload_only_rewrites_text_content():
    payload = {
        "model": "nvidia-routed",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Email ada@example.com before calling."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Call +1 555 0100."},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send email to the address requested by the user.",
                },
            }
        ],
    }
    redactor = RecordingRedactor()

    redacted = redact_openai_chat_payload(payload, redactor)

    assert redacted["messages"][0]["content"] == "<redacted:1>"
    assert redacted["messages"][1]["content"] == "<redacted:2>"
    assert redacted["messages"][2]["content"][0]["text"] == "<redacted:3>"
    assert redacted["messages"][2]["content"][1]["image_url"]["url"] == "https://example.test/a.png"
    assert redacted["tools"] == payload["tools"]
    assert payload["messages"][1]["content"] == "Email ada@example.com before calling."
    assert redactor.seen == [
        "You are concise.",
        "Email ada@example.com before calling.",
        "Call +1 555 0100.",
    ]


def test_noop_redactor_leaves_payload_equal():
    payload = {"messages": [{"role": "user", "content": "Email ada@example.com"}]}

    redacted = redact_openai_chat_payload(payload, NoopRedactor())

    assert redacted == payload
    assert redacted is not payload


def test_redact_json_request_body_rewrites_valid_json_body():
    body = json.dumps(
        {"messages": [{"role": "user", "content": "My SSN is 123-45-6789."}]}
    ).encode()
    redactor = RecordingRedactor()

    redacted_body, changed = redact_json_request_body(body, redactor)

    assert changed is True
    assert json.loads(redacted_body) == {
        "messages": [{"role": "user", "content": "<redacted:1>"}]
    }


def test_redact_json_request_body_ignores_invalid_json():
    redacted_body, changed = redact_json_request_body(b"not-json", RecordingRedactor())

    assert changed is False
    assert redacted_body == b"not-json"
