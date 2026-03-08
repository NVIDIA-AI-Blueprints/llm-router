"""Tests for SQLite telemetry module."""

import pytest

import model_router_toolkit.telemetry as telemetry_mod


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    """Reset module-level state between tests."""
    original_path = telemetry_mod._DB_PATH
    original_conn = telemetry_mod._conn
    yield
    telemetry_mod._DB_PATH = original_path
    if telemetry_mod._conn is not None and telemetry_mod._conn is not original_conn:
        telemetry_mod._conn.close()
    telemetry_mod._conn = original_conn


def _enable_telemetry(tmp_path):
    db_path = tmp_path / "test_telemetry.db"
    telemetry_mod._DB_PATH = db_path
    telemetry_mod._conn = None
    return db_path


class TestTelemetry:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ROUTER_TELEMETRY_DB", raising=False)
        telemetry_mod._DB_PATH = None
        assert telemetry_mod.enabled() is False

    def test_enabled_with_env_var(self, tmp_path):
        _enable_telemetry(tmp_path)
        assert telemetry_mod.enabled() is True

    def test_raises_when_not_enabled(self):
        telemetry_mod._DB_PATH = None
        telemetry_mod._conn = None
        with pytest.raises(RuntimeError, match="Telemetry not enabled"):
            telemetry_mod._get_conn()

    def test_create_session(self, tmp_path):
        _enable_telemetry(tmp_path)
        session_id = telemetry_mod.create_session()
        assert isinstance(session_id, int)
        assert session_id >= 1

    def test_log_chat(self, tmp_path):
        _enable_telemetry(tmp_path)
        session_id = telemetry_mod.create_session()
        event_id = telemetry_mod.log_chat(session_id, "What is 2+2?", "nem-think", latency_ms=42.5)
        assert isinstance(event_id, int)
        assert event_id >= 1

    def test_get_stats_aggregation(self, tmp_path):
        _enable_telemetry(tmp_path)
        sid = telemetry_mod.create_session()
        telemetry_mod.log_chat(sid, "q1", "model-a", latency_ms=10.0)
        telemetry_mod.log_chat(sid, "q2", "model-b", latency_ms=20.0)
        telemetry_mod.log_chat(sid, "q3", "model-a", latency_ms=30.0)

        stats = telemetry_mod.get_stats()
        assert stats["total_events"] == 3
        assert stats["total_sessions"] == 1
        assert stats["avg_latency_ms"] == pytest.approx(20.0, abs=0.1)
        assert stats["by_model"]["model-a"] == 2
        assert stats["by_model"]["model-b"] == 1
