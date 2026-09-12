import pytest
from fastapi.testclient import TestClient

from voice_router.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEMETRY_PATH", str(tmp_path / "calls.jsonl"))
    app = create_app()
    # force mock-only: real providers skip themselves without keys, but make it explicit
    for cap in ("stt", "tts"):
        for name in list(app.state.engine.providers[cap]):
            if not name.startswith("mock"):
                del app.state.engine.providers[cap][name]
    return TestClient(app)
