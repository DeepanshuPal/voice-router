from voice_router.providers.base import sine_wav


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_models_lists_auto(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "auto" in ids
    assert any(i.startswith("mock-stt:") for i in ids)


def test_route_dry_run(client):
    r = client.post("/v1/route", json={"capability": "stt", "language": "en",
                                       "strategy": "cost", "units": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "cost"
    assert body["choices"][0]["provider"] == "mock-stt"


def test_transcribe_roundtrip(client):
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", sine_wav(), "audio/wav")},
                    data={"model": "auto", "language": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "mock-stt"
    assert "mock transcript" in body["text"]


def test_speech_returns_audio(client):
    r = client.post("/v1/audio/speech",
                    json={"model": "auto", "voice": "alloy", "input": "hello world"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert r.headers["x-voice-router-provider"] == "mock-tts"


def test_telemetry_records_calls(client):
    client.post("/v1/audio/speech",
                json={"model": "auto", "voice": "alloy", "input": "one call"})
    summary = client.get("/v1/telemetry").json()
    assert summary["total_calls"] >= 1
    assert "tts:mock-tts" in summary["providers"]
