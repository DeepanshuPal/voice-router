import pytest
from benchmarks.harness import corpus_error_rate, normalize_for_scoring, wer


def test_content_equivalent_punctuation_case_and_numbers_score_zero():
    ref = 'oliver has twenty five apples and can not wait'
    hyp = "Oliver has 25 apples, and can't wait!"
    assert normalize_for_scoring(ref) == normalize_for_scoring(hyp)
    assert wer(ref, hyp) == 0


def test_common_contractions_are_expanded_symmetrically():
    assert wer('we will not stop because we can not fail', "We won't stop, because we can't fail.") == 0


def test_corpus_wer_weights_reference_tokens_not_utterances():
    pairs = [('alpha', 'wrong'), ('alpha beta gamma delta epsilon zeta eta theta iota',
                                 'alpha beta gamma delta epsilon zeta eta theta iota')]
    assert corpus_error_rate(pairs, 'en') == 0.1
    assert sum(wer(a, b) for a, b in pairs) / 2 == 0.5


def test_latency_summary_separates_protocols_and_reports_iqr():
    from benchmarks.harness import latency_summary
    runs=[]
    for protocol, values in [('sync_batch',[10,20,30,40,50]),('async_batch',[100,200,300,400,500])]:
        runs += [{'provider':'p','protocol':protocol,'status':'ok','request_to_completion_ms':v} for v in values]
    rows=latency_summary(runs)
    assert len(rows)==2
    assert {r['protocol'] for r in rows}=={'sync_batch','async_batch'}
    assert {r['median_ms'] for r in rows}=={30,300}
    assert all(r['iqr_ms']>0 for r in rows)


def test_adapter_declares_protocol_instead_of_harness_name_guess():
    from voice_router.providers.stt import AssemblyAISTT, DeepgramSTT, DeepgramFluxSTT
    assert AssemblyAISTT.protocol == "async_batch"
    assert DeepgramSTT.protocol == "sync_batch"
    assert DeepgramFluxSTT.protocol == "streaming_websocket"

@pytest.mark.asyncio
async def test_stt_harness_rejects_fewer_than_five_repeats(tmp_path):
    from benchmarks.harness import run
    import pytest
    with pytest.raises(SystemExit, match="at least five"):
        await run(tmp_path, None, "en", tmp_path / "out.json", repeats=4)

@pytest.mark.asyncio
async def test_tts_harness_rejects_fewer_than_five_repeats():
    from benchmarks.tts_harness import run
    import pytest
    with pytest.raises(SystemExit, match="at least five"):
        await run("en", repeats=4)

def test_legacy_ambiguous_latency_field_is_not_summarized():
    from benchmarks.harness import latency_summary
    runs=[{'provider':'p','protocol':'sync_batch','status':'ok','latency_ms':v} for v in range(5)]
    assert latency_summary(runs) == []

@pytest.mark.asyncio
async def test_gladia_non_json_poll_becomes_provider_error(monkeypatch):
    import httpx
    from voice_router.config import ProviderSpec
    from voice_router.providers.base import ProviderError
    from voice_router.providers.stt import GladiaSTT

    responses = iter([
        httpx.Response(200, json={"audio_url": "https://audio.invalid/a"}),
        httpx.Response(201, json={"result_url": "https://result.invalid/r"}),
        httpx.Response(502, text="upstream unavailable"),
    ])
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return next(responses)
        async def get(self, *args, **kwargs): return next(responses)
    async def no_wait(*args, **kwargs):
        yield 0
    monkeypatch.setenv("GLADIA_API_KEY", "test")
    monkeypatch.setattr("voice_router.providers.stt.httpx.AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr("voice_router.providers.stt._adaptive_poll_delays", no_wait)
    spec=ProviderSpec("gladia","stt",["v2"],"GLADIA_API_KEY",languages=["en"])
    with pytest.raises(ProviderError, match="gladia poll 502"):
        await GladiaSTT(spec).transcribe(b"wav","v2","en")

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_name,spec,responses,error",
    [
        ("AssemblyAISTT", ("assemblyai", ["universal-2"], "ASSEMBLYAI_API_KEY"),
         [
             (200, {"upload_url": "https://audio.invalid/a"}),
             (200, {"id": "job"}),
             (502, None),
         ], "assemblyai poll 502"),
        ("SpeechmaticsSTT", ("speechmatics", ["enhanced"], "SPEECHMATICS_API_KEY"),
         [(201, {"id": "job"}), (502, None)], "speechmatics poll 502"),
        ("RevSTT", ("rev", ["machine"], "REV_API_KEY"),
         [(201, {"id": "job"}), (502, None)], "rev poll 502"),
    ],
)
async def test_async_stt_poll_http_error_becomes_provider_error(
        monkeypatch, adapter_name, spec, responses, error):
    import httpx
    from voice_router.config import ProviderSpec
    from voice_router.providers.base import ProviderError
    from voice_router.providers import stt

    queue = iter(httpx.Response(code, json=body) if body is not None
                 else httpx.Response(code, text="upstream unavailable")
                 for code, body in responses)
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return next(queue)
        async def get(self, *args, **kwargs): return next(queue)
    async def no_wait(*args, **kwargs):
        yield 0
    name, models, env_key = spec
    monkeypatch.setenv(env_key, "test")
    monkeypatch.setattr("voice_router.providers.stt.httpx.AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr("voice_router.providers.stt._adaptive_poll_delays", no_wait)
    adapter = getattr(stt, adapter_name)(ProviderSpec(name, "stt", models, env_key, languages=["en"]))
    with pytest.raises(ProviderError, match=error):
        await adapter.transcribe(b"wav", models[0], "en")

def test_provider_allowlist_filters_both_capabilities(monkeypatch):
    from voice_router.config import load_config
    from voice_router.providers.registry import build_providers
    monkeypatch.setenv('GROQ_API_KEY', 'x')
    monkeypatch.setenv('HUME_API_KEY', 'x')
    monkeypatch.setenv('BENCHMARK_PROVIDER_ALLOWLIST', 'groq-whisper,hume')
    providers = build_providers(load_config())
    assert set(providers['stt']) == {'groq-whisper'}
    assert set(providers['tts']) == {'hume'}

@pytest.mark.asyncio
async def test_llm_harness_rejects_fewer_than_five_repeats():
    from benchmarks.llm_harness import run
    with pytest.raises(SystemExit, match='at least five'):
        await run(repeats=4)
