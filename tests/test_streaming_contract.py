import io
import struct
import wave

import pytest
from pathlib import Path

from voice_router.config import ProviderSpec
from voice_router.providers.base import STTProvider, ProviderUnavailable, StreamingSTTEvent, StreamingSTTResult
from voice_router.providers.stt import DeepgramFluxSTT


def wav_bytes(frames=1600, rate=16000):
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(b'\0\0'*frames)
    return out.getvalue()


def test_streaming_result_exposes_client_observed_boundaries():
    result=StreamingSTTResult('hello','streaming_websocket',(StreamingSTTEvent('connection_open',12),StreamingSTTEvent('first_partial',80,'hel'),StreamingSTTEvent('final',240,'hello')),1000,250)
    assert result.first_ms('first_partial')==80
    assert result.first_ms('stable_partial') is None


@pytest.mark.asyncio
async def test_batch_adapter_rejects_streaming_instead_of_faking_it():
    spec=ProviderSpec('batch','stt',['m'],'',languages=['en'])
    with pytest.raises(ProviderUnavailable,match='does not implement streaming'):
        await STTProvider(spec).transcribe_streaming(wav_bytes(),'m','en')


def test_flux_wav_decoder_requires_mono_16bit_and_preserves_duration():
    pcm,rate,frames=DeepgramFluxSTT._pcm(wav_bytes())
    assert rate==16000 and frames==1600 and len(pcm)==3200


def float_wav_bytes(samples=(0.0, 0.5, -0.5), rate=16000):
    data = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32) + b"\x00\x00"
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE" +
            b"fmt " + struct.pack("<I", len(fmt)) + fmt +
            b"data" + struct.pack("<I", len(data)) + data)


def test_flux_wav_decoder_converts_ieee_float_to_linear16():
    pcm, rate, frames = DeepgramFluxSTT._pcm(float_wav_bytes())
    assert rate == 16000 and frames == 3
    assert struct.unpack("<3h", pcm) == (0, 16384, -16384)


def test_flux_wav_decoder_accepts_committed_fleurs_format():
    audio = (Path(__file__).parent.parent / "benchmarks" / "samples" / "en-1.wav").read_bytes()
    pcm, rate, frames = DeepgramFluxSTT._pcm(audio)
    assert rate == 16000 and frames > 0 and len(pcm) == frames * 2

@pytest.mark.asyncio
async def test_arena_requires_allowlist_and_preserves_existing_clips(tmp_path, monkeypatch):
    import benchmarks.arena_audio as arena
    from voice_router.config import ProviderSpec
    from voice_router.providers.base import TTSProvider
    class Fake(TTSProvider):
        async def synthesize(self, text, model, voice): return b"RIFFfake"
    refs=tmp_path/"refs"; refs.mkdir(); (refs/"en-1.txt").write_text("hello")
    out=tmp_path/"arena"/"audio"; out.mkdir(parents=True)
    manifest=out.parent/"manifest.json"
    manifest.write_text('{"language":"en","clips":[{"provider":"fake","model":"m","sample":"en-1","file":"audio/fake/en-1.wav","chars":5}]}')
    monkeypatch.setattr(arena,"REFERENCES",refs); monkeypatch.setattr(arena,"OUT",out); monkeypatch.setattr(arena,"MANIFEST",manifest)
    adapter=Fake(ProviderSpec("fake","tts",["m"],"",languages=["en"]))
    monkeypatch.setattr(arena,"build_providers",lambda cfg:{"tts":{"fake":adapter}})
    result=await arena.run(["fake"],missing_only=True)
    assert len(result["clips"])==1
