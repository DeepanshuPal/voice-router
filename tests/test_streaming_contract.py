import io
import wave

import pytest

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
