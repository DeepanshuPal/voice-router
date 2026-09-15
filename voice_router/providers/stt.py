"""STT adapters: Deepgram, OpenAI Whisper, Groq Whisper, AssemblyAI, Gladia,
Speechmatics, Rev AI, Cartesia Ink, Smallest Pulse, and a zero-key mock."""

from __future__ import annotations

import asyncio
import io
import os
import time
import wave
from urllib.parse import urlencode

import httpx
import json

from .base import (ProviderError, ProviderUnavailable, STTProvider,
                   StreamingSTTEvent, StreamingSTTResult)


async def _adaptive_poll_delays(timeout_s: float = 180.0):
    """Yield after adaptive waits: 250ms initially, capped at 4s."""
    delay = 0.25
    elapsed = 0.0
    while elapsed < timeout_s:
        await asyncio.sleep(delay)
        elapsed += delay
        yield elapsed
        delay = min(delay * 1.7, 4.0)


class DeepgramSTT(STTProvider):
    BASE = "https://api.deepgram.com/v1/listen"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("DEEPGRAM_API_KEY not set")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.BASE,
                params={"model": model, "language": language,
                        "punctuate": "true", "smart_format": "false"},
                headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
                content=audio,
            )
        if resp.status_code != 200:
            raise ProviderError(f"deepgram {resp.status_code}: {resp.text[:200]}")
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


class DeepgramFluxSTT(STTProvider):
    protocol = "streaming_websocket"
    """Deepgram Flux over its required Listen v2 WebSocket endpoint.

    WAV input is decoded to PCM and sent in 80 ms chunks at realtime pace.
    Timing events are client-observed. An ``Update`` becomes ``stable_partial``
    only after the same non-empty transcript appears twice consecutively; this
    avoids pretending every mutable partial is stable.
    """

    BASE = "wss://api.deepgram.com/v2/listen"
    CHUNK_MS = 80

    @staticmethod
    def _pcm(audio: bytes) -> tuple[bytes, int, int]:
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                    raise ProviderError("deepgram-flux requires mono 16-bit PCM WAV")
                rate = wav.getframerate()
                return wav.readframes(wav.getnframes()), rate, wav.getnframes()
        except (wave.Error, EOFError) as exc:
            raise ProviderError(f"deepgram-flux invalid WAV: {exc}") from exc

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        return (await self.transcribe_streaming(audio, model, language)).transcript

    async def transcribe_streaming(
        self, audio: bytes, model: str, language: str
    ) -> StreamingSTTResult:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("DEEPGRAM_API_KEY not set")
        try:
            import websockets
        except ImportError as exc:
            raise ProviderUnavailable("websockets dependency is not installed") from exc

        pcm, rate, frames = self._pcm(audio)
        flux_model = model or ("flux-general-en" if language == "en" else "flux-general-multi")
        params: list[tuple[str, str]] = [
            ("model", flux_model), ("encoding", "linear16"),
            ("sample_rate", str(rate)),
        ]
        if flux_model == "flux-general-multi" and language:
            params.append(("language_hint", language))
        url = f"{self.BASE}?{urlencode(params)}"
        started = time.perf_counter()
        events: list[StreamingSTTEvent] = []
        final = ""
        last_update = None
        stable_emitted = False
        bytes_per_chunk = max(2, int(rate * 2 * self.CHUNK_MS / 1000))

        try:
            async with websockets.connect(
                url, additional_headers={"Authorization": f"Token {key}"},
                open_timeout=15, close_timeout=10, max_size=8 * 1024 * 1024,
            ) as ws:
                events.append(StreamingSTTEvent(
                    "connection_open", (time.perf_counter() - started) * 1000))

                async def send_audio():
                    target = time.perf_counter()
                    for offset in range(0, len(pcm), bytes_per_chunk):
                        chunk = pcm[offset:offset + bytes_per_chunk]
                        if chunk:
                            await ws.send(chunk)
                        target += self.CHUNK_MS / 1000
                        await asyncio.sleep(max(0, target - time.perf_counter()))
                    await ws.send(json.dumps({"type": "CloseStream"}))

                sender = asyncio.create_task(send_audio())
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        message = json.loads(raw)
                        if message.get("type") != "TurnInfo":
                            continue
                        event_name = message.get("event", "Update")
                        transcript = (message.get("transcript") or "").strip()
                        elapsed = (time.perf_counter() - started) * 1000
                        kind = {
                            "EagerEndOfTurn": "eager_end_of_turn",
                            "TurnResumed": "turn_resumed",
                            "EndOfTurn": "final",
                            "StartOfTurn": "start_of_turn",
                        }.get(event_name, "partial")
                        if transcript and kind == "partial":
                            if not any(e.kind == "first_partial" for e in events):
                                events.append(StreamingSTTEvent("first_partial", elapsed, transcript))
                            if transcript == last_update and not stable_emitted:
                                events.append(StreamingSTTEvent("stable_partial", elapsed, transcript))
                                stable_emitted = True
                            last_update = transcript
                        events.append(StreamingSTTEvent(
                            kind, elapsed, transcript,
                            {k: message[k] for k in ("event", "turn_index",
                             "audio_window_start", "audio_window_end",
                             "end_of_turn_confidence", "sequence_id") if k in message},
                        ))
                        if kind == "final" and transcript:
                            final = transcript
                finally:
                    await sender
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"deepgram-flux stream failed: {str(exc)[:180]}") from exc

        completion = (time.perf_counter() - started) * 1000
        if not final:
            final = next((e.transcript for e in reversed(events) if e.transcript), "")
        return StreamingSTTResult(
            transcript=final, protocol="streaming_websocket", events=tuple(events),
            audio_duration_ms=frames / rate * 1000, completion_ms=completion,
        )


class OpenAISTT(STTProvider):
    BASE = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                headers={"Authorization": f"Bearer {key}"},
                data={"model": model, "language": language},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
        if resp.status_code != 200:
            raise ProviderError(f"openai {resp.status_code}: {resp.text[:200]}")
        return resp.json()["text"]


class GroqSTT(STTProvider):
    BASE = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("GROQ_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                headers={"Authorization": f"Bearer {key}"},
                data={"model": model, "language": language},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
        if resp.status_code != 200:
            raise ProviderError(f"groq {resp.status_code}: {resp.text[:200]}")
        return resp.json()["text"]


class AssemblyAISTT(STTProvider):
    """Async API: upload, submit, poll."""
    protocol = "async_batch"

    BASE = "https://api.assemblyai.com/v2"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("ASSEMBLYAI_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            up = await client.post(f"{self.BASE}/upload", headers={"Authorization": key},
                                   content=audio)
            if up.status_code != 200:
                raise ProviderError(f"assemblyai upload {up.status_code}: {up.text[:150]}")
            sub = await client.post(f"{self.BASE}/transcript", headers={"Authorization": key},
                                    json={"audio_url": up.json()["upload_url"],
                                          "speech_models": [model],
                                          "language_code": language})
            if sub.status_code != 200:
                raise ProviderError(f"assemblyai submit {sub.status_code}: {sub.text[:150]}")
            tid = sub.json()["id"]
            async for _ in _adaptive_poll_delays():
                poll = await client.get(f"{self.BASE}/transcript/{tid}",
                                        headers={"Authorization": key})
                if poll.status_code != 200:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"assemblyai poll {poll.status_code}: {detail}")
                try:
                    d = poll.json()
                except ValueError as exc:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"assemblyai poll returned non-JSON: {detail}") from exc
                if d.get("status") == "completed":
                    return d["text"]
                if d.get("status") == "error":
                    raise ProviderError(f"assemblyai: {d.get('error', '')[:150]}")
        raise ProviderError("assemblyai: poll timeout")


class GladiaSTT(STTProvider):
    """Async API: upload, pre-recorded job, poll result_url."""
    protocol = "async_batch"

    BASE = "https://api.gladia.io/v2"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("GLADIA_API_KEY not set")
        h = {"x-gladia-key": key}
        async with httpx.AsyncClient(timeout=60) as client:
            up = await client.post(f"{self.BASE}/upload", headers=h,
                                   files={"audio": ("audio.wav", audio, "audio/wav")})
            if up.status_code != 200:
                raise ProviderError(f"gladia upload {up.status_code}: {up.text[:150]}")
            sub = await client.post(f"{self.BASE}/pre-recorded", headers=h,
                                    json={"audio_url": up.json()["audio_url"],
                                          "language_config": {"languages": [language]}})
            if sub.status_code not in (200, 201):
                raise ProviderError(f"gladia submit {sub.status_code}: {sub.text[:150]}")
            result_url = sub.json()["result_url"]
            async for _ in _adaptive_poll_delays():
                poll = await client.get(result_url, headers=h)
                if poll.status_code != 200:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"gladia poll {poll.status_code}: {detail}")
                try:
                    d = poll.json()
                except ValueError as exc:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"gladia poll returned non-JSON: {detail}") from exc
                if d.get("status") == "done":
                    return d["result"]["transcription"]["full_transcript"]
                if d.get("status") == "error":
                    raise ProviderError(f"gladia: {str(d.get('message'))[:150]}")
        raise ProviderError("gladia: poll timeout")


class SpeechmaticsSTT(STTProvider):
    """Batch API v2: multipart submit, poll, fetch txt transcript."""
    protocol = "async_batch"

    BASE = "https://asr.api.speechmatics.com/v2/jobs"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("SPEECHMATICS_API_KEY not set")
        h = {"Authorization": f"Bearer {key}"}
        cfg = json.dumps({"type": "transcription",
                          "transcription_config": {"language": language,
                                                   "operating_point": model}})
        async with httpx.AsyncClient(timeout=60) as client:
            sub = await client.post(self.BASE, headers=h,
                                    data={"config": cfg},
                                    files={"data_file": ("audio.wav", audio, "audio/wav")})
            if sub.status_code not in (200, 201):
                raise ProviderError(f"speechmatics submit {sub.status_code}: {sub.text[:150]}")
            jid = sub.json()["id"]
            async for _ in _adaptive_poll_delays():
                poll = await client.get(f"{self.BASE}/{jid}", headers=h)
                if poll.status_code != 200:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"speechmatics poll {poll.status_code}: {detail}")
                try:
                    st = poll.json()
                except ValueError as exc:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"speechmatics poll returned non-JSON: {detail}") from exc
                status = st.get("job", {}).get("status")
                if status == "done":
                    t = await client.get(f"{self.BASE}/{jid}/transcript", headers=h,
                                         params={"format": "txt"})
                    return t.text.strip()
                if status == "rejected":
                    raise ProviderError(f"speechmatics rejected: {str(st['job'])[:150]}")
        raise ProviderError("speechmatics: poll timeout")


class RevSTT(STTProvider):
    """Async API: multipart submit, poll, fetch text/plain transcript."""
    protocol = "async_batch"

    BASE = "https://api.rev.ai/speechtotext/v1/jobs"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("REV_API_KEY not set")
        h = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(timeout=60) as client:
            sub = await client.post(self.BASE, headers=h,
                                    data={"options": json.dumps({"language": language,
                                                                 "model": model})},
                                    files={"media": ("audio.wav", audio, "audio/wav")})
            if sub.status_code not in (200, 201):
                raise ProviderError(f"rev submit {sub.status_code}: {sub.text[:150]}")
            jid = sub.json()["id"]
            async for _ in _adaptive_poll_delays():
                poll = await client.get(f"{self.BASE}/{jid}", headers=h)
                if poll.status_code != 200:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"rev poll {poll.status_code}: {detail}")
                try:
                    st = poll.json()
                except ValueError as exc:
                    detail = poll.text[:150].replace("\n", " ")
                    raise ProviderError(f"rev poll returned non-JSON: {detail}") from exc
                if st.get("status") == "transcribed":
                    t = await client.get(f"{self.BASE}/{jid}/transcript",
                                         headers={**h, "Accept": "text/plain"})
                    import re
                    parts = []
                    for line in t.text.splitlines():
                        m = re.match(r"^Speaker \d+\s+\d{2}:\d{2}:\d{2}\s+(.*)$", line.strip())
                        if m:
                            parts.append(m.group(1))
                        elif line.strip():
                            parts.append(line.strip())
                    return " ".join(parts)
                if st.get("status") == "failed":
                    raise ProviderError(f"rev failed: {str(st.get('failure_detail'))[:150]}")
        raise ProviderError("rev: poll timeout")



class CartesiaSTT(STTProvider):
    """Ink-Whisper batch STT: multipart POST, JSON transcript back."""

    BASE = "https://api.cartesia.ai/stt"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("CARTESIA_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                headers={"X-API-Key": key, "Cartesia-Version": "2025-04-16"},
                data={"model": model, "language": language},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
        if resp.status_code != 200:
            raise ProviderError(f"cartesia-stt {resp.status_code}: {resp.text[:200]}")
        return resp.json()["text"]


class SmallestSTT(STTProvider):
    """Smallest AI Pulse: raw-bytes POST, JSON transcript back."""

    BASE = "https://api.smallest.ai/waves/v1/stt/"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("SMALLEST_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                params={"model": model, "language": language},
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/octet-stream"},
                content=audio,
            )
        if resp.status_code != 200:
            raise ProviderError(f"smallest-stt {resp.status_code}: {resp.text[:200]}")
        return resp.json()["transcription"]


class MockSTT(STTProvider):
    """Always-on provider so the routing path runs end to end with no keys."""

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        return f"[mock transcript: {len(audio)} bytes, language={language}]"


REGISTRY = {
    "deepgram": DeepgramSTT,
    "deepgram-flux": DeepgramFluxSTT,
    "assemblyai": AssemblyAISTT,
    "gladia": GladiaSTT,
    "speechmatics": SpeechmaticsSTT,
    "rev": RevSTT,
    "cartesia-stt": CartesiaSTT,
    "smallest-stt": SmallestSTT,
    "openai-whisper": OpenAISTT,
    "groq-whisper": GroqSTT,
    "mock-stt": MockSTT,
}
