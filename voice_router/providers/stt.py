"""STT adapters: Deepgram, OpenAI Whisper, Groq Whisper, AssemblyAI, Gladia,
Speechmatics, Rev AI, Cartesia Ink, and a zero-key mock."""

from __future__ import annotations

import asyncio
import os

import httpx
import json

from .base import ProviderError, ProviderUnavailable, STTProvider


class DeepgramSTT(STTProvider):
    BASE = "https://api.deepgram.com/v1/listen"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("DEEPGRAM_API_KEY not set")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.BASE,
                params={"model": model, "language": language},
                headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
                content=audio,
            )
        if resp.status_code != 200:
            raise ProviderError(f"deepgram {resp.status_code}: {resp.text[:200]}")
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


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
            for _ in range(45):
                await asyncio.sleep(4)
                d = (await client.get(f"{self.BASE}/transcript/{tid}",
                                      headers={"Authorization": key})).json()
                if d["status"] == "completed":
                    return d["text"]
                if d["status"] == "error":
                    raise ProviderError(f"assemblyai: {d.get('error', '')[:150]}")
        raise ProviderError("assemblyai: poll timeout")


class GladiaSTT(STTProvider):
    """Async API: upload, pre-recorded job, poll result_url."""

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
            for _ in range(45):
                await asyncio.sleep(4)
                d = (await client.get(result_url, headers=h)).json()
                if d.get("status") == "done":
                    return d["result"]["transcription"]["full_transcript"]
                if d.get("status") == "error":
                    raise ProviderError(f"gladia: {str(d.get('message'))[:150]}")
        raise ProviderError("gladia: poll timeout")


class SpeechmaticsSTT(STTProvider):
    """Batch API v2: multipart submit, poll, fetch txt transcript."""

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
            for _ in range(45):
                await asyncio.sleep(4)
                st = (await client.get(f"{self.BASE}/{jid}", headers=h)).json()
                if st["job"]["status"] == "done":
                    t = await client.get(f"{self.BASE}/{jid}/transcript", headers=h,
                                         params={"format": "txt"})
                    return t.text.strip()
                if st["job"]["status"] == "rejected":
                    raise ProviderError(f"speechmatics rejected: {str(st['job'])[:150]}")
        raise ProviderError("speechmatics: poll timeout")


class RevSTT(STTProvider):
    """Async API: multipart submit, poll, fetch text/plain transcript."""

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
            for _ in range(45):
                await asyncio.sleep(4)
                st = (await client.get(f"{self.BASE}/{jid}", headers=h)).json()
                if st["status"] == "transcribed":
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
                if st["status"] == "failed":
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


class MockSTT(STTProvider):
    """Always-on provider so the routing path runs end to end with no keys."""

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        return f"[mock transcript: {len(audio)} bytes, language={language}]"


REGISTRY = {
    "deepgram": DeepgramSTT,
    "assemblyai": AssemblyAISTT,
    "gladia": GladiaSTT,
    "speechmatics": SpeechmaticsSTT,
    "rev": RevSTT,
    "cartesia-stt": CartesiaSTT,
    "openai-whisper": OpenAISTT,
    "groq-whisper": GroqSTT,
    "mock-stt": MockSTT,
}
