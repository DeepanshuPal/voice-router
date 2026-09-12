"""Wire a Pipecat voice agent through voice-router.

Pipecat's OpenAI-compatible services take a base_url - point them at the
router and your existing pipeline gets per-call STT/TTS routing, fallback,
and telemetry with zero changes to the pipeline itself.

Run the router first:  uvicorn voice_router.main:app
Then:                  pip install voice-router[agents] && python examples/pipecat_agent.py
"""

import asyncio

ROUTER_BASE = "http://localhost:8000/v1"


async def main():
    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.services.openai.stt import OpenAISTTService
        from pipecat.services.openai.tts import OpenAITTSService
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.transports.daily import DailyTransport
    except ImportError:
        raise SystemExit("pip install voice-router[agents] first")

    # The router is keyless-facing: your agent holds ONE credential set
    # (the router's), and the router holds the provider keys.
    stt = OpenAISTTService(base_url=ROUTER_BASE, api_key="router", model="auto")
    tts = OpenAITTSService(base_url=ROUTER_BASE, api_key="router", model="auto", voice="alloy")
    llm = OpenAILLMService(base_url=ROUTER_BASE, api_key="router", model="auto")

    transport = DailyTransport(room_url="https://your.daily.co/room", token="...")
    pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
    # ... standard Pipecat runner from here. Your agent never knows which
    # provider actually served each leg, and swaps cost nothing.


if __name__ == "__main__":
    asyncio.run(main())
