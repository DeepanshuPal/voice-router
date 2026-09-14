# Provider coverage: proposed $300 escalation

**Prepared:** 15 September 2026  
**Status:** proposal only; no purchase or signup is authorized.

Spend only after the measurement rebuild passes adversarial review. The goal is not maximum logo count. It is one defensible streaming-first comparison set plus a small reserve for reruns.

| Priority | Provider | Proposed cap | Measurement unlocked | Observable current block |
|---|---|---:|---|---|
| 1 | Deepgram | $75 | Nova-3 batch plus Flux streaming STT and Aura streaming TTS | Existing free quota is finite; Flux lacks the new streaming harness |
| 2 | ElevenLabs | $50 | Scribe v2 STT and streaming TTS TTFB | No free-tier API access on our account; paid plan required on 2026-09-14 |
| 3 | OpenAI | $50 | Whisper-1 / GPT transcription and streaming-capable audio comparison where API permits | No funded API account under the $0 rule |
| 4 | Soniox | $40 | stt-rt-v5 streaming plus TTS | API returned 402; console balance displayed $0.00 on 2026-09-14 |
| 5 | Cartesia | $35 | Ink streaming STT and Sonic streaming TTS repeats | Free monthly credits were near exhaustion on 2026-09-14 |
| 6 | Hume | $25 | Octave streaming TTS TTFB and repeat coverage | Free monthly quota is limited |
| 7 | Rime | $25 | Streaming TTS TTFB and repeat coverage | Current promotional balance exists but is finite |
| | **Total** | **$300** | | |

Excluded from this first paid tranche: Azure, AWS, Google Cloud, xAI, Sarvam, Rev, Inworld, Alibaba, Modulate, Gradium, Fish Audio, Speechify, MiniMax, Palabra, Bland, Maya, Meta and Nari. Reasons: card/account overhead, personal OAuth, captcha, batch-only redundancy, or lower priority than the streaming measurement spine. Their absence remains an explicit not-run fact, not a fabricated score.

Before any purchase, confirm the exact provider, amount, billing identity, renewal behavior, and cancellation/refund terms. Unused caps do not roll automatically to another provider without approval.

## Speko-scale paid-access estimate

**Researched:** 15 September 2026  
**Scope:** price research only. No account, subscription, card, or spend action was taken.  
**Comparison surface:** Speko currently describes 20 STT, 36 TTS, 14 LLM and 10 speech-to-speech catalog rows. The requested planning shorthand is 59 benchmark models across 10 languages with streaming measurements. Catalog counts and benchmarked counts are not identical, so this estimate budgets provider access, not a claim that every catalog row can run in every language.

### The short answer

The subscription floor is much lower than the total operating cost because most model APIs are usage-based:

| Scenario | Known recurring minimum | Known one-time/prepaid entry | Prudent first-month envelope | What it means |
|---|---:|---:|---:|---|
| A. Seven-provider streaming spine | **$11-$14/mo** | **$5 + Soniox's undisclosed top-up floor** | **$100-$150** | Deepgram, ElevenLabs, OpenAI, Soniox, Cartesia, Hume and Rime. The range treats Hume Starter ($3) as optional because Hume can be metered from Free; ElevenLabs Starter ($6) + Cartesia Pro ($5) are the hard published subscriptions. |
| B. Full market-relevant STT + TTS set | **$64-$67/mo** | **at least $5**, plus undisclosed xAI/Soniox/Modulate/Nari/Maya entry terms | **$200-$300** | Adds the self-serve paid floors that actually unlock breadth: Inworld Creator $25, Gradium XS $13, Speechify Starter $10 and MiniMax Audio Starter $5. Most other providers are $0-base pay-as-you-go. |
| C. Speko-parity surface, including LLM and S2S providers | **$64-$67/mo** | **at least $15**, plus the undisclosed balances above | **$300-$500** | LLM serverless providers mostly add usage, not subscriptions. Cerebras publicly starts self-serve payment at $10; OpenAI prepaid starts at $5. The upper envelope leaves room for repeated streaming runs, reruns after harness failures, and models priced per token rather than per minute. |

**Planning recommendation:** treat **$300 as the realistic first-month benchmark wallet**, not as a $300 subscription stack. The known recurring plan floor for broad coverage is about **$67/month**; the rest should remain capped variable credit. A full 5-repeat, multilingual streaming run should be costed from the final corpus duration and prompt/character counts before purchase. The current stop condition still prevents running it.

### Provider-by-provider minimums

`$0 base` means usage-based access with no published monthly subscription minimum. It does not mean a benchmark run is free. `Undisclosed` means the official page did not expose a self-serve entry amount; it must not be guessed.

| Provider | Basic paid path | Monthly minimum | Billing shape | Official evidence |
|---|---|---:|---|---|
| Deepgram | Pay As You Go | $0 | Usage; no minimum. Growth starts at $4,000/year and is unnecessary for benchmarking. | [Deepgram pricing](https://deepgram.com/pricing) |
| ElevenLabs | Starter | $6 | Monthly subscription; API model usage is priced in USD. | [ElevenLabs API pricing](https://elevenlabs.io/pricing/api) |
| OpenAI | Prepaid API credits | $0 recurring; $5 initial | Usage against prepaid balance; official minimum purchase $5, default $10. | [OpenAI prepaid billing](https://help.openai.com/en/articles/8264778-what-is-prepaid-billing) |
| Soniox | Pay as you go | $0 base | Usage; real-time STT is published at $0.12/hour. Funding/top-up minimum was not published on the pricing page. | [Soniox pricing](https://soniox.com/pricing) |
| Cartesia | Pro | $5 | Monthly subscription with 100K model credits. | [Cartesia pricing](https://www.cartesia.ai/pricing) |
| Hume | Starter | $3 | Monthly plan; usage beyond inclusions is metered. Keep as optional while Free access is sufficient. | [Hume pricing](https://www.hume.ai/pricing) |
| Rime | Starter | $0 base | Usage only; $0.03/1K chars for Mist and $0.05/1K for Coda. | [Rime pricing](https://www.rime.ai/pricing) |
| AssemblyAI | Pay as you go | $0 | No contract, minimum, or monthly subscription. | [AssemblyAI billing](https://www.assemblyai.com/docs/billing-and-pricing) |
| Gladia | Starter | $0 base | Prepaid wallet; $0.61/hour async, $0.75/hour real-time. | [Gladia pricing](https://www.gladia.io/pricing) |
| Speechmatics | Pro/pay as you go | $0 base | Usage billed after free credit; no published monthly plan floor. | [Speechmatics pricing](https://www.speechmatics.com/pricing) |
| Groq | Developer | $0 base | Pay per token/audio usage, invoiced in arrears; no immediate upgrade charge. | [Groq plans](https://groq.com/groqcloud), [billing FAQ](https://console.groq.com/docs/billing-faqs) |
| Smallest AI | Pay As You Go | $0 | No monthly commitment; model usage metered. | [Smallest model pricing](https://smallest.ai/pricing-page-2) |
| Google Cloud Speech / Gemini | Paid billing account | $0 base | Usage-based. Card/billing account required for paid tiers; no monthly subscription quoted. | [Google STT pricing](https://cloud.google.com/speech-to-text/pricing), [Gemini billing](https://ai.google.dev/gemini-api/docs/billing) |
| Alibaba Model Studio | Pay as you go | $0 base | Per-call/second/character usage. | [Alibaba Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |
| xAI | Prepaid or monthly invoiced | Undisclosed | Usage; official docs publish the billing modes but not a minimum credit purchase. | [xAI billing](https://docs.x.ai/console/billing), [xAI pricing](https://docs.x.ai/developers/pricing) |
| Inworld | Creator | $25 | Monthly subscription including $25 credits and lower STT/TTS rates. | [Inworld pricing](https://inworld.ai/pricing) |
| Gradium | XS | $13 | Monthly credit subscription; STT consumes 3 credits/sec and TTS 1 credit/character. | [Gradium pricing](https://gradium.ai/pricing), [credit rules](https://docs.gradium.ai/guides/credits) |
| Modulate | Credit/service tier | Undisclosed | Published model rates ($0.03/hour batch, $0.06/hour streaming), but the official page describes annual service-tier commitments without a public entry amount. | [Modulate model pricing](https://platform.modulate.ai/pricing) |
| Meta Model API | Pay as you go | $0 base | Usage; Muse Voice billed per audio minute with no minimum/upfront commitment. | [Meta pricing and limits](https://dev.meta.ai/docs/pricing-rate-limits/) |
| Nari Labs | Early Access paid path | Undisclosed | Free public beta exists; paid access is Early Access. Model page quotes $0.12/hour for fast ASR but no plan minimum. | [Nari Model APIs](https://narilabs.com/product/model-apis/) |
| Speechify | Starter | $10 | Monthly; 1M TTS characters included, then published per-character rate. | [Speechify API pricing](https://speechify.com/pricing-api/) |
| Fish Audio | API pay as you go | $0 | No subscription fee or monthly minimum; S2.1 Pro is $15/M UTF-8 bytes. | [Fish Audio API pricing](https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits) |
| MiniMax | Audio Starter | $5 | Monthly audio-points plan; pay-as-you-go is also available for standard API keys. | [MiniMax audio subscription](https://platform.minimax.io/docs/guides/pricing-speech), [pay as you go](https://platform.minimax.io/docs/guides/pricing-paygo) |
| Palabra | API pay as you go | $0 base | TTS/STT usage rates and $50 signup credit are published separately from event subscription plans. | [Palabra pricing](https://www.palabra.ai/pricing) |
| Bland | Start | $0 platform fee | Usage per connected minute. Build is $299/month but is not needed for benchmark access. | [Bland pricing](https://www.bland.ai/pricing) |
| Maya 2 Native | Paid access | Undisclosed | No authoritative public self-serve plan minimum found. Do not use Speko's approximate per-character rate as an access-floor claim. | Provider minimum not publicly verified |

### Additional providers needed for Speko's LLM rows

| Provider | Basic paid path | Entry floor | Official evidence |
|---|---|---:|---|
| Anthropic | API usage billing | $0 recurring; purchase minimum not publicly verified here | [Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing) |
| Baseten | Serverless Model APIs | $0 base | Per-token serverless model pricing; Basic account tier exists without a published monthly subscription. [Baseten pricing](https://www.baseten.co/pricing/) |
| Cerebras | Developer/self-serve | $10 payment | Official pricing says self-serve payment starts at $10. [Cerebras pricing](https://www.cerebras.ai/pricing) |
| Together AI | Serverless | $0 base | Usage-based with no minimum/provisioning cost. [Together serverless pricing](https://docs.together.ai/docs/serverless/overview) |

### What the totals exclude

- Provider usage itself. The exact bill depends on total audio minutes, generated characters, LLM tokens, repeats, failed/retried calls, and whether a model charges for session time or processed audio.
- Taxes, FX, card verification holds and regional surcharges.
- Enterprise-only models, higher concurrency, SLAs or annual commitments.
- H100 rental/ownership for open-weight rows.
- Labor and infrastructure for the local sidecar, dataset storage, egress and result review.
- Any spend before P0-1 through P0-4 and adversarial measurement review pass.

### Sources and confidence

All linked prices above were checked on vendor-owned pricing or documentation pages on 15 September 2026. Confidence is high for explicit self-serve amounts and `$0/no minimum` statements. Confidence is intentionally low/`Undisclosed` where the operator exposes only a model usage rate, a contact-sales path, or an authenticated top-up flow. Those unknowns are why the parity total is presented as a floor plus an operating envelope rather than a false exact number.
