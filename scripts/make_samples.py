"""Generate synthetic WAV samples for pipeline testing. For real benchmark
numbers, replace these with actual speech clips (the more accents, the better)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_router.providers.base import sine_wav  # noqa: E402

out = Path("benchmarks/samples")
out.mkdir(parents=True, exist_ok=True)
for i, freq in enumerate((220, 330, 440), 1):
    (out / f"sample-{i}.wav").write_bytes(sine_wav(seconds=1.5, freq=freq))
    print(f"wrote {out}/sample-{i}.wav")
