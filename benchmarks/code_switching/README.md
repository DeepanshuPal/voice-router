# Hinglish code-switching STT benchmark

This corpus starts with 12 real Hindi-English code-switched clips from the
MUCS-Hinglish test split (CC-BY-4.0). Selection is deterministic and favors
within-utterance mixing: each reference has at least three Latin-script English
words, three Devanagari words, and an English-token ratio between 20% and 60%.
Distinct speakers are selected first.

`manifest.json` preserves the source shard and segment ID, speaker ID, duration,
script counts, and exact selection rule. Audio is normalized to 16 kHz mono
PCM. References are the dataset's mixed-script transcripts, unchanged except
for the trailing newline.

This commit contains the corpus only. Numbers appear on the public board only
after the nightly/manual provider matrix runs. That separation keeps code and
data commits at zero API cost and prevents push-triggered free-quota burn.

Source: https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish
