# Benchmark reference transcripts

One `.txt` per sample clip in `../samples/`, named `<lang>-<n>.txt`.
Sources: FLEURS test split (CC-BY-4.0), normalized (lowercased, punctuation stripped) for WER scoring.

Last regenerated after adding AssemblyAI, Gladia, Speechmatics, and Rev AI credentials.

Scoring: languages with whitespace-delimited words use WER; ja/zh/ko/th use
CER (character error rate) because FLEURS ja references are phrase-spaced,
which makes whitespace-tokenized WER meaningless. See docs/methodology.html.
