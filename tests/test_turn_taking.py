from pathlib import Path

from benchmarks.turn_taking.run import load_examples


def test_turn_corpus_is_balanced_and_auditable():
    corpus, examples = load_examples(Path("benchmarks/turn_taking/corpus.json"))
    assert corpus["split_ratio"] == 0.55
    assert len(examples) == 24
    assert sum(e.expected_end for e in examples) == 12
    for complete, prefix in zip(examples[::2], examples[1::2]):
        assert complete.kind == "complete"
        assert prefix.kind == "derived-prefix"
        assert len(prefix.audio) < len(complete.audio)
        assert len(prefix.text.split()) < len(complete.text.split())
