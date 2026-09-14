from benchmarks.harness import corpus_error_rate, normalize_for_scoring, wer


def test_content_equivalent_punctuation_case_and_numbers_score_zero():
    ref = 'oliver has twenty five apples and can not wait'
    hyp = "Oliver has 25 apples, and can't wait!"
    assert normalize_for_scoring(ref) == normalize_for_scoring(hyp)
    assert wer(ref, hyp) == 0


def test_common_contractions_are_expanded_symmetrically():
    assert wer('we will not stop because we can not fail', "We won't stop, because we can't fail.") == 0


def test_corpus_wer_weights_reference_tokens_not_utterances():
    pairs = [('alpha', 'wrong'), ('alpha beta gamma delta epsilon zeta eta theta iota',
                                 'alpha beta gamma delta epsilon zeta eta theta iota')]
    assert corpus_error_rate(pairs, 'en') == 0.1
    assert sum(wer(a, b) for a, b in pairs) / 2 == 0.5


def test_latency_summary_separates_protocols_and_reports_iqr():
    from benchmarks.harness import latency_summary
    runs=[]
    for protocol, values in [('sync_batch',[10,20,30,40,50]),('async_batch',[100,200,300,400,500])]:
        runs += [{'provider':'p','protocol':protocol,'status':'ok','latency_ms':v} for v in values]
    rows=latency_summary(runs)
    assert len(rows)==2
    assert {r['protocol'] for r in rows}=={'sync_batch','async_batch'}
    assert {r['median_ms'] for r in rows}=={30,300}
    assert all(r['iqr_ms']>0 for r in rows)
