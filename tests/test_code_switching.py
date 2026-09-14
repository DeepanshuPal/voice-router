from benchmarks.code_switching.run import metrics, tokens

def test_mixed_script_metrics_reward_both_scripts():
    ref='आज meeting का final agenda share करें'
    perfect=metrics(ref,ref)
    no_english=metrics(ref,'आज का करें')
    assert perfect['mixed_wer']==0
    assert perfect['script_balance']==1
    assert no_english['english_token_recall']==0
    assert no_english['script_balance']<perfect['script_balance']

def test_normalization_keeps_devanagari_and_latin():
    assert tokens('FINAL एजेंडा, v2!')==['final','एजेंडा','v','2']
