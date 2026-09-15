"""Fail closed unless a measurement artifact and its adversarial review match.

This verifies structure and binding, not scientific validity. A human reviewer
still has to perform the attacks in docs/measurement-review.md and sign the
exact artifact digest. Generators must call this before exposing measurements.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

REQUIRED_ATTACKS={
 'normalization_equivalence','independent_corpus_error_rate','largest_error_inspection',
 'serial_repeats','protocol_separation','raw_timing_recompute','streaming_event_validation',
 'tts_first_audio_validation','public_diff','deployed_site_verification',
}

def digest(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(artifact:Path, review:Path)->dict:
    payload=json.loads(artifact.read_text()); decision=json.loads(review.read_text())
    errors=[]; measurement=payload.get('measurement',{})
    if payload.get('sample_data'): errors.append('sample_data cannot be published')
    if measurement.get('execution')!='serial': errors.append('execution must be serial')
    if int(measurement.get('repeats_per_clip',0))<5: errors.append('at least five repeats required')
    if not measurement.get('runner_region'): errors.append('runner_region required')
    if decision.get('artifact_sha256')!=digest(artifact): errors.append('review digest does not match artifact')
    if decision.get('disposition')!='approved': errors.append('review disposition is not approved')
    for field in ('reviewer','reviewed_at','artifact_commit'):
        if not decision.get(field): errors.append(f'review {field} required')
    attempted=set(decision.get('attacks_attempted',[])); missing=REQUIRED_ATTACKS-attempted
    if missing: errors.append('missing review attacks: '+', '.join(sorted(missing)))
    runs=payload.get('runs',[])
    if not runs: errors.append('raw runs required')
    for i,row in enumerate(runs):
        if row.get('status')!='ok': continue
        for field in ('provider','model','protocol','repeat'):
            if field not in row: errors.append(f'run {i} missing {field}')
        if row.get('reference') is not None and row.get('hypothesis') is None: errors.append(f'run {i} missing hypothesis')
        if row.get('protocol','').startswith('streaming') and not row.get('events'): errors.append(f'streaming run {i} missing events')
    if errors: raise SystemExit('publication gate rejected:\n- '+'\n- '.join(errors))
    return decision

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--review',type=Path,required=True)
    args=ap.parse_args(); d=verify(args.artifact,args.review); print(f"approved by {d['reviewer']} for {d['artifact_sha256']}")
if __name__=='__main__': main()
