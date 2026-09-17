import hashlib, json
import pytest
from scripts.verify_publication_gate import REQUIRED_ATTACKS, verify

def write(tmp_path,payload,review):
 a=tmp_path/'result.json'; a.write_text(json.dumps(payload)); review['artifact_sha256']=hashlib.sha256(a.read_bytes()).hexdigest(); r=tmp_path/'review.json'; r.write_text(json.dumps(review)); return a,r

def valid():
 return ({'sample_data':False,'measurement':{'execution':'serial','repeats_per_clip':5,'runner_region':'test-region'},'runs':[{'status':'ok','provider':'p','model':'m','protocol':'streaming_websocket','repeat':1,'reference':'a','hypothesis':'a','events':[{'kind':'final','elapsed_ms':1}]}]}, {'disposition':'approved','reviewer':'independent reviewer','reviewed_at':'2026-09-15','artifact_commit':'abc','attacks_attempted':sorted(REQUIRED_ATTACKS)})

def test_gate_accepts_review_bound_to_exact_artifact(tmp_path):
 p,r=valid(); a,review=write(tmp_path,p,r); assert verify(a,review)['disposition']=='approved'

def test_gate_rejects_stale_review_after_artifact_change(tmp_path):
 p,r=valid(); a,review=write(tmp_path,p,r); a.write_text(a.read_text()+'\n')
 with pytest.raises(SystemExit,match='digest does not match'): verify(a,review)

def test_gate_rejects_missing_attack_and_stream_events(tmp_path):
 p,r=valid(); p['runs'][0]['events']=[]; r['attacks_attempted'].remove('streaming_event_validation'); a,review=write(tmp_path,p,r)
 with pytest.raises(SystemExit,match='missing review attacks'): verify(a,review)

def test_reviewed_generator_publishes_only_bound_artifact(tmp_path):
 import subprocess
 p,r=valid(); a,review=write(tmp_path,p,r); out=tmp_path/'site'
 subprocess.run(['python','scripts/generate_leaderboard.py','--results',str(a),'--review',str(review),'--out',str(out)],check=True)
 assert 'reviewed subset' in (out/'index.html').read_text().lower()
 assert json.loads((out/'data/status.json').read_text())['status']=='reviewed-subset'
 assert json.loads((out/'data/results.json').read_text())['runs'][0]['provider']=='p'
