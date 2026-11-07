"""Package a finite CPU screen with pinned upstream source, without model training."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
upstream = ROOT/'reference/upstream/farnoosh-run3'
directory = ROOT/'kaggle/public-decoder-v6'
directory.mkdir(parents=True, exist_ok=True)
files = {p.relative_to(ROOT).as_posix(): p for p in (ROOT/'src').glob('*.py')}
files['scripts/screen_public_decoder.py'] = ROOT/'scripts/screen_public_decoder.py'
for name in ('solarseg/__init__.py', 'solarseg/postprocess.py', 'LICENSE'):
    files['upstream/'+name] = upstream/name
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, path in files.items():
        archive.writestr(name, path.read_bytes())
payload = base64.b64encode(buffer.getvalue()).decode('ascii')
runner = f'''import base64, io, json, os, runpy, sys, zipfile
from pathlib import Path
root = Path('/kaggle/working/solar')
root.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode('{payload}'))).extractall(root)
os.chdir(root)
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'timm==1.0.22', 'segmentation-models-pytorch==0.5.0', 'pycocotools==2.0.10'], check=True)
refined, = Path('/kaggle/input').rglob('refinement_complete.json')
full, = Path('/kaggle/input').rglob('refinement_full_check_complete.json')
policy, = Path('/kaggle/input').rglob('refinement_frozen_policies.json')
Path('baseline.json').write_text(json.dumps(json.loads(policy.read_text())['refined']))
sys.argv = ['screen_public_decoder.py', '--calibration', str(refined.parent/'runs/refine_v2/convnext_unetpp/fold0/oof'), '--evaluation', str(refined.parent/'runs/refine_v2/convnext_unetpp/fold1/oof'), '--upstream', str(root/'upstream'), '--baseline-policy', 'baseline.json', '--output', 'decoder_screen']
runpy.run_path(str(root/'scripts/screen_public_decoder.py'), run_name='__main__')
'''
(directory/'run.py').write_text(runner, encoding='utf8')
(directory/'kernel-metadata.json').write_text(json.dumps({
    'id': 'arjunvijayprakash/solar-filament-public-decoder-v6',
    'title': 'Solar filament public decoder v6', 'code_file': 'run.py',
    'language': 'python', 'kernel_type': 'script', 'is_private': True,
    'enable_gpu': False, 'enable_internet': True, 'dataset_sources': [],
    'competition_sources': [], 'kernel_sources': [
        'arjunvijayprakash/solar-filament-refine-v2',
        'arjunvijayprakash/solar-filament-refinement-comparison',
        'arjunvijayprakash/solar-filament-refinement-full-check']}, indent=2))
(directory/'source_provenance.json').write_text(json.dumps({
    'upstream_url': 'https://github.com/FarnooshMemari/solar-filament-segmentation',
    'upstream_commit': subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
    'license': 'MIT, notice included in payload',
    'files': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
    'payload_sha256': hashlib.sha256(buffer.getvalue()).hexdigest()}, indent=2))
print(directory)
