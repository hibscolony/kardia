from pathlib import Path
import zipfile,json,hashlib
root=Path.cwd()
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'public/models').glob('*.onnx')}
(root/'artifacts/model_checksums.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(root.parent/'KARDIA-source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in root.rglob('*'):
  if not p.is_file(): continue
  rel=p.relative_to(root)
  if rel.parts[0] in {'node_modules','dist','.git'} or p.name.endswith('.tsbuildinfo'): continue
  if rel.parts[0]=='artifacts' and rel.name not in {'reference_predictions.json','verification.json','model_checksums.json'}: continue
  z.write(p,Path('kardia')/rel)
with zipfile.ZipFile(root.parent/'KARDIA-github-pages-build.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in (root/'dist').rglob('*'):
  if p.is_file(): z.write(p,p.relative_to(root/'dist'))
for name in ['KARDIA-source.zip','KARDIA-github-pages-build.zip','KARDIA-preview.jpg']:
 p=root.parent/name;print(name,p.stat().st_size)
