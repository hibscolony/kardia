import json,base64
from pathlib import Path
n=json.loads(Path('E:/Downloads/notebook9963638c15.ipynb').read_text())
p=Path('public/data'); figures=[]
for i,c in enumerate(n['cells']):
 for j,o in enumerate(c.get('outputs',[])):
  if 'image/png' in o.get('data',{}):
   name=f'notebook-cell-{i}-{j}.png'; (p/name).write_bytes(base64.b64decode(o['data']['image/png'])); print(name, ''.join(c['source'])[:100]); figures.append(name)
(p/'figures-all.json').write_text(json.dumps(figures))
