import zipfile,json,csv,io
from pathlib import Path
p=Path('public/data'); figs=[]
with zipfile.ZipFile('E:/Downloads/KARDIA_V3_Dual_Scheme_Outputs.zip') as z:
 for n in [4,3]:
  prefix=f'kardia_v3_outputs/{n}class/'
  name=f'confusion-{n}class.png'; (p/name).write_bytes(z.read(prefix+'confusion_matrix.png'));figs.append(name)
  (p/f'metrics-{n}class.json').write_bytes(z.read(prefix+'metrics.json'))
  print(n,json.loads(z.read(prefix+'metrics.json'))['accuracy'])
 (p/'split_records.json').write_bytes(z.read('kardia_v3_outputs/split_records.json'))
(p/'figures.json').write_text(json.dumps(figs))
