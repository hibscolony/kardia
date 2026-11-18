from pathlib import Path
import ast, json, base64, hashlib
import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt
ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=ROOT/'artifacts'/'mitdb'; DATA_DIR.mkdir(exist_ok=True)
CACHE=ROOT/'artifacts'/'cache'; CACHE.mkdir(exist_ok=True)
FS=360; LEFT=112; RIGHT=144
LABELS_4=['N','S','V','F']
AAMI={'N':'N','L':'N','R':'N','e':'N','j':'N','A':'S','a':'S','J':'S','S':'S','V':'V','E':'V','F':'F'}
EXTRA_BEATS={'/','f','Q'}
SOS=butter(3,[0.5,40],btype='bandpass',fs=FS,output='sos')
source=(ROOT/'research_notebook.py').read_text(encoding='utf-8')
tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_record')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original notebook extract_record>','exec'))
if not (DATA_DIR/'100.dat').exists(): wfdb.dl_database('mitdb',str(DATA_DIR),records=['100'])
x,rr,y=extract_record('100')
indices=list(range(10,18))
for c in [1,2]:
    ids=np.where(y==c)[0]
    indices.extend(ids[:3].tolist())
indices=sorted(set(indices))
r=wfdb.rdrecord(str(DATA_DIR/'100'))
obj={'version':'kardia-tensors-v1','source':'MIT-BIH record 100 · PhysioNet','preprocessing':'kardia-v3-notebook','rrFormat':'raw','samplingRate':360,'ecg':x[indices].tolist(),'rr':rr[indices].tolist(),'annotations':[LABELS_4[i] for i in y[indices]],'beatIds':[int(i) for i in indices]}
(ROOT/'public/data/demo.json').write_text(json.dumps(obj,separators=(',',':')))
(ROOT/'public/data/waveform.json').write_text(json.dumps({'source':'MIT-BIH record 100','samplingRate':360,'lead':'MLII','startSample':3600,'samples':r.p_signal[3600:6480,0].tolist()},separators=(',',':')))
scalers={}
for n in [3,4]:
    p=ROOT/f'public/models/rr_scaler_{n}class.npz'
    a=np.load(p); scalers[str(n)]={k:a[k].tolist() for k in a.files}
(ROOT/'public/models/scalers.json').write_text(json.dumps(scalers))
# Reference predictions are verification fixtures, never loaded by the application.
import onnxruntime as ort
from scipy.special import softmax
meta=json.loads((ROOT/'public/models/model_metadata.json').read_text())
reference={}
for n in [3,4]:
    s=scalers[str(n)]
    # StandardScaler float32 transform uses two in-place operations.
    z=rr[indices].copy(); z-=np.array(s['mean']); z/=np.array(s['scale']); z=np.clip(z,-8,8).astype(np.float32)
    logits=[]
    for seed in [42,123]:
        session=ort.InferenceSession(str(ROOT/f'public/models/kardia_{n}class_seed{seed}.onnx'),providers=['CPUExecutionProvider'])
        # Export attention graph may fix the batch size: verify batch=1.
        logits.append(np.concatenate([session.run(None,{'ecg':a[None],'rr':b[None]})[0] for a,b in zip(x[indices],z)]))
    probs=softmax(np.log(np.maximum(np.mean([softmax(l,axis=-1) for l in logits],axis=0),1e-8))+meta['schemes'][str(n)]['logit_bias'],axis=-1)
    reference[str(n)]=probs.tolist()
(ROOT/'artifacts/reference_predictions.json').write_text(json.dumps(reference))
print('Prepared',len(indices),'real demo beats. Reference inference succeeded for all four models.')
