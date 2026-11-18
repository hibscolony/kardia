# 01 â€” Setup
import os, sys, subprocess, random, json, time, itertools, shutil, warnings
from pathlib import Path
from collections import Counter

try:
    import wfdb
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'wfdb>=4.0,<6'])
    import wfdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from contextlib import nullcontext

SEEDS = [42, 123]          # Dua model/seed per skema, dirata-ratakan. Untuk cepat: [42]
FAST_RUN = False          # True: sanity check, BUKAN hasil LKTI
EPOCHS = 24 if not FAST_RUN else 2
PATIENCE = 7
BATCH_SIZE = 192
LEARNING_RATE = 7e-4
SAMPLER_POWER = 0.55     # 0 = sampling alami, 1 = sepenuhnya class-balanced
ALPHA_CONFORMAL = 0.10
FS = 360
LEFT, RIGHT = 112, 144  # 256 sampel
PAST_BEATS = 1           # 1 denyut sebelumnya + denyut sekarang = 4 kanal
LABELS_4 = ['N', 'S', 'V', 'F']
LABELS_3 = ['N', 'S', 'V']
OUTPUT = Path('/kaggle/working/kardia_v3_outputs') if Path('/kaggle/working').exists() else Path('/mnt/data/kardia_v3_outputs')
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE = OUTPUT.parent / 'kardia_v3_cache'
CACHE.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('PyTorch:', torch.__version__, '| device:', DEVICE, '| output:', OUTPUT)
print('FAST_RUN:', FAST_RUN, '| seeds:', SEEDS, '| epochs:', EPOCHS)
if DEVICE.type != 'cuda':
    print('PERINGATAN: GPU tidak aktif. Aktifkan Kaggle GPU T4 untuk training penuh.')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_all(SEEDS[0])

DS1 = [str(x) for x in [101,106,108,109,112,114,115,116,118,119,122,124,
                        201,203,205,207,208,209,215,220,223,230]]
DS2_CLASSIC = [str(x) for x in [100,103,105,111,113,117,121,123,200,202,210,
                                212,213,214,219,221,222,228,231,232,233,234]]
TEST_IDS = [r for r in DS2_CLASSIC if r != '202']
VAL_IDS = ['223', '207', '205']
CAL_IDS = ['124', '118', '106']
TRAIN_IDS = sorted(set(DS1) - set(VAL_IDS) - set(CAL_IDS))
ALL_IDS = sorted(set(DS1 + TEST_IDS))
for a, b in itertools.combinations([TRAIN_IDS, VAL_IDS, CAL_IDS, TEST_IDS], 2):
    assert not set(a).intersection(b), 'KEBOCORAN REKAMAN TERDETEKSI!'
assert '208' in TRAIN_IDS and '202' not in TEST_IDS
print('Split rekaman:', {'train':TRAIN_IDS, 'val':VAL_IDS, 'cal':CAL_IDS, 'test':TEST_IDS})
with open(OUTPUT/'split_records.json', 'w') as f:
    json.dump({'train':TRAIN_IDS,'validation':VAL_IDS,'calibration':CAL_IDS,
               'test':TEST_IDS,'excluded_same_patient':['202'],
               'split_note':'Record 208 dipertahankan pada training; test tidak dipakai tuning.'},f,indent=2)

RAW_DIR = OUTPUT.parent / 'mitdb_wfdb'

def find_wfdb_dir():
    roots = [RAW_DIR, Path('/kaggle/input'), Path('/mnt/data')]
    for root in roots:
        if not root.exists(): continue
        candidates = [root/'100.hea']
        if root.is_dir():
            candidates += list(root.rglob('100.hea'))
        for f in candidates:
            base = f.parent
            if all((base/(f'100.{ext}')).exists() for ext in ['hea','dat','atr']):
                return base
    return None

DATA_DIR = find_wfdb_dir()
if DATA_DIR is None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print('Download MIT-BIH dari PhysioNet...')
    try:
        wfdb.dl_database('mitdb', str(RAW_DIR), records=ALL_IDS,
                         annotators=['atr'], keep_subdirs=False, overwrite=False)
    except Exception as exc:
        raise RuntimeError(
            'Unduhan MIT-BIH gagal. Aktifkan Kaggle Internet atau Add Input dataset WFDB '
            '(.hea/.dat/.atr). Error asli: ' + repr(exc)
        ) from exc
    DATA_DIR = find_wfdb_dir()
if DATA_DIR is None: raise FileNotFoundError('Dataset MIT-BIH tidak ditemukan.')
missing_files = [f'{rid}.{ext}' for rid in ALL_IDS for ext in ['hea','dat','atr']
                 if not (DATA_DIR/f'{rid}.{ext}').exists()]
if missing_files:
    raise FileNotFoundError('Dataset belum lengkap, contoh: ' + str(missing_files[:8]))
print('Dataset:', DATA_DIR, '| rekaman:', len(ALL_IDS))

AAMI = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V','F':'F'
}
EXTRA_BEATS = {'/', 'f', 'Q'}  # titik denyut untuk menghitung RR (tanpa memakai label target)
SOS = butter(3, [0.5,40], btype='bandpass', fs=FS, output='sos')
REC_DATA = {}


def extract_record(rid):
    cfile = CACHE/f'v3_causal_{rid}.npz'
    if cfile.exists():
        with np.load(cfile, allow_pickle=False) as obj:
            return obj['X'], obj['RR'], obj['y']

    record = wfdb.rdrecord(str(DATA_DIR/rid))
    ann = wfdb.rdann(str(DATA_DIR/rid), 'atr')
    if int(record.fs) != FS: raise ValueError(f'Frekuensi rekaman {rid} bukan 360 Hz')
    if record.p_signal.shape[1] < 2:
        raise ValueError(f'Rekaman {rid} tidak memiliki 2 lead ECG')
    lead1 = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
    lead2 = next(i for i in range(record.p_signal.shape[1]) if i != lead1)
    signal = record.p_signal[:, [lead1, lead2]].astype(np.float32)
    # Hilangkan NaN sebelum filtering; celah panjang tetap menjadi keterbatasan data
    for ch in range(signal.shape[1]):
        valid = np.isfinite(signal[:,ch])
        if not valid.all():
            signal[:,ch] = np.interp(
                np.arange(len(signal)), np.where(valid)[0], signal[valid,ch]
            )
    filtered = sosfiltfilt(SOS, signal, axis=0).astype(np.float32)
    n_time = len(signal)

    beat_idxs = [i for i, sym in enumerate(ann.symbol)
                 if sym in AAMI or sym in EXTRA_BEATS]
    peaks = np.asarray([ann.sample[i] for i in beat_idxs], dtype=np.int64)
    target = []
    waveforms = []
    rr_rows = []

    def segment(p):
        window = filtered[int(p)-LEFT:int(p)+RIGHT,:].T.copy()
        # Skala robust per lead; train tidak memakai statistik test / rekaman lainnya
        baseline = np.median(window, axis=1, keepdims=True)
        window -= baseline
        scale = np.percentile(np.abs(window), 95, axis=1, keepdims=True)
        window /= np.maximum(scale, 0.05)
        return np.clip(window, -5, 5).astype(np.float32)

    for j, original_idx in enumerate(beat_idxs):
        sym = ann.symbol[original_idx]
        if sym not in AAMI: continue
        p = int(peaks[j])
        if p-LEFT < 0 or p+RIGHT > n_time: continue
        previous_p = int(peaks[j-1]) if j >= 1 else p
        if previous_p-LEFT < 0 or previous_p+RIGHT > n_time:
            previous_p = p

        current = segment(p)
        previous = segment(previous_p)
        waveforms.append(np.concatenate([current, previous], axis=0))
        target.append(LABELS_4.index(AAMI[sym]))

        hist = np.diff(peaks[max(0, j-8):j+1]).astype(np.float64)/FS
        prev_rr = float(hist[-1]) if len(hist) else 0.8
        rr2 = float(hist[-2]) if len(hist) >= 2 else prev_rr
        median_rr = float(np.median(hist)) if len(hist) else prev_rr
        mean_rr = float(np.mean(hist)) if len(hist) else prev_rr
        std_rr = float(np.std(hist)) if len(hist) >= 2 else 0.0
        rr_rows.append([
            prev_rr, rr2, prev_rr/max(rr2,1e-3),
            prev_rr/max(median_rr,1e-3), median_rr, mean_rr,
            std_rr/max(mean_rr,1e-3), prev_rr-rr2
        ])

    X = np.asarray(waveforms, dtype=np.float32)
    RR = np.asarray(rr_rows, dtype=np.float32)
    y = np.asarray(target, dtype=np.int64)
    assert X.ndim==3 and X.shape[1:] == (4, LEFT+RIGHT)
    assert len(X)==len(RR)==len(y)
    np.savez_compressed(cfile, X=X, RR=RR, y=y)
    return X, RR, y

for i, rid in enumerate(ALL_IDS):
    REC_DATA[rid] = extract_record(rid)
    if (i+1)%10==0 or (i+1)==len(ALL_IDS):
        print(f'Loaded {i+1}/{len(ALL_IDS)} rekaman')

rows = []
for rid in ALL_IDS:
    y = REC_DATA[rid][2]
    cnt = np.bincount(y,minlength=4)
    rows.append({'record':rid,'split':('train' if rid in TRAIN_IDS else
                 'val' if rid in VAL_IDS else 'cal' if rid in CAL_IDS else 'test'),
                 **dict(zip(LABELS_4, cnt.astype(int)))})
per_record = pd.DataFrame(rows)
per_record.to_csv(OUTPUT/'distribution_per_record.csv',index=False)
print(per_record.groupby('split')[LABELS_4].sum().to_string())
assert per_record.loc[per_record.record=='208','F'].iloc[0] > 300, 'Cek rekaman 208'

def combine(ids, allowed):
    xlist, rlist, ylist, rec_list = [], [], [], []
    for rid in ids:
        x, rr, y4 = REC_DATA[rid]
        idx = np.isin(y4, allowed)
        xlist.append(x[idx]); rlist.append(rr[idx]); ylist.append(y4[idx])
        rec_list.extend([rid]*int(idx.sum()))
    return (np.concatenate(xlist), np.concatenate(rlist),
            np.concatenate(ylist), np.asarray(rec_list))


def make_scheme(n_classes):
    allowed = list(range(n_classes))
    data = {}
    for split, ids in [('train', TRAIN_IDS), ('val', VAL_IDS),
                       ('cal', CAL_IDS), ('test', TEST_IDS)]:
        x,rr,y,rec=combine(ids, allowed)
        if FAST_RUN:
            # Per rekaman dipilih sebagian, menjaga sampel F/S pada training sanity check
            rng = np.random.default_rng(2026+n_classes)
            keep=[]
            for rid in np.unique(rec):
                inds=np.flatnonzero(rec==rid)
                if len(inds)>180:
                    inds=np.sort(rng.choice(inds, size=180, replace=False))
                keep.append(inds)
            keep=np.sort(np.concatenate(keep))
            x,rr,y,rec=x[keep],rr[keep],y[keep],rec[keep]
        data[split] = {'X':x, 'RR':rr, 'y':y, 'record':rec}

    scaler = StandardScaler()
    scaler.fit(data['train']['RR'])
    for s in data:
        data[s]['RR'] = np.clip(
            scaler.transform(data[s]['RR']), -8, 8
        ).astype(np.float32)
    counts = np.bincount(data['train']['y'],minlength=n_classes)
    assert (counts>0).all(), 'Ada kelas tanpa sampel training'
    print(f'\nSKEMA {n_classes} kelas:')
    for split, a in data.items():
        print(f'{split:>5}: {len(a["y"]):>6} beats',
              dict(zip(LABELS_4[:n_classes],np.bincount(a['y'],minlength=n_classes))))
    return data, scaler

class MultiScaleBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        assert cout % 3 == 0
        branch_out = cout//3
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(cin,branch_out,k,stride=stride,padding=k//2,bias=False),
                          nn.GroupNorm(8 if branch_out%8==0 else 4,branch_out),nn.GELU())
            for k in [5,11,21]
        ])
        self.skip = nn.Sequential(nn.Conv1d(cin,cout,1,stride=stride,bias=False),
                                  nn.GroupNorm(8,cout))
        self.act = nn.GELU()
    def forward(self,x):
        return self.act(torch.cat([b(x) for b in self.branches],dim=1)+self.skip(x))

class KardiaDualNet(nn.Module):
    def __init__(self, n_classes, rr_dim=8, d=144):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(4,48,9,stride=2,padding=4,bias=False),
                                nn.GroupNorm(8,48),nn.GELU())
        self.cnn=nn.Sequential(
            MultiScaleBlock(48,96,2),
            MultiScaleBlock(96,d,2),
            MultiScaleBlock(d,d,1)
        )
        self.temporal=nn.MultiheadAttention(d,4,dropout=0.1,batch_first=True)
        self.temporal_norm=nn.LayerNorm(d)
        self.rr_token=nn.Linear(1,d)
        self.rr_position=nn.Parameter(torch.randn(1,rr_dim,d)*0.015)
        self.cross=nn.MultiheadAttention(d,4,dropout=0.1,batch_first=True)
        self.cross_norm=nn.LayerNorm(d)
        self.pool=nn.Linear(d,1)
        self.rr_global=nn.Sequential(nn.Linear(rr_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.head=nn.Sequential(nn.Linear(d*3,192),nn.LayerNorm(192),nn.GELU(),
                                nn.Dropout(0.3),nn.Linear(192,n_classes))

    def forward(self, ecg, rr):
        z=self.cnn(self.stem(ecg)).transpose(1,2)
        att,_=self.temporal(z,z,z,need_weights=False)
        z=self.temporal_norm(z+att)
        tokens=self.rr_token(rr.unsqueeze(-1))+self.rr_position
        cross,_=self.cross(query=z,key=tokens,value=tokens,need_weights=False)
        z=self.cross_norm(z+cross)
        weight=torch.softmax(self.pool(z),dim=1)
        pooled=torch.sum(z*weight,dim=1)
        maxpool=z.amax(dim=1)
        return self.head(torch.cat([pooled,maxpool,self.rr_global(rr)],dim=1))

with torch.no_grad():
    check = KardiaDualNet(4).to(DEVICE)(torch.randn(2,4,256,device=DEVICE),
                                       torch.randn(2,8,device=DEVICE))
    assert check.shape==(2,4)
print('Forward pass berhasil. Parameter:',sum(p.numel() for p in KardiaDualNet(4).parameters()))

def make_loader(split, train, seed=42):
    X=torch.from_numpy(np.ascontiguousarray(split['X']))
    RR=torch.from_numpy(np.ascontiguousarray(split['RR']))
    y=torch.from_numpy(np.ascontiguousarray(split['y'])).long()
    ds=TensorDataset(X,RR,y)
    if not train:
        return DataLoader(ds,batch_size=BATCH_SIZE*2,shuffle=False,num_workers=0,
                          pin_memory=(DEVICE.type=='cuda'))
    counts=np.bincount(split['y'], minlength=int(y.max())+1)
    w=(counts[split['y']].astype(float)+1.0)**(-SAMPLER_POWER)
    sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),
                                  num_samples=len(y),replacement=True,
                                  generator=torch.Generator().manual_seed(seed))
    return DataLoader(ds,batch_size=BATCH_SIZE,sampler=sampler,num_workers=0,
                      pin_memory=(DEVICE.type=='cuda'))


def augment_batch(x):
    # Hanya untuk train; tidak diterapkan pada val/cal/test
    x=x*torch.empty(x.shape[0],x.shape[1],1,device=x.device).uniform_(0.90,1.10)
    x=x+torch.randn_like(x)*torch.empty(x.shape[0],1,1,device=x.device).uniform_(0.0,0.045)
    # Partial dropout pada konteks denyut lalu (bukan current beat)
    if x.shape[1]>=4:
        mask=(torch.rand(x.shape[0],1,1,device=x.device)<0.045)
        x[:,2:,:]=torch.where(mask,torch.zeros_like(x[:,2:,:]),x[:,2:,:])
    return x

@torch.no_grad()
def collect_logits(model, loader):
    model.eval(); logits=[]
    for x,rr,y in loader:
        x=x.to(DEVICE,non_blocking=True); rr=rr.to(DEVICE,non_blocking=True)
        logits.append(model(x,rr).float().cpu().numpy())
    return np.concatenate(logits,axis=0)


def train_seed(n_classes, data, seed, folder):
    seed_all(seed)
    model=KardiaDualNet(n_classes).to(DEVICE)
    tr_loader=make_loader(data['train'],True,seed)
    va_loader=make_loader(data['val'],False)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=2e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=(DEVICE.type=='cuda'))
    best=-1.0; best_epoch=0; wait=0; history=[]
    file=folder/f'model_seed{seed}.pt'
    t0=time.time()

    for epoch in range(1,EPOCHS+1):
        model.train(); total_loss=0.0; count=0
        for xb,rb,yb in tr_loader:
            xb=augment_batch(xb.to(DEVICE,non_blocking=True))
            rb=rb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ctx=torch.autocast(device_type='cuda',dtype=torch.float16) if DEVICE.type=='cuda' else nullcontext()
            with ctx:
                logits=model(xb,rb)
                loss=F.cross_entropy(logits,yb,label_smoothing=0.025)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
            total_loss+=float(loss.item())*len(yb); count+=len(yb)

        schedule.step()
        val_logits=collect_logits(model,va_loader)
        pred=val_logits.argmax(1)
        val_f1=f1_score(data['val']['y'],pred,labels=list(range(n_classes)),
                        average='macro',zero_division=0)
        history.append({'seed':seed,'epoch':epoch,'train_loss':total_loss/count,
                        'val_macro_f1':float(val_f1), 'lr':opt.param_groups[0]['lr']})
        print(f'[{n_classes} kelas, seed={seed}] epoch {epoch:02d} | '
              f'loss {total_loss/count:.4f} | val macro-F1 {val_f1:.4f} | '
              f'best {max(best,val_f1):.4f}',flush=True)

        if val_f1>best+0.00005:
            best=val_f1;best_epoch=epoch;wait=0
            torch.save({'model':model.state_dict(), 'epoch':epoch,
                        'val_macro_f1':float(val_f1), 'seed':seed,
                        'n_classes':n_classes}, file)
        else: wait+=1
        if wait>=PATIENCE:
            print('Early stopping:',epoch)
            break

    ckpt=torch.load(file,map_location=DEVICE,weights_only=True)
    model.load_state_dict(ckpt['model'])
    print(f'Best seed {seed}: val macro-F1={best:.4f}, epoch={best_epoch}, '
          f'time={((time.time()-t0)/60):.1f} min')
    return model, pd.DataFrame(history), float(best)

def softmax_np(z):
    z=z-z.max(axis=1,keepdims=True)
    ex=np.exp(z)
    return ex/ex.sum(axis=1,keepdims=True)


def choose_fusion(val_logits, y_val, n_classes):
    probs=[softmax_np(logit) for logit in val_logits]
    candidates=[(f'seed_{i}',p) for i,p in enumerate(probs)]
    if len(probs)>1: candidates.append(('ensemble',np.mean(probs,axis=0)))
    best=None
    for name,p in candidates:
        f1=f1_score(y_val,p.argmax(1),labels=list(range(n_classes)),average='macro',zero_division=0)
        if best is None or f1>best[2]+1e-5:
            best=(name,p,float(f1))
    return best[0],best[2]


def tune_logits(val_probs, y_val, n_classes):
    # Koreksi log-p (setara prior adjustment) dilakukan validation only
    grids = [[-1.2,-0.6,0.0,0.6,1.2],[-0.5,0.0,0.5]]
    if n_classes==4: grids.append([-1.2,-0.6,0.0,0.6,1.2])
    base=np.log(np.clip(val_probs,1e-8,1.0))
    best=(-1e9,None,None)
    for rest in itertools.product(*grids):
        bias=np.array([0.0,*rest],dtype=np.float64)
        pred=np.argmax(base+bias[None,:],axis=1)
        macro=f1_score(y_val,pred,labels=list(range(n_classes)),average='macro',zero_division=0)
        score=float(macro-0.003*np.mean(np.abs(bias)))
        if score>best[0]:best=(score,bias,float(macro))
    return best[1],best[2]


def apply_bias(probs,bias):
    return softmax_np(np.log(np.clip(probs,1e-8,1.0))+bias[None,:])

def conformal_evaluation(p_cal,y_cal,p_test,y_test,alpha=0.1):
    score=1-p_cal[np.arange(len(y_cal)),y_cal]
    rank=min(int(np.ceil((len(score)+1)*(1-alpha))),len(score))
    q=float(np.partition(score,rank-1)[rank-1])
    s=(1-p_test<=q)
    return {'alpha':alpha,'qhat':q,'coverage':float(s[np.arange(len(y_test)),y_test].mean()),
            'avg_set_size':float(s.sum(axis=1).mean()),
            'singleton_rate':float((s.sum(axis=1)==1).mean())}


def save_report(y_true,probs,labels,folder):
    pred=probs.argmax(axis=1)
    report=classification_report(y_true,pred,labels=list(range(len(labels))),
                                 target_names=labels,digits=4,zero_division=0,output_dict=True)
    pd.DataFrame(report).T.to_csv(folder/'classification_report.csv')
    cm=confusion_matrix(y_true,pred,labels=list(range(len(labels))))
    fig,ax=plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,cmap='Blues',values_format='d',colorbar=False)
    ax.set_title(f'KARDIA V3 â€” {len(labels)} kelas')
    fig.tight_layout();fig.savefig(folder/'confusion_matrix.png',dpi=180);plt.close(fig)
    return {'accuracy':float(accuracy_score(y_true,pred)),
            'macro_f1':float(f1_score(y_true,pred,average='macro',labels=list(range(len(labels))),zero_division=0)),
            'weighted_f1':float(f1_score(y_true,pred,average='weighted',zero_division=0)),
            'per_class_f1':{lbl:float(report[lbl]['f1-score']) for lbl in labels},
            'report':report},pred


def run_scheme(n_classes):
    folder=OUTPUT/f'{n_classes}class'
    folder.mkdir(exist_ok=True)
    labels=LABELS_4[:n_classes]
    data,rr_scaler=make_scheme(n_classes)
    np.savez(folder/'rr_scaler.npz',mean=rr_scaler.mean_,scale=rr_scaler.scale_)
    pd.DataFrame({s:np.bincount(data[s]['y'],minlength=n_classes) for s in data},
                 index=labels).to_csv(folder/'class_distribution.csv')

    models=[]; histories=[]; single_val_scores=[]
    va_loader=make_loader(data['val'],False)
    ca_loader=make_loader(data['cal'],False)
    te_loader=make_loader(data['test'],False)
    for seed in SEEDS:
        model,hist,val_f1=train_seed(n_classes,data,seed,folder)
        histories.append(hist);single_val_scores.append(val_f1)
        models.append(model)
    pd.concat(histories,ignore_index=True).to_csv(folder/'training_history.csv',index=False)
    val_logits=[collect_logits(m,va_loader) for m in models]
    selected,best_val=choose_fusion(val_logits,data['val']['y'],n_classes)

    def selected_probs(loader):
        if selected=='ensemble':
            return np.mean([softmax_np(collect_logits(m,loader)) for m in models],axis=0)
        idx=int(selected.replace('seed_',''))
        return softmax_np(collect_logits(models[idx],loader))

    if selected=='ensemble':val_probs=np.mean([softmax_np(z) for z in val_logits],axis=0)
    else:val_probs=softmax_np(val_logits[int(selected.replace('seed_',''))])
    bias,val_tuned_f1=tune_logits(val_probs,data['val']['y'],n_classes)
    print(f'Pilihan validasi {n_classes} kelas: {selected} | '
          f'val F1 raw={best_val:.4f} | tuned={val_tuned_f1:.4f} | bias={bias}')

    cal_probs=apply_bias(selected_probs(ca_loader),bias)
    test_probs=apply_bias(selected_probs(te_loader),bias)
    metrics,pred=save_report(data['test']['y'],test_probs,labels,folder)
    cp=conformal_evaluation(cal_probs,data['cal']['y'],test_probs,data['test']['y'],
                            ALPHA_CONFORMAL)
    print('\n=========== HASIL FINAL',n_classes,'KELAS ===========')
    print('Accuracy',round(metrics['accuracy'],4),'Macro F1',round(metrics['macro_f1'],4),
          'Weighted F1',round(metrics['weighted_f1'],4))
    print(classification_report(data['test']['y'],pred,labels=list(range(n_classes)),
                                target_names=labels,digits=4,zero_division=0))
    print('Conformal (marginal, not clinical guarantee):',cp)

    preds=pd.DataFrame({'record':data['test']['record'],
                        'true_label':[labels[i] for i in data['test']['y']],
                        'pred_label':[labels[i] for i in pred],
                        'correct':(pred==data['test']['y']).astype(int)})
    for i,lbl in enumerate(labels):preds[f'p_{lbl}']=test_probs[:,i]
    preds.to_csv(folder/'test_predictions.csv',index=False)

    out_metrics={k:v for k,v in metrics.items() if k!='report'}
    out_metrics.update({'n_classes':n_classes,'labels':labels,
                        'n_test_beats':int(len(pred)),
                        'selected_on_val':selected,
                        'val_macro_f1_before_tuning':best_val,
                        'val_macro_f1_after_tuning':val_tuned_f1,
                        'logit_bias':bias.tolist(),
                        'seed_val_f1':dict(zip(map(str,SEEDS),single_val_scores)),
                        'conformal':cp,'test_protocol':'DS2 without record 202; held out',
                        'rpeak_source':'MIT-BIH expert annotations',
                        'fast_run':bool(FAST_RUN)})
    with open(folder/'metrics.json','w') as f:json.dump(out_metrics,f,indent=2)
    del models
    if DEVICE.type=='cuda':torch.cuda.empty_cache()
    return out_metrics, data, (rr_scaler,selected,bias)

results={}
results[4],data4,_ = run_scheme(4)
results[3],data3,_ = run_scheme(3)

comparison=pd.DataFrame([
    {'scheme':'A - 4 classes N/S/V/F',
     'classes':'N,S,V,F','n_test':results[4]['n_test_beats'],
     'accuracy':results[4]['accuracy'],'macro_f1':results[4]['macro_f1'],
     'weighted_f1':results[4]['weighted_f1'],
     'macro_f1_ge_070':results[4]['macro_f1']>=0.70},
    {'scheme':'B - 3 classes N/S/V (F excluded)',
     'classes':'N,S,V','n_test':results[3]['n_test_beats'],
     'accuracy':results[3]['accuracy'],'macro_f1':results[3]['macro_f1'],
     'weighted_f1':results[3]['weighted_f1'],
     'macro_f1_ge_070':results[3]['macro_f1']>=0.70}
])
comparison.to_csv(OUTPUT/'comparison_4class_vs_3class.csv',index=False)
print('\n===== KARDIA V3: DUA EKSPERIMEN =====')
print(comparison.to_string(index=False))
for cls_count in (4,3):
    goal='TERCAPAI' if results[cls_count]['macro_f1']>=0.70 else 'BELUM TERCAPAI'
    print(f'Skema {cls_count} kelas â€” target Macro F1 >= 0.70: {goal}')

fig,ax=plt.subplots(figsize=(8,4))
ax.bar(['4 kelas','3 kelas'],comparison['macro_f1'].to_numpy())
ax.axhline(0.70,linestyle='--',linewidth=1.5,label='Target riset 0.70')
ax.set_ylim(0,1);ax.set_ylabel('Macro F1');ax.set_title('KARDIA V3: Target pada skema berbeda')
ax.legend();fig.tight_layout();fig.savefig(OUTPUT/'macro_f1_comparison.png',dpi=180);plt.show()

archive_base=str(OUTPUT.parent/'KARDIA_V3_Dual_Scheme_Outputs')
zip_path=shutil.make_archive(archive_base,'zip',root_dir=OUTPUT.parent,base_dir=OUTPUT.name)
print('\nâœ… Semua eksperimen selesai.')
print('Hasil:',OUTPUT)
print('ZIP:',zip_path,'| size MB:',round(Path(zip_path).stat().st_size/1024**2,2))
try:
    from IPython.display import FileLink,display
    display(FileLink(zip_path))
except Exception:
    pass


# ==================================================
# FIX ONNX EXPORT â€” KARDIA
# UnsupportedOperatorError: native_multi_head_attention
# ==================================================

import torch

# Matikan jalur native MultiheadAttention saat ekspor
torch.backends.mha.set_fastpath_enabled(False)

print("âœ… MultiheadAttention fastpath disabled!")
print("âœ… Model siap diekspor ke ONNX")
print("PyTorch version:", torch.__version__)


# ================================================================
# KARDIA â€” EKSPOR CHECKPOINT KAGGLE KE MODEL ONNX UNTUK GITHUB PAGES
# Letakkan setelah cell yang mendefinisikan KardiaDualNet.
# Memerlukan folder /kaggle/working/kardia_v3_outputs dari training.
# ================================================================
import sys, subprocess, json, shutil
from pathlib import Path
import numpy as np
import torch

# Instal paket pendukung bila belum tersedia (Internet Kaggle ON).
for package in ('onnx', 'onnxruntime'):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

import onnx
import onnxruntime as ort

assert 'KardiaDualNet' in globals(), (
    'Kelas KardiaDualNet belum tersedia: jalankan cell arsitektur dari notebook KARDIA.'
)

SOURCE = Path('/kaggle/working/kardia_v3_outputs')
if not SOURCE.exists():
    SOURCE = Path('/mnt/data/kardia_v3_outputs')
if not SOURCE.exists():
    raise FileNotFoundError('Folder kardia_v3_outputs tidak ditemukan. Jalankan eksperimen dulu.')

DEST = SOURCE.parent / 'kardia_web_onnx'
DEST.mkdir(exist_ok=True)

# Hasil validasi yang disimpan dalam metrics.json, bukan hasil yang dikarang.
all_metadata = {
    'project': 'KARDIA',
    'model_family': 'MultiScale CNN + Temporal Self-Attention + RR Cross-Attention',
    'model_inputs': {'ecg': ['batch', 4, 256], 'rr': ['batch', 8]},
    'input_notes': (
        'ecg: 2 lead x (denyut kini, denyut sebelumnya), urutan kanal '
        '[current_lead1, current_lead2, previous_lead1, previous_lead2]. '
        'RR: 8 fitur temporal yang distandardisasi memakai scaler per skema.'
    ),
    'sampling_rate_hz': 360,
    'beat_window_samples': {'left': 112, 'right': 144},
    'source_preprocessing': (
        'Bandpass Butterworth order 3 0.5-40 Hz (zero-phase sosfiltfilt), '
        'median subtraction, normalization by 95th percentile abs with minimum 0.05, '
        'clamp to [-5,5], RR feature normalization by training scaler and clamp [-8,8]. '
        'R-peaks in original evaluation are from MIT-BIH expert annotations.'
    ),
    'warning': 'Research prototype only. Never claim general raw-upload clinical diagnosis.',
    'schemes': {}
}

for n_classes in (4, 3):
    folder = SOURCE / f'{n_classes}class'
    with open(folder / 'metrics.json', encoding='utf-8') as f:
        metrics = json.load(f)
    labels = metrics['labels']
    selected = metrics['selected_on_val']
    all_metadata['schemes'][str(n_classes)] = {
        'labels': labels,
        'selected_on_validation': selected,
        'logit_bias': metrics['logit_bias'],
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'n_test_beats': metrics['n_test_beats'],
        'inference_rule': (
            'Compute softmax(logits) for each seed; average probabilities for ensemble; '
            'then compute softmax(log(max(mean_probs,1e-8)) + logit_bias).'
        )
    }
    shutil.copyfile(folder / 'rr_scaler.npz', DEST / f'rr_scaler_{n_classes}class.npz')
    for seed in (42, 123):
        ckpt_path = folder / f'model_seed{seed}.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model = KardiaDualNet(n_classes=n_classes).cpu().eval()
        model.load_state_dict(checkpoint['model'], strict=True)

        torch.manual_seed(100 + n_classes + seed)
        ecg = torch.randn(2, 4, 256, dtype=torch.float32)
        rr = torch.randn(2, 8, dtype=torch.float32)
        target = DEST / f'kardia_{n_classes}class_seed{seed}.onnx'

        with torch.no_grad():
            pytorch_logits = model(ecg, rr).detach().numpy()
            torch.onnx.export(
                model, (ecg, rr), str(target),
                input_names=['ecg', 'rr'], output_names=['logits'],
                opset_version=17, export_params=True,
                do_constant_folding=True, dynamo=False,
                dynamic_axes={
                    'ecg': {0: 'batch'}, 'rr': {0: 'batch'},
                    'logits': {0: 'batch'}
                }
            )
        onnx.checker.check_model(onnx.load(str(target)))
        session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
        onnx_logits = session.run(None, {'ecg': ecg.numpy(), 'rr': rr.numpy()})[0]
        max_abs_error = float(np.max(np.abs(onnx_logits - pytorch_logits)))
        print(f'{target.name}: PyTorch vs ONNX | max_abs_error={max_abs_error:.7g}')
        if not np.allclose(onnx_logits, pytorch_logits, atol=2e-3, rtol=2e-3):
            raise AssertionError(f'Parity failed for {target.name}: {max_abs_error}')

with open(DEST / 'model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(all_metadata, f, indent=2, ensure_ascii=False)

zip_path = shutil.make_archive(str(DEST), 'zip', root_dir=DEST.parent, base_dir=DEST.name)
print('\nâœ… Ekspor dan parity test selesai!')
print('Model ZIP:', zip_path)
print('Lampirkan ZIP ini ke ASTRA bersama hasil eksperimen dan referensi desain.')
try:
    from IPython.display import display, FileLink
    display(FileLink(zip_path))
except ImportError:
    pass


