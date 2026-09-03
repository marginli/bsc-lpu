#!/usr/bin/env python3
"""指令 14：對可以跑的腦區逐一跑指令 4–7，產出總表。"""
import numpy as np, json, csv, collections, time
from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; D06="/home/wanjuli/claude_linux/BSC_plan/D06"
OUT="part4/out"; GX=GY=512; B=4; NP=9; CUTS=np.arange(4,26,1.0)
lab=np.load(f"{D03}/work/labels_fc_1um.npy")
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]; n2i={n:i for i,n in enumerate(names)}
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); n2x={r["name"]:i for i,r in enumerate(rows)}
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")
LN=collections.defaultdict(list)
for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")): LN[r["region"]].append(n2x[r["neuron"]])
est=json.load(open(f"{OUT}/c14_estimate.json"))
rng=np.random.default_rng(0); res=[]; t0=time.time()
for nm in names[1:]:
    n_ln=len(LN.get(nm,[]))
    if n_ln<10:
        res.append(dict(region=nm,n_ln=n_ln,status="LN 太少（<10）",verdict="候選 hub")); continue
    zz,yy,xx=np.nonzero(lab==n2i[nm]); sh=tuple(s//B+1 for s in lab.shape)
    m=np.zeros(sh,bool); m[zz//B,yy//B,xx//B]=True
    sel=np.isin(nid,LN[nm]); v=vox[sel]; n=nid[sel]
    iz=(v//(GX*GY))*2//B; iy=((v//GX)%GY)*2//B; ix=(v%GX)*2//B
    k=np.unique(np.stack([iz,iy,ix,n]),axis=1)
    f=np.zeros(sh,np.float32); np.add.at(f,(k[0],k[1],k[2]),1.0)
    sm=ndimage.uniform_filter(f,3)
    P=np.argwhere(m&(sm>=np.percentile(sm[m],75))).astype(np.float32)
    thr=0.01*m.sum(); runs=[]
    for t in range(NP):
        Q=P if t==0 else P[rng.permutation(len(P))]
        Z=linkage(pdist(Q),"average")
        runs.append([int((np.bincount(fcluster(Z,c,"distance"))[1:]>thr).sum()) for c in CUTS])
        if t==0: root=float(Z[:,2].max())
    A=np.array(runs); pl=A[:,(CUTS>=14)&(CUTS<=20)]
    res.append(dict(region=nm,n_ln=n_ln,n_vox=int(m.sum()),n_hot=int(len(P)),root=round(root,2),
                    lo=int(A.min()),hi=int(A.max()),plateau=[int(pl.min()),int(pl.max())],
                    span=int((A.max(0)-A.min(0)).max()),status="跑完",
                    verdict="單一 LPU" if pl.max()==1 else f"候選 {pl.min()}–{pl.max()} 個 LPU"))
    print(f"  {nm:10s} LN {n_ln:5d} 熱點 {len(P):5d} 全範圍 {A.min()}–{A.max():<3d} 平台 {pl.min()}–{pl.max()} 根 {root:6.2f}  ({time.time()-t0:.0f}s)",flush=True)
json.dump(res,open(f"{OUT}/c14_table.json","w"),ensure_ascii=False,indent=1)
ran=[r for r in res if r["status"]=="跑完"]
print(f"\n跑完 {len(ran)} 區、略過 {len(res)-len(ran)} 區，用時 {time.time()-t0:.0f} 秒")
print("判定分布：", collections.Counter(r["verdict"] for r in res).most_common())
