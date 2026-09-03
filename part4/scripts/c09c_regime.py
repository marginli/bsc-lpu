#!/usr/bin/env python3
"""指令 9 診斷：招募步驟有沒有「收一部分」的操作區間？掃膨脹格數 × 重疊門檻。"""
import numpy as np, json, csv
from scipy import ndimage
D06="/home/wanjuli/claude_linux/BSC_plan/D06"; OUT="part4/out"; GX=GY=512
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); n2x={r["name"]:i for i,r in enumerate(rows)}
S=[n2x[r["neuron"]] for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")) if r["region"]=="AL_R"]
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")
sel=np.isin(nid,S); v=vox[sel]; n=nid[sel]
ix=(v%GX).astype(np.int32); iy=((v//GX)%GY).astype(np.int32); iz=(v//(GX*GY)).astype(np.int32)
x0,y0,z0=ix.min(),iy.min(),iz.min(); sh=(iz.max()-z0+9,iy.max()-y0+9,ix.max()-x0+9)
M={}
for i in np.unique(n):
    k=(n==i); a=np.zeros(sh,bool); a[iz[k]-z0+4,iy[k]-y0+4,ix[k]-x0+4]=True; M[i]=a
sz={i:int(a.sum()) for i,a in M.items()}; ids=np.array(sorted(M))
st=ndimage.generate_binary_structure(3,1); seed=int(np.load(f"{OUT}/c08_seed.npy")[0])
res={}
print("招募到的顆數（池子 723 顆）")
print("  膨脹\\門檻  " + "".join(f"{t:>8.2f}" for t in (0.30,0.40,0.50,0.60,0.70)))
for g in (2,3,4,5):
    row=[]
    for t in (0.30,0.40,0.50,0.60,0.70):
        src=M[seed].copy(); got={seed}
        for _ in range(40):
            D=ndimage.binary_dilation(src,st,g)
            new=[i for i in ids if i not in got and (D&M[i]).sum()/sz[i]>t]
            if not new: break
            for i in new: src|=M[i]; got.add(i)
        row.append(len(got))
    res[g]=row
    print(f"  {g:4d}      " + "".join(f"{x:>8d}" for x in row))
json.dump(res,open(f"{OUT}/c09c_regime.json","w"),indent=1)
print("\n「收一部分」＝結果落在 2 到 722 之間的格子數：",
      sum(1 for g in res for x in res[g] if 2<=x<=722), "／", sum(len(v) for v in res.values()))
