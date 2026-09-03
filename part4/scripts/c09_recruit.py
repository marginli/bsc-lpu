#!/usr/bin/env python3
"""指令 9：膨脹敏感度檢查 + 滾雪球招募。撐大 source、撐大 2 格、分母用 target 原體積。"""
import numpy as np, json, csv
from scipy import ndimage
D06="/home/wanjuli/claude_linux/BSC_plan/D06"; OUT="part4/out"; GX=GY=512; GROW=2
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); n2x={r["name"]:i for i,r in enumerate(rows)}
S=[n2x[r["neuron"]] for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")) if r["region"]=="AL_R"]
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")
sel=np.isin(nid,S); v=vox[sel]; n=nid[sel]
ix=(v%GX).astype(np.int32); iy=((v//GX)%GY).astype(np.int32); iz=(v//(GX*GY)).astype(np.int32)
x0,y0,z0=ix.min(),iy.min(),iz.min(); sh=(iz.max()-z0+9,iy.max()-y0+9,ix.max()-x0+9)
def mk(i):
    k=(n==i); a=np.zeros(sh,bool); a[iz[k]-z0+4,iy[k]-y0+4,ix[k]-x0+4]=True; return a
st=ndimage.generate_binary_structure(3,1)
ids=np.array(sorted(set(n.tolist())))
print("撐大幾格 → 體積變成幾倍（抽 60 顆的中位數）：")
smp=np.random.default_rng(0).choice(ids,60,replace=False); M={i:mk(i) for i in smp}
fac={k:float(np.median([ndimage.binary_dilation(M[i],st,k).sum()/M[i].sum() for i in smp])) for k in (1,2,3)}
print("   "+"、".join(f"{k} 格 {v:.1f} 倍" for k,v in fac.items()))
print("\n敏感度檢查（撐大 source，分母用 target 原體積）：")
print("  撐大  有交集   重疊率中位  最大   >50%")
for g in (0,1,2,3):
    D={i:(ndimage.binary_dilation(M[i],st,g) if g else M[i]) for i in smp}
    fr=np.array([(D[a]&M[b]).sum()/M[b].sum() for a in smp for b in smp if a!=b])
    print(f"  {g:4d}  {(fr>0).mean()*100:6.1f}%  {np.median(fr):10.3f}  {fr.max():5.2f}  {(fr>0.5).mean()*100:6.2f}%")
seed=int(np.load(f"{OUT}/c08_seed.npy")[0])
print(f"\n開始招募，種子＝{rows[seed]['name']}，撐大 {GROW} 格，門檻 50%")
src=mk(seed); got={seed}; rnd=0; log=[]
sizes={i:None for i in ids}
while True:
    rnd+=1; D=ndimage.binary_dilation(src,st,GROW); new=[]
    for i in ids:
        if i in got: continue
        m=mk(i)
        if (D&m).sum()/m.sum()>0.5: new.append(i)
    if not new: break
    for i in new: src|=mk(i); got.add(i)
    log.append(len(new)); print(f"   第 {rnd} 輪：收了 {len(new)} 顆，累計 {len(got)} 顆")
    if rnd>40: print("   超過 40 輪，停止"); break
print(f"招募結束：{len(got)} 顆（AL_R 的 LN 候選共 {len(ids)} 顆，佔 {len(got)/len(ids)*100:.0f}%），共 {rnd-1} 輪")
np.save(f"{OUT}/c09_recruited.npy",np.array(sorted(got)))
json.dump({"seed":rows[seed]["name"],"grow":GROW,"n_recruited":len(got),"n_pool":len(ids),
           "rounds":log,"vol_factor":{str(k):round(v,1) for k,v in fac.items()}},
          open(f"{OUT}/c09.json","w"),ensure_ascii=False,indent=1)
