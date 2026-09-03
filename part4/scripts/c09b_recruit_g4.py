#!/usr/bin/env python3
"""指令 9 修正版：撐大 2 格招募不到任何一顆，改試「撐大 4 格」。"""
import numpy as np, json, csv, sys
from scipy import ndimage
D06="/home/wanjuli/claude_linux/BSC_plan/D06"; OUT="part4/out"; GX=GY=512
GROW=int(sys.argv[1]) if len(sys.argv)>1 else 4
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); n2x={r["name"]:i for i,r in enumerate(rows)}
S=[n2x[r["neuron"]] for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")) if r["region"]=="AL_R"]
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")
sel=np.isin(nid,S); v=vox[sel]; n=nid[sel]
ix=(v%GX).astype(np.int32); iy=((v//GX)%GY).astype(np.int32); iz=(v//(GX*GY)).astype(np.int32)
x0,y0,z0=ix.min(),iy.min(),iz.min(); sh=(iz.max()-z0+9,iy.max()-y0+9,ix.max()-x0+9)
M={}
for i in np.unique(n):
    k=(n==i); a=np.zeros(sh,bool); a[iz[k]-z0+4,iy[k]-y0+4,ix[k]-x0+4]=True; M[i]=a
sz={i:int(a.sum()) for i,a in M.items()}
st=ndimage.generate_binary_structure(3,1)
seed=int(np.load(f"{OUT}/c08_seed.npy")[0]); ids=np.array(sorted(M))
print(f"種子 {rows[seed]['name']}，撐大 {GROW} 格，門檻 50%，池子 {len(ids)} 顆")
src=M[seed].copy(); got={seed}; log=[]
for rnd in range(1,41):
    D=ndimage.binary_dilation(src,st,GROW)
    new=[i for i in ids if i not in got and (D&M[i]).sum()/sz[i]>0.5]
    if not new: break
    for i in new: src|=M[i]; got.add(i)
    log.append(len(new)); print(f"   第 {rnd} 輪：收 {len(new):4d} 顆，累計 {len(got):4d}")
print(f"結束：招募到 {len(got)} 顆 / {len(ids)} 顆（{len(got)/len(ids)*100:.0f}%），{len(log)} 輪")
json.dump({"grow":GROW,"seed":rows[seed]["name"],"n_recruited":len(got),"n_pool":len(ids),"rounds":log},
          open(f"{OUT}/c09b_g{GROW}.json","w"),ensure_ascii=False,indent=1)
np.save(f"{OUT}/c09b_recruited_g{GROW}.npy",np.array(sorted(got)))
