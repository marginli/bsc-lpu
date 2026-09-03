#!/usr/bin/env python3
"""指令 8：挑起始種子。先照論文字面試，失敗才用替代規則。"""
import numpy as np, json, csv
from scipy import ndimage
D06="/home/wanjuli/claude_linux/BSC_plan/D06"; OUT="part4/out"; GX=GY=512; B=4
mask=np.load(f"{OUT}/c04_mask.npy"); P=np.load(f"{OUT}/c06_P.npy"); cl=np.load(f"{OUT}/c06_cl.npy")
b=np.bincount(cl)[1:]; thr=0.01*mask.sum()
keep=[i+1 for i,c in enumerate(b) if c>thr]
small=keep[int(np.argmin(b[np.array(keep)-1]))]
print(f"最小的候選 LPU：第 {small} 群，{int(b[small-1])} 個體素")
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); n2x={r["name"]:i for i,r in enumerate(rows)}
S=[n2x[r["neuron"]] for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")) if r["region"]=="AL_R"]
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")
sel=np.isin(nid,S); v=vox[sel]; n=nid[sel]
iz=(v//(GX*GY))*2//B; iy=((v//GX)%GY)*2//B; ix=(v%GX)*2//B
neu={}
for i in np.unique(n):
    k=(n==i); neu[i]=set(zip(iz[k].tolist(),iy[k].tolist(),ix[k].tolist()))
exact=set(map(tuple,P[cl==small].tolist()))
big=np.zeros(mask.shape,bool); pts=P[cl==small]; big[pts[:,0],pts[:,1],pts[:,2]]=True
loose=set(map(tuple,np.argwhere(ndimage.binary_fill_holes(ndimage.binary_dilation(big,iterations=2))).tolist()))
for nm,R in (("嚴格（該群的體素集合）",exact),("放寬（膨脹兩格＋填洞）",loose)):
    fr=np.array([len(vs&R)/len(vs) for vs in neu.values()])
    print(f"{nm}：比例中位 {np.median(fr)*100:.1f}%、最大 {fr.max()*100:.1f}%、達到 100% 的 {int((fr>=0.999).sum())} 顆")
fr={i:len(vs&exact)/len(vs) for i,vs in neu.items()}
top10=sorted(fr,key=lambda i:-fr[i])[:10]
sizes={i:len(neu[i]) for i in top10}
seed=min(top10,key=lambda i:sizes[i])
print("\n照字面挑不到 → 改用替代規則（比例最高的前十顆裡取體積最小的）")
print("【替代規則，不是論文的條件】")
print(f"選中：{rows[seed]['name']}  比例 {fr[seed]*100:.1f}%  體積 {sizes[seed]} 個分析體素")
print("前十顆：")
for i in top10:
    print(f"   {rows[i]['name']:22s} 比例 {fr[i]*100:5.1f}%  體積 {sizes[i]:5d}" + ("   ← 選中" if i==seed else ""))
json.dump({"cluster":int(small),"seed":rows[seed]["name"],"seed_frac":round(fr[seed],4),
           "seed_size":int(sizes[seed]),"rule":"替代規則：比例最高前十顆取體積最小",
           "top10":[{"name":rows[i]["name"],"frac":round(fr[i],4),"size":int(sizes[i])} for i in top10]},
          open(f"{OUT}/c08.json","w"),ensure_ascii=False,indent=1)
np.save(f"{OUT}/c08_seed.npy",np.array([seed]))
