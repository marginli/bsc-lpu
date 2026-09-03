#!/usr/bin/env python3
"""指令 2：把論文的目視判準翻成比例規則，產出 LN 候選名單與 f 的敏感度曲線。"""
import numpy as np, json, csv, os
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; D06="/home/wanjuli/claude_linux/BSC_plan/D06"
OUT="/home/wanjuli/claude_linux/BSC_plan/specific_topics/LPU/part4/out"
GX=GY=512; F=0.80

lab=np.load(f"{D03}/work/labels_fc_1um.npy")
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]; R=len(names)
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); N=len(rows)
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")

# 每個體素查腦區標籤（D06 的 2 µm 體素中心落在 1 µm 格點的 2*i）
ix=vox%GX; iy=(vox//GX)%GY; iz=vox//(GX*GY)
L=np.zeros(len(vox),np.int16)
ok=((2*iz<lab.shape[0])&(2*iy<lab.shape[1])&(2*ix<lab.shape[2]))
L[ok]=lab[2*iz[ok],2*iy[ok],2*ix[ok]]
print(f"落在某個腦區內的體素：{(L>0).mean()*100:.1f}%")

# 神經元 × 腦區的體素數（0 欄是腦區外，歸零不算進分母）
C=np.bincount(nid.astype(np.int64)*R+L,minlength=N*R).reshape(N,R); C[:,0]=0
tot=C.sum(1); top=C.max(1); arg=C.argmax(1); big=tot>0
print(f"完全沒落進任何腦區的神經元：{int((~big).sum())} 顆")

sel=big&(top>=F*tot)
with open(f"{OUT}/c02_ln_candidates.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["neuron","region","frac","n_vox_in_region","n_vox_in_any_region"])
    for i in np.flatnonzero(sel):
        w.writerow([rows[i]["name"],names[arg[i]],f"{top[i]/tot[i]:.4f}",int(top[i]),int(tot[i])])
print(f"f={F}: LN 候選 {int(sel.sum())} 顆（{sel.mean()*100:.1f}%），涵蓋 {len(set(arg[sel].tolist()))} 個腦區")

with open(f"{OUT}/c02_f_sweep.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["f","n_neurons","pct","n_regions"])
    prev=None; mono=True
    for f in np.arange(0.50,1.001,0.01):
        s=big&(top>=f*tot); n=int(s.sum())
        w.writerow([f"{f:.2f}",n,f"{s.mean()*100:.2f}",len(set(arg[s].tolist()))])
        if prev is not None and n>prev: mono=False
        prev=n
print("f 掃描曲線單調遞減：",mono)
for f in (0.70,0.80,0.90,1.00):
    s=big&(top>=f*tot)
    print(f"   f={f:.2f}  {int(s.sum()):5d} 顆  {s.mean()*100:5.1f}%  {len(set(arg[s].tolist())):2d} 個腦區")
np.save(f"{OUT}/c02_C.npy",C)
