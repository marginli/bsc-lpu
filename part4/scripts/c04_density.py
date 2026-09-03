#!/usr/bin/env python3
"""指令 4：AL_R 的密度場 v(r)。先算記憶體再決定要不要併格。"""
import numpy as np, json, csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; D06="/home/wanjuli/claude_linux/BSC_plan/D06"
OUT="/home/wanjuli/claude_linux/BSC_plan/specific_topics/LPU/part4/out"
GX=GY=512; REGION="AL_R"; INK="#1b2733"

lab=np.load(f"{D03}/work/labels_fc_1um.npy")
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]
n2i={n:i for i,n in enumerate(names)}
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv")))
name2idx={r["name"]:i for i,r in enumerate(rows)}
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")
S=[name2idx[r["neuron"]] for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")) if r["region"]==REGION]
print(f"{REGION} 的 LN 候選 {len(S)} 顆")

# 先在原始格點上估規模
zz,yy,xx=np.nonzero(lab==n2i[REGION])
n_native=len(zz); n_hot=int(n_native*0.25)
mem=n_hot*(n_hot-1)/2*8/2**30
print(f"原始格點：區內 {n_native:,} 個體素，熱點約 {n_hot:,} 個")
print(f"  → 距離矩陣 {n_hot*(n_hot-1)/2:,.0f} 個 float64 = {mem:.0f} GB")
B=1 if mem<=16 else 4
print(f"  → {'不併格' if B==1 else f'超過 16 GB，把 {B}x{B}x{B} 個格子併成一個分析體素'}")

sh=tuple(s//B+1 for s in lab.shape)
mask=np.zeros(sh,bool); mask[zz//B,yy//B,xx//B]=True
sel=np.isin(nid,S); v=vox[sel]; n=nid[sel]
iz=(v//(GX*GY))*2//B; iy=((v//GX)%GY)*2//B; ix=(v%GX)*2//B
# v(r)＝有幾顆「不同的」神經元佔到這一格（重複註冊次數，不是纖維條數）
key=np.unique(np.stack([iz,iy,ix,n]),axis=1)          # 同一顆神經元在同一格只算一次
f=np.zeros(sh,np.float32); np.add.at(f,(key[0],key[1],key[2]),1.0)
sm=ndimage.uniform_filter(f,size=3)                    # 3x3x3 移動平均（格子數不變）
print(f"分析體素：區內 {int(mask.sum()):,} 個")
print(f"v(r) 最大 {sm[mask].max():.1f}、平均 {sm[mask].mean():.1f}")
print("v(r) 的定義：重複註冊次數（一顆神經元在同一格只算一次），非纖維條數")
np.save(f"{OUT}/c04_mask.npy",mask); np.save(f"{OUT}/c04_v.npy",sm)
json.dump({"region":REGION,"n_ln":len(S),"B":B,"n_native":n_native,"mem_GB":round(mem),
           "n_analysis_vox":int(mask.sum()),"vmax":float(sm[mask].max()),"vmean":float(sm[mask].mean())},
          open(f"{OUT}/c04.json","w"),ensure_ascii=False,indent=1)

zs,ys,xs=np.nonzero(mask); sl=int(np.median(zs))
m2=mask[sl]; v2=np.where(m2,sm[sl],np.nan)
ys2,xs2=np.nonzero(m2); y0,y1,x0,x1=ys2.min()-3,ys2.max()+4,xs2.min()-3,xs2.max()+4
fig,axes=plt.subplots(1,2,figsize=(8.6,4.2),dpi=170)
axes[0].imshow(m2[y0:y1,x0:x1],cmap="Greys",vmin=0,vmax=1.6,interpolation="nearest")
axes[0].set_title(f"{REGION}: the neuropil mask",fontsize=10,color=INK)
im=axes[1].imshow(v2[y0:y1,x0:x1],cmap="turbo",interpolation="nearest")
axes[1].set_title(f"v(r): LN coverage,  n = {len(S)} LNs",fontsize=10,color=INK)
fig.colorbar(im,ax=axes[1],fraction=.046).ax.tick_params(labelsize=8)
for a in axes: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/c04_density.png"); plt.close(fig)
print("圖已存：c04_density.png")
