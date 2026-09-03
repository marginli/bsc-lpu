#!/usr/bin/env python3
"""指令 10＋11：去本體與初級神經突、抽等值面（三個門檻）、算 c 值。"""
import numpy as np, json, csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; D06="/home/wanjuli/claude_linux/BSC_plan/D06"
OUT="part4/out"; GX=GY=512; B=4; REG="AL_R"; DEPTH_CUT=10
lab=np.load(f"{D03}/work/labels_fc_1um.npy")
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]; n2i={n:i for i,n in enumerate(names)}
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); n2x={r["name"]:i for i,r in enumerate(rows)}
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy"); dep=np.load(f"{D06}/work/dep.npy")
got=set(np.load(f"{OUT}/c09b_recruited_g4.npy").tolist())
sel=np.isin(nid,list(got)); v=vox[sel]; n=nid[sel]; d=dep[sel]
ix=(v%GX).astype(np.int64); iy=((v//GX)%GY).astype(np.int64); iz=(v//(GX*GY)).astype(np.int64)
ok=((2*iz<lab.shape[0])&(2*iy<lab.shape[1])&(2*ix<lab.shape[2]))
L=np.zeros(len(v),np.int16); L[ok]=lab[2*iz[ok],2*iy[ok],2*ix[ok]]
print(f"招募到的 {len(got)} 顆，共 {len(v):,} 個體素")
print("一、先量：依離本體的路徑深度分段")
print("   深度       體素數    落在該區外   落在任何腦區外")
for lo,hi in [(0,5),(5,10),(10,20),(20,35),(35,65),(65,101)]:
    k=(d>=lo)&(d<hi)
    if k.sum(): print(f"   {lo:3d}–{hi:<3d} {k.sum():10,d} {(L[k]!=n2i[REG]).mean()*100:11.1f}% {(L[k]==0).mean()*100:14.1f}%")
keep=d>=DEPTH_CUT
print(f"   → 用「深度 < {DEPTH_CUT} 丟掉」當去除規則（這個 10 是依上表決定的，不是論文給的）")
print(f"     丟掉 {int((~keep).sum()):,} 個體素（{(~keep).mean()*100:.1f}%）")
sh=tuple(s//B+1 for s in lab.shape)
kk=np.unique(np.stack([iz[keep]*2//B,iy[keep]*2//B,ix[keep]*2//B,n[keep]]),axis=1)
f=np.zeros(sh,np.float32); np.add.at(f,(kk[0],kk[1],kk[2]),1.0)
sm=ndimage.uniform_filter(f,3)
mask=np.load(f"{OUT}/c04_mask.npy")
zz,yy,xx=np.nonzero(lab==n2i[REG]); reg=np.zeros(sh,bool); reg[zz//B,yy//B,xx//B]=True
nz=sm[sm>0]
print("\n二、等值面三個門檻")
print("   門檻(百分位)   值    區域體素   佔腦區")
out={}
for p in (25,50,75):
    t=float(np.percentile(nz,p)); body=sm>=t
    body=ndimage.binary_fill_holes(body)
    out[p]=dict(thr=round(t,2),n=int(body.sum()),pct=round(float(body.sum()/reg.sum()*100),1))
    print(f"   {p:>3d}        {t:6.1f}  {body.sum():8,d}  {body.sum()/reg.sum()*100:7.1f}%")
    # 指令 11：c 值 —— 距離轉換取最內側 80%
    dt=ndimage.distance_transform_edt(body)
    idx=np.argsort(-dt[body])                      # 由內而外
    vals=sm[body]; nA=int(round(0.8*len(idx)))
    A=idx[:nA]; Bx=idx[nA:]
    c=(vals[A].mean())/(vals[Bx].mean()) if len(Bx) and vals[Bx].mean()>0 else float("nan")
    out[p].update(nA=int(nA),nB=int(len(Bx)),sumA=float(vals[A].sum()),sumB=float(vals[Bx].sum()),c=round(float(c),3))
print("\n三、c 值（中心 80% 的平均密度 ÷ 周邊的平均密度，判準 c > 2）")
print("   門檻   N_A     N_B    ΣA/N_A   ΣB/N_B     c     通過?")
for p in (25,50,75):
    o=out[p]; print(f"   {p:>3d}  {o['nA']:7,d} {o['nB']:7,d} {o['sumA']/o['nA']:8.1f} {o['sumB']/o['nB']:8.1f} {o['c']:7.3f}   {'是' if o['c']>2 else '否'}")
json.dump({"depth_cut":DEPTH_CUT,"levels":out},open(f"{OUT}/c10_11.json","w"),ensure_ascii=False,indent=1)
print(f"\n最鬆與最緊的體積比：{out[25]['n']/out[75]['n']:.1f} 倍")
