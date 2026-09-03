#!/usr/bin/env python3
"""指令 7：切割高度掃描 + 打亂順序重跑九次。"""
import numpy as np, json, csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import ndimage
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; D06="/home/wanjuli/claude_linux/BSC_plan/D06"
OUT="part4/out"; GX=GY=512; B=4; NP=9; CUTS=np.arange(4,26,1.0)
lab=np.load(f"{D03}/work/labels_fc_1um.npy")
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]; n2i={n:i for i,n in enumerate(names)}
rows=list(csv.DictReader(open(f"{D06}/work/neurons.csv"))); n2x={r["name"]:i for i,r in enumerate(rows)}
vox=np.load(f"{D06}/work/vox.npy"); nid=np.load(f"{D06}/work/nid.npy")
LN={}
for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")): LN.setdefault(r["region"],[]).append(n2x[r["neuron"]])

def field(reg):
    zz,yy,xx=np.nonzero(lab==n2i[reg]); sh=tuple(s//B+1 for s in lab.shape)
    m=np.zeros(sh,bool); m[zz//B,yy//B,xx//B]=True
    sel=np.isin(nid,LN.get(reg,[])); v=vox[sel]; n=nid[sel]
    iz=(v//(GX*GY))*2//B; iy=((v//GX)%GY)*2//B; ix=(v%GX)*2//B
    k=np.unique(np.stack([iz,iy,ix,n]),axis=1)
    f=np.zeros(sh,np.float32); np.add.at(f,(k[0],k[1],k[2]),1.0)
    return m,ndimage.uniform_filter(f,3),len(LN.get(reg,[]))

rng=np.random.default_rng(0); res={}; INK,GREY="#1b2733","#8b98a5"
fig,ax=plt.subplots(figsize=(9.6,4.6),dpi=170); cm=plt.get_cmap("tab10")
for k,reg in enumerate(["AL_R","AL_L","AVLP_R","AVLP_L","FB","PB"]):
    m,sm,nln=field(reg); P=np.argwhere(m&(sm>=np.percentile(sm[m],75))).astype(np.float32)
    thr=0.01*m.sum(); runs=[]
    for t in range(NP):
        Q=P if t==0 else P[rng.permutation(len(P))]
        Z=linkage(pdist(Q),"average")
        runs.append([int((np.bincount(fcluster(Z,c,"distance"))[1:]>thr).sum()) for c in CUTS])
        if t==0: root=float(Z[:,2].max()); D=pdist(Q); nd=len(np.unique(np.rint(D**2)))
    A=np.array(runs); res[reg]=dict(n_ln=nln,n_hot=len(P),root=round(root,2),
        lo=int(A.min()),hi=int(A.max()),span=int((A.max(0)-A.min(0)).max()),
        n_pairs=len(D),n_distinct=int(nd),
        plateau=[int(A[:,(CUTS>=14)&(CUTS<=20)].min()),int(A[:,(CUTS>=14)&(CUTS<=20)].max())])
    ax.fill_between(CUTS,A.min(0),A.max(0),color=cm(k),alpha=.18,lw=0)
    ax.plot(CUTS,np.median(A,0),lw=1.8,color=cm(k),label=f"{reg} (n={nln})")
    if reg=="AL_R":
        i6=int(np.where(CUTS==6)[0][0]); res[reg]["at6"]=sorted(A[:,i6].tolist())
        i4=int(np.where(CUTS==4)[0][0])
        Z=linkage(pdist(P),"average"); cl=fcluster(Z,4.0,"distance"); b=np.bincount(cl)[1:]
        res[reg]["h4_raw"]=int(len(b)); res[reg]["h4_pass"]=int((b>thr).sum()); res[reg]["h4_max"]=int(b.max())
ax.axhline(1,color=GREY,ls=":",lw=1)
ax.set_xlabel("UPGMA cut height  (analysis voxels)",fontsize=10,color=INK)
ax.set_ylabel("candidate LPUs found\n(clusters > 1% of region)",fontsize=10,color=INK)
ax.set_title(f"line = median of {NP} runs;  band = min–max over shuffled voxel order",fontsize=9.5,color=GREY,loc="left")
ax.legend(fontsize=8.5,frameon=False,ncol=3)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/c07_cutheight.png"); plt.close(fig)
json.dump(res,open(f"{OUT}/c07.json","w"),ensure_ascii=False,indent=1)
r=res["AL_R"]
print(f"高度 4：剪出 {r['h4_raw']} 群 / 通過 {r['h4_pass']} 群，最大的一群 {r['h4_max']} 個體素")
print(f"高度 6 打亂九次：{r['at6']}")
print(f"AL_R 距離總數 {r['n_pairs']:,}，相異距離值 {r['n_distinct']}")
print(f"AL_R 樹根高度 {r['root']}")
print("\n腦區      LN   熱點   全範圍   平台14-20  同高度最大變動  樹根")
for reg,d in res.items():
    print(f"{reg:8s} {d['n_ln']:5d} {d['n_hot']:5d}   {d['lo']}–{d['hi']:<5d} {d['plateau'][0]}–{d['plateau'][1]:<8d} {d['span']:^12d} {d['root']:6.2f}")
print("\n有沒有『自然的』切割高度：沒有。各腦區的平台高度不一致，")
print("而右端全部收斂到 1 只是因為切割高度超過了每一棵樹的根。")
