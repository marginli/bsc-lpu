#!/usr/bin/env python3
"""指令 3：全腦 LN 普查與 f 敏感度掃描。"""
import numpy as np, json, collections, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D03="/home/wanjuli/claude_linux/BSC_plan/D03"
OUT="/home/wanjuli/claude_linux/BSC_plan/specific_topics/LPU/part4/out"
INK,ACC,GREY="#1b2733","#2563eb","#8b98a5"
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]
C=np.load(f"{OUT}/c02_C.npy"); tot=C.sum(1); top=C.max(1); arg=C.argmax(1); big=tot>0
lab=np.load(f"{D03}/work/labels_fc_1um.npy")
u,cnt=np.unique(lab,return_counts=True); vol={names[i]:int(v) for i,v in zip(u,cnt)}

def census(f):
    s=big&(top>=f*tot); return collections.Counter(names[a] for a in arg[s]), int(s.sum())
c,_=census(0.80)
nz=[n for n in names[1:] if c.get(n,0)>0]; zr=[n for n in names[1:] if c.get(n,0)==0]
vn=sum(vol.get(n,0) for n in nz); vz=sum(vol.get(n,0) for n in zr)
print(f"1. 非零腦區 {len(nz)} 個，零 {len(zr)} 個")
print("   零的是：", "、".join(zr))
print(f"2. 非零腦區體積佔比 {vn/(vn+vz)*100:.1f}%，零的佔 {vz/(vn+vz)*100:.1f}%")
print("   最多的三個：", "、".join(f"{n} {c[n]}" for n,_ in sorted(c.items(),key=lambda t:-t[1])[:3]))
WATCH=["MB_CA_R","MB_CA_L","MB_PED_R","MB_PED_L","MB_VL_R","MB_VL_L","MB_ML_R","MB_ML_L",
       "EB","NO","AOTU_R","AOTU_L","AL_R","AL_L"]
print("3. f 掃描（觀察名單）")
print("      f  " + "".join(f"{w:>9s}" for w in ["MB 合計","EB","NO","AOTU 合計","AL_R","AL_L"]))
for f in (0.70,0.80,0.90,0.95,1.00):
    cc,_=census(f)
    mb=sum(cc.get(x,0) for x in WATCH[:8]); ao=cc.get("AOTU_R",0)+cc.get("AOTU_L",0)
    print(f"   {f:.2f}  " + "".join(f"{v:>9d}" for v in [mb,cc.get('EB',0),cc.get('NO',0),ao,cc.get('AL_R',0),cc.get('AL_L',0)]))

# 圖 A：普查
srt=sorted([(n,c[n]) for n in nz],key=lambda t:t[1])
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.4,7.2),dpi=170,gridspec_kw={"width_ratios":[1.6,1]})
W={"MB_CA_R","MB_CA_L","MB_PED_R","MB_PED_L","MB_VL_R","MB_VL_L","MB_ML_R","MB_ML_L","EB","NO","AOTU_R","AOTU_L"}
ax.barh(range(len(srt)),[v for _,v in srt],color=["#d97706" if n in W else ACC for n,_ in srt],height=.72)
for i,(n,v) in enumerate(srt): ax.text(v*1.14,i,str(v),va="center",fontsize=8,color=INK)
ax.set_yticks(range(len(srt))); ax.set_yticklabels([n for n,_ in srt],fontsize=8)
ax.set_xscale("log"); ax.set_xlim(.7,4000)
ax.set_xlabel("neurons with >=80% of their arbour in this region  (log)",fontsize=9.5,color=INK)
ax.set_title(f"{len(nz)} of 75 regions have at least one",fontsize=10.5,color=INK,loc="left")
bx.axis("off"); bx.set_title(f"the other {len(zr)} regions: zero",fontsize=10.5,color="#dc2626",loc="left")
per=(len(zr)+3)//4
for ci in range(4):
    for ri,n in enumerate(zr[ci*per:(ci+1)*per]):
        bx.text(ci*.26,.95-ri*.068,n,fontsize=7.4,transform=bx.transAxes,
                color="#dc2626" if n in W else GREY,fontweight="bold" if n in W else "normal")
for a in (ax,):
    for sp in ("top","right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/c03_census.png"); plt.close(fig)

# 圖 B：f 敏感度
fs=np.arange(.50,1.001,.01)
g=[(big&(top>=f*tot)).mean()*100 for f in fs]
nr=[len({arg[i] for i in np.flatnonzero(big&(top>=f*tot))}) for f in fs]
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.2,4.2),dpi=170)
ax.plot(fs*100,g,color="#16a34a",lw=2.2); ax.axhline(26,color="#dc2626",ls="--",lw=1.5)
ax.text(100,27.5,"the paper: ~26%",ha="right",fontsize=9,color="#dc2626",fontweight="bold")
ax.set_ylabel("neurons judged LN  (%)",fontsize=9.5,color=INK)
bx.plot(fs*100,nr,color=ACC,lw=2.2); bx.set_ylabel("regions with at least one LN",fontsize=9.5,color=INK)
for a in (ax,bx):
    a.set_xlabel("f = % of the neuron's arbour that must stay in one region",fontsize=9.5,color=INK)
    a.set_xlim(50,101); a.set_ylim(0,62)
    for sp in ("top","right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/c03_fsweep.png"); plt.close(fig)
print("4. 圖已存：c03_census.png、c03_fsweep.png")
