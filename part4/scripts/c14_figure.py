#!/usr/bin/env python3
"""指令 14 第三項之 1：全腦判定圖。"""
import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; OUT="part4/out"
lab=np.load(f"{D03}/work/labels_fc_1um.npy")
names=json.load(open(f"{D03}/D03_labels_meta.json"))["names"]; n2i={n:i+1 for i,n in enumerate(names)}
tab={r["region"]:r for r in json.load(open(f"{OUT}/c14_table.json"))}
EN={"單一 LPU":"one LPU","多個候選":"more than one candidate","候選 hub":"no LN population (candidate hub)"}
def cls(r):
    if r["status"]!="跑完": return "候選 hub"
    return "單一 LPU" if r["verdict"]=="單一 LPU" else "多個候選"
kind={n:cls(tab[n]) for n in names}
# axis0=前後, axis1=背腹, axis2=左右（指令 1 驗出來的），三軸都是「越大越前／背／左」。
# from_high 決定攝影機站在軸的哪一端，也就是哪一面擋住哪一面——從錯的一端走，
# 前側的腦區會被後側整個蓋掉（AL_R 正視只剩 100 格看得到、EB 是 0 格）。
def proj(axis,from_high=False):
    a=np.moveaxis(lab,axis,0)
    if from_high: a=a[::-1]
    out=np.zeros(a.shape[1:],np.int16)
    for i in range(a.shape[0]):
        s=a[i]; m=(out==0)&(s>0); out[m]=s[m]
    return out
rgb={"單一 LPU":(0.15,0.39,0.92),"多個候選":(0.85,0.47,0.02),"候選 hub":(0.58,0.64,0.70)}
fig,axes=plt.subplots(1,2,figsize=(11.5,5.4),dpi=170)
# 一張投影圖要對，三件事都得指定：攝影機站哪一端（from_high，決定遮擋）、
# 上下（axis1 越大越背側，而 imshow 第 0 列在最上面）、左右（由攝影機位置決定）。
for ax,(axis,from_high,upright,ttl) in zip(axes,[
        (0,True,np.flipud,"front view  (from the front; the fly's right is on the left)"),
        (2,False,lambda a:np.fliplr(np.rot90(a,1)),"side view  (from the right; anterior to the left)")]):
    P=proj(axis,from_high); img=np.ones(P.shape+(3,))
    for n in names:
        m=(P==n2i[n])
        if m.any(): img[m]=rgb[kind[n]]
    img[P==0]=1.0
    ax.imshow(upright(img)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl,fontsize=10,color="#1b2733")
    for sp in ax.spines.values(): sp.set_color("#c3cbd6")
c=lambda k: sum(1 for n in names if kind[n]==k)
fig.legend(handles=[Patch(color=rgb[k],label=f"{EN[k]}  ({c(k)})") for k in ("單一 LPU","多個候選","候選 hub")],
           loc="lower center",ncol=3,frameon=False,fontsize=10)
fig.suptitle("Our own whole-brain pass  ·  75 Ito regions  ·  21 clustered, 54 had <10 LN candidates",
             fontsize=10.5,color="#5a6b7b")
fig.tight_layout(rect=[0,.06,1,.96]); fig.savefig(f"{OUT}/c14_brain.png"); plt.close(fig)
print("判定分布：", {k:c(k) for k in ("單一 LPU","多個候選","候選 hub")})
print("圖已存：c14_brain.png")
