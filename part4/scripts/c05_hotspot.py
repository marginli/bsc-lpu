#!/usr/bin/env python3
"""指令 5：五數綜合、取上四分位、得到熱點。"""
import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="part4/out"; INK,GREY="#1b2733","#8b98a5"
mask=np.load(f"{OUT}/c04_mask.npy"); sm=np.load(f"{OUT}/c04_v.npy")
vals=sm[mask]; q=np.percentile(vals,[0,25,50,75,100])
hot=mask&(sm>=q[3])
print("1. 五數綜合：" + " ／ ".join(f"{x:.1f}" for x in q))
print(f"2. 熱點 {int(hot.sum()):,} 個體素，佔區內 {hot.sum()/mask.sum()*100:.1f}%")
print("3. 為什麼是這個百分比：因為『上四分位』的定義就是「排到 75% 的那個位置」，")
print("   取大於等於它的體素，剩下的必然是最上面的四分之一。跟資料長什麼樣無關。")
np.save(f"{OUT}/c05_hot.npy",hot)
json.dump({"fivenum":[round(float(x),1) for x in q],"n_hot":int(hot.sum()),
           "pct":round(float(hot.sum()/mask.sum()*100),1)},open(f"{OUT}/c05.json","w"),indent=1)
fig,ax=plt.subplots(figsize=(9.2,3.8),dpi=170)
ax.hist(vals,bins=70,color="#c7dbff",edgecolor="#93b4f5",linewidth=.4)
for lv,lb,col in zip(q[1:4],["lower quartile","median","upper quartile  Q_U"],[GREY,GREY,"#dc2626"]):
    ax.axvline(lv,color=col,lw=1.8 if col!=GREY else 1.1,ls="-" if col!=GREY else "--")
    ax.text(lv,ax.get_ylim()[1]*.93,f" {lb}\n {lv:.1f}",color=col,fontsize=9,
            fontweight="bold" if col!=GREY else "normal")
ax.axvspan(q[3],q[4],color="#dc2626",alpha=.07)
ax.set_xlabel("v(r) in AL_R   (distinct LNs per voxel, smoothed)",fontsize=10,color=INK)
ax.set_ylabel("voxels",fontsize=10,color=INK)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/c05_fivenum.png"); plt.close(fig)
print("圖已存：c05_fivenum.png")
