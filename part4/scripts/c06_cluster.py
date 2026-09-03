#!/usr/bin/env python3
"""指令 6：距離矩陣、UPGMA、1% 過濾。切割高度先用 8。"""
import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
OUT="part4/out"; CUT=8.0
mask=np.load(f"{OUT}/c04_mask.npy"); hot=np.load(f"{OUT}/c05_hot.npy")
P=np.argwhere(hot)
thr=0.01*mask.sum()
print(f"1. 規則 ② 的門檻：{thr:.1f} 個體素（腦區內 {int(mask.sum()):,} 個的 1%）")
print(f"   注意分母是整個腦區（{int(mask.sum()):,}），不是熱點（{len(P):,}）")
D=pdist(P.astype(np.float32)); print(f"   距離矩陣 {len(D):,} 個 float64 = {len(D)*8/2**20:.0f} MB")
Z=linkage(D,method="average")     # average linkage = UPGMA，論文指定
cl=fcluster(Z,CUT,"distance"); b=np.bincount(cl)[1:]
keep=[i+1 for i,c in enumerate(b) if c>thr]
print(f"2. 高度 {CUT:g}：規則 ① 剪出 {len(b)} 群，規則 ② 通過 {len(keep)} 群")
print(f"3. 各群大小：{sorted(b[np.array(keep)-1].tolist())}  合計 {int(b[np.array(keep)-1].sum()):,}（熱點 {len(P):,}）")
np.save(f"{OUT}/c06_cl.npy",cl); np.save(f"{OUT}/c06_P.npy",P)
json.dump({"cut":CUT,"thr":float(thr),"n_raw":int(len(b)),"n_pass":len(keep),
           "sizes":sorted(int(x) for x in b[np.array(keep)-1])},open(f"{OUT}/c06.json","w"),indent=1)
fig,ax=plt.subplots(figsize=(7.6,7.2),dpi=170)
mz,my,mx=np.nonzero(mask); sl=int(np.median(P[:,0]))
ax.scatter(mx[mz==sl],my[mz==sl],s=6,c="#e8ecf1")
cm=plt.get_cmap("tab10")
for k,c in enumerate(keep):
    p=P[(cl==c)&(P[:,0]==sl)]
    ax.scatter(p[:,2],p[:,1],s=9,color=cm(k),label=f"cluster {k+1}  ({int(b[c-1])})")
ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"AL_R: hot-spot voxels, UPGMA cut = {CUT:g}",fontsize=11,color="#1b2733")
ax.legend(fontsize=8.5,frameon=False,loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/c06_clusters.png"); plt.close(fig)
print("4. 圖已存：c06_clusters.png")
