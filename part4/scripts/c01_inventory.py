#!/usr/bin/env python3
"""指令 1：盤點資料、確認座標框與單位。不做分析。"""
import numpy as np, json, csv, os

D03 = "/home/wanjuli/claude_linux/BSC_plan/D03"
D06 = "/home/wanjuli/claude_linux/BSC_plan/D06"
R = {}

# 1. 腦區與標籤體積
lab = np.load(f"{D03}/work/labels_fc_1um.npy", mmap_mode="r")
meta = json.load(open(f"{D03}/work/labels_fc_meta.json"))
names = json.load(open(f"{D03}/D03_labels_meta.json"))["names"]
R["腦區數"] = len(names)
R["標籤體積形狀 (z,y,x)"] = tuple(lab.shape)
R["格距"] = meta["step_um"]
R["原點"] = meta["origin_um"]

# 2. 神經元與體素
vox = np.load(f"{D06}/work/vox.npy"); nid = np.load(f"{D06}/work/nid.npy")
rows = list(csv.DictReader(open(f"{D06}/work/neurons.csv")))
u, c = np.unique(vox, return_counts=True)
R["神經元數"] = len(rows)
R["(體素,神經元) 紀錄數"] = len(vox)
R["相異體素數"] = len(u)
R["平均每體素被幾顆登記"] = round(float(len(vox) / len(u)), 2)
R["最擠的體素被幾顆登記"] = int(c.max())

# 3. 單位：從 affine 的奇異值反推
aff = json.load(open(f"{D03}/D03_fc_to_fcwb_affine.json"))
M = None
for k in ("M", "matrix", "A", "affine"):
    if k in aff:
        M = np.array(aff[k], float); break
if M is None:                       # 檔案結構未知，先把鍵印出來
    R["affine 檔案的鍵"] = list(aff.keys())
else:
    sv = np.linalg.svd(M[:3, :3] if M.shape[0] > 3 else M, compute_uv=False)
    R["affine 奇異值"] = [round(float(x), 3) for x in sv]
    R["det"] = round(float(np.linalg.det(M[:3, :3] if M.shape[0] > 3 else M)), 4)
    R["1 格 = 幾個物理微米"] = f"{sv.min():.2f}–{sv.max():.2f}"

# 4. 軸向：用三組解剖關係反推
n2i = {n: i + 1 for i, n in enumerate(names)}
def cen(nm):
    z, y, x = np.nonzero(np.asarray(lab) == n2i[nm])
    return np.array([z.mean(), y.mean(), x.mean()])
me_r, me_l = cen("ME_R"), cen("ME_L")
ca, gng, al = cen("MB_CA_R"), cen("GNG"), cen("AL_R")
d_me = np.abs(me_r - me_l)
R["ME_R 與 ME_L 的重心差 (z,y,x)"] = [round(float(v), 1) for v in (me_r - me_l)]
R["→ 左右軸"] = "axis " + str(int(d_me.argmax()))
R["MB_CA_R − GNG (z,y,x)"] = [round(float(v), 1) for v in (ca - gng)]
R["AL_R − MB_CA_R (z,y,x)"] = [round(float(v), 1) for v in (al - ca)]

# 5. D06 體素邊長換算成標籤格
R["D06 體素邊長（標籤格）"] = 2   # build_lpu.py 的 STEP=2.0，以 2*i 查 1 µm 格點

for k, v in R.items():
    print(f"{k:34s} {v}")
json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in R.items()},
          open("/home/wanjuli/claude_linux/BSC_plan/specific_topics/LPU/part4/out/c01.json", "w"),
          ensure_ascii=False, indent=1)
