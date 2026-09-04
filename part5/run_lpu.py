#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_lpu.py — 主程式

PART 3 的十四條指令，收成十二個函式（指令 10＋11、12＋13 各併成一個），
順序就是 PART 2 那七個步驟的順序。
所有參數都在 lpu_lib.Config 裡；所有資料存取都在 lpu_lib.Brain 裡。
這個檔案只負責「先做什麼、再做什麼、把什麼印出來」。

用法：
    python3 run_lpu.py check            # 只檢查環境與資料，不算任何東西
    python3 run_lpu.py all              # 從頭跑到尾
    python3 run_lpu.py 4 5 6            # 只跑指定的幾條指令
    python3 run_lpu.py all --d03 /path/to/D03 --d06 /path/to/D06 --out ./out
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import lpu_lib as L
from lpu_lib import Brain, Config

REGION = "AL_R"              # 逐步示範用的腦區
SWEEP_REGIONS = ["AL_R", "AL_L", "AVLP_R", "AVLP_L", "FB", "PB"]


# ══════════════════════════════════════════════════════════════════
# 前置檢查
# ══════════════════════════════════════════════════════════════════

REQUIRED_FILES = [
    ("d03", "work/labels_fc_1um.npy", "分區標籤體積"),
    ("d03", "work/labels_fc_meta.json", "標籤的格距與原點"),
    ("d03", "D03_labels_meta.json", "75 個腦區的名字"),
    ("d03", "D03_fc_to_fcwb_affine.json", "座標框轉換矩陣"),
    ("d06", "work/vox.npy", "骨架體素編號"),
    ("d06", "work/nid.npy", "每個體素屬於哪顆神經元"),
    ("d06", "work/dep.npy", "每個體素離細胞本體多遠"),
    ("d06", "work/neurons.csv", "28,573 顆神經元的名字"),
]


def check(cfg: Config) -> int:
    """換一台機器之後，第一件該做的事。不載入資料，只確認東西都在。"""
    bad = 0
    print("── 一、Python 與套件 " + "─" * 44)
    print(f"   python      {sys.version.split()[0]}")
    for mod, floor in (("numpy", "1.24"), ("scipy", "1.10"), ("matplotlib", "3.6")):
        try:
            m = __import__(mod)
            print(f"   {mod:<11} {m.__version__}   （測過的下限 {floor}）")
        except ImportError:
            print(f"   {mod:<11} 沒有安裝   ← 必要")
            bad += 1

    print("── 二、資料檔 " + "─" * 51)
    total = 0
    for which, rel, what in REQUIRED_FILES:
        root = Path(cfg.d03 if which == "d03" else cfg.d06)
        p = root / rel
        alt = p.with_suffix(".npz")
        hit = p if p.exists() else (alt if alt.exists() else None)
        if hit is None:
            print(f"   [缺]  {p}\n         （{what}）")
            bad += 1
        else:
            mb = hit.stat().st_size / 2 ** 20
            total += mb
            print(f"   [有]  {hit.name:<24} {mb:8.1f} MB   {what}")
    print(f"         {'合計':<24} {total:8.1f} MB")

    print("── 三、記憶體 " + "─" * 51)
    peak = 2.5
    print(f"   這條管線實測的峰值是 {peak} GB（指令 14 跑到 ME_L 的兩兩距離時）。")
    print("   其他每一步都在 1.1 GB 以下。")
    try:
        free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
        print(f"   目前可用約 {free:.1f} GB " + ("→ 夠" if free > 4 else "→ 偏少，建議 4 GB 以上"))
    except (ValueError, AttributeError):
        print("   （無法在這個平台上查可用記憶體）")

    print("── 四、輸出目錄 " + "─" * 49)
    print(f"   {cfg.ensure_out().resolve()}")
    print()
    print(cfg.describe())
    print()
    print("結果：" + ("全部通過，可以跑 `python3 run_lpu.py all`"
                     if bad == 0 else f"有 {bad} 項缺少，先補齊再跑"))
    return bad


# ══════════════════════════════════════════════════════════════════
# 指令 1：盤點
# ══════════════════════════════════════════════════════════════════

def step01(brain: Brain) -> dict:
    """盤點資料、確認座標框與單位。不做任何分析。"""
    cfg, lab = brain.cfg, brain.lab
    R = {}
    R["腦區數"] = len(brain.region_names)
    R["標籤體積形狀 (z,y,x)"] = list(lab.shape)
    R["格距"] = brain.meta["step_um"]
    R["原點"] = brain.meta["origin_um"]

    u, c = np.unique(brain.vox, return_counts=True)
    R["神經元數"] = len(brain.neurons)
    R["(體素,神經元) 紀錄數"] = len(brain.vox)
    R["相異體素數"] = len(u)
    R["平均每體素被幾顆登記"] = round(float(len(brain.vox) / len(u)), 2)
    R["最擠的體素被幾顆登記"] = int(c.max())

    # 單位：從 affine 的奇異值反推。一格不是一微米，這件事後面每一步都會用到。
    aff = json.loads((Path(cfg.d03) / "D03_fc_to_fcwb_affine.json").read_text())
    M = next((np.array(aff[k], float) for k in ("M", "matrix", "A", "affine") if k in aff), None)
    if M is None:
        R["affine 檔案的鍵"] = list(aff.keys())
    else:
        A = M[:3, :3] if M.shape[0] > 3 else M
        sv = np.linalg.svd(A, compute_uv=False)
        R["affine 奇異值"] = [round(float(x), 3) for x in sv]
        R["det"] = round(float(np.linalg.det(A)), 4)
        R["1 格 = 幾個物理微米"] = f"{sv.min():.2f}–{sv.max():.2f}"

    # 軸向：用解剖上已知的三組關係反推，不要相信檔頭寫的順序
    def cen(nm):
        z, y, x = brain.region_mask(nm)
        return np.array([z.mean(), y.mean(), x.mean()])
    d = cen("ME_R") - cen("ME_L")
    R["ME_R 與 ME_L 的重心差 (z,y,x)"] = [round(float(v), 1) for v in d]
    R["→ 左右軸"] = "axis " + str(int(np.abs(d).argmax()))
    ca, gng, al = cen("MB_CA_R"), cen("GNG"), cen("AL_R")
    R["MB_CA_R − GNG (z,y,x)"] = [round(float(v), 1) for v in (ca - gng)]
    R["AL_R − MB_CA_R (z,y,x)"] = [round(float(v), 1) for v in (al - ca)]
    R["D06 體素邊長（標籤格）"] = cfg.d06_step

    for k, v in R.items():
        print(f"{k:34s} {v}")
    cfg.save("c01", R)
    return R


# ══════════════════════════════════════════════════════════════════
# 指令 2：把目視判準翻成比例規則
# ══════════════════════════════════════════════════════════════════

def step02(brain: Brain) -> dict:
    cfg = brain.cfg
    Lv = brain.region_of_voxel
    print(f"落在某個腦區內的體素：{(Lv > 0).mean() * 100:.1f}%")
    C = brain.contact
    sel, arg, frac = L.ln_selection(C, cfg.f_ln)
    print(f"完全沒落進任何腦區的神經元：{int((C.sum(1) == 0).sum())} 顆")

    out = cfg.ensure_out()
    with open(out / "c02_ln_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["neuron", "region", "frac", "n_vox_in_region", "n_vox_in_any_region"])
        for i in np.flatnonzero(sel):
            w.writerow([brain.neurons[i]["name"], brain.names[arg[i]],
                        f"{frac[i]:.4f}", int(C[i].max()), int(C[i].sum())])
    print(f"f={cfg.f_ln}: LN 候選 {int(sel.sum())} 顆（{sel.mean() * 100:.1f}%），"
          f"涵蓋 {len(set(arg[sel].tolist()))} 個腦區")

    # 敏感度：門檻一動，答案動多少
    with open(out / "c02_f_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["f", "n_neurons", "pct", "n_regions"])
        prev, mono = None, True
        for f in np.arange(0.50, 1.001, 0.01):
            s, a, _ = L.ln_selection(C, f)
            n = int(s.sum())
            w.writerow([f"{f:.2f}", n, f"{s.mean() * 100:.2f}", len(set(a[s].tolist()))])
            if prev is not None and n > prev:
                mono = False
            prev = n
    print("f 掃描曲線單調遞減：", mono)
    for f in (0.70, 0.80, 0.90, 1.00):
        s, a, _ = L.ln_selection(C, f)
        print(f"   f={f:.2f}  {int(s.sum()):5d} 顆  {s.mean() * 100:5.1f}%  "
              f"{len(set(a[s].tolist())):2d} 個腦區")
    np.save(out / "c02_C.npy", C)
    r = dict(n_ln=int(sel.sum()), n_regions=len(set(arg[sel].tolist())), monotone=mono,
             pct_in_region=round(float((Lv > 0).mean() * 100), 1))
    cfg.save("c02", r)
    return r


# ══════════════════════════════════════════════════════════════════
# 指令 3：全腦普查
# ══════════════════════════════════════════════════════════════════

WATCH = ["MB_CA_R", "MB_CA_L", "MB_PED_R", "MB_PED_L", "MB_VL_R", "MB_VL_L",
         "MB_ML_R", "MB_ML_L", "EB", "NO", "AOTU_R", "AOTU_L"]


def step03(brain: Brain) -> dict:
    cfg = brain.cfg
    C = brain.contact
    names = brain.names
    u, cnt = np.unique(brain.lab, return_counts=True)
    vol = {names[i]: int(v) for i, v in zip(u, cnt)}

    c, _ = L.ln_by_region(C, names, cfg.f_ln)
    nz = [n for n in brain.region_names if c.get(n, 0) > 0]
    zr = [n for n in brain.region_names if c.get(n, 0) == 0]
    vn = sum(vol.get(n, 0) for n in nz)
    vz = sum(vol.get(n, 0) for n in zr)
    print(f"1. 非零腦區 {len(nz)} 個，零 {len(zr)} 個")
    print("   零的是：", "、".join(zr))
    print(f"2. 非零腦區體積佔比 {vn / (vn + vz) * 100:.1f}%，零的佔 {vz / (vn + vz) * 100:.1f}%")
    print("   最多的三個：", "、".join(f"{n} {v}" for n, v in
                                      sorted(c.items(), key=lambda t: -t[1])[:3]))
    print("3. f 掃描（觀察名單）")
    cols = ["MB 合計", "EB", "NO", "AOTU 合計", "AL_R", "AL_L"]
    print("      f  " + "".join(f"{w:>9s}" for w in cols))
    for f in (0.70, 0.80, 0.90, 0.95, 1.00):
        cc, _ = L.ln_by_region(C, names, f)
        mb = sum(cc.get(x, 0) for x in WATCH[:8])
        ao = cc.get("AOTU_R", 0) + cc.get("AOTU_L", 0)
        vals = [mb, cc.get("EB", 0), cc.get("NO", 0), ao, cc.get("AL_R", 0), cc.get("AL_L", 0)]
        print(f"   {f:.2f}  " + "".join(f"{v:>9d}" for v in vals))

    _fig_census(brain, c, nz, zr)
    _fig_fsweep(brain, C)
    r = dict(n_nonzero=len(nz), n_zero=len(zr), zero=zr,
             vol_pct_nonzero=round(vn / (vn + vz) * 100, 1))
    cfg.save("c03", r)
    return r


def _fig_census(brain, c, nz, zr):
    plt = L.setup_matplotlib()
    srt = sorted([(n, c[n]) for n in nz], key=lambda t: t[1])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 7.2), dpi=170,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    W = set(WATCH)
    ax.barh(range(len(srt)), [v for _, v in srt],
            color=[L.WARN if n in W else L.ACC for n, _ in srt], height=.72)
    for i, (n, v) in enumerate(srt):
        ax.text(v * 1.14, i, str(v), va="center", fontsize=8, color=L.INK)
    ax.set_yticks(range(len(srt)))
    ax.set_yticklabels([n for n, _ in srt], fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(.7, 4000)
    ax.set_xlabel("neurons with >=80% of their arbour in this region  (log)",
                  fontsize=9.5, color=L.INK)
    ax.set_title(f"{len(nz)} of {len(brain.region_names)} regions have at least one",
                 fontsize=10.5, color=L.INK, loc="left")
    bx.axis("off")
    bx.set_title(f"the other {len(zr)} regions: zero", fontsize=10.5, color=L.BAD, loc="left")
    per = (len(zr) + 3) // 4
    for ci in range(4):
        for ri, n in enumerate(zr[ci * per:(ci + 1) * per]):
            bx.text(ci * .26, .95 - ri * .068, n, fontsize=7.4, transform=bx.transAxes,
                    color=L.BAD if n in W else L.GREY,
                    fontweight="bold" if n in W else "normal")
    L.despine(ax)
    fig.tight_layout()
    L.save_fig(fig, brain.cfg, "c03_census")


def _fig_fsweep(brain, C):
    plt = L.setup_matplotlib()
    fs = np.arange(.50, 1.001, .01)
    g, nr = [], []
    for f in fs:
        s, a, _ = L.ln_selection(C, f)
        g.append(s.mean() * 100)
        nr.append(len(set(a[s].tolist())))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=170)
    ax.plot(fs * 100, g, color="#16a34a", lw=2.2)
    ax.axhline(26, color=L.BAD, ls="--", lw=1.5)
    ax.text(100, 27.5, "the paper: ~26%", ha="right", fontsize=9,
            color=L.BAD, fontweight="bold")
    ax.set_ylabel("neurons judged LN  (%)", fontsize=9.5, color=L.INK)
    bx.plot(fs * 100, nr, color=L.ACC, lw=2.2)
    bx.set_ylabel("regions with at least one LN", fontsize=9.5, color=L.INK)
    for a in (ax, bx):
        a.set_xlabel("f = % of the neuron's arbour that must stay in one region",
                     fontsize=9.5, color=L.INK)
        a.set_xlim(50, 101); a.set_ylim(0, 62)
        L.despine(a)
    fig.tight_layout()
    L.save_fig(fig, brain.cfg, "c03_fsweep")


# ══════════════════════════════════════════════════════════════════
# 指令 4：密度場
# ══════════════════════════════════════════════════════════════════

def _ln_ids(brain: Brain, region: str) -> list:
    """從指令 2 的名單裡取出某一區的 LN 候選。"""
    p = brain.cfg.ensure_out() / "c02_ln_candidates.csv"
    with open(p, newline="", encoding="utf-8") as fh:
        return [brain.n2x[r["neuron"]] for r in csv.DictReader(fh) if r["region"] == region]


def step04(brain: Brain) -> dict:
    cfg = brain.cfg
    S = _ln_ids(brain, REGION)
    print(f"{REGION} 的 LN 候選 {len(S)} 顆")

    # 先在原始格點上估規模，再決定要不要併格。這一步是 AI 自己加的。
    zz, _, _ = brain.region_mask(REGION)
    n_native = len(zz)
    mem = L.estimate_memory(int(n_native * 0.25))
    print(f"原始格點：區內 {n_native:,} 個體素，熱點約 {int(n_native * 0.25):,} 個")
    print(f"  → 距離矩陣 {int(n_native * 0.25) * (int(n_native * 0.25) - 1) / 2:,.0f} 個 float64 = {mem:.0f} GB")
    if mem <= 16:
        cfg.bin_factor = 1
        print("  → 不併格")
    else:
        print(f"  → 超過 16 GB，把 {cfg.bin_factor}x{cfg.bin_factor}x{cfg.bin_factor} 個格子併成一個分析體素")

    mask, sm = L.density_field(brain, S, REGION)
    print(f"分析體素：區內 {int(mask.sum()):,} 個")
    print(f"v(r) 最大 {sm[mask].max():.1f}、平均 {sm[mask].mean():.1f}")
    print("v(r) 的定義：有幾顆不同的 LN 佔到這一格（不是纖維條數）")
    out = cfg.ensure_out()
    np.save(out / "c04_mask.npy", mask)
    np.save(out / "c04_v.npy", sm)
    r = dict(region=REGION, n_ln=len(S), B=cfg.bin_factor, n_native=n_native,
             mem_GB=round(mem), n_analysis_vox=int(mask.sum()),
             vmax=float(sm[mask].max()), vmean=float(sm[mask].mean()))
    cfg.save("c04", r)
    _fig_density(brain, mask, sm, len(S))
    return r


def _fig_density(brain, mask, sm, n_ln):
    plt = L.setup_matplotlib()
    zs, _, _ = np.nonzero(mask)
    sl = int(np.median(zs))
    m2 = mask[sl]
    v2 = np.where(m2, sm[sl], np.nan)
    ys, xs = np.nonzero(m2)
    y0, y1, x0, x1 = ys.min() - 3, ys.max() + 4, xs.min() - 3, xs.max() + 4
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), dpi=170)
    axes[0].imshow(m2[y0:y1, x0:x1], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    axes[0].set_title(f"{REGION}: the neuropil mask", fontsize=10, color=L.INK)
    im = axes[1].imshow(v2[y0:y1, x0:x1], cmap="turbo", interpolation="nearest")
    axes[1].set_title(f"v(r): LN coverage,  n = {n_ln} LNs", fontsize=10, color=L.INK)
    fig.colorbar(im, ax=axes[1], fraction=.046).ax.tick_params(labelsize=8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    L.save_fig(fig, brain.cfg, "c04_density")


# ══════════════════════════════════════════════════════════════════
# 指令 5：五數綜合與熱點
# ══════════════════════════════════════════════════════════════════

def step05(brain: Brain) -> dict:
    cfg = brain.cfg
    out = cfg.ensure_out()
    mask = np.load(out / "c04_mask.npy")
    sm = np.load(out / "c04_v.npy")
    vals = sm[mask]
    q = L.five_number(vals)
    hot = L.hotspots(mask, sm, cfg.quartile)
    print("1. 五數綜合：" + " ／ ".join(f"{x:.1f}" for x in q))
    print(f"2. 熱點 {int(hot.sum()):,} 個體素，佔區內 {hot.sum() / mask.sum() * 100:.1f}%")
    print("3. 為什麼是這個百分比：因為『上四分位』的定義就是「排到 75% 的那個位置」，")
    print("   取大於等於它的體素，剩下的必然是最上面的四分之一。跟資料長什麼樣無關。")
    np.save(out / "c05_hot.npy", hot)
    r = dict(fivenum=[round(x, 1) for x in q], n_hot=int(hot.sum()),
             pct=round(float(hot.sum() / mask.sum() * 100), 1))
    cfg.save("c05", r)
    _fig_fivenum(brain, vals, q)
    return r


def _fig_fivenum(brain, vals, q):
    plt = L.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=170)
    ax.hist(vals, bins=70, color="#c7dbff", edgecolor="#93b4f5", linewidth=.4)
    for lv, lb, col in zip(q[1:4], ["lower quartile", "median", "upper quartile  Q_U"],
                           [L.GREY, L.GREY, L.BAD]):
        ax.axvline(lv, color=col, lw=1.8 if col != L.GREY else 1.1,
                   ls="-" if col != L.GREY else "--")
        ax.text(lv, ax.get_ylim()[1] * .93, f" {lb}\n {lv:.1f}", color=col, fontsize=9,
                fontweight="bold" if col != L.GREY else "normal")
    ax.axvspan(q[3], q[4], color=L.BAD, alpha=.07)
    ax.set_xlabel(f"v(r) in {REGION}   (distinct LNs per voxel, smoothed)",
                  fontsize=10, color=L.INK)
    ax.set_ylabel("voxels", fontsize=10, color=L.INK)
    L.despine(ax)
    fig.tight_layout()
    L.save_fig(fig, brain.cfg, "c05_fivenum")


# ══════════════════════════════════════════════════════════════════
# 指令 6：UPGMA 分群
# ══════════════════════════════════════════════════════════════════

CUT_DEMO = 8.0


def step06(brain: Brain) -> dict:
    cfg = brain.cfg
    out = cfg.ensure_out()
    mask = np.load(out / "c04_mask.npy")
    hot = np.load(out / "c05_hot.npy")
    P = np.argwhere(hot)
    thr = cfg.min_cluster_frac * mask.sum()
    print(f"1. 規則 ② 的門檻：{thr:.1f} 個體素（腦區內 {int(mask.sum()):,} 個的 1%）")
    print(f"   注意分母是整個腦區（{int(mask.sum()):,}），不是熱點（{len(P):,}）")
    print(f"   距離矩陣 {len(P) * (len(P) - 1) // 2:,} 個 float64 = "
          f"{L.estimate_memory(len(P)) * 1024:.0f} MB")
    cl, sizes, keep, _ = L.cluster_hotspots(P, thr, CUT_DEMO)
    print(f"2. 高度 {CUT_DEMO:g}：規則 ① 剪出 {len(sizes)} 群，規則 ② 通過 {len(keep)} 群")
    ks = sorted(int(x) for x in sizes[np.array(keep) - 1])
    print(f"3. 各群大小：{ks}  合計 {sum(ks):,}（熱點 {len(P):,}）")
    np.save(out / "c06_cl.npy", cl)
    np.save(out / "c06_P.npy", P)
    r = dict(cut=CUT_DEMO, thr=float(thr), n_raw=int(len(sizes)), n_pass=len(keep), sizes=ks)
    cfg.save("c06", r)
    _fig_clusters(brain, mask, P, cl, keep, sizes)
    return r


def _fig_clusters(brain, mask, P, cl, keep, sizes):
    plt = L.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 7.2), dpi=170)
    mz, my, mx = np.nonzero(mask)
    sl = int(np.median(P[:, 0]))
    ax.scatter(mx[mz == sl], my[mz == sl], s=6, c="#e8ecf1")
    cm = plt.get_cmap("tab10")
    for k, c in enumerate(keep):
        p = P[(cl == c) & (P[:, 0] == sl)]
        ax.scatter(p[:, 2], p[:, 1], s=9, color=cm(k),
                   label=f"cluster {k + 1}  ({int(sizes[c - 1])})")
    ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{REGION}: hot-spot voxels, UPGMA cut = {CUT_DEMO:g}",
                 fontsize=11, color=L.INK)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    L.save_fig(fig, brain.cfg, "c06_clusters")


# ══════════════════════════════════════════════════════════════════
# 指令 7：切割高度掃描
# ══════════════════════════════════════════════════════════════════

def step07(brain: Brain) -> dict:
    cfg = brain.cfg
    plt = L.setup_matplotlib()
    rng = np.random.default_rng(cfg.rng_seed)
    res = {}
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=170)
    cm = plt.get_cmap("tab10")
    cuts = cfg.cuts
    for k, reg in enumerate(SWEEP_REGIONS):
        ids = _ln_ids(brain, reg)
        mask, sm = L.density_field(brain, ids, reg)
        P = np.argwhere(L.hotspots(mask, sm, cfg.quartile)).astype(np.float32)
        thr = cfg.min_cluster_frac * mask.sum()
        d = L.cut_sweep(P, thr, cfg, rng, detail=True)
        A = d.pop("runs")
        res[reg] = dict(n_ln=len(ids), n_hot=len(P), **d)
        if reg == REGION:
            # 這幾個數字 PART 4 引用過，留著才對得回去
            i6 = int(np.where(cuts == 6)[0][0])
            res[reg]["at6"] = sorted(A[:, i6].tolist())
            _, sz4, keep4, _ = L.cluster_hotspots(P, thr, 4.0)
            res[reg].update(h4_raw=int(len(sz4)), h4_pass=len(keep4), h4_max=int(sz4.max()))
        ax.fill_between(cuts, A.min(0), A.max(0), color=cm(k), alpha=.18, lw=0)
        ax.plot(cuts, np.median(A, 0), lw=1.8, color=cm(k), label=f"{reg} (n={len(ids)})")
        print(f"  {reg:8s} LN {len(ids):5d} 熱點 {len(P):5d} "
              f"全範圍 {d['lo']}–{d['hi']:<3d} 平台 {d['plateau'][0]}–{d['plateau'][1]} "
              f"根 {d['root']:6.2f}", flush=True)
    ax.axhline(1, color=L.GREY, ls=":", lw=1)
    r = res[REGION]
    print(f"\n高度 4：剪出 {r['h4_raw']} 群 / 通過 {r['h4_pass']} 群，"
          f"最大的一群 {r['h4_max']} 個體素")
    print(f"高度 6 打亂 {cfg.n_shuffle} 次：{r['at6']}")
    print(f"{REGION} 距離總數 {r['n_pairs']:,}，相異距離值 {r['n_distinct']}")
    print(f"{REGION} 樹根高度 {r['root']}")
    ax.set_xlabel("UPGMA cut height  (analysis voxels)", fontsize=10, color=L.INK)
    ax.set_ylabel("candidate LPUs found\n(clusters > 1% of region)", fontsize=10, color=L.INK)
    ax.set_title(f"line = median of {cfg.n_shuffle} runs;  band = min-max over shuffled voxel order",
                 fontsize=9.5, color=L.GREY, loc="left")
    ax.legend(fontsize=8.5, frameon=False, ncol=3)
    L.despine(ax)
    fig.tight_layout()
    L.save_fig(fig, cfg, "c07_cutheight")
    cfg.save("c07", res)
    print("\n有沒有『自然的』切割高度：沒有。各腦區的平台高度不一致，")
    print("而右端全部收斂到 1 只是因為切割高度超過了每一棵樹的根。")
    return res


# ══════════════════════════════════════════════════════════════════
# 指令 8：挑種子
# ══════════════════════════════════════════════════════════════════

def step08(brain: Brain) -> dict:
    cfg = brain.cfg
    out = cfg.ensure_out()
    mask = np.load(out / "c04_mask.npy")
    P = np.load(out / "c06_P.npy")
    cl = np.load(out / "c06_cl.npy")
    sizes = np.bincount(cl)[1:]
    thr = cfg.min_cluster_frac * mask.sum()
    keep = [i + 1 for i, c in enumerate(sizes) if c > thr]
    small = keep[int(np.argmin(sizes[np.array(keep) - 1]))]
    print(f"最小的候選 LPU：第 {small} 群，{int(sizes[small - 1])} 個體素")

    S = _ln_ids(brain, REGION)
    neu = L.neuron_voxel_sets(brain, S)
    pick = L.pick_seed(neu, P[cl == small])
    fr = pick["frac"]
    print(f"嚴格（該群的體素集合）：比例中位 {np.median(list(fr.values())) * 100:.1f}%、"
          f"最大 {max(fr.values()) * 100:.1f}%、"
          f"達到 100% 的 {sum(1 for v in fr.values() if v >= 0.999)} 顆")
    print("\n照字面挑不到 → 改用替代規則（比例最高的前十顆裡取體積最小的）")
    print("【替代規則，不是論文的條件】")
    seed = pick["seed"]
    print(f"選中：{brain.neurons[seed]['name']}  比例 {fr[seed] * 100:.1f}%  "
          f"體積 {pick['size'][seed]} 個分析體素")
    print("前十顆：")
    for i in pick["top"]:
        print(f"   {brain.neurons[i]['name']:22s} 比例 {fr[i] * 100:5.1f}%  "
              f"體積 {pick['size'][i]:5d}" + ("   ← 選中" if i == seed else ""))
    np.save(out / "c08_seed.npy", np.array([seed]))
    r = dict(cluster=int(small), seed=brain.neurons[seed]["name"],
             seed_frac=round(fr[seed], 4), seed_size=int(pick["size"][seed]),
             rule="替代規則：比例最高前十顆取體積最小",
             top10=[dict(name=brain.neurons[i]["name"], frac=round(fr[i], 4),
                         size=int(pick["size"][i])) for i in pick["top"]])
    cfg.save("c08", r)
    return r


# ══════════════════════════════════════════════════════════════════
# 指令 9：招募 —— 這一步在這份資料上沒有可用的操作區間
# ══════════════════════════════════════════════════════════════════

GROWS = (2, 3, 4, 5)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def step09(brain: Brain, force: bool = False) -> dict:
    cfg = brain.cfg
    out = cfg.ensure_out()
    S = _ln_ids(brain, REGION)
    M, sizes = L.neuron_masks(brain, S)
    seed = int(np.load(out / "c08_seed.npy")[0])
    ids = np.array(sorted(M))
    print(f"種子 {brain.neurons[seed]['name']}，池子 {len(ids)} 顆")

    # 先量：撐大幾格，體積會變成幾倍
    import scipy.ndimage as ndi
    st = ndi.generate_binary_structure(3, 1)
    smp = np.random.default_rng(cfg.rng_seed).choice(ids, 60, replace=False)
    fac = {k: float(np.median([ndi.binary_dilation(M[i], st, k).sum() / sizes[i] for i in smp]))
           for k in (1, 2, 3)}
    print("撐大幾格 → 體積變成幾倍（抽 60 顆的中位數）：")
    print("   " + "、".join(f"{k} 格 {v:.1f} 倍" for k, v in fac.items()))

    # 掃「撐大幾格 × 重疊門檻」，看有沒有「只收一部分」的格子
    print(f"\n招募到的顆數（池子 {len(ids)} 顆）")
    print("  膨脹\\門檻  " + "".join(f"{t:>8.2f}" for t in THRESHOLDS))
    regime = L.recruit_regime(M, sizes, seed, GROWS, THRESHOLDS)
    for g in GROWS:
        print(f"  {g:4d}      " + "".join(f"{x:>8d}" for x in regime[g]))
    partial = sum(1 for g in regime for x in regime[g] if 2 <= x <= len(ids) - 1)
    total = sum(len(v) for v in regime.values())
    print(f"\n「收一部分」＝結果落在 2 到 {len(ids) - 1} 之間的格子數：{partial} ／ {total}")
    cfg.save("c09c_regime", {str(k): v for k, v in regime.items()})
    cfg.save("c09", dict(vol_factor={str(k): round(v, 1) for k, v in fac.items()},
                         n_pool=len(ids), partial_cells=partial, total_cells=total))

    if partial == 0:
        print()
        print("┌" + "─" * 70)
        print("│ 這一步沒有可用的操作區間。")
        print(f"│ {total} 種參數組合，每一種的結果不是 1（只有種子）就是 {len(ids)}（整池）。")
        print("│ 沒有任何一種設定收到「一部分」。")
        print("│")
        print("│ 這不是參數沒調好。論文要求從「最小的」LN 出發，但一顆小的神經元")
        print("│ 撐大之後仍然裝不下一顆大的一半體積，所以要嘛收不到，要嘛一旦撐得")
        print("│ 夠大就整池全收。")
        print("│")
        print("│ 主程式在這裡停下來，不往下產出看起來完整、其實沒有意義的結果。")
        print("│ 要繼續，請加上 --force-recruit：接下來的一切都建立在")
        print(f"│ 「撐大 {cfg.grow} 格」這一種讀法上，而那是唯一跑得動的讀法，不是驗證過的。")
        print("└" + "─" * 70)
        if not force:
            return dict(halted=True, partial_cells=0, regime=regime)

    got, log = L.recruit(M, sizes, seed, cfg.grow, cfg.overlap)
    print(f"招募結束：{len(got)} 顆 / {len(ids)} 顆（{len(got) / len(ids) * 100:.0f}%），{len(log)} 輪")
    np.save(out / f"c09b_recruited_g{cfg.grow}.npy", np.array(sorted(got)))
    r = dict(halted=False, grow=cfg.grow, seed=brain.neurons[seed]["name"],
             n_recruited=len(got), n_pool=len(ids), rounds=log, partial_cells=partial)
    cfg.save(f"c09b_g{cfg.grow}", r)
    return r


# ══════════════════════════════════════════════════════════════════
# 指令 10＋11：去初級神經突、抽等值面、算 c 值
# ══════════════════════════════════════════════════════════════════

BANDS = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 65), (65, 101)]
LEVELS = (25, 50, 75)


def step10_11(brain: Brain) -> dict:
    cfg = brain.cfg
    out = cfg.ensure_out()
    from scipy import ndimage
    got = set(np.load(out / f"c09b_recruited_g{cfg.grow}.npy").tolist())
    sel = np.isin(brain.nid, list(got))
    Lv = brain.region_of_voxel[sel]
    d = brain.dep[sel]
    ri = brain.n2i[REGION]
    print(f"招募到的 {len(got)} 顆，共 {int(sel.sum()):,} 個體素")

    print("一、先量：依離本體的路徑深度分段")
    print("   深度       體素數    落在該區外   落在任何腦區外")
    for row in L.depth_profile(d, Lv, ri, BANDS):
        print(f"   {row['lo']:3d}–{row['hi']:<3d} {row['n']:10,d} "
              f"{row['outside_region'] * 100:11.1f}% {row['outside_any'] * 100:14.1f}%")
    keep = d >= cfg.depth_cut
    print(f"   → 用「深度 < {cfg.depth_cut} 丟掉」當去除規則"
          f"（這個 {cfg.depth_cut} 是依上表決定的，不是論文給的）")
    print(f"     丟掉 {int((~keep).sum()):,} 個體素（{(~keep).mean() * 100:.1f}%）")

    idx = np.flatnonzero(sel)[keep]
    kmask = np.zeros(len(brain.nid), bool); kmask[idx] = True
    kz, ky, kx = brain.binned_coords(kmask)
    n = brain.nid[kmask]
    key = np.unique(np.stack([kz, ky, kx, n]), axis=1)
    f = np.zeros(brain.binned_shape(), np.float32)
    np.add.at(f, (key[0], key[1], key[2]), 1.0)
    sm = ndimage.uniform_filter(f, cfg.smooth_size)
    reg = brain.binned_mask(REGION)
    nz = sm[sm > 0]

    print("\n二、等值面三個門檻")
    print("   門檻(百分位)   值    區域體素   佔腦區")
    res = {}
    for p in LEVELS:
        t = float(np.percentile(nz, p))
        body = ndimage.binary_fill_holes(sm >= t)
        res[p] = dict(thr=round(t, 2), n=int(body.sum()),
                      pct=round(float(body.sum() / reg.sum() * 100), 1))
        print(f"   {p:>3d}        {t:6.1f}  {body.sum():8,d}  {body.sum() / reg.sum() * 100:7.1f}%")
        res[p].update(L.c_value(body, sm, cfg.core_frac))

    print(f"\n三、c 值（中心 {cfg.core_frac:.0%} 的平均密度 ÷ 周邊的平均密度，判準 c > {cfg.c_threshold:g}）")
    print("   門檻   N_A     N_B     平均A    平均B      c     通過?")
    for p in LEVELS:
        o = res[p]
        print(f"   {p:>3d}  {o['nA']:7,d} {o['nB']:7,d} {o['meanA']:8.1f} {o['meanB']:8.1f} "
              f"{o['c']:7.3f}   {'是' if o['c'] > cfg.c_threshold else '否'}")
    print(f"\n最鬆與最緊的體積比：{res[25]['n'] / res[75]['n']:.1f} 倍")
    r = dict(depth_cut=cfg.depth_cut, levels={str(k): v for k, v in res.items()})
    cfg.save("c10_11", r)
    return r


# ══════════════════════════════════════════════════════════════════
# 指令 12＋13：長程神經束（降級版）與合併判斷
# ══════════════════════════════════════════════════════════════════

NAMED_GROUPS = {"MB_R": ["MB_CA_R", "MB_PED_R", "MB_VL_R", "MB_ML_R"],
                "MB_L": ["MB_CA_L", "MB_PED_L", "MB_VL_L", "MB_ML_L"],
                "EBLT": ["EB", "BU_R", "BU_L"]}


def step12_13(brain: Brain) -> dict:
    cfg = brain.cfg
    C, names = brain.contact, brain.names
    print("═══ 指令 12：長程神經束驗證（降級版） ═══")
    print("完整的綁束演算法需要每顆神經元在兩區的終點平均位置與最短路徑，")
    print("D06 沒有存路徑，只有體素與離本體的深度 → 無法照補充材料實作。")
    print(f"降級為：『有幾顆神經元同時碰到 {REGION} 和另一個腦區』（每區至少 {cfg.min_vox_touch} 個體素）。")
    print("與原版的差別：沒有做軌跡分群，所以算出來的是「連結對象」不是「神經束」。\n")
    touch = C >= cfg.min_vox_touch
    al = brain.n2i[REGION]
    par = touch[:, al]
    cnt = collections.Counter()
    for j in np.flatnonzero(touch[par].sum(0)):
        if j and j != al:
            cnt[names[j]] = int(touch[par][:, j].sum())
    print(f"碰到 {REGION} 的神經元 {int(par.sum()):,} 顆，其中同時碰到別區的：")
    for k, v in cnt.most_common(8):
        print(f"   {REGION} ↔ {k:10s} {v:5d} 顆")

    print("\n═══ 指令 13：合併判斷 ═══")
    c0, n0 = L.ln_by_region(C, names, cfg.f_ln)
    print("一、二、指定的三組")
    for g, mem in NAMED_GROUPS.items():
        print(f"   {g:6s} 合併前 " + "、".join(f"{m} {c0.get(m, 0)}" for m in mem))
    C2, nm = C.copy(), list(names)
    for g, mem in NAMED_GROUPS.items():
        ii = [brain.n2i[m] for m in mem]
        C2[:, ii[0]] = C[:, ii].sum(1)
        nm[ii[0]] = g
        for j in ii[1:]:
            C2[:, j] = 0
    c1, n1 = L.ln_by_region(C2, nm, cfg.f_ln)
    for g in NAMED_GROUPS:
        print(f"   {g:6s} 合併後 {c1.get(g, 0)}")
    print(f"   全腦 LN 候選 {n0} → {n1}（多了 {n1 - n0}）")

    print("\n三、全腦相鄰腦區兩兩試合併，看 LN 數增加最多的前十組")
    adj = L.adjacency(brain.lab)
    print(f"   空間相鄰的腦區配對：{len(adj)} 組")
    gain = L.merge_gain(C, names, adj, cfg.f_ln, n0)
    print("   增加最多的前十組：")
    print("   Δ全腦   合併對象                      合併後  合併前(相加)")
    top10 = []
    for g, a, b, after in gain[:10]:
        before = c0.get(a, 0) + c0.get(b, 0)
        print(f"   {g:5d}   {a:10s}+ {b:12s} {after:6d} {before:11d}")
        top10.append(dict(gain=int(g), a=a, b=b, after=int(after), before=int(before)))
    r = dict(tract_relaxed=dict(cnt.most_common(10)),
             merge_named={g: int(c1.get(g, 0)) for g in NAMED_GROUPS},
             n0=n0, n1=n1, top10=top10)
    cfg.save("c12_13", r)
    return r


# ══════════════════════════════════════════════════════════════════
# 指令 14：全腦
# ══════════════════════════════════════════════════════════════════

def step14_estimate(brain: Brain) -> dict:
    """全腦跑之前先估算。指令 14 要求估完停下來等人確認。"""
    cfg = brain.cfg
    out = cfg.ensure_out()
    with open(out / "c02_ln_candidates.csv", newline="", encoding="utf-8") as fh:
        ln = collections.Counter(r["region"] for r in csv.DictReader(fh))
    rows = []
    for nm in brain.region_names:
        coarse = int(brain.binned_mask(nm).sum())
        hot = int(coarse * 0.25)
        rows.append((nm, ln.get(nm, 0), coarse, hot, L.estimate_memory(hot)))
    rows.sort(key=lambda r: -r[4])
    print(f"全腦 {len(brain.region_names)} 區的規模估算"
          f"（分析體素 = {cfg.bin_factor}×{cfg.bin_factor}×{cfg.bin_factor} 併格後）")
    print(" 腦區        LN    分析體素    熱點   距離矩陣(GB)   會不會卡")
    big, small, ok = [], [], []
    for nm, l, c, h, m in rows:
        tag = ""
        if m > 16:
            tag = "體素太多"; big.append(nm)
        elif l < cfg.min_ln_to_cluster:
            tag = "LN 太少"; small.append(nm)
        else:
            ok.append(nm)
        if m > 1 or l < cfg.min_ln_to_cluster or nm in (REGION, "FB", "PB"):
            print(f" {nm:10s} {l:5d} {c:10,d} {h:8,d} {m:12.1f}   {tag}")
    print(f"\n分類：可以跑 {len(ok)} 區、體素太多 {len(big)} 區、LN 太少 {len(small)} 區")
    print("體素太多的：", "、".join(big) if big else "（無）")
    print(f"LN 少於 {cfg.min_ln_to_cluster} 顆的 {len(small)} 區：",
          "、".join(small[:12]), "…" if len(small) > 12 else "")
    tot = sum(m for *_, m in rows)
    print(f"\n全部 {len(rows)} 區的距離矩陣加總約 {tot:.1f} GB（不是同時，是逐區）")
    print(f"最大的一區單獨就要 {rows[0][4]:.1f} GB")
    r = dict(ok=ok, too_big=big, too_few=small,
             per_region=[dict(region=n, ln=l, vox=c, hot=h, mem_GB=round(m, 2))
                         for n, l, c, h, m in rows])
    cfg.save("c14_estimate", r)
    print("\n【依指令 14 第二項，估算完成】")
    print("處理方式：")
    print(f"  · 體素太多的區：再併一次格，距離矩陣降到 1/64")
    print(f"  · LN 少於 {cfg.min_ln_to_cluster} 顆的區：不分群，直接判定"
          f"「沒有自己的 LN 族群」→ 候選 hub")
    return r


def step14_run(brain: Brain) -> list:
    cfg = brain.cfg
    out = cfg.ensure_out()
    LN = collections.defaultdict(list)
    with open(out / "c02_ln_candidates.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            LN[r["region"]].append(brain.n2x[r["neuron"]])
    rng = np.random.default_rng(cfg.rng_seed)
    res, t0 = [], time.time()
    for nm in brain.region_names:
        n_ln = len(LN.get(nm, []))
        if n_ln < cfg.min_ln_to_cluster:
            res.append(dict(region=nm, n_ln=n_ln,
                            status=f"LN 太少（<{cfg.min_ln_to_cluster}）", verdict="候選 hub"))
            continue
        mask, sm = L.density_field(brain, LN[nm], nm)
        P = np.argwhere(L.hotspots(mask, sm, cfg.quartile)).astype(np.float32)
        d = L.cut_sweep(P, cfg.min_cluster_frac * mask.sum(), cfg, rng)
        A = d.pop("runs")
        pl = d["plateau"]
        res.append(dict(region=nm, n_ln=n_ln, n_vox=int(mask.sum()), n_hot=int(len(P)),
                        status="跑完",
                        verdict="單一 LPU" if pl[1] == 1 else f"候選 {pl[0]}–{pl[1]} 個 LPU",
                        **d))
        print(f"  {nm:10s} LN {n_ln:5d} 熱點 {len(P):5d} 全範圍 {d['lo']}–{d['hi']:<3d} "
              f"平台 {pl[0]}–{pl[1]} 根 {d['root']:6.2f}  ({time.time() - t0:.0f}s)", flush=True)
    cfg.save("c14_table", res)
    ran = [r for r in res if r["status"] == "跑完"]
    kinds = collections.Counter(
        "候選 hub" if r["status"] != "跑完" else
        ("單一 LPU" if r["verdict"] == "單一 LPU" else "多個候選") for r in res)
    print(f"\n跑完 {len(ran)} 區、略過 {len(res) - len(ran)} 區，用時 {time.time() - t0:.0f} 秒")
    print("判定分布：", collections.Counter(r["verdict"] for r in res).most_common())
    cfg.save("c14", dict(n_ran=len(ran), n_skipped=len(res) - len(ran), **kinds))
    return res


def step14_figure(brain: Brain):
    cfg = brain.cfg
    plt = L.setup_matplotlib()
    from matplotlib.patches import Patch
    tab = {r["region"]: r for r in json.loads(
        (cfg.ensure_out() / "c14_table.json").read_text(encoding="utf-8"))}
    EN = {"單一 LPU": "one LPU", "多個候選": "more than one candidate",
          "候選 hub": "no LN population (candidate hub)"}
    rgb = {"單一 LPU": (0.15, 0.39, 0.92), "多個候選": (0.85, 0.47, 0.02),
           "候選 hub": (0.58, 0.64, 0.70)}

    def cls(r):
        if r["status"] != "跑完":
            return "候選 hub"
        return "單一 LPU" if r["verdict"] == "單一 LPU" else "多個候選"
    kind = {n: cls(tab[n]) for n in brain.region_names}

    # axis0=前後、axis1=背腹、axis2=左右，是指令 1 驗出來的，不是猜的
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=170)
    views = [(0, "front view  (looking along the anterior-posterior axis)"),
             (2, "side view  (looking along the left-right axis)")]
    for ax, (axis, ttl) in zip(axes, views):
        P = L.project(brain.lab, axis)
        img = np.ones(P.shape + (3,))
        for n in brain.region_names:
            m = (P == brain.n2i[n])
            if m.any():
                img[m] = rgb[kind[n]]
        img[P == 0] = 1.0
        ax.imshow(img if axis == 0 else np.rot90(img, 1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=10, color=L.INK)
        for sp in ax.spines.values():
            sp.set_color("#c3cbd6")
    n_of = lambda k: sum(1 for n in brain.region_names if kind[n] == k)
    fig.legend(handles=[Patch(color=rgb[k], label=f"{EN[k]}  ({n_of(k)})")
                        for k in ("單一 LPU", "多個候選", "候選 hub")],
               loc="lower center", ncol=3, frameon=False, fontsize=10)
    n_ran = sum(1 for n in brain.region_names if tab[n]["status"] == "跑完")
    fig.suptitle(f"Our own whole-brain pass  ·  {len(brain.region_names)} Ito regions  ·  "
                 f"{n_ran} clustered, {len(brain.region_names) - n_ran} had "
                 f"<{cfg.min_ln_to_cluster} LN candidates",
                 fontsize=10.5, color="#5a6b7b")
    fig.tight_layout(rect=[0, .06, 1, .96])
    L.save_fig(fig, cfg, "c14_brain")
    print("判定分布：", {k: n_of(k) for k in ("單一 LPU", "多個候選", "候選 hub")})


def step14(brain: Brain):
    step14_estimate(brain)
    print()
    step14_run(brain)
    print()
    step14_figure(brain)


# ══════════════════════════════════════════════════════════════════
# 打包 —— 把要搬到另一台機器的東西壓成一份
# ══════════════════════════════════════════════════════════════════

def pack(cfg: Config, to: str) -> int:
    """把 D03 與 D06 裡真正用得到的八個檔案壓成一份可攜的資料包。

    程式本身只有幾十 KB，搬過去沒有用——這條管線真正的重量在資料。
    原始 277 MB，壓成 .npz 之後約 37 MB（標籤體積壓到 0.6%，因為它幾乎
    全是零與大片同值）。程式讀得懂 .npz，所以另一台機器不必再解開。
    """
    import shutil
    dst = Path(to)
    total_raw = total_new = 0
    for which, rel, what in REQUIRED_FILES:
        src = Path(cfg.d03 if which == "d03" else cfg.d06) / rel
        if not src.exists():
            print(f"   [缺] {src}")
            return 1
        out = dst / which.upper() / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = src.stat().st_size
        if src.suffix == ".npy":
            out = out.with_suffix(".npz")
            np.savez_compressed(out, np.load(src))
        else:
            shutil.copy2(src, out)
        new = out.stat().st_size
        total_raw += raw
        total_new += new
        print(f"   {src.name:<24} {raw / 2 ** 20:8.1f} MB → {new / 2 ** 20:7.1f} MB   {what}")
    print(f"   {'合計':<24} {total_raw / 2 ** 20:8.1f} MB → {total_new / 2 ** 20:7.1f} MB "
          f"（{total_new / total_raw * 100:.0f}%）")
    print(f"\n資料包在 {dst.resolve()}")
    print("到另一台機器之後：")
    print(f"    ./run_lpu.sh <資料包>/D03 <資料包>/D06")
    return 0


# ══════════════════════════════════════════════════════════════════
# 核對 —— 把 PART 3 的「怎麼確認它做對了」變成可以執行的
# ══════════════════════════════════════════════════════════════════

def verify(cfg: Config) -> int:
    """比對 out/ 裡的結果與 expected.json。

    PART 4 最大的一次教訓是「驗收表比指令本身有用」：指令 4 的表對不上，
    才追出密度場的定義寫錯了。所以把那些表留在這裡，而不是只留在紙上。
    """
    spec = json.loads((Path(__file__).parent / "expected.json").read_text(encoding="utf-8"))
    cache, bad, miss, ok = {}, [], [], 0
    for ck in spec["checks"]:
        head, *rest = ck["path"].split("/")
        if head not in cache:
            p = cfg.ensure_out() / f"{head}.json"
            cache[head] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        node = cache[head]
        for k in rest:
            if node is None:
                break
            node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            miss.append(ck)
            continue
        want, tol = ck["want"], ck.get("tol")
        if tol is not None:
            a = node if isinstance(node, list) else [node]
            b = want if isinstance(want, list) else [want]
            good = len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))
        else:
            good = node == want
        if good:
            ok += 1
        else:
            bad.append((ck, node))

    for ck, got in bad:
        print(f"  [不符] 指令 {ck['指令']:>2}  {ck['說明']}")
        print(f"         應為 {ck['want']}，實得 {got}")
    for ck in miss:
        print(f"  [沒跑] 指令 {ck['指令']:>2}  {ck['說明']}  （缺 {ck['path'].split('/')[0]}.json）")
    total = len(spec["checks"])
    print(f"\n核對 {total} 項：通過 {ok}、不符 {len(bad)}、還沒跑 {len(miss)}")
    if miss and not bad:
        print("（先把還沒跑的步驟跑完，再核對一次）")
    if not bad and not miss:
        print("全部通過。")
    return len(bad)


# ══════════════════════════════════════════════════════════════════
# 排程
# ══════════════════════════════════════════════════════════════════

STEPS = {
    "1": ("盤點資料、確認座標框與單位", step01),
    "2": ("把目視判準翻成比例規則", step02),
    "3": ("全腦 LN 普查與 f 敏感度", step03),
    "4": (f"{REGION} 的密度場 v(r)", step04),
    "5": ("五數綜合、取上四分位", step05),
    "6": ("UPGMA 分群與 1% 過濾", step06),
    "7": ("切割高度掃描 × 打亂順序", step07),
    "8": ("挑招募的起始種子", step08),
    "9": ("滾雪球招募（含操作區間掃描）", None),   # 需要 force 參數，特別處理
    "10": ("去初級神經突、等值面、c 值", step10_11),
    "12": ("長程神經束驗證與合併判斷", step12_13),
    "14": ("全腦逐區跑一遍", step14),
}
ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="LPU 管線：PART 3 的十四條指令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="步驟：" + "  ".join(f"{k}={v[0]}" for k, v in STEPS.items()))
    ap.add_argument("steps", nargs="*", default=["all"],
                    help="要跑的步驟編號，或 all／check／verify／pack")
    ap.add_argument("--d03", default=os.environ.get("LPU_D03"), help="D03 標準腦與分區的目錄")
    ap.add_argument("--d06", default=os.environ.get("LPU_D06"), help="D06 骨架體素的目錄")
    ap.add_argument("--out", default=os.environ.get("LPU_OUT", "out"), help="輸出目錄")
    ap.add_argument("--pack-to", help="配合 pack：資料包要放到哪個目錄")
    ap.add_argument("--force-recruit", action="store_true",
                    help="指令 9 沒有操作區間時仍往下跑（結果只是一種讀法）")
    a = ap.parse_args(argv)

    cfg = Config(out=Path(a.out))
    if a.d03:
        cfg.d03 = Path(a.d03)
    if a.d06:
        cfg.d06 = Path(a.d06)

    if "check" in a.steps:
        return 1 if check(cfg) else 0
    if "verify" in a.steps:
        return 1 if verify(cfg) else 0
    if "pack" in a.steps:
        if not a.pack_to:
            print("pack 需要 --pack-to <目錄>")
            return 2
        return pack(cfg, a.pack_to)

    want = ORDER if a.steps == ["all"] or "all" in a.steps else a.steps
    bad = [s for s in want if s not in STEPS]
    if bad:
        print(f"不認得的步驟：{bad}。可用的是 {list(STEPS)} 或 all／check")
        return 2

    print("=" * 72)
    print(cfg.describe())
    print("=" * 72)
    brain = Brain(cfg)
    t0 = time.time()
    for s in want:
        title, fn = STEPS[s]
        print(f"\n{'━' * 72}\n指令 {s}：{title}\n{'━' * 72}")
        if s == "9":
            r = step09(brain, force=a.force_recruit)
            if r.get("halted"):
                print("\n（`all` 在指令 9 停下。加 --force-recruit 才會往下跑。）")
                return 3
        else:
            fn(brain)
    print(f"\n{'=' * 72}\n全部完成，用時 {time.time() - t0:.0f} 秒。輸出在 {cfg.out.resolve()}")
    if want is ORDER:                     # 只有整條跑完才核對
        print(f"{'━' * 72}\n核對\n{'━' * 72}")
        return 1 if verify(cfg) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
