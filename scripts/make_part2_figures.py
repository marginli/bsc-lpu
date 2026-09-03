#!/usr/bin/env python3
"""PART 2 的真實資料插圖：把論文的前三個步驟在本機資料上實際跑一次。

資料來源（皆為本機自製，可公開）：
  D03  標準腦分區（FCWBNP／Ito 2014），以及搬到 FC12_warp 座標框的標籤體積
  D06  28,573 顆 FlyCircuit 骨架的 2 µm 體素化結果與神經元→腦區對照表

「LN 候選」的定義：D06 的 n_LPU_touched == 1，
即該神經元的體素只落在單一個腦區內——這正是論文對 local neuron 的判準
（processes are restricted within a single brain region），
只是分區用的是 Ito 2014 的 75 區，不是 Chiang 2011 的 58 區。

座標框注意：FC12_warp 的一格不是一個真實微米。
D03 的 affine 行列式 0.154，奇異值 0.626／0.605／0.406，
所以 FC 的 1 格約等於 0.4–0.63 個 FCWB 微米。本腳本一律以「格」為單位。

圖內不放中文，說明寫在 HTML 圖說裡。
"""
import csv, json, os
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D03 = "/home/wanjuli/claude_linux/BSC_plan/D03"
D06 = "/home/wanjuli/claude_linux/BSC_plan/D06"
OUT = "/home/wanjuli/claude_linux/BSC_plan/specific_topics/LPU/assets"
GX, GY = 512, 512
B = 4                      # 分析體素 = 4 個 FC 格，控制距離矩陣大小
INK, ACC, GREY = "#1b2733", "#2563eb", "#8b98a5"


def load():
    lab = np.load(os.path.join(D03, "work", "labels_fc_1um.npy"))
    names = ["Exterior"] + json.load(open(os.path.join(D03, "D03_labels_meta.json")))["names"]
    rows = list(csv.DictReader(open(os.path.join(D06, "work", "neurons.csv"))))
    vox = np.load(os.path.join(D06, "work", "vox.npy"))
    nid = np.load(os.path.join(D06, "work", "nid.npy"))
    n2l = {r["neuron"]: r for r in csv.DictReader(open(os.path.join(D06, "D06_neuron_to_LPU.csv")))}
    return lab, names, {n: i for i, n in enumerate(names)}, rows, vox, nid, n2l


def ln_of(region, rows, n2l):
    """該腦區的 LN 候選：體素只落在這一個腦區裡的神經元。"""
    return [i for i, r in enumerate(rows)
            if (m := n2l.get(r["name"])) and m["n_LPU_touched"] == "1"
            and m["primary_LPU"] == region]


def field_of(region, lab, name2id, rows, vox, nid, n2l):
    """回傳 (遮罩, 平滑後的 v(r), LN 候選數)。"""
    sh = tuple(s // B + 1 for s in lab.shape)
    mask = np.zeros(sh, bool)
    zz, yy, xx = np.nonzero(lab == name2id[region])
    mask[zz // B, yy // B, xx // B] = True
    S = ln_of(region, rows, n2l)
    f = np.zeros(sh, np.float32)
    if S:
        sel = np.isin(nid, S); v = vox[sel]
        iz = (v // (GX * GY)) * 2 // B; iy = ((v // GX) % GY) * 2 // B; ix = (v % GX) * 2 // B
        np.add.at(f, (iz, iy, ix), 1.0)
    return mask, ndimage.uniform_filter(f, size=3), len(S)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY); ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_facecolor("white")


# ── 圖 A：各腦區的 LN 候選數 ────────────────────────────────────────────
def fig_census(rows, n2l, names):
    """全部 75 個腦區的 LN 候選普查——不是抽樣。"""
    import collections
    c = collections.Counter(m["primary_LPU"] for r in rows
                            if (m := n2l.get(r["name"])) and m["n_LPU_touched"] == "1")
    nz = sorted([(n, c[n]) for n in names if c.get(n, 0) > 0], key=lambda t: t[1])
    zero = [n for n in names if c.get(n, 0) == 0]
    # 圖 5A 上被標成「−」而我們也拿來對照的幾個
    WATCH = {"MB_CA_R", "MB_CA_L", "MB_PED_R", "MB_PED_L", "MB_VL_R", "MB_VL_L",
             "MB_ML_R", "MB_ML_L", "EB", "AOTU_L", "AOTU_R", "NO"}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 6.6), dpi=170,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    ys = range(len(nz))
    ax.barh(list(ys), [v for _, v in nz],
            color=["#d97706" if n in WATCH else ACC for n, _ in nz], height=.7)
    for i, (n, v) in enumerate(nz):
        ax.text(v * 1.14, i, str(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _ in nz], fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(.7, 1600)
    ax.set_xlabel("neurons confined to this region  (log scale)", fontsize=9.5, color=INK)
    ax.set_title(f"{len(nz)} of {len(names)} regions have at least one",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)

    bx.axis("off")
    bx.set_title(f"the other {len(zero)} regions: zero",
                 fontsize=10.5, color="#dc2626", loc="left")
    per = (len(zero) + 3) // 4
    cols = [zero[i * per:(i + 1) * per] for i in range(4)]
    for ci, col in enumerate(cols):
        for ri, n in enumerate(col):
            bx.text(ci * .26, .93 - ri * .073, n, fontsize=7.4, transform=bx.transAxes,
                    color="#dc2626" if n in WATCH else GREY,
                    fontweight="bold" if n in WATCH else "normal")
    bx.text(0, -.02, "orange / red = marked \u2212 in the paper's Figure 5A",
            fontsize=8, transform=bx.transAxes, color=GREY)
    fig.tight_layout()
    fig.savefig(f"{OUT}/p2-ln-census.png"); plt.close(fig)
    print("寫出 p2-ln-census.png：非零 %d／%d，LN 候選共 %d"
          % (len(nz), len(names), sum(v for _, v in nz)))


# ── 圖 A2：LN 比例對「進入腦區」門檻的敏感度 ─────────────────────────────
# Chiang 是目視判讀，我們是設一個體素門檻。這張圖量的就是這個代用品的代價。
MERGE = {                       # Ito 明顯把 Chiang 的一區切開的幾組
    "MB":   ["MB_CA", "MB_PED", "MB_VL", "MB_ML"],   # Chiang: Cal + MB
    "SOG":  ["GNG", "SAD", "PRW", "FLA", "CAN"],     # Chiang: SOG
    "VLP":  ["AVLP", "PVLP"],                        # Chiang: VLP
    "EBLT": ["EB", "BU", "GA"],                      # Chiang: EB + Lat Tri（BU＝舊稱側三角）
}


def neuron_region_counts(lab, names, rows, vox, nid):
    """神經元 × 腦區的體素數表。"""
    R, N = len(names), len(rows)
    ix = vox % GX; iy = (vox // GX) % GY; iz = vox // (GX * GY)
    L = np.zeros(len(vox), np.int16)
    ok = ((2*iz < lab.shape[0]) & (2*iy < lab.shape[1]) & (2*ix < lab.shape[2]))
    L[ok] = lab[2*iz[ok], 2*iy[ok], 2*ix[ok]]
    k = nid.astype(np.int64) * R + L
    C = np.bincount(k, minlength=N * R).reshape(N, R); C[:, 0] = 0
    return C


def coarsen(C, names):
    """把 MERGE 列出的組併起來，逼近 Chiang 的分區粗細。"""
    n2i = {n: i for i, n in enumerate(names)}
    C2 = C.copy(); drop = set()
    for mem in MERGE.values():
        for side in ("_R", "_L", ""):
            idx = [n2i[m + side] for m in mem if m + side in n2i]
            if len(idx) < 2: continue
            C2[:, idx[0]] = C[:, idx].sum(1); drop |= set(idx[1:])
    keep = [i for i in range(C.shape[1]) if i not in drop]
    return C2[:, keep], len(keep) - 1


def fig_threshold(lab, names, rows, vox, nid):
    """兩種 LN 代用判準的參數敏感度。左：絕對體素門檻。右：比例門檻。"""
    C = neuron_region_counts(lab, names, rows, vox, nid)
    Cc, kc = coarsen(C, names)
    ts = np.unique(np.round(np.logspace(0, np.log10(400), 90)).astype(int))
    f75 = np.array([((C  >= t).sum(1) == 1).mean() * 100 for t in ts])
    f61 = np.array([((Cc >= t).sum(1) == 1).mean() * 100 for t in ts])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=170,
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    ax.axhline(26, color="#dc2626", lw=1.5, ls="--")
    ax.text(390, 27.2, "the paper: ~26%", ha="right", fontsize=9,
            color="#dc2626", fontweight="bold")
    ax.plot(ts, f75, color=ACC, lw=2.2, label=f"{len(names)-1} Ito regions")
    ax.plot(ts, f61, color="#d97706", lw=2.2, label=f"{kc} regions (merged toward 58)")
    pk = ts[f75.argmax()]
    ax.plot([pk], [f75.max()], "v", color=ACC, ms=8)
    ax.text(pk, f75.max() + 2.5, f"peak {f75.max():.0f}% at t\u2248{pk}",
            ha="center", fontsize=9, color=ACC, fontweight="bold")
    ax.plot([5], [f75[list(ts).index(5)]], "o", color=ACC, ms=7)
    ax.annotate("our figure: 9.3%\n(t = 5)", (5, f75[list(ts).index(5)]),
                textcoords="offset points", xytext=(6, -34), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    # 26% 被穿過兩次
    for i in range(1, len(ts)):
        if (f75[i-1] - 26) * (f75[i] - 26) < 0:
            ax.plot([ts[i]], [26], "x", color="#dc2626", ms=9, mew=2.2)
    ax.set_xscale("log"); ax.set_xlim(1, 400); ax.set_ylim(0, 58)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("t = voxels required to count as entering a region  (log)", fontsize=9.5, color=INK)
    ax.set_ylabel("neurons confined to one region  (%)", fontsize=9.5, color=INK)
    ax.set_title("an absolute voxel threshold", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)

    tot = C.sum(1); top = C.max(1); big = tot > 0
    fs = np.linspace(.5, 1.0, 60)
    g = [((big & (top >= f * tot)).mean()) * 100 for f in fs]
    bx.axhline(26, color="#dc2626", lw=1.5, ls="--")
    bx.plot(fs * 100, g, color="#16a34a", lw=2.2)
    for f in (.95, 1.0):
        v = (big & (top >= f * tot)).mean() * 100
        bx.plot([f * 100], [v], "o", color="#16a34a", ms=6)
        bx.text(f * 100 - 1, v + 2, f"{v:.1f}%", ha="right", fontsize=9,
                color="#16a34a", fontweight="bold")
    bx.set_xlim(50, 101); bx.set_ylim(0, 58)
    bx.set_xlabel("f = % of the neuron's arbour that must stay in one region", fontsize=9.5, color=INK)
    bx.set_title("a proportional threshold", fontsize=10.5, color=INK, loc="left")
    style(bx)

    fig.tight_layout(); fig.savefig(f"{OUT}/p2-threshold.png"); plt.close(fig)
    cross = [int(ts[i]) for i in range(1, len(ts)) if (f75[i-1]-26)*(f75[i]-26) < 0]
    print("寫出 p2-threshold.png：t=5 → %.1f%%，峰值 %.1f%% 在 t≈%d，穿過 26%% 的 t = %s"
          % (f75[list(ts).index(5)], f75.max(), pk, cross))


# ── 圖 B：v(r) 的空間分布 ───────────────────────────────────────────────
def fig_density(lab, name2id, rows, vox, nid, n2l, region="AL_R"):
    mask, sm, n = field_of(region, lab, name2id, rows, vox, nid, n2l)
    zz, yy, xx = np.nonzero(mask)
    sl = int(np.median(zz))
    m2, v2 = mask[sl], np.where(mask[sl], sm[sl], np.nan)
    ys, xs = np.nonzero(m2)
    y0, y1, x0, x1 = ys.min() - 3, ys.max() + 4, xs.min() - 3, xs.max() + 4
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), dpi=170)
    axes[0].imshow(m2[y0:y1, x0:x1], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    axes[0].set_title(f"{region}: the neuropil mask", fontsize=10, color=INK)
    im = axes[1].imshow(v2[y0:y1, x0:x1], cmap="turbo", interpolation="nearest")
    axes[1].set_title(f"v(r): LN coverage,  n = {n} LNs", fontsize=10, color=INK)
    fig.colorbar(im, ax=axes[1], fraction=.046).ax.tick_params(labelsize=8, colors=GREY)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color(GREY)
    fig.tight_layout(); fig.savefig(f"{OUT}/p2-density.png"); plt.close(fig)
    print("寫出 p2-density.png")


# ── 圖 C：五數綜合與上四分位 ────────────────────────────────────────────
def fig_fivenum(lab, name2id, rows, vox, nid, n2l, region="AL_R"):
    mask, sm, n = field_of(region, lab, name2id, rows, vox, nid, n2l)
    vals = sm[mask]
    q = np.percentile(vals, [0, 25, 50, 75, 100])
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=170)
    ax.hist(vals, bins=70, color="#c7dbff", edgecolor="#93b4f5", linewidth=.4)
    for lv, lb, col in zip(q[1:4], ["Q1", "median", "Q_U (upper quartile)"],
                           [GREY, GREY, "#dc2626"]):
        ax.axvline(lv, color=col, lw=1.8 if col != GREY else 1.1,
                   ls="-" if col != GREY else "--")
        ax.text(lv, ax.get_ylim()[1] * .93, f" {lb}\n {lv:.1f}", color=col,
                fontsize=9, fontweight="bold" if col != GREY else "normal")
    ax.axvspan(q[3], q[4], color="#dc2626", alpha=.07)
    ax.set_xlabel(f"v(r) in {region}   (smoothed LN coverage per voxel)", fontsize=10, color=INK)
    ax.set_ylabel("voxels", fontsize=10, color=INK)
    style(ax); fig.tight_layout()
    fig.savefig(f"{OUT}/p2-fivenum.png"); plt.close(fig)
    print("寫出 p2-fivenum.png  五數綜合", np.round(q, 2))


# ── 圖 D：切割高度決定答案（核心圖）────────────────────────────────────
def fig_cutheight(lab, name2id, rows, vox, nid, n2l):
    regions = ["AL_R", "AL_L", "AVLP_R", "AVLP_L", "FB", "PB"]
    cuts = np.arange(4, 26, 1.0)
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=170)
    cmap = plt.get_cmap("tab10")
    res = {}
    for k, region in enumerate(regions):
        mask, sm, n = field_of(region, lab, name2id, rows, vox, nid, n2l)
        q = np.percentile(sm[mask], 75)
        Z = np.argwhere(mask & (sm >= q)).astype(np.float32)
        L = linkage(pdist(Z), method="average")
        thr = 0.01 * mask.sum()
        ys = []
        for c in cuts:
            cl = fcluster(L, t=c, criterion="distance")
            ys.append(int((np.bincount(cl)[1:] > thr).sum()))
        res[region] = ys
        ax.plot(cuts, ys, "-o", ms=3.2, lw=1.7, color=cmap(k), label=f"{region} (n={n})")
    ax.set_xlabel("UPGMA cut height  (analysis voxels)", fontsize=10, color=INK)
    ax.set_ylabel("candidate LPUs found\n(clusters > 1% of region)", fontsize=10, color=INK)
    ax.axhline(1, color=GREY, ls=":", lw=1)
    ax.legend(fontsize=8.5, frameon=False, ncol=3)
    style(ax); fig.tight_layout()
    fig.savefig(f"{OUT}/p2-cutheight.png"); plt.close(fig)
    print("寫出 p2-cutheight.png")
    for r, ys in res.items():
        print(f"   {r:8s} 群數範圍 {min(ys)}–{max(ys)}")
    return res


# ── 圖 E：熱點分群的樣子 ────────────────────────────────────────────────
def fig_clusters(lab, name2id, rows, vox, nid, n2l, region="AL_R", cut=8.0):
    mask, sm, n = field_of(region, lab, name2id, rows, vox, nid, n2l)
    q = np.percentile(sm[mask], 75)
    Z = np.argwhere(mask & (sm >= q))
    L = linkage(pdist(Z.astype(np.float32)), method="average")
    cl = fcluster(L, t=cut, criterion="distance")
    keep = np.flatnonzero(np.bincount(cl)[1:] > 0.01 * mask.sum()) + 1
    zz, yy, xx = np.nonzero(mask)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=170)
    ax.scatter(xx, -yy, s=1.2, c="#e4e8ee", linewidths=0)
    cmap = plt.get_cmap("tab10")
    for j, c in enumerate(keep):
        p = Z[cl == c]
        ax.scatter(p[:, 2], -p[:, 1], s=2.4, color=cmap(j % 10),
                   linewidths=0, label=f"cluster {j+1}  ({len(p)})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{region}: hot-spot voxels, UPGMA cut = {cut:g}", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ax.spines.values(): s.set_color(GREY)
    fig.tight_layout(); fig.savefig(f"{OUT}/p2-clusters.png"); plt.close(fig)
    print(f"寫出 p2-clusters.png  {region} 在 cut={cut} 得到 {len(keep)} 群")


def main():
    lab, names, name2id, rows, vox, nid, n2l = load()
    fig_census(rows, n2l, names[1:])
    fig_threshold(lab, names, rows, vox, nid)
    fig_density(lab, name2id, rows, vox, nid, n2l)
    fig_fivenum(lab, name2id, rows, vox, nid, n2l)
    fig_cutheight(lab, name2id, rows, vox, nid, n2l)
    fig_clusters(lab, name2id, rows, vox, nid, n2l)


if __name__ == "__main__":
    main()
