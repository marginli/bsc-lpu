#!/usr/bin/env python3
"""PART 2 的真實資料插圖：把論文的前三個步驟在本機資料上實際跑一次。

資料來源（皆為本機自製，可公開）：
  D03  標準腦分區（FCWBNP／Ito 2014），以及搬到 FC12_warp 座標框的標籤體積
  D06  28,573 顆 FlyCircuit 骨架的 2 µm 體素化結果與神經元→腦區對照表

「LN 候選」的定義：該神經元落在腦區內的體素，有 F_LN 以上集中在同一個腦區。
論文的原句是 processes are restricted within a single brain region，
用比例而不是體素數，是因為比例不受神經元大小影響。
分區用的是 Ito 2014 的 75 區，不是 Chiang 2011 的 58 區。

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
F_LN = 0.80                # LN 候選：至少八成的分枝要待在同一個腦區
INK, ACC, GREY = "#1b2733", "#2563eb", "#8b98a5"


def load():
    lab = np.load(os.path.join(D03, "work", "labels_fc_1um.npy"))
    names = ["Exterior"] + json.load(open(os.path.join(D03, "D03_labels_meta.json")))["names"]
    rows = list(csv.DictReader(open(os.path.join(D06, "work", "neurons.csv"))))
    vox = np.load(os.path.join(D06, "work", "vox.npy"))
    nid = np.load(os.path.join(D06, "work", "nid.npy"))
    n2l = {r["neuron"]: r for r in csv.DictReader(open(os.path.join(D06, "D06_neuron_to_LPU.csv")))}
    return lab, names, {n: i for i, n in enumerate(names)}, rows, vox, nid, n2l


def ln_table(lab, names, rows, vox, nid, f=F_LN):
    """回傳 {腦區名: [神經元索引…]}：至少 f 比例的分枝待在該區的神經元。"""
    C = neuron_region_counts(lab, names, rows, vox, nid)
    tot = C.sum(1); top = C.max(1); arg = C.argmax(1)
    out = {}
    for i in np.flatnonzero((tot > 0) & (top >= f * tot)):
        out.setdefault(names[arg[i]], []).append(int(i))
    return out


LNTAB = {}                 # 腦區 -> LN 候選的神經元索引；main() 裡建好


def ln_of(region, *_):
    """該腦區的 LN 候選（索引串列）。"""
    return LNTAB.get(region, [])


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
def fig_census(names):
    """全部 75 個腦區的 LN 候選普查——不是抽樣。"""
    c = {n: len(v) for n, v in LNTAB.items()}
    nz = sorted([(n, c[n]) for n in names if c.get(n, 0) > 0], key=lambda t: t[1])
    zero = [n for n in names if c.get(n, 0) == 0]
    WATCH = {"MB_CA_R", "MB_CA_L", "MB_PED_R", "MB_PED_L", "MB_VL_R", "MB_VL_L",
             "MB_ML_R", "MB_ML_L", "EB", "AOTU_L", "AOTU_R", "NO"}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 7.2), dpi=170,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    ys = range(len(nz))
    ax.barh(list(ys), [v for _, v in nz],
            color=["#d97706" if n in WATCH else ACC for n, _ in nz], height=.72)
    for i, (n, v) in enumerate(nz):
        ax.text(v * 1.14, i, str(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys)); ax.set_yticklabels([n for n, _ in nz], fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(.7, 4000)
    ax.set_xlabel(f"neurons with \u2265{F_LN:.0%} of their arbour in this region  (log scale)",
                  fontsize=9.5, color=INK)
    ax.set_title(f"{len(nz)} of {len(names)} regions have at least one",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)

    bx.axis("off")
    bx.set_title(f"the other {len(zero)} regions: zero",
                 fontsize=10.5, color="#dc2626", loc="left")
    per = (len(zero) + 3) // 4
    for ci in range(4):
        for ri, n in enumerate(zero[ci * per:(ci + 1) * per]):
            bx.text(ci * .26, .95 - ri * .068, n, fontsize=7.4, transform=bx.transAxes,
                    color="#dc2626" if n in WATCH else GREY,
                    fontweight="bold" if n in WATCH else "normal")
    bx.text(0, -.02, "orange / red = marked \u2212 in the paper's Figure 5A",
            fontsize=8, transform=bx.transAxes, color=GREY)
    fig.tight_layout(); fig.savefig(f"{OUT}/p2-ln-census.png"); plt.close(fig)
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


def fig_criterion(lab, names, rows, vox, nid):
    """比例判準的敏感度：f 從 50% 掃到 100%。"""
    C = neuron_region_counts(lab, names, rows, vox, nid)
    tot = C.sum(1); top = C.max(1); arg = C.argmax(1); big = tot > 0
    fs = np.unique(np.append(np.linspace(.5, 1.0, 60), F_LN))
    g = np.array([(big & (top >= f * tot)).mean() * 100 for f in fs])
    nreg = np.array([len({arg[i] for i in np.flatnonzero(big & (top >= f * tot))})
                     for f in fs])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=170)
    for a_, y, lab_, col in ((ax, g, "neurons judged LN  (%)", "#16a34a"),
                             (bx, nreg, "regions with at least one LN", ACC)):
        a_.plot(fs * 100, y, color=col, lw=2.2)
        i = int(np.argmin(abs(fs - F_LN)))
        a_.plot([F_LN * 100], [y[i]], "o", color=col, ms=8)
        txt = f"f = {F_LN:.0%}\n{y[i]:.1f}%" if col == "#16a34a" else f"f = {F_LN:.0%}\n{y[i]:.0f} regions"
        a_.annotate(txt, (F_LN * 100, y[i]), textcoords="offset points",
                    xytext=(-14, -34), ha="right", fontsize=9.5, color=INK,
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
        a_.set_xlabel("f = % of the neuron's arbour that must stay in one region",
                      fontsize=9.5, color=INK)
        a_.set_ylabel(lab_, fontsize=9.5, color=INK)
        a_.set_xlim(50, 101); style(a_)
    ax.axhline(26, color="#dc2626", lw=1.5, ls="--")
    ax.text(100, 27.5, "the paper: ~26%  (crossed at f \u2248 78%)", ha="right",
            fontsize=9, color="#dc2626", fontweight="bold")
    ax.set_ylim(0, 62); bx.set_ylim(0, 62)
    fig.tight_layout(); fig.savefig(f"{OUT}/p2-criterion.png"); plt.close(fig)
    i = int(np.argmin(abs(fs - F_LN)))
    print("寫出 p2-criterion.png：f=%.0f%% → %.1f%% 的神經元、%d 個腦區"
          % (F_LN * 100, g[i], nreg[i]))


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
    LNTAB.update(ln_table(lab, names, rows, vox, nid))
    fig_census(names[1:])
    fig_criterion(lab, names, rows, vox, nid)
    fig_density(lab, name2id, rows, vox, nid, n2l)
    fig_fivenum(lab, name2id, rows, vox, nid, n2l)
    fig_cutheight(lab, name2id, rows, vox, nid, n2l)
    fig_clusters(lab, name2id, rows, vox, nid, n2l)


if __name__ == "__main__":
    main()
