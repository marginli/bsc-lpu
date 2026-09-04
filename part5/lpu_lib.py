#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lpu_lib.py — LPU 管線的設定、資料物件與函式庫

這個檔案分成四段：

  一、Config     整條管線的所有可調參數。全部集中在這裡。
  二、Brain      持有資料（分區標籤、神經元骨架體素），並負責座標換算。
  三、七個步驟   論文〈Defining an LPU〉那七步，一步一個函式。
  四、畫圖       出圖的輔助函式。圖內一律不放中文。

配套檔案：run_lpu.py（主程式）、run_lpu.sh（在終端機一鍵跑完）。
"""
from __future__ import annotations

import collections
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


# ══════════════════════════════════════════════════════════════════
# 一、Config —— 整條管線的所有決定
# ══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """管線的全部參數。

    分成三組：論文明確給定的、論文沒給而我們自己決定的、以及資料格式常數。
    第二組就是 PART 4〈結論五〉那張表——**整條管線的結論取決於這幾個數字**，
    所以把它們集中在這裡，而不是散在程式各處。
    """

    # ── 資料在哪。換一台機器時，通常只要改這三行（或用命令列參數覆寫）──
    d03: Path = Path("/home/wanjuli/claude_linux/BSC_plan/D03")
    d06: Path = Path("/home/wanjuli/claude_linux/BSC_plan/D06")
    out: Path = Path("out")

    # ── 論文明確給定的參數 ──
    quartile: float = 75.0          # 步驟 3：取上四分位當熱點
    smooth_size: int = 3            # 步驟 2：3x3x3 移動平均
    min_cluster_frac: float = 0.01  # 步驟 4：群要大於腦區體積的 1%
    overlap: float = 0.50           # 步驟 5：重疊超過 50% 才招募
    core_frac: float = 0.80         # 步驟 7：中心 80% 對周邊 20%
    c_threshold: float = 2.0        # 步驟 7：c > 2 才算 LPU

    # ── 論文沒給、我們自己決定的（PART 4 結論五）──
    f_ln: float = 0.80              # 指令 2：多少比例的分枝待在同一區才算 LN
    bin_factor: int = 4             # 指令 4：幾格併成一個分析體素
    n_shuffle: int = 9              # 指令 7：打亂體素順序重跑幾次
    grow: int = 4                   # 指令 9：招募時撐大幾格（2 格招不到任何一顆）
    depth_cut: int = 10             # 指令 10：離本體幾格以內視為初級神經突
    min_ln_to_cluster: int = 10     # 指令 14：LN 少於幾顆就不分群
    min_vox_touch: int = 5          # 指令 12：碰到幾個體素才算「碰到」這一區
    rng_seed: int = 0               # 打亂順序用的亂數種子

    # 切割高度：論文沒給。掃 4 到 25，平台取 14–20。
    cut_lo: float = 4.0
    cut_hi: float = 26.0
    cut_step: float = 1.0
    plateau: tuple = (14.0, 20.0)

    # ── 資料格式常數（跟著 D03／D06 走，不要改）──
    d06_grid: int = 512             # D06 體素編號的 X、Y 邊長
    d06_step: int = 2               # 一個 D06 體素等於幾個 1 µm 標籤格

    # ---------------------------------------------------------------
    @property
    def cuts(self) -> np.ndarray:
        return np.arange(self.cut_lo, self.cut_hi, self.cut_step)

    @property
    def plateau_slice(self) -> np.ndarray:
        c = self.cuts
        return (c >= self.plateau[0]) & (c <= self.plateau[1])

    def describe(self) -> str:
        """把「我們自己決定的」那一組印出來。每次跑都印，免得忘記。"""
        rows = [
            ("f_ln", self.f_ln, "指令 2  LN 判準的比例門檻"),
            ("bin_factor", self.bin_factor, "指令 4  幾格併成一個分析體素"),
            ("n_shuffle", self.n_shuffle, "指令 7  打亂順序重跑幾次"),
            ("cut range", f"{self.cut_lo:g}–{self.cut_hi - self.cut_step:g}", "指令 7  切割高度掃描範圍"),
            ("plateau", f"{self.plateau[0]:g}–{self.plateau[1]:g}", "指令 7  取哪一段當平台"),
            ("grow", self.grow, "指令 9  招募時撐大幾格"),
            ("depth_cut", self.depth_cut, "指令 10 深度小於此值視為初級神經突"),
            ("min_ln_to_cluster", self.min_ln_to_cluster, "指令 14 少於幾顆 LN 就不分群"),
            ("min_vox_touch", self.min_vox_touch, "指令 12 碰到幾個體素才算碰到"),
            ("rng_seed", self.rng_seed, "亂數種子"),
        ]
        w = max(len(r[0]) for r in rows)
        head = "論文沒有給、由我們自己決定的參數（換掉任何一個，結論就可能改變）"
        body = "\n".join(f"  {k:<{w}}  = {v!s:<8} {note}" for k, v, note in rows)
        return f"{head}\n{body}"

    def ensure_out(self) -> Path:
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        return self.out

    def save(self, name: str, obj) -> None:
        """把結果寫成 JSON。所有步驟都用這個，格式才會一致。"""
        p = self.ensure_out() / f"{name}.json"
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# 二、Brain —— 資料，以及只做一次的座標換算
# ══════════════════════════════════════════════════════════════════

class Brain:
    """持有分區標籤與神經元骨架體素，並負責兩套座標之間的換算。

    原本十六支程式各自重複這段載入與換算，共約九十行。集中在這裡之後，
    座標怎麼換只有一個地方寫得出來，也就只有一個地方會錯。

    兩套座標：
      · 標籤格：D03 的 1 µm 格點，形狀 (z, y, x) = (311, 611, 956)
      · D06 體素：邊長 2 個標籤格，用一個整數編號 vox 表示，
                  ix = vox % 512、iy = (vox // 512) % 512、iz = vox // 512²
        兩者的關係是「D06 體素 (iz,iy,ix) 的中心落在標籤格 (2iz, 2iy, 2ix)」。
    """

    def __init__(self, cfg: Config, verbose: bool = True):
        self.cfg = cfg
        d03, d06 = Path(cfg.d03), Path(cfg.d06)

        # 分區標籤。1.8 億個 uint8，約 174 MB。
        self.lab = _load_array(d03 / "work" / "labels_fc_1um.npy")
        self.meta = json.loads((d03 / "work" / "labels_fc_meta.json").read_text())
        # 第 0 號是腦區外，所以名字要往後推一格
        self.names = ["Exterior"] + json.loads(
            (d03 / "D03_labels_meta.json").read_text())["names"]
        self.n2i = {n: i for i, n in enumerate(self.names)}
        self.region_names = self.names[1:]

        # 神經元清單與骨架體素
        with open(d06 / "work" / "neurons.csv", newline="", encoding="utf-8") as fh:
            self.neurons = list(csv.DictReader(fh))
        self.n2x = {r["name"]: i for i, r in enumerate(self.neurons)}
        self.vox = _load_array(d06 / "work" / "vox.npy")
        self.nid = _load_array(d06 / "work" / "nid.npy")

        # 體素編號 → 三個軸的索引。只算這一次。
        # ix ＝ 左右、iy ＝ 背腹、iz ＝ 前後（軸序由指令 1 的解剖關係反推，標頭沒寫）
        G = cfg.d06_grid
        self.ix = (self.vox % G).astype(np.int64)
        self.iy = ((self.vox // G) % G).astype(np.int64)
        self.iz = (self.vox // (G * G)).astype(np.int64)

        self._dep = None
        self._L = None
        self._C = None
        if verbose:
            print(f"載入：{len(self.region_names)} 個腦區、{len(self.neurons):,} 顆神經元、"
                  f"{len(self.vox):,} 筆（體素,神經元）紀錄")

    # ── 延後載入：只有指令 10 用得到深度 ──
    @property
    def dep(self) -> np.ndarray:
        """每筆紀錄離細胞本體的路徑深度。"""
        if self._dep is None:
            self._dep = _load_array(Path(self.cfg.d06) / "work" / "dep.npy")
        return self._dep

    # ── 座標換算 ──
    @property
    def region_of_voxel(self) -> np.ndarray:
        """每一筆（體素,神經元）紀錄落在哪個腦區。0 代表腦區外。"""
        if self._L is None:
            s, lab = self.cfg.d06_step, self.lab
            ok = ((s * self.iz < lab.shape[0]) & (s * self.iy < lab.shape[1])
                  & (s * self.ix < lab.shape[2]))
            L = np.zeros(len(self.vox), np.int16)
            L[ok] = lab[s * self.iz[ok], s * self.iy[ok], s * self.ix[ok]]
            self._L = L
        return self._L

    @property
    def contact(self) -> np.ndarray:
        """神經元 × 腦區的體素數矩陣 C。第 0 欄（腦區外）已歸零。

        這張表是後面好幾步的共同輸入：LN 判準、神經束驗證、合併判斷都靠它。
        """
        if self._C is None:
            N, R = len(self.neurons), len(self.names)
            L = self.region_of_voxel
            C = np.bincount(self.nid.astype(np.int64) * R + L,
                            minlength=N * R).reshape(N, R)
            C[:, 0] = 0
            self._C = C
        return self._C

    def region_mask(self, name: str) -> tuple:
        """腦區在標籤格上的體素座標 (zz, yy, xx)。"""
        return np.nonzero(self.lab == self.n2i[name])

    def binned_shape(self) -> tuple:
        B = self.cfg.bin_factor
        return tuple(s // B + 1 for s in self.lab.shape)

    def binned_mask(self, name: str) -> np.ndarray:
        """腦區在分析格（B×B×B 併格後）上的遮罩。"""
        B = self.cfg.bin_factor
        zz, yy, xx = self.region_mask(name)
        m = np.zeros(self.binned_shape(), bool)
        m[zz // B, yy // B, xx // B] = True
        return m

    def binned_coords(self, sel: np.ndarray) -> tuple:
        """把選中的紀錄換算到分析格座標。"""
        s, B = self.cfg.d06_step, self.cfg.bin_factor
        return (self.iz[sel] * s // B, self.iy[sel] * s // B, self.ix[sel] * s // B)


def _load_array(p: Path) -> np.ndarray:
    """讀 .npy；若不存在就讀同名的 .npz。

    整批資料原始 277 MB，壓縮後只有 37 MB——標籤體積幾乎全是零與大片同值，
    壓到 0.6%。要搬到另一台機器時用 `run_lpu.py pack` 打包成 .npz 就好，
    程式這一端不必改。"""
    p = Path(p)
    if p.exists():
        return np.load(p)
    z = p.with_suffix(".npz")
    if z.exists():
        with np.load(z) as f:
            return f[f.files[0]]
    raise FileNotFoundError(f"找不到 {p}（也沒有 {z}）")


# ══════════════════════════════════════════════════════════════════
# 三、七個步驟
# ══════════════════════════════════════════════════════════════════

# ── 步驟 0：誰是 LN ────────────────────────────────────────────────

def ln_selection(C: np.ndarray, f: float) -> tuple:
    """比例規則：一顆神經元有 f 以上的「區內體素」集中在同一區，就算那一區的 LN。

    論文的原文是目視判準（纖維完全侷限在該區內）。完全侷限在真實資料上
    幾乎沒有神經元通得過，所以改成比例。f 是我們自己選的，不是論文的數字。

    回傳 (選中與否, 該區, 比例)。
    """
    tot = C.sum(1)
    top = C.max(1)
    arg = C.argmax(1)
    inside = tot > 0
    sel = inside & (top >= f * tot)
    frac = np.divide(top, tot, out=np.zeros(len(tot), float), where=inside)
    return sel, arg, frac


def ln_by_region(C: np.ndarray, names: list, f: float) -> tuple:
    """回傳 (每區的 LN 顆數 Counter, 全腦 LN 總數)。"""
    sel, arg, _ = ln_selection(C, f)
    return collections.Counter(names[a] for a in arg[sel]), int(sel.sum())


# ── 步驟 1、2：密度場 v(r) ─────────────────────────────────────────

def density_field(brain: Brain, ln_ids, region: str) -> tuple:
    """在分析格上堆出密度場 v(r)，再做 3×3×3 移動平均。

    v(r) 的定義：**有幾顆不同的 LN 佔到這一格**。同一顆神經元在同一格
    只算一次（那個 np.unique 就是在做這件事）。論文寫的是「穿過該體素的
    纖維條數」——我們沒有纖維的拓樸，只有體素，所以這是近似，不是同一個量。

    回傳 (腦區遮罩, 平滑後的 v(r))。
    """
    cfg = brain.cfg
    mask = brain.binned_mask(region)
    sel = np.isin(brain.nid, ln_ids)
    kz, ky, kx = brain.binned_coords(sel)
    n = brain.nid[sel]
    # 同一顆神經元在同一格只算一次
    key = np.unique(np.stack([kz, ky, kx, n]), axis=1)
    f = np.zeros(brain.binned_shape(), np.float32)
    np.add.at(f, (key[0], key[1], key[2]), 1.0)
    return mask, ndimage.uniform_filter(f, size=cfg.smooth_size)


def estimate_memory(n_hot: int) -> float:
    """兩兩距離矩陣要多少 GB。n 個點會產生 n(n−1)/2 個 float64。"""
    return n_hot * (n_hot - 1) / 2 * 8 / 2 ** 30


# ── 步驟 3：熱點 ──────────────────────────────────────────────────

def five_number(values: np.ndarray) -> list:
    return [float(x) for x in np.percentile(values, [0, 25, 50, 75, 100])]


def hotspots(mask: np.ndarray, v: np.ndarray, q: float) -> np.ndarray:
    """取上四分位以上的體素。

    佔比必然接近 25%，因為「上四分位」的定義就是排到 75% 的那個位置。
    這個 25% 不是資料的性質，是分位數定義的必然結果。
    """
    return mask & (v >= np.percentile(v[mask], q))


# ── 步驟 4：UPGMA 分群 ────────────────────────────────────────────

def cluster_hotspots(P: np.ndarray, thr: float, cut: float) -> tuple:
    """對熱點體素做 UPGMA，切在高度 cut，只留大於 thr 個體素的群。

    average linkage 就是 UPGMA，這是論文指定的。
    thr 的分母是**整個腦區**的體素數，不是熱點數。
    """
    Z = linkage(pdist(P.astype(np.float32)), method="average")
    cl = fcluster(Z, cut, "distance")
    sizes = np.bincount(cl)[1:]
    keep = [i + 1 for i, c in enumerate(sizes) if c > thr]
    return cl, sizes, keep, Z


def cut_sweep(P: np.ndarray, thr: float, cfg: Config, rng, detail: bool = False) -> dict:
    """掃切割高度，並把體素順序打亂重跑 n_shuffle 次。

    為什麼要打亂：UPGMA 遇到距離相同的兩對時，選哪一對取決於它們在陣列裡
    的位置。體素順序一換，樹就可能不一樣。打亂重跑是在量這件事有多嚴重。
    """
    cuts = cfg.cuts
    runs, root, extra = [], None, {}
    for t in range(cfg.n_shuffle):
        Q = P if t == 0 else P[rng.permutation(len(P))]
        D = pdist(Q)
        Z = linkage(D, "average")
        runs.append([int((np.bincount(fcluster(Z, c, "distance"))[1:] > thr).sum())
                     for c in cuts])
        if t == 0:
            root = float(Z[:, 2].max())
            if detail:
                # 相異距離值有幾種：體素在整齊的格子上，距離會大量重複，
                # 重複就會有平手，平手就要靠陣列順序決定——這是打亂重跑的理由。
                extra = dict(n_pairs=int(len(D)),
                             n_distinct=int(len(np.unique(np.rint(D ** 2)))))
        del D
    A = np.array(runs)
    pl = A[:, cfg.plateau_slice]
    return dict(runs=A, root=round(root, 2), lo=int(A.min()), hi=int(A.max()),
                span=int((A.max(0) - A.min(0)).max()),
                plateau=[int(pl.min()), int(pl.max())], **extra)


# ── 步驟 5：種子與招募 ────────────────────────────────────────────

def neuron_masks(brain: Brain, ids, pad: int = 4) -> tuple:
    """把每顆神經元畫成一塊小立方體裡的二值遮罩（原始 D06 格，不併格）。

    所有神經元共用同一個包住全體的方框，這樣兩顆之間才能直接做布林運算。
    """
    sel = np.isin(brain.nid, list(ids))
    n = brain.nid[sel]
    ix, iy, iz = brain.ix[sel], brain.iy[sel], brain.iz[sel]
    x0, y0, z0 = ix.min(), iy.min(), iz.min()
    sh = (int(iz.max() - z0) + 2 * pad + 1,
          int(iy.max() - y0) + 2 * pad + 1,
          int(ix.max() - x0) + 2 * pad + 1)
    M = {}
    for i in np.unique(n):
        k = (n == i)
        a = np.zeros(sh, bool)
        a[iz[k] - z0 + pad, iy[k] - y0 + pad, ix[k] - x0 + pad] = True
        M[int(i)] = a
    return M, {i: int(a.sum()) for i, a in M.items()}


def neuron_voxel_sets(brain: Brain, ids) -> dict:
    """每顆神經元在**分析格**上佔了哪些格子（集合）。用來跟分群的結果比對。"""
    sel = np.isin(brain.nid, list(ids))
    kz, ky, kx = brain.binned_coords(sel)
    n = brain.nid[sel]
    out = {}
    for i in np.unique(n):
        k = (n == i)
        out[int(i)] = set(zip(kz[k].tolist(), ky[k].tolist(), kx[k].tolist()))
    return out


def pick_seed(neu: dict, cluster_voxels: np.ndarray, top_n: int = 10) -> dict:
    """挑招募的起始神經元。

    論文的作法是「目視挑一顆纖維侷限在該候選 LPU 內、體積最小的 LN」。
    在我們的資料上沒有任何一顆的比例接近 100%，照字面挑不到，所以改成
    **替代規則：比例最高的前 top_n 顆裡取體積最小的**。這不是論文的條件。
    """
    target = set(map(tuple, cluster_voxels.tolist()))
    frac = {i: len(vs & target) / len(vs) for i, vs in neu.items()}
    top = sorted(frac, key=lambda i: -frac[i])[:top_n]
    size = {i: len(neu[i]) for i in top}
    seed = min(top, key=lambda i: size[i])
    return dict(seed=seed, frac=frac, top=top, size=size)


def recruit(M: dict, sizes: dict, seed: int, grow: int, overlap: float,
            max_rounds: int = 40, verbose: bool = True) -> tuple:
    """滾雪球：把目前的 source 撐大 grow 格，凡是有一半以上體積落在裡面的
    LN 就併進來，然後重複，直到沒有新的可收。

    分母用 target 神經元**原本的**體積——只撐大 source，不撐大 target，
    否則兩邊都撐大會算出超過 1 的比例。
    """
    st = ndimage.generate_binary_structure(3, 1)
    ids = np.array(sorted(M))
    src = M[seed].copy()
    got = {seed}
    log = []
    for rnd in range(1, max_rounds + 1):
        D = ndimage.binary_dilation(src, st, grow)
        new = [i for i in ids if i not in got and (D & M[i]).sum() / sizes[i] > overlap]
        if not new:
            break
        for i in new:
            src |= M[i]
            got.add(i)
        log.append(len(new))
        if verbose:
            print(f"     第 {rnd} 輪：收 {len(new):4d} 顆，累計 {len(got):4d}")
    return got, log


def recruit_regime(M: dict, sizes: dict, seed: int, grows, thresholds) -> dict:
    """掃「撐大幾格 × 重疊門檻」，看有沒有「只收一部分」的操作區間。

    這是 PART 4 最重要的一次檢查：如果每一格不是 1 就是全部，代表這一步
    在這份資料上沒有可用的設定，不是參數沒調好。
    """
    res = {}
    for g in grows:
        row = []
        for t in thresholds:
            got, _ = recruit(M, sizes, seed, g, t, verbose=False)
            row.append(len(got))
        res[g] = row
    return res


# ── 步驟 6：去初級神經突、抽等值面 ────────────────────────────────

def depth_profile(dep: np.ndarray, L: np.ndarray, region_idx: int, bands) -> list:
    """依離本體的路徑深度分段，看每一段有多少體素落在該區外。

    初級神經突與細胞本體是「從本體出發、還沒進到神經纖維網」的那一段，
    所以它們在深度上靠近 0，在空間上常常落在腦區外。這張表就是在找那個界線。
    """
    rows = []
    for lo, hi in bands:
        k = (dep >= lo) & (dep < hi)
        if k.sum():
            rows.append(dict(lo=lo, hi=hi, n=int(k.sum()),
                             outside_region=float((L[k] != region_idx).mean()),
                             outside_any=float((L[k] == 0).mean())))
    return rows


def c_value(body: np.ndarray, v: np.ndarray, core_frac: float) -> dict:
    """c 值：中心區的平均密度 ÷ 周邊的平均密度。論文的判準是 c > 2。

    「均勻內縮到剩 core_frac」用距離轉換做：離表面越遠的越內側，
    由內而外排序後取前 core_frac 當中心區 A，其餘當周邊 B。
    """
    dt = ndimage.distance_transform_edt(body)
    vals = v[body]
    order = np.argsort(-dt[body])          # 由內而外
    nA = int(round(core_frac * len(order)))
    A, Bx = order[:nA], order[nA:]
    mB = vals[Bx].mean() if len(Bx) else 0.0
    c = float(vals[A].mean() / mB) if mB > 0 else float("nan")
    return dict(nA=int(nA), nB=int(len(Bx)),
                meanA=float(vals[A].mean()), meanB=float(mB), c=round(c, 3))


# ── 步驟 7：連結與合併 ────────────────────────────────────────────

def adjacency(lab: np.ndarray) -> set:
    """哪些腦區在空間上相鄰：沿三個軸各比一次相鄰格的標籤。"""
    adj = set()
    for ax in range(3):
        a = np.moveaxis(lab, ax, 0)
        x, y = a[:-1].ravel(), a[1:].ravel()
        k = (x > 0) & (y > 0) & (x != y)
        adj |= {tuple(sorted(t)) for t in zip(x[k].tolist(), y[k].tolist())}
    return adj


def merge_gain(C: np.ndarray, names: list, pairs, f: float, base: int) -> list:
    """把兩個腦區當成一個來看，全腦的 LN 顆數會增加多少。

    增加得多，代表這兩區之間有大量神經元的分枝被邊界切開——也就是說，
    這條邊界可能不該存在。這是**線索**，不是論文的合併判準。
    """
    out = []
    for i, j in pairs:
        Cx = C.copy()
        Cx[:, i] = C[:, i] + C[:, j]
        Cx[:, j] = 0
        nm = list(names)
        nm[i] = names[i] + "+" + names[j]
        cc, nn = ln_by_region(Cx, nm, f)
        out.append((nn - base, names[i], names[j], cc.get(nm[i], 0)))
    out.sort(reverse=True)
    return out


# ══════════════════════════════════════════════════════════════════
# 四、畫圖
# ══════════════════════════════════════════════════════════════════
# 規矩：圖內一律不放中文。新機器不一定裝了中文字型，缺字會畫成空白方塊。

INK, ACC, GREY, WARN, BAD = "#1b2733", "#2563eb", "#8b98a5", "#d97706", "#dc2626"


def setup_matplotlib():
    """一定要在 import pyplot 之前設好 Agg，否則沒有螢幕的機器會直接報錯。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def despine(ax, which=("top", "right")):
    for sp in which:
        ax.spines[sp].set_visible(False)


def save_fig(fig, cfg: Config, name: str):
    plt = setup_matplotlib()
    p = cfg.ensure_out() / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"   圖已存：{p.name}")


def project(lab: np.ndarray, axis: int) -> np.ndarray:
    """沿一個軸做「第一個非零」的投影，得到一張腦區圖。"""
    a = np.moveaxis(lab, axis, 0)
    out = np.zeros(a.shape[1:], np.int16)
    for i in range(a.shape[0]):
        s = a[i]
        m = (out == 0) & (s > 0)
        out[m] = s[m]
    return out
