#!/usr/bin/env python3
"""檢查 STEP 4 的種子條件：論文要求「纖維完全侷限在該候選 LPU 內」，做得到嗎？

候選 LPU 在這個時間點只是 STEP 3 分出來的一群高密度體素（前 25%），
而一顆 LN 的分枝必然同時鋪在密的地方和疏的地方。
"""
import numpy as np, importlib.util, os
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "make_part2_figures.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
GX, GY = 512, 512


def run(region="AL_R", cut=8.0):
    lab, names, n2i, rows, vox, nid, n2l = m.load()
    m.LNTAB.update(m.ln_table(lab, names, rows, vox, nid))
    mask, sm, _ = m.field_of(region, lab, n2i, rows, vox, nid, n2l)
    q = np.percentile(sm[mask], 75)
    P = np.argwhere(mask & (sm >= q))
    Z = linkage(pdist(P.astype(np.float32)), "average")
    cl = fcluster(Z, cut, "distance")
    sizes = np.bincount(cl)[1:]; thr = 0.01 * mask.sum()
    keep = [i + 1 for i, c in enumerate(sizes) if c > thr]
    small = keep[int(np.argmin(sizes[np.array(keep) - 1]))]
    print(f"{region} 在切割高度 {cut:g}：{len(keep)} 群過 1%，"
          f"最小的一群 {sizes[small-1]} 個體素")

    S = m.LNTAB[region]; sel = np.isin(nid, S); v = vox[sel]; n = nid[sel]
    iz = (v // (GX*GY)) * 2 // m.B; iy = ((v // GX) % GY) * 2 // m.B; ix = (v % GX) * 2 // m.B
    hot = set(map(tuple, P.tolist()))
    sm_set = set(map(tuple, P[cl == small].tolist()))
    reg = set(map(tuple, np.argwhere(mask).tolist()))
    big = np.zeros(mask.shape, bool); pts = P[cl == small]
    big[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    loose = set(map(tuple, np.argwhere(
        ndimage.binary_fill_holes(ndimage.binary_dilation(big, iterations=2))).tolist()))

    out = {k: [] for k in ("最小群", "最小群(放寬)", "全部熱點", "整個腦區")}
    for i in np.unique(n):
        k = (n == i); vs = set(zip(iz[k].tolist(), iy[k].tolist(), ix[k].tolist()))
        out["最小群"].append(len(vs & sm_set) / len(vs))
        out["最小群(放寬)"].append(len(vs & loose) / len(vs))
        out["全部熱點"].append(len(vs & hot) / len(vs))
        out["整個腦區"].append(len(vs & reg) / len(vs))
    print(f"{len(S)} 顆 LN 候選，各有多少比例的體素落在……")
    for k, a in out.items():
        a = np.array(a)
        print(f"   {k:14s} 中位 {np.median(a):.3f}  最大 {a.max():.3f}  "
              f"達到 100% 的 {int((a >= 0.999).sum())} 顆")


if __name__ == "__main__":
    run()
