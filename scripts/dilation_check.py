#!/usr/bin/env python3
"""量 STEP 4 的膨脹：不把體素撐大的話，兩顆神經元的重疊率有多低。

判準是「B 有超過 50% 的體積落在 A 之內」。這是個比例，
所以「整體等比放大」對它毫無作用；唯一有作用的是把每個體素各自撐大
（形態學膨脹），讓貼著跑卻不共格的纖維產生交集。
"""
import numpy as np, importlib.util, os
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "make_part2_figures.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
GX, GY = 512, 512


def ref(a, k=1):
    """參考形狀撐大 k 格後的體積倍率——倍率取決於形狀的表面體積比，不是常數。"""
    p = np.pad(a, k + 1)
    return ndimage.binary_dilation(p, ndimage.generate_binary_structure(3, 1), iterations=k).sum() / a.sum()


def run(region="AL_R", nsample=60, seed=0):
    lab, names, n2i, rows, vox, nid, n2l = m.load()
    m.LNTAB.update(m.ln_table(lab, names, rows, vox, nid))
    S = m.LNTAB[region]
    sel = np.isin(nid, S); v = vox[sel]; n = nid[sel]
    ix = (v % GX).astype(np.int32); iy = ((v // GX) % GY).astype(np.int32); iz = (v // (GX*GY)).astype(np.int32)
    x0, y0, z0 = ix.min(), iy.min(), iz.min()
    sh = (iz.max()-z0+9, iy.max()-y0+9, ix.max()-x0+9)
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.array(sorted(set(n.tolist()))), size=min(nsample, len(S)), replace=False)

    def mask_of(i):
        k = (n == i); a = np.zeros(sh, bool)
        a[iz[k]-z0+4, iy[k]-y0+4, ix[k]-x0+4] = True; return a

    # 先看「一個體素被幾顆神經元佔用」——體素比纖維粗，共用是常態
    uv, cnt = np.unique(v, return_counts=True)
    print(f"{region}：{len(S)} 顆共佔 {len(uv)} 個相異體素；"
          f"每格被佔用的神經元數 中位 {int(np.median(cnt))}、平均 {cnt.mean():.1f}、最大 {cnt.max()}")
    print(f"   只被 1 顆佔用 {(cnt==1).sum()}（{(cnt==1).mean()*100:.1f}%）、"
          f"2 顆以上 {(cnt>=2).sum()}（{(cnt>=2).mean()*100:.1f}%）、"
          f"10 顆以上 {(cnt>=10).sum()}（{(cnt>=10).mean()*100:.1f}%）")

    M = {i: mask_of(i) for i in ids}
    st = ndimage.generate_binary_structure(3, 1)
    D2 = {i: ndimage.binary_dilation(M[i], st, iterations=2) for i in ids}
    print("\n撐大 2 格時，『撐大誰』的差別：")
    for name, f in (("只撐大 source", lambda a, b: (D2[a] & M[b]).sum() / M[b].sum()),
                    ("只撐大 target", lambda a, b: (M[a] & D2[b]).sum() / M[b].sum()),
                    ("兩邊都撐大，分母用原體積", lambda a, b: (D2[a] & D2[b]).sum() / M[b].sum()),
                    ("兩邊都撐大，分母用撐大後體積", lambda a, b: (D2[a] & D2[b]).sum() / D2[b].sum())):
        fr = np.array([f(a, b) for a in ids for b in ids if a != b])
        print(f"   {name:26s} 中位 {np.median(fr):6.3f}   >50% 的配對 {(fr>0.5).mean()*100:5.1f}%")
    print("\n撐大幾格 → 體積幾倍（中位）：", {k: round(float(np.median(
        [ndimage.binary_dilation(M[i], st, iterations=k).sum()/M[i].sum() for i in ids])), 1)
        for k in (1, 2, 3, 4)})
    f1 = np.array([ndimage.binary_dilation(M[i], st, 1).sum()/M[i].sum() for i in ids])
    print(f"   撐大 1 格的倍率分布：中位 {np.median(f1):.1f}，範圍 {f1.min():.1f}–{f1.max():.1f}")
    # 這個倍率不是常數，取決於形狀的表面體積比
    print("   對照（撐大 1 格）：", {
        "單一體素": round(float(ref(np.ones((1, 1, 1), bool))), 1),
        "一格寬直線": round(float(ref(np.ones((1, 1, 50), bool))), 1),
        **{f"實心{L}立方": round(float(ref(np.ones((L, L, L), bool))), 1) for L in (4, 8, 16, 32)}})
    print()
    print(f"{region}：LN 候選 {len(S)} 顆，抽 {len(ids)} 顆；每顆體素數中位 "
          f"{int(np.median([M[i].sum() for i in ids]))}")
    print(" 膨脹  有交集  重疊率中位  最大重疊  >50% 的配對")
    for grow in (0, 1, 2, 3):
        D = {i: (ndimage.binary_dilation(M[i], st, iterations=grow) if grow else M[i]) for i in ids}
        fr = np.array([(D[a] & M[b]).sum() / M[b].sum() for a in ids for b in ids if a != b])
        print(f" {grow:4d}  {(fr>0).mean()*100:6.1f}%  {np.median(fr):10.3f}  {fr.max():8.2f}  {(fr>0.5).mean()*100:11.2f}%")


if __name__ == "__main__":
    run()
