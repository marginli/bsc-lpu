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

    M = {i: mask_of(i) for i in ids}
    st = ndimage.generate_binary_structure(3, 1)
    print(f"{region}：LN 候選 {len(S)} 顆，抽 {len(ids)} 顆；每顆體素數中位 "
          f"{int(np.median([M[i].sum() for i in ids]))}")
    print(" 膨脹  有交集  重疊率中位  最大重疊  >50% 的配對")
    for grow in (0, 1, 2, 3):
        D = {i: (ndimage.binary_dilation(M[i], st, iterations=grow) if grow else M[i]) for i in ids}
        fr = np.array([(D[a] & M[b]).sum() / M[b].sum() for a in ids for b in ids if a != b])
        print(f" {grow:4d}  {(fr>0).mean()*100:6.1f}%  {np.median(fr):10.3f}  {fr.max():8.2f}  {(fr>0.5).mean()*100:11.2f}%")


if __name__ == "__main__":
    run()
