#!/usr/bin/env python3
"""STEP 5 為什麼要去掉細胞本體與初級神經突：量最靠近本體的那一段落在哪裡。

果蠅的神經細胞本體排在腦的表層，不屬於任何 neuropil；初級神經突是從表層
伸進纖維網的通勤路線。留著它們，密度場會在腦區外多出一團，等值面被拉出去。
"""
import numpy as np, importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "make_part2_figures.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
D06 = "/home/wanjuli/claude_linux/BSC_plan/D06"
GX, GY = 512, 512


def run(region="AL_R"):
    lab, names, n2i, rows, vox, nid, n2l = m.load()
    m.LNTAB.update(m.ln_table(lab, names, rows, vox, nid))
    dep = np.load(os.path.join(D06, "work", "dep.npy"))
    S = m.LNTAB[region]; sel = np.isin(nid, S)
    v = vox[sel]; d = dep[sel]
    ix = (v % GX).astype(np.int64); iy = ((v // GX) % GY).astype(np.int64)
    iz = (v // (GX * GY)).astype(np.int64)
    ok = ((2*iz < lab.shape[0]) & (2*iy < lab.shape[1]) & (2*ix < lab.shape[2]))
    L = np.zeros(len(v), np.int16); L[ok] = lab[2*iz[ok], 2*iy[ok], 2*ix[ok]]
    inside = (L == n2i[region])
    print(f"{region}：{len(S)} 顆 LN 候選，共 {len(v):,} 個體素；"
          f"整體落在該區外 {(~inside).mean()*100:.1f}%")
    print(" 離本體的路徑深度    體素數     落在該區外   落在任何腦區外")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 35), (35, 65), (65, 100)]:
        k = (d >= lo) & (d < hi)
        if not k.sum():
            continue
        print(f"   {lo:3d}\u2013{hi:<3d} {k.sum():12,d} {(~inside[k]).mean()*100:12.1f}%"
              f" {(L[k] == 0).mean()*100:14.1f}%")


if __name__ == "__main__":
    run()
