#!/usr/bin/env python3
"""合併之後 LN 族群才出現：量「切太細就沒有 LN」這件事。

LN 的定義是相對於某一塊區域的。蕈狀體的 Kenyon cell 樹突在萼、軸突在葉，
所以把萼和葉當兩個區時，KC 兩邊都不算 LN；合成一區，它們就成了 LN 族群。
"""
import numpy as np, collections, importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "make_part2_figures.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

GROUPS = {
    "MB_R":  ["MB_CA_R", "MB_PED_R", "MB_VL_R", "MB_ML_R"],
    "MB_L":  ["MB_CA_L", "MB_PED_L", "MB_VL_L", "MB_ML_L"],
    "EBLT":  ["EB", "BU_R", "BU_L"],
}


def ln_count(C, names, f=None):
    f = m.F_LN if f is None else f
    tot = C.sum(1); top = C.max(1); arg = C.argmax(1); big = tot > 0
    ok = big & (top >= f * tot)
    return collections.Counter(names[a] for a in arg[ok]), int(ok.sum())


def run():
    lab, names, n2i, rows, vox, nid, n2l = m.load()
    C = m.neuron_region_counts(lab, names, rows, vox, nid)
    idx = {n: i for i, n in enumerate(names)}
    c0, n0 = ln_count(C, names)
    print("合併前（Ito 的細分區）：")
    for g, mem in GROUPS.items():
        print(f"   {g:6s} <- " + "、".join(f"{x} {c0.get(x, 0)}" for x in mem))
    print(f"   全腦 LN 候選 {n0}")
    C2 = C.copy(); nm = list(names)
    for g, mem in GROUPS.items():
        ix = [idx[x] for x in mem if x in idx]
        C2[:, ix[0]] = C[:, ix].sum(1); nm[ix[0]] = g
        for j in ix[1:]:
            C2[:, j] = 0
    c1, n1 = ln_count(C2, nm)
    print("合併後：")
    for g in GROUPS:
        print(f"   {g:6s} {c1.get(g, 0)}")
    print(f"   全腦 LN 候選 {n1}（多了 {n1 - n0}）")


if __name__ == "__main__":
    run()
