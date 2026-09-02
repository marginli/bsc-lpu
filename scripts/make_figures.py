#!/usr/bin/env python3
"""產生 LPU 專題課程 PART 1 的兩張圖。

資料來源（皆可公開，見 BSC_plan/D03/D03_資料說明卡.md）：
  FCWB 標準腦模板         Zenodo record 10568 (Ostrovsky & Jefferis 2014)  CC0
  FCWBNP 腦區分區標籤     natverse / Ito et al. 2014 Neuron 81:755-765     GPL-3 + 原論文

產出：
  assets/fig1-mb-four-vs-one.png   開場圖：MB 形態四塊 vs 功能一塊
  assets/fig2-neuropils-75.png     替身圖：Ito 2014 的 75 個腦區標籤（41 個基本名）

圖內不放任何中文，全部說明寫在 HTML 的圖說裡（避免 matplotlib 的 CJK 字型問題）。

座標軸（本腳本啟動時會重新驗證，不符就中止）：
  axis 0 (565)  左右   低 = 右腦
  axis 1 (328)  背腹   低 = 背側
  axis 2 (109)  前後   低 = 前側   ← 正視圖沿此軸投影
"""
import csv
import sys
from pathlib import Path

import numpy as np
import nrrd
from PIL import Image

D03 = Path("/home/wanjuli/claude_linux/BSC_plan/D03")
OUT = Path("/home/wanjuli/claude_linux/BSC_plan/specific_topics/LPU/assets")

BG = np.array([255, 255, 255], np.float64)      # 圖底
GHOST = np.array([223, 228, 234], np.float64)   # 其餘腦區的淡灰剪影

# 沿用網站的四個強調色
MB_PARTS = {
    "MB_CA":  "#2563eb",
    "MB_PED": "#9333ea",
    "MB_VL":  "#d97706",
    "MB_ML":  "#16a34a",
}
MB_ONE = "#2563eb"


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float64)


def load():
    labels, _ = nrrd.read(str(D03 / "D03_FCWBNP_labels_1um.nrrd"))
    rows = list(csv.DictReader(open(D03 / "D03_region_table.csv", encoding="utf-8")))
    table = {int(r["id"]): r for r in rows}
    return labels, table


def check_axes(labels, table):
    """把說明裡的座標軸假設真的驗一次，不對就中止。"""
    def centroid(name):
        rid = next(k for k, v in table.items() if v["name"] == name)
        return np.argwhere(labels == rid).mean(0)

    me_r, me_l = centroid("ME_R"), centroid("ME_L")
    ca, gng = centroid("MB_CA_R"), centroid("GNG")
    al = centroid("AL_R")

    ok = (
        abs(me_l[0] - me_r[0]) > 300      # 左右視葉在 axis0 上分得最開
        and ca[1] < gng[1]                # 蕈狀體萼比顎神經節背側
        and al[2] < ca[2]                 # 觸角葉比萼前側
    )
    if not ok:
        sys.exit("座標軸與假設不符，請重新確認後再產圖。")
    print("座標軸驗證通過：axis0=左右、axis1=背腹（小=背）、axis2=前後（小=前）")


def project(labels, keep, axis=2):
    """沿 axis 由近而遠，取每個像素第一個落在 keep 裡的標籤。

    axis=2 為正視圖（由前往後），axis=0 為矢狀視圖（由外側往內側）。
    回傳 (lab2d, depth2d)：標籤圖與該像素的深度（用來做明暗）。
    """
    masked = np.where(np.isin(labels, list(keep)), labels, 0)
    hit = masked > 0
    any_hit = hit.any(axis=axis)
    first = np.argmax(hit, axis=axis)              # 第一個 True 的位置
    idx = list(np.indices(first.shape))
    idx.insert(axis, first)
    lab2d = np.where(any_hit, masked[tuple(idx)], 0)
    depth = np.where(any_hit, first, 0)
    return lab2d, depth


def bbox(mask, margin=6):
    """回傳把 mask 包起來、外加 margin 的切片。"""
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    r0 = max(rows[0] - margin, 0)
    r1 = min(rows[-1] + margin + 1, mask.shape[0])
    c0 = max(cols[0] - margin, 0)
    c1 = min(cols[-1] + margin + 1, mask.shape[1])
    return slice(r0, r1), slice(c0, c1)


def shade(depth, mask):
    """依深度做輕微明暗：前面亮、後面暗。回傳 0.72–1.06 的係數。"""
    f = np.ones(depth.shape)
    if mask.any():
        d = depth[mask].astype(np.float64)
        lo, hi = d.min(), d.max()
        if hi > lo:
            n = (depth - lo) / (hi - lo)
            f = np.clip(1.06 - 0.34 * n, 0.72, 1.06)
    return f


def compose(shape, layers):
    """layers: [(mask2d, rgb, shade_factor), ...] 後面的蓋前面的。"""
    img = np.repeat(BG[None, None, :], shape[0], 0).repeat(shape[1], 1)
    for mask, rgb, f in layers:
        if not mask.any():
            continue
        col = np.clip(rgb[None, None, :] * f[..., None], 0, 255)
        img = np.where(mask[..., None], col, img)
    return img


def to_image(img, scale=2):
    """(x, y, 3) 的資料軸 → 正立的影像：背側在上、右腦在左。"""
    arr = np.transpose(img, (1, 0, 2))          # → (背腹, 左右, 3)
    im = Image.fromarray(arr.astype(np.uint8))
    w, h = im.size
    return im.resize((w * scale, h * scale), Image.LANCZOS)


def pad(im, m=24):
    out = Image.new("RGB", (im.width + 2 * m, im.height + 2 * m), tuple(BG.astype(int)))
    out.paste(im, (m, m))
    return out


def fig1(labels, table):
    """開場圖：左＝MB 依形態切成四塊，右＝同一顆 MB 一整塊。"""
    ids_by_base = {}
    for rid, r in table.items():
        ids_by_base.setdefault(r["base_name"], []).append(rid)

    mb_ids = [i for b in MB_PARTS for i in ids_by_base[b]]
    other_ids = [rid for rid in table if rid not in mb_ids]

    # 背景剪影：整顆腦扣掉 MB
    g_lab, g_dep = project(labels, other_ids)
    g_mask = g_lab > 0
    g_f = shade(g_dep, g_mask)

    # MB 本身
    m_lab, m_dep = project(labels, mb_ids)
    m_mask = m_lab > 0
    m_f = shade(m_dep, m_mask)

    shape = m_lab.shape
    crop = bbox(g_mask | m_mask, margin=8)
    panels = []
    for merged in (False, True):
        layers = [(g_mask, GHOST, g_f)]
        if merged:
            layers.append((m_mask, hex2rgb(MB_ONE), m_f))
        else:
            for base, hx in MB_PARTS.items():
                sub = np.isin(m_lab, ids_by_base[base])
                layers.append((sub, hex2rgb(hx), m_f))
        panels.append(pad(to_image(compose(shape, layers)[crop])))

    gap = 28
    w = panels[0].width * 2 + gap
    canvas = Image.new("RGB", (w, panels[0].height), tuple(BG.astype(int)))
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + gap, 0))
    p = OUT / "fig1-mb-four-vs-one.png"
    canvas.save(p, optimize=True)
    print(f"寫出 {p.name}  {canvas.size[0]}×{canvas.size[1]}")


def fig1b(labels, table):
    """輔助圖：右半腦矢狀視圖，四個部分才分得清楚。"""
    ids_by_base = {}
    for rid, r in table.items():
        ids_by_base.setdefault(r["base_name"], []).append(rid)
    mb_ids = [i for b in MB_PARTS for i in ids_by_base[b]]
    other_ids = [rid for rid in table if rid not in mb_ids]

    half = labels.copy()
    half[283:, :, :] = 0                      # 只留右半腦（中線在 x≈282）

    g_lab, g_dep = project(half, other_ids, axis=0)
    m_lab, m_dep = project(half, mb_ids, axis=0)
    g_mask, m_mask = g_lab > 0, m_lab > 0

    layers = [(g_mask, GHOST, shade(g_dep, g_mask))]
    for base, hx in MB_PARTS.items():
        layers.append((np.isin(m_lab, ids_by_base[base]), hex2rgb(hx), shade(m_dep, m_mask)))

    img = compose(g_lab.shape, layers)        # 軸為 (背腹, 前後)，已是正立
    crop = bbox(m_mask, margin=40)
    im = Image.fromarray(img[crop].astype(np.uint8))
    im = pad(im.resize((im.width * 5, im.height * 5), Image.LANCZOS))
    p = OUT / "fig1b-mb-sagittal.png"
    im.save(p, optimize=True)
    print(f"寫出 {p.name}  {im.size[0]}×{im.size[1]}")


def fig2(labels, table):
    """替身圖：75 個標籤，依 41 個基本名配色（左右同色）。"""
    bases = sorted({r["base_name"] for r in table.values()})
    # 黃金角取色相，讓相鄰編號的顏色差得夠開
    import colorsys
    colors = {}
    for k, b in enumerate(bases):
        h = (k * 0.61803398875) % 1.0
        s = 0.52 + 0.30 * ((k * 7) % 3) / 2.0
        v = 0.72 + 0.22 * ((k * 5) % 3) / 2.0
        colors[b] = np.array(colorsys.hsv_to_rgb(h, s, v)) * 255

    lab, dep = project(labels, list(table))
    mask = lab > 0
    f = shade(dep, mask)

    layers = []
    for b in bases:
        ids = [rid for rid, r in table.items() if r["base_name"] == b]
        layers.append((np.isin(lab, ids), colors[b], f))

    im = pad(to_image(compose(lab.shape, layers)[bbox(mask, margin=8)]))
    p = OUT / "fig2-neuropils-75.png"
    im.save(p, optimize=True)
    print(f"寫出 {p.name}  {im.size[0]}×{im.size[1]}　（{len(bases)} 個基本名／{len(table)} 個標籤）")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, table = load()
    check_axes(labels, table)
    fig1(labels, table)
    fig1b(labels, table)
    fig2(labels, table)


if __name__ == "__main__":
    main()
