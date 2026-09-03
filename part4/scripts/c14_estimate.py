#!/usr/bin/env python3
"""指令 14 第二項：全腦跑之前先估算。指令要求估完停下來等人確認。"""
import numpy as np, json, csv, collections
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; OUT="part4/out"; B=4
lab=np.load(f"{D03}/work/labels_fc_1um.npy")
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]
n2i={n:i for i,n in enumerate(names)}
ln=collections.Counter(r["region"] for r in csv.DictReader(open(f"{OUT}/c02_ln_candidates.csv")))
rowsout=[]
for nm in names[1:]:
    zz,yy,xx=np.nonzero(lab==n2i[nm])
    coarse=len(set(zip((zz//B).tolist(),(yy//B).tolist(),(xx//B).tolist())))
    hot=int(coarse*0.25); mem=hot*(hot-1)/2*8/2**30
    rowsout.append((nm,ln.get(nm,0),coarse,hot,mem))
rowsout.sort(key=lambda r:-r[4])
print("全腦 75 區的規模估算（分析體素 = 4×4×4 併格後）")
print(" 腦區        LN    分析體素    熱點   距離矩陣(GB)   會不會卡")
big=[];small=[];ok=[]
for nm,l,c,h,m in rowsout:
    tag=""
    if m>16: tag="體素太多"; big.append(nm)
    elif l<10: tag="LN 太少"; small.append(nm)
    else: ok.append(nm)
    if m>1 or l<10 or nm in ("AL_R","FB","PB"):
        print(f" {nm:10s} {l:5d} {c:10,d} {h:8,d} {m:12.1f}   {tag}")
print(f"\n分類：可以跑 {len(ok)} 區、體素太多 {len(big)} 區、LN 太少 {len(small)} 區")
print("體素太多的：", "、".join(big) if big else "（無）")
print(f"LN 少於 10 顆的 {len(small)} 區：", "、".join(small[:12]), "…" if len(small)>12 else "")
tot=sum(m for *_,m in rowsout)
print(f"\n全部 75 區的距離矩陣加總約 {tot:.1f} GB（不是同時，是逐區）")
print("最大的一區單獨就要 %.1f GB"%rowsout[0][4])
json.dump({"ok":ok,"too_big":big,"too_few":small,
           "per_region":[{"region":n,"ln":l,"vox":c,"hot":h,"mem_GB":round(m,2)} for n,l,c,h,m in rowsout]},
          open(f"{OUT}/c14_estimate.json","w"),ensure_ascii=False,indent=1)
print("\n【依指令 14 第二項，估算完成，停在這裡等確認】")
print("建議的處理方式：")
print("  · 體素太多的區：再併一次格（8×8×8），距離矩陣降到 1/64")
print("  · LN 少於 10 顆的區：不跑分群，直接判定「沒有自己的 LN 族群」→ 候選 hub")
