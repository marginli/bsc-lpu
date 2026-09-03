#!/usr/bin/env python3
"""指令 12（降級版神經束驗證）＋指令 13（合併判斷）。"""
import numpy as np, json, csv, collections
D03="/home/wanjuli/claude_linux/BSC_plan/D03"; OUT="part4/out"
names=["Exterior"]+json.load(open(f"{D03}/D03_labels_meta.json"))["names"]
C=np.load(f"{OUT}/c02_C.npy"); idx={n:i for i,n in enumerate(names)}
MINVOX=5
print("═══ 指令 12：長程神經束驗證（降級版） ═══")
print("完整的綁束演算法需要每顆神經元在兩區的終點平均位置與最短路徑，")
print("D06 沒有存路徑，只有體素與離本體的深度 → 無法照補充材料實作。")
print("降級為：『有幾顆神經元同時碰到 AL_R 和另一個腦區』（每區至少 5 個體素）。")
print("與原版的差別：沒有做軌跡分群，所以算出來的是「連結對象」不是「神經束」。\n")
touch=(C>=MINVOX)
al=idx["AL_R"]; par=touch[:,al]
cnt=collections.Counter()
for j in np.flatnonzero(touch[par].sum(0)):
    if j and j!=al: cnt[names[j]]=int(touch[par][:,j].sum())
print(f"碰到 AL_R 的神經元 {int(par.sum()):,} 顆，其中同時碰到別區的：")
for k,v in cnt.most_common(8): print(f"   AL_R ↔ {k:10s} {v:5d} 顆")
print("\n═══ 指令 13：合併判斷 ═══")
tot=C.sum(1); top=C.max(1); arg=C.argmax(1); big=tot>0
def ln(Cx,nm,f=.80):
    t=Cx.sum(1); tp=Cx.max(1); ag=Cx.argmax(1); b=t>0; s=b&(tp>=f*t)
    return collections.Counter(nm[a] for a in ag[s]), int(s.sum())
c0,n0=ln(C,names)
G={"MB_R":["MB_CA_R","MB_PED_R","MB_VL_R","MB_ML_R"],
   "MB_L":["MB_CA_L","MB_PED_L","MB_VL_L","MB_ML_L"],
   "EBLT":["EB","BU_R","BU_L"]}
print("一、二、指定的三組")
for g,mem in G.items(): print(f"   {g:6s} 合併前 " + "、".join(f"{m} {c0.get(m,0)}" for m in mem))
C2=C.copy(); nm=list(names)
for g,mem in G.items():
    ii=[idx[m] for m in mem]; C2[:,ii[0]]=C[:,ii].sum(1); nm[ii[0]]=g
    for j in ii[1:]: C2[:,j]=0
c1,n1=ln(C2,nm)
for g in G: print(f"   {g:6s} 合併後 {c1.get(g,0)}")
print(f"   全腦 LN 候選 {n0} → {n1}（多了 {n1-n0}）")
print("\n三、全腦相鄰腦區兩兩試合併，看 LN 數增加最多的前十組")
lab=np.load(f"{D03}/work/labels_fc_1um.npy")
from scipy import ndimage
adj=set()
for ax in range(3):
    a=np.moveaxis(lab,ax,0)
    x=a[:-1].ravel(); y=a[1:].ravel(); k=(x>0)&(y>0)&(x!=y)
    adj|=set(map(lambda t:tuple(sorted(t)),zip(x[k].tolist(),y[k].tolist())))
print(f"   空間相鄰的腦區配對：{len(adj)} 組")
gain=[]
for i,j in adj:
    Cx=C.copy(); Cx[:,i]=C[:,i]+C[:,j]; Cx[:,j]=0
    nmx=list(names); nmx[i]=names[i]+"+"+names[j]
    cc,nn=ln(Cx,nmx)
    gain.append((nn-n0, names[i], names[j], cc.get(nmx[i],0), c0.get(names[i],0)+c0.get(names[j],0)))
gain.sort(reverse=True)
print("   增加最多的前十組：")
print("   Δ全腦   合併對象                      合併後  合併前(相加)")
for g,a,b,after,before in gain[:10]:
    print(f"   {g:5d}   {a:10s}+ {b:12s} {after:6d} {before:11d}")
json.dump({"tract_relaxed":dict(cnt.most_common(10)),
           "merge_named":{g:int(c1.get(g,0)) for g in G},"n0":n0,"n1":n1,
           "top10":[{"gain":int(g),"a":a,"b":b,"after":int(af),"before":int(bf)} for g,a,b,af,bf in gain[:10]]},
          open(f"{OUT}/c12_13.json","w"),ensure_ascii=False,indent=1)
