# -*- coding: utf-8 -*-
"""稽核：頁面上的英文引文是否逐字出現在論文全文裡。

論文是雙欄排版，pdftotext -layout 會把左右欄放在同一行，
所以要用多個切欄位置各還原一次再比對（欄界每行不同）。
用法：python3 scripts/quote_audit.py paper.txt page1.html page2.html ...
"""
import io,re,sys

def norm(t):
    t=re.sub(r'\s+',' ',t)
    t=re.sub(r'(\w)-\s+(\w)',r'\1\2',t)   # 接回被換行切斷的字（demar- cated）
    t=re.sub(r'\s+([,.;:%\)])',r'\1',t)      # 去掉標點前的空格（剝 <b> 標籤造成的）
    t=re.sub(r'([\(])\s+',r'\1',t)
    return t.lower()

def corpora(path):
    """論文是雙欄，pdftotext -layout 把左右欄放在同一行、以 3 個以上空白分隔。
    以空白欄界切開，分別把每一欄串起來還原成連續的段落。"""
    raw=io.open(path,encoding='utf-8',errors='replace').read()
    yield norm(raw)                                     # 原樣（補充材料是單欄）
    lines=raw.split('\n')
    frags=[re.split(r'\s{3,}',l.strip()) for l in lines]
    for idx in (0,-1):                                  # 左欄、右欄
        yield norm(' '.join(f[idx] if f else '' for f in frags))

def main(paper,pages):
    corp=list(corpora(paper)); bad=0
    for pg in pages:
        s=io.open(pg,encoding='utf-8').read()
        # 區塊標籤換成換行（免得相鄰儲存格的字黏成同一句），行內標籤直接拿掉（引文常被 <b> 切開）
        # 程式碼不是引文：<pre> 與行內 <code> 一律先剝掉，否則 Python 會被當成英文句子
        t=re.sub(r'<(script|style|svg|pre)\b.*?</\1>',' ',s,flags=re.S)
        t=re.sub(r'<code\b[^>]*>.*?</code>',' ',t,flags=re.S)
        t=re.sub(r'</?(td|th|tr|table|p|div|li|ol|ul|h[1-6]|figcaption|figure|br|blockquote|nav)\b[^>]*>','\n',t)
        txt=re.sub(r'<[^>]+>','',t)
        print(pg); seen=set()
        for c in re.findall(r'[A-Za-z][A-Za-z0-9 ,\'’\.\-\(\)=<>%…]{25,}',txt):
            c2=re.sub(r'\s+',' ',c).strip(' .,—…'); k=norm(c2)
            if len(c2)<28 or k in seen: continue
            seen.add(k)
            parts=[norm(p) for p in re.split(r'…|\.\.\.',c2) if len(p.strip())>=20] or [k]
            hit=any(all(p in co for p in parts) for co in corp)
            print(('  OK  ' if hit else '  ??  ')+c2[:92]); bad += 0 if hit else 1
    print('未對到:',bad); return bad

if __name__=='__main__':
    sys.exit(1 if main(sys.argv[1],sys.argv[2:]) else 0)
