# LPU：果蠅腦的「在地處理單元」

BSC 專題課程。圍繞單一概念 **LPU（local processing unit）** 的教材網頁，
出自 Chiang et al. (2011) *Current Biology* 21(1), 1–11，
doi:10.1016/j.cub.2010.11.056。

線上版：https://marginli.github.io/bsc-lpu/

## 五個部分

| 頁面 | 內容 |
|---|---|
| `index.html` | 首頁，五個按鈕 |
| `what-is-lpu.html` | PART 1　什麼是 LPU——定義、與 neuropil／hub 的差別、全腦 41 個 LPU |
| `how-to-draw-lpu.html` | PART 2　如何畫出 LPU——論文的七步驟流程，以及照著跑會發現的**十件必須自己決定的事** |
| `agent-prompts.html` | PART 3　要怎麼交代 AI Agent——把流程寫成 14 條指令，重點在那十個空缺各寫在哪一句 |
| `agent-run.html` | PART 4　真的讓 AI Agent 跑一遍——實跑紀錄，含失敗與修正（此頁適用規範第 2 節的例外） |
| `python-lpu.html` | PART 5　讀懂 AI 寫出來的程式——十六支整合成三個檔案，用五個問題讀完 |

`assets/lpu.css` 為六頁共用樣式，沿用 BSC 線上課程的視覺慣例。

## 目錄

| 目錄 | 內容 |
|---|---|
| `scripts/` | PART 1、2 的產圖程式，以及兩支可沿用的稽核程式（`quote_audit.py` 引文對回論文全文、`svg_audit.py` 檢查 SVG 有沒有超出 viewBox） |
| `part4/` | PART 4 實跑用的十六支程式與其產出（`scripts/`、`out/`） |
| `part5/` | 整合後的三個檔案＋驗收表（`run_lpu.sh`／`run_lpu.py`／`lpu_lib.py`／`expected.json`），另見 `part5/README.md` |
| `_notes/` | 製作過程的修正紀錄。**進 repo，但由 workflow 的 `rm -rf _notes` 擋在網站之外** |

## 狀態

**五個 PART 都已完成。** 所有圖都是用本機資料（D03 的 FCWBNP 分區標籤 ＋ D06 的
28,573 顆骨架體素）自己算出來的真實結果，不重製論文原圖。

`part5/` 的管線可重跑：`./run_lpu.sh`，實測 165 秒、47 項驗收全過，
判定為 8 區單一 LPU、13 區多個候選、54 區候選 hub。

**已知的天花板**：STEP 7（長程神經束驗證）因為 D06 沒有存路徑而是降級版
（用腦區相鄰關係代替）；到目前為止沒有任何一個候選 LPU 真的做過 STEP 7。
整條流程通得過「換參數穩、打亂體素順序穩、c 值高於隨機對照」三道檢查的候選
只有一個——AVLP_R 的 138 顆 LN 族群。

`robots.txt` 擋搜尋引擎收錄。

## 發布

推到 `main` 由 GitHub Actions 自動部署到 GitHub Pages。
