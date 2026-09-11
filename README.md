# LPU：果蠅腦的「在地處理單元」

BSC 專題課程。圍繞單一概念 **LPU（local processing unit）** 的教材網頁，
出自 Chiang et al. (2011) *Current Biology* 21(1), 1–11，
doi:10.1016/j.cub.2010.11.056。

線上版：https://marginli.github.io/bsc-lpu/

## 五個部分

| 頁面 | 內容 |
|---|---|
| `index.html` | 首頁：**11 分半的說明影片** ＋ 五個按鈕 |
| `what-is-lpu.html` | PART 1　什麼是 LPU——定義、與 neuropil／hub 的差別、全腦 41 個 LPU |
| `how-to-draw-lpu.html` | PART 2　如何畫出 LPU——論文的七步驟流程，以及照著跑會發現的**十件必須自己決定的事** |
| `agent-prompts.html` | PART 3　要怎麼交代 AI Agent——把流程寫成 14 條指令，重點在那十個空缺各寫在哪一句 |
| `agent-run.html` | PART 4　真的讓 AI Agent 跑一遍——實跑紀錄，含失敗與修正（此頁適用規範第 2 節的例外） |
| `python-lpu.html` | PART 5　讀懂 AI 寫出來的程式——十六支整合成三個檔案，用五個問題讀完 |

`assets/lpu.css` 為六頁共用樣式，沿用 BSC 線上課程的視覺慣例。

首頁的說明影片 `assets/lpu-overview.mp4`（11 分 29 秒、1080p、12 MB、17 張投影片）與字幕 `assets/lpu-overview.vtt` 由 `scripts/make_video.py` 產生：投影片以 headless Chrome 截圖、旁白用 `edge-tts` 合成、再用 `ffmpeg`（`libopenh264`）合成。改內容就重跑那支程式。

## 目錄

| 目錄 | 內容 |
|---|---|
| `downloads/` | 學員下載用的程式包 `lpu-part5-code.zip`（44 KB）。**資料包不在這裡**——38 MB 的 `lpu-part5-data.zip` 放在 GitHub Release `data-v1`，不進版本控制的歷史 |
| `scripts/` | PART 1、2 的產圖程式、說明影片的產生器（`make_video.py` ＋ `video_slides.py`），以及**四支可以直接沿用到下一個專題的稽核程式**：`page_audit.py`（標籤平衡／連結錨點／中英夾雜／SVG 內的 `<b>`）、`content_audit.py`（術語第一次出現／表格有沒有標我們用哪一個／口語擬人／抽所有數字）、`quote_audit.py`（引文對回論文全文）、`svg_audit.py`（SVG 有沒有超出 viewBox）|
| `part4/` | PART 4 實跑用的十六支程式與其產出（`scripts/`、`out/`） |
| `part5/` | 整合後的三個檔案＋驗收表（`run_lpu.sh`／`run_lpu.py`／`lpu_lib.py`／`expected.json`），另見 `part5/README.md` |
| `_notes/` | 製作過程的修正紀錄。**進 repo，但由 workflow 的 `rm -rf _notes` 擋在網站之外** |

## 狀態

**五個 PART 都已完成。** 所有圖都是用本機資料（D03 的 FCWBNP 分區標籤 ＋ D06 的
28,573 顆骨架體素）自己算出來的真實結果，不重製論文原圖。

**PART 2 到 PART 5 用的是同一個 v(r) 定義**（同一顆神經元在同一格只算一次，即圖 S5D 圖說的
「重複註冊次數」），所以 PART 2 的圖、PART 3 的驗收表、PART 4 的實跑、PART 5 的程式
可以逐項對照：右觸角葉 v(r) 最大 142.8、熱點 3,130 個分析體素、切割高度 8 分成五群。

`part5/` 的管線可重跑：`./run_lpu.sh`，實測 170 秒、56 項驗收全過，
判定為 8 區單一 LPU、13 區多個候選、48 區 hub、6 區兩者皆非。

**七個步驟都跑完了，包含 STEP 7。** 長程神經束驗證需要神經元的樹狀拓樸，
D06 柵格化之後只剩體素集合，所以一度是降級版；原始的 FlyCircuit SWC 有
`idpar` 欄，拿回來就做得到（28,573 顆掃出 17,690 條跨兩區路徑、2,420 條神經束）。
「hub」與「兩者皆非」的分別就是這樣來的——論文對 hub 的定義是
「有長程神經束但沒有 LN 族群」，所以連神經束也沒有的 6 區不能叫 hub。

**通得過全部四道檢查的候選只有一個**——AVLP_R 的 138 顆 LN 族群：
換參數穩、打亂體素順序穩、c 值高於隨機對照、而且有自己的長程神經束
（20 條，與同一個腦區另一個候選的 30 條完全不重疊；隨機對照最少也共用 8 條）。

**還沒解掉的**：STEP 7 的綁束方式論文沒交代（相容關係不遞移時怎麼辦），
換一種綁法束數會差一個數量級。

## 學員要下載的東西

PART 5 的〈下載：把這兩包拿走〉一節（`python-lpu.html#download`）給兩個連結：

| 包 | 大小 | 放在哪 | 內容 |
|---|---|---|---|
| `lpu-part5-code.zip` | 44 KB | `downloads/`，由 Pages 提供 | `run_lpu.sh`／`run_lpu.py`／`lpu_lib.py`／`expected.json`／`README.md` ＋ `SHA256SUMS.txt` |
| `lpu-part5-data.zip` | 38 MB | GitHub Release **`data-v1`** | 八個資料檔壓成 `.npz`（D03 的分區標籤、D06 的骨架體素）＋ `out/c12_paths.csv` ＋ `來源與出處.md` ＋ `SHA256SUMS.txt` |

**資料包用 Release 不用 repo**：38 MB 進了 git 歷史就拿不掉，而資料是會換版的。
換版時重打一包、`gh release create data-v2`，再改頁面上那一個連結即可。

兩包合起來在一台空目錄的機器上實跑過：**170 秒、56 項驗收全過，
16 份 JSON 與 7 張圖與 `part5/out/` 逐位元組相同**。

重打資料包：`python3 part5/run_lpu.py pack --pack-to <某個目錄>`，
再把 `part5/out/c12_paths.csv` 放進 `<某個目錄>/out/`，連同 `來源與出處.md` 一起壓。

`robots.txt` 擋搜尋引擎收錄。

## 發布

推到 `main` 由 GitHub Actions 自動部署到 GitHub Pages。
