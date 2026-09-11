#!/usr/bin/env python3
"""產生首頁那支 6 分鐘的說明影片。

   內容在 video_slides.py（投影片要點＋旁白稿），這支只負責產出：
     1. 每張投影片用 headless Chrome 截成 1920×1080 的 PNG
     2. 每段旁白用 edge-tts 合成（zh-TW-HsiaoChenNeural，語速 -8%）
     3. 依各段長度把 PNG 排成影像軌，與音軌一起用 ffmpeg 合成 MP4
     4. 依各段長度切出 WebVTT 字幕

   改了投影片或旁白就重跑：  python3 scripts/make_video.py
   產出：assets/lpu-overview.mp4、assets/lpu-overview.vtt、assets/poster.jpg

   注意：edge-tts 會把旁白文字送到微軟的語音服務（內容本來就是公開教材）。
"""
import io, os, re, json, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from video_slides import SLIDES          # noqa: E402

ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "_video")      # 中間檔，不進 repo（.gitignore）
OUT = os.path.join(ROOT, "assets")
VOICE, RATE, PAD = "zh-TW-HsiaoChenNeural", "-8%", 0.6
FIX = [("B S C", "BSC"), ("L P U", "LPU"), ("A I", "AI"),
       ("V L P", "VLP"), ("U P G M A", "UPGMA"), ("D 0 6", "D06")]

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{width:1920px;height:1080px;background:#fbfcfd;font:400 17px/1.7 "Noto Sans CJK TC","Noto Sans TC",sans-serif;color:#1b2733;overflow:hidden}
.bar{height:10px;background:linear-gradient(90deg,#2563eb 0%,#7c3aed 55%,#d97706 100%)}
.wrap{padding:56px 84px 0;height:1070px;display:flex;flex-direction:column}
.eyebrow{color:#2563eb;font-weight:700;font-size:24px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px}
h1{font-size:58px;line-height:1.28;font-weight:700;letter-spacing:-.01em}
h1 b{color:#2563eb}
.body{display:flex;gap:60px;margin-top:40px;flex:1;min-height:0}
.col{flex:1;min-width:0}
ul{list-style:none}
li{position:relative;padding-left:40px;margin-bottom:26px;font-size:31px;line-height:1.62;color:#2b3a48}
li:before{content:"";position:absolute;left:0;top:18px;width:20px;height:5px;border-radius:3px;background:#2563eb}
li b{color:#0f1b26;font-weight:700}
.fig{flex:0 0 720px;display:flex;flex-direction:column;justify-content:center}
.fig img{width:100%;max-height:700px;object-fit:contain;border:1px solid #e4e8ee;border-radius:14px;background:#fff}
.cap{font-size:19px;color:#7a8794;margin-top:14px;line-height:1.6}
.full li{font-size:34px}
.foot{display:flex;justify-content:space-between;align-items:center;padding:22px 0 26px;color:#9aa7b4;font-size:19px;border-top:1px solid #e8ecf1;margin-top:auto}
.pg{font-variant-numeric:tabular-nums}
"""


def slide_html(s, i, n):
    fig = (f'<div class="fig"><img src="file://{s["img"]}">'
           f'<div class="cap">{s["imgcap"]}</div></div>') if s["img"] else ""
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<style>{CSS}</style></head><body><div class="bar"></div><div class="wrap">'
            f'<div class="eyebrow">{s["eyebrow"]}</div><h1>{s["title"]}</h1>'
            f'<div class="body"><div class="{"col" if s["img"] else "col full"}">'
            f'<ul>{"".join(f"<li>{b}</li>" for b in s["bullets"])}</ul></div>{fig}</div>'
            f'<div class="foot"><div>BSC 腦空間體研究中心 · 專題課程 LPU　｜　marginli.github.io/bsc-lpu</div>'
            f'<div class="pg">{i + 1} / {n}</div></div></div></body></html>')


def ts(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def main():
    os.makedirs(f"{WORK}/png", exist_ok=True)
    os.makedirs(f"{WORK}/aud", exist_ok=True)
    durs = []
    for i, s in enumerate(SLIDES):
        p = f"{WORK}/s{i:02d}.html"
        io.open(p, "w", encoding="utf-8").write(slide_html(s, i, len(SLIDES)))
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--hide-scrollbars", "--window-size=1920,1080", "--virtual-time-budget=3000",
                        f"--screenshot={WORK}/png/s{i:02d}.png", f"file://{p}"], capture_output=True)
        a = f"{WORK}/aud/a{i:02d}.mp3"
        if not os.path.exists(a):
            subprocess.run(["edge-tts", "--voice", VOICE, f"--rate={RATE}",
                            "--text", s["say"], "--write-media", a], capture_output=True, timeout=300)
        durs.append(float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a],
            capture_output=True, text=True).stdout.strip()))
        print(f"   投影片 {i + 1}/{len(SLIDES)}　旁白 {durs[-1]:.1f} 秒")

    with io.open(f"{WORK}/ca.txt", "w") as f:
        for i in range(len(durs)):
            f.write(f"file 'aud/a{i:02d}.mp3'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", f"{WORK}/ca.txt",
                    "-af", f"apad=pad_dur={PAD}", "-c:a", "aac", "-b:a", "128k",
                    f"{WORK}/voice.m4a"], capture_output=True, cwd=WORK)
    with io.open(f"{WORK}/cv.txt", "w") as f:
        for i, d in enumerate(durs):
            f.write(f"file 'png/s{i:02d}.png'\nduration {d + PAD:.3f}\n")
        f.write(f"file 'png/s{len(durs) - 1:02d}.png'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", f"{WORK}/cv.txt",
                    "-i", f"{WORK}/voice.m4a", "-c:v", "libopenh264", "-b:v", "900k",
                    "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "24", "-c:a", "copy",
                    "-shortest", "-movflags", "+faststart", f"{OUT}/lpu-overview.mp4"],
                   capture_output=True, cwd=WORK)

    vtt, t = ["WEBVTT", ""], 0.0
    for s, d in zip(SLIDES, durs):
        sents = [x for x in re.split(r"(?<=[。！？])", s["say"]) if x.strip()]
        tot, cur = sum(len(x) for x in sents), t
        for x in sents:
            dur = d * len(x) / tot
            for a, b in FIX:
                x = x.replace(a, b)
            vtt += [f"{ts(cur)} --> {ts(cur + dur)}", x.strip(), ""]
            cur += dur
        t += d + PAD
    io.open(f"{OUT}/lpu-overview.vtt", "w", encoding="utf-8").write("\n".join(vtt))
    from PIL import Image
    Image.open(f"{WORK}/png/s00.png").convert("RGB").save(f"{OUT}/poster.jpg", quality=86, optimize=True)
    print(f"\n完成：{sum(durs) + len(durs) * PAD:.0f} 秒，"
          f"{os.path.getsize(f'{OUT}/lpu-overview.mp4') / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
