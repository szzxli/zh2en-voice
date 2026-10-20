"""声音档案核心：提示句、质量校验、裁剪/降噪、保存 profile。"""
import json, time
from pathlib import Path
import numpy as np
import soundfile as sf

PROMPTS = [
    "今天天气不错，我打算下午去公园散散步，顺便买杯咖啡。",
    "哇，这个消息太让人激动了！我们终于成功了！",
    "唉，说实话，我有点累了，真想好好睡一觉。",
    "请问一下，去火车站应该怎么走？大概需要多长时间？",
    "我觉得这个方案还可以再改进一下，你怎么看？",
    "一二三四五，上山打老虎，老虎没打着，打到小松鼠。",
]
SR = 16000
MIN_S, MAX_S = 3.0, 15.0

def check(a: np.ndarray, sr: int):
    """返回 (ok, 指标, 问题列表)"""
    issues = []
    dur = len(a) / sr
    peak = float(np.max(np.abs(a))) if len(a) else 0
    frame = int(sr * 0.02)
    fr = a[: len(a) // frame * frame].reshape(-1, frame)
    e = np.sqrt((fr ** 2).mean(1) + 1e-12)
    noise, speech = np.percentile(e, 10), np.percentile(e, 90)
    snr = 20 * np.log10(speech / max(noise, 1e-6))
    clip = float((np.abs(a) > 0.99).mean())
    if sr < 16000: issues.append(f"采样率 {sr} < 16000")
    if dur < MIN_S: issues.append(f"时长 {dur:.1f}s 过短(≥{MIN_S}s)")
    if dur > MAX_S: issues.append(f"时长 {dur:.1f}s 过长(≤{MAX_S}s)")
    if peak < 0.05: issues.append("音量过低，请靠近麦克风")
    if clip > 0.001: issues.append("出现削波，请调小增益/远离麦克风")
    if snr < 20: issues.append(f"信噪比 {snr:.1f}dB 过低(≥20dB)，请换安静环境")
    return not issues, {"duration": round(dur, 2), "peak": round(peak, 3), "snr_db": round(float(snr), 1), "clip_ratio": clip}, issues

def trim(a, sr, th_db=-40):
    e = np.abs(a); th = 10 ** (th_db / 20) * e.max()
    idx = np.where(e > th)[0]
    if not len(idx): return a
    pad = int(0.15 * sr)
    return a[max(idx[0] - pad, 0): idx[-1] + pad]

def denoise(a, sr):
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=a, sr=sr, stationary=True, prop_decrease=0.8).astype(np.float32)
    except ImportError:
        print("  (未安装 noisereduce，跳过降噪)"); return a

def save_profile(d, items, sr=SR):
    """items: [(audio, text, ok, issues, metrics)] -> 写 samples/、ref.wav、ref.txt、profile.json"""
    d = Path(d); (d / "samples").mkdir(parents=True, exist_ok=True)
    kept = []
    for i, (audio, txt, ok, iss, m) in enumerate(items):
        p = d / "samples" / f"{i:02d}.wav"; sf.write(p, audio, sr)
        (d / "samples" / f"{i:02d}.txt").write_text(txt, encoding="utf-8")
        kept.append({"file": p.name, "text": txt, "ok": ok, "issues": iss, **m})
    good = [k for k in kept if k["ok"]] or kept
    best = max(good, key=lambda k: (k["snr_db"], -abs(k["duration"] - 8)))
    ref, _ = sf.read(d / "samples" / best["file"], dtype="float32")
    sf.write(d / "ref.wav", ref, sr); (d / "ref.txt").write_text(best["text"], encoding="utf-8")
    (d / "profile.json").write_text(json.dumps({"name": d.name, "sample_rate": sr, "ref": best, "samples": kept,
        "language": "zh", "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    return best
