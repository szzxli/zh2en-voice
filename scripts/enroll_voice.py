"""声音样本采集：引导录制若干段中文朗读，做采样率/时长/音量/信噪比校验与降噪，
保存 voices/<name>/{samples/*.wav, ref.wav, ref.txt, profile.json}。
用法: python scripts/enroll_voice.py --name me [--from-dir 已有wav目录] [--denoise]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

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

def record(seconds):
    import sounddevice as sd
    input("  按回车开始录音…"); print(f"  ● 录音中（最长 {seconds}s，读完按 Ctrl+C 提前结束）")
    buf = []
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=lambda d, *_: buf.append(d[:, 0].copy())):
            t = time.time()
            while time.time() - t < seconds: time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    return np.concatenate(buf) if buf else np.zeros(0, np.float32)

def load(p):
    import soundfile as sf
    a, sr = sf.read(p, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR); a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a, sr

def main(argv=None):
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="me"); ap.add_argument("--out", default="voices")
    ap.add_argument("--n", type=int, default=4, help="录制段数(3~6)")
    ap.add_argument("--from-dir", help="使用已有 wav（文件名同名 .txt 为转写）")
    ap.add_argument("--denoise", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.out) / a.name; (d / "samples").mkdir(parents=True, exist_ok=True)
    items = []
    if a.from_dir:
        for w in sorted(Path(a.from_dir).glob("*.wav")):
            t = w.with_suffix(".txt"); items.append((load(w), t.read_text(encoding="utf-8").strip() if t.exists() else ""))
    else:
        print("请在安静环境下，用自然、带情绪的语气朗读以下句子：")
        for i, txt in enumerate(PROMPTS[: a.n]):
            while True:
                print(f"\n[{i+1}/{a.n}] 「{txt}」")
                audio = record(MAX_S)
                ok, m, iss = check(trim(audio, SR), SR)
                print("  指标:", m)
                if ok or input(f"  问题: {iss}  重录? [Y/n] ").lower() == "n": break
            items.append(((audio, SR), txt))
    kept = []
    for i, ((audio, orig_sr), txt) in enumerate(items):
        audio = trim(audio, SR)
        if a.denoise: audio = denoise(audio, SR)
        ok, m, iss = check(audio, orig_sr if a.from_dir else SR)
        p = d / "samples" / f"{i:02d}.wav"; sf.write(p, audio, SR)
        (d / "samples" / f"{i:02d}.txt").write_text(txt, encoding="utf-8")
        kept.append({"file": str(p.name), "text": txt, "ok": ok, "issues": iss, **m})
        print(f"  {p.name}: {'✓' if ok else '✗ ' + ';'.join(iss)}")
    good = [k for k in kept if k["ok"]] or kept
    if not good: sys.exit("没有可用样本")
    # 参考音频：选 SNR 最高的单段（零样本克隆推荐 5~10s 单段，过长反而影响 CosyVoice2/F5）
    best = max(good, key=lambda k: (k["snr_db"], -abs(k["duration"] - 8)))
    ref, _ = sf.read(d / "samples" / best["file"], dtype="float32")
    sf.write(d / "ref.wav", ref, SR); (d / "ref.txt").write_text(best["text"], encoding="utf-8")
    (d / "profile.json").write_text(json.dumps({"name": a.name, "sample_rate": SR, "ref": best, "samples": kept,
        "language": "zh", "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 声音档案已保存到 {d}（参考段 {best['file']}，{best['duration']}s，SNR {best['snr_db']}dB）")

if __name__ == "__main__":
    main()
