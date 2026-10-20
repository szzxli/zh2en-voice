"""声音样本采集：引导录制若干段中文朗读，做采样率/时长/音量/信噪比校验与降噪，
保存 voices/<name>/{samples/*.wav, ref.wav, ref.txt, profile.json}。
用法: python scripts/enroll_voice.py --name me [--from-dir 已有wav目录] [--denoise]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zh2en.enroll import PROMPTS, SR, MIN_S, MAX_S, check, trim, denoise, save_profile

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
    proc = []
    for (audio, orig_sr), txt in items:
        audio = trim(audio, SR)
        if a.denoise: audio = denoise(audio, SR)
        ok, m, iss = check(audio, orig_sr if a.from_dir else SR)
        print(f"  {txt[:12]}…: {'✓' if ok else '✗ ' + ';'.join(iss)}")
        proc.append((audio, txt, ok, iss, m))
    if not proc: sys.exit("没有可用样本")
    best = save_profile(d, proc, SR)
    print(f"\n✅ 声音档案已保存到 {d}（参考段 {best['file']}，{best['duration']}s，SNR {best['snr_db']}dB）")

if __name__ == "__main__":
    main()
