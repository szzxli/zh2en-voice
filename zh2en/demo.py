"""可测试 demo：显示中文提示 → 录音(回车开始/VAD 结束) 或 --input-wav → ASR → 保留语气译英 → 克隆音色 TTS → 播放并保存 out.wav。
没有声音档案时，本次录音（文本已知=提示句）同时作为克隆参考；绝不回退预设音色。

python -m zh2en.demo                                   # 麦克风
python -m zh2en.demo --input-wav x.wav --prompt-text "x.wav 的中文原文" --no-play
"""
import argparse, json, logging, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
from .config import load_config
from .enroll import PROMPTS, SR, check, trim, save_profile

def load_wav16k(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR); a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a, sr

def record_until_silence(cfg, max_s=15.0):
    import sounddevice as sd
    from .audio.vad import EnergyVAD
    input("按回车开始录音（说完停顿约 0.8 秒自动结束）…")
    vad = EnergyVAD(SR, min_silence_ms=800, max_segment_s=max_s, thresh=0.015)
    blk, got, buf = int(SR * 0.05), [], []
    print("● 录音中…")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=blk) as st:
        t = time.time()
        while not got and time.time() - t < max_s + 2:
            x, _ = st.read(blk); x = x[:, 0].copy(); buf.append(x); got = vad.accept(x)
    return (got[0] if got else np.concatenate(buf)), SR

class Demo:
    """CLI 与网页共用的核心。组件延迟加载，模型只加载一次。"""
    def __init__(self, cfg):
        self.cfg = cfg
        from .asr import build_asr
        from .mt import build_mt
        t = time.perf_counter(); self.asr = build_asr(cfg); self.mt = build_mt(cfg)
        self.load_s = {"asr+mt": time.perf_counter() - t}
        self.tts = None
        self.profile_dir = Path(cfg["tts"]["voice_profile"])

    def has_profile(self):
        return (self.profile_dir / "profile.json").exists()

    def _ensure_tts(self):
        from .tts import build_tts, VoiceProfile
        if self.tts is None:
            t = time.perf_counter(); self.tts = build_tts(self.cfg); self.load_s["tts"] = time.perf_counter() - t

    def run(self, audio, prompt_text, out_wav="out.wav", orig_sr=SR):
        r = {"prompt_text": prompt_text, "latency_ms": {}}
        L = r["latency_ms"]
        audio = trim(audio, SR)
        ok, metrics, issues = check(audio, orig_sr)
        r["input_check"] = {"ok": ok, **metrics, "issues": issues}
        if not self.has_profile():
            # 首次：本次录音即克隆参考（文本已知）
            t = time.perf_counter()
            save_profile(self.profile_dir, [(audio, prompt_text, ok, issues, metrics)], SR)
            L["enroll"] = (time.perf_counter() - t) * 1e3; r["enrolled"] = True
            if self.tts is not None:
                from .tts import VoiceProfile; self.tts.set_profile(VoiceProfile(self.profile_dir))
        self._ensure_tts()
        t0 = time.perf_counter()
        zh = self.asr.transcribe(audio, SR); t1 = time.perf_counter()
        L["asr"] = (t1 - t0) * 1e3; r["zh"] = zh
        if not zh:
            r["error"] = "未识别到语音"; return r
        en, chunks, first_tts = [], [], None
        sents = self.mt.translate_sentences(zh)
        t_mt = time.perf_counter(); mt_first = None; tts_total = 0.0
        for s in sents:
            now = time.perf_counter()
            if mt_first is None: mt_first = now - t_mt
            en.append(s); ts = time.perf_counter()
            for c in self.tts.synth(s):
                if first_tts is None:
                    first_tts = time.perf_counter() - ts; L["e2e_first_audio"] = (time.perf_counter() - t0) * 1e3
                chunks.append(c)
            tts_total += time.perf_counter() - ts
        L["mt_first_sentence"] = mt_first * 1e3
        L["tts_first_chunk"] = (first_tts or 0) * 1e3
        L["tts_total"] = tts_total * 1e3
        L["total"] = (time.perf_counter() - t0) * 1e3
        wav = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        sf.write(out_wav, wav, self.tts.sr)
        r.update(en=" ".join(en), out_wav=str(Path(out_wav).resolve()), audio_s=round(len(wav) / self.tts.sr, 2),
                 rtf_tts=round(tts_total / max(len(wav) / self.tts.sr, 1e-6), 2), tts_backend=self.cfg["tts"]["backend"])
        r["latency_ms"] = {k: round(v) for k, v in L.items()}
        return r, wav

def main(argv=None):
    ap = argparse.ArgumentParser("zh2en.demo")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--input-wav", help="用 wav 代替麦克风")
    ap.add_argument("--prompt-text", help="提示/朗读文本（--input-wav 时填该音频的原文）")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="重复运行次数（第 2 次起为热启动延迟）")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(a.config)
    prompt = a.prompt_text or random.choice(PROMPTS)
    print(f"\n请朗读：「{prompt}」\n")
    if a.input_wav:
        audio, orig_sr = load_wav16k(a.input_wav)
    else:
        audio, orig_sr = record_until_silence(cfg)
    d = Demo(cfg)
    if not d.has_profile():
        print("（尚无声音档案：本次录音将同时作为你的克隆参考音色）")
    for i in range(a.repeat):
        res = d.run(audio, prompt, a.out, orig_sr)
        if isinstance(res, dict):
            print(json.dumps(res, ensure_ascii=False, indent=2)); sys.exit(1)
        r, wav = res
        print(f"\n--- 第 {i+1} 次 ---")
        print("识别:", r["zh"]); print("英文:", r["en"])
        print("延迟(ms):", r["latency_ms"], "| TTS RTF:", r["rtf_tts"], "| 模型加载(s):", {k: round(v, 1) for k, v in d.load_s.items()})
        print("输出:", r["out_wav"])
    Path(a.out).with_suffix(".json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    if not a.no_play:
        from .audio.io import play; play(wav, d.tts.sr)

if __name__ == "__main__":
    main()
