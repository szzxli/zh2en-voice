"""离线基准：读取一段中文 wav，测量 ASR / MT 首句 / TTS 首包耗时。
python scripts/bench.py test.wav"""
import sys, time, soundfile as sf, numpy as np
sys.path.insert(0, ".")
from zh2en.config import load_config
from zh2en.asr import build_asr
from zh2en.mt import build_mt
from zh2en.tts import build_tts
cfg = load_config(); a, sr = sf.read(sys.argv[1], dtype="float32")
asr, mt, tts = build_asr(cfg), build_mt(cfg), build_tts(cfg)
t = time.perf_counter(); zh = asr.transcribe(a); t1 = time.perf_counter()
s = next(mt.translate_sentences(zh)); t2 = time.perf_counter()
next(tts.synth(s)); t3 = time.perf_counter()
print(f"ASR {1e3*(t1-t):.0f}ms  MT首句 {1e3*(t2-t1):.0f}ms  TTS首包 {1e3*(t3-t2):.0f}ms  合计 {1e3*(t3-t):.0f}ms")
print(zh, "->", s)
