import asyncio, json, sys, numpy as np, soundfile as sf
sys.path.insert(0, ".")
from zh2en.mt import split_sentences, MockMT
from zh2en.audio.vad import EnergyVAD
from zh2en.pipeline import Pipeline
from zh2en.asr import MockASR
from zh2en.tts import build_tts, VoiceProfile
import importlib.util; spec = importlib.util.spec_from_file_location("enroll", "scripts/enroll_voice.py")
enroll = importlib.util.module_from_spec(spec); spec.loader.exec_module(enroll)

def test_split():
    assert list(split_sentences(iter(["Wow! That's ", "amazing. Really? ", "yes"]), min_chars=3)) == ["Wow!", "That's amazing.", "Really?", "yes"]

def make_profile(tmp):
    src = tmp / "src"; src.mkdir()
    rng = np.random.default_rng(0); t = np.arange(16000 * 5) / 16000
    a = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + 0.001 * rng.standard_normal(len(t))).astype(np.float32)
    sf.write(src / "a.wav", a, 16000); (src / "a.txt").write_text("今天天气不错", encoding="utf-8")
    enroll.main(["--name", "t", "--out", str(tmp / "voices"), "--from-dir", str(src)])
    return tmp / "voices" / "t"

def test_enroll_and_pipeline(tmp_path):
    d = make_profile(tmp_path)
    assert json.loads((d / "profile.json").read_text())["ref"]["ok"]
    cfg = {"tts": {"backend": "mock", "voice_profile": str(d)}, "mt": {}}
    tts = build_tts(cfg)
    out = []
    async def run():
        async def sink(c, sr, txt): out.append(txt)
        p = Pipeline(cfg, MockASR(), MockMT({}), tts, EnergyVAD(), sink); ts = p.start()
        tone = 0.2 * np.sin(np.arange(16000) / 5).astype(np.float32)
        for c in np.split(np.concatenate([tone, np.zeros(8000, np.float32)]), 15): await p.feed(c)
        for _ in range(50):
            await asyncio.sleep(0.02)
            if len(out) >= 2: break
        for t in ts: t.cancel()
    asyncio.run(run())
    assert out == ["What a lovely day!", "Let's go for a walk in the park."]

def test_requires_profile(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        build_tts({"tts": {"backend": "mock", "voice_profile": str(tmp_path / "none")}})
