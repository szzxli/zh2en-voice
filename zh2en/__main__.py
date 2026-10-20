import argparse, asyncio, logging
from .config import load_config
from .asr import build_asr
from .mt import build_mt
from .tts import build_tts
from .audio.vad import build_vad

def main():
    ap = argparse.ArgumentParser("zh2en-voice")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--source", choices=["local", "web"])
    ap.add_argument("--cert"); ap.add_argument("--key")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(a.config)
    src = a.source or cfg["audio"]["source"]
    build = lambda: (build_asr(cfg), build_mt(cfg), build_tts(cfg), build_vad(cfg))
    if src == "web":
        from .server import run; return run(cfg, build, a.cert, a.key)
    from .pipeline import Pipeline
    from .audio.io import mic_stream, play
    async def go():
        asr, mt, tts, vad = build()
        async def sink(chunk, sr, text): await asyncio.to_thread(play, chunk, sr)
        p = Pipeline(cfg, asr, mt, tts, vad, sink); p.start()
        print("🎙  开始说中文（Ctrl+C 退出）")
        async for c in mic_stream(cfg["audio"]["sample_rate"], cfg["audio"]["block_ms"]):
            await p.feed(c)
    asyncio.run(go())

if __name__ == "__main__":
    main()
