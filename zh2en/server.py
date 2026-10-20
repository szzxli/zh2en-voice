"""aiohttp 服务：
- /            PWA demo 页面（提示句→录音→识别/英文→播放克隆音）
- /api/prompt  GET 随机提示句与是否已有声音档案
- /api/demo    POST PCM16 单声道 16k 原始字节（?prompt=...），返回 JSON（含 base64 wav）
- /ws          实时流式模式（PCM16@16k 上行，float32 音频下行）
"""
import asyncio, base64, io, json, random, ssl, numpy as np, soundfile as sf
from pathlib import Path
from aiohttp import web
from .pipeline import Pipeline

WEB = Path(__file__).resolve().parent.parent / "web"

def make_app(cfg, build):
    app = web.Application(client_max_size=20 * 1024 * 1024)
    state = {"demo": None, "lock": asyncio.Lock()}

    async def get_demo():
        if state["demo"] is None:
            from .demo import Demo
            state["demo"] = await asyncio.to_thread(Demo, cfg)
        return state["demo"]

    async def index(_): return web.FileResponse(WEB / "index.html")
    async def prompt(_):
        from .enroll import PROMPTS
        d = await get_demo()
        return web.json_response({"prompt": random.choice(PROMPTS), "has_profile": d.has_profile()})

    async def demo(req):
        pcm = np.frombuffer(await req.read(), np.int16).astype(np.float32) / 32768
        text = req.query.get("prompt", "")
        d = await get_demo()
        async with state["lock"]:
            res = await asyncio.to_thread(d.run, pcm, text, "out_web.wav")
        if isinstance(res, dict):
            return web.json_response(res, status=422)
        r, wav = res
        buf = io.BytesIO(); sf.write(buf, wav, d.tts.sr, format="WAV")
        r["wav_b64"] = base64.b64encode(buf.getvalue()).decode()
        return web.json_response(r)

    async def ws(req):
        sock = web.WebSocketResponse(); await sock.prepare(req)
        asr, mt, tts, vad = build()
        async def sink(chunk, sr, text):
            await sock.send_str(json.dumps({"sr": sr, "text": text}))
            await sock.send_bytes(chunk.astype(np.float32).tobytes())
        p = Pipeline(cfg, asr, mt, tts, vad, sink); tasks = p.start()
        async for msg in sock:
            if msg.type == web.WSMsgType.BINARY:
                await p.feed(np.frombuffer(msg.data, np.int16).astype(np.float32) / 32768)
        for t in tasks: t.cancel()
        return sock

    app.add_routes([web.get("/", index), web.get("/api/prompt", prompt), web.post("/api/demo", demo),
                    web.get("/ws", ws), web.static("/", WEB)])
    return app

def run(cfg, build, cert=None, key=None):
    ctx = None
    if cert:
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH); ctx.load_cert_chain(cert, key)
    web.run_app(make_app(cfg, build), host=cfg["server"]["host"], port=cfg["server"]["port"], ssl_context=ctx)
