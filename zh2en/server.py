"""WebSocket 服务：手机浏览器打开 https://<电脑IP>:8765/ 收音，PCM16@16k 上行，float32 音频下行。"""
import asyncio, json, ssl, numpy as np
from pathlib import Path
from aiohttp import web
from .pipeline import Pipeline

def make_app(cfg, build):
    app = web.Application()
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    async def index(_): return web.Response(text=html, content_type="text/html")
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
    app.add_routes([web.get("/", index), web.get("/ws", ws)])
    return app

def run(cfg, build, cert=None, key=None):
    ctx = None
    if cert:
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH); ctx.load_cert_chain(cert, key)
    web.run_app(make_app(cfg, build), host=cfg["server"]["host"], port=cfg["server"]["port"], ssl_context=ctx)
