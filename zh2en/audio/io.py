"""本机麦克风输入 / 扬声器输出（sounddevice）。"""
import asyncio, numpy as np

async def mic_stream(sr=16000, block_ms=100):
    import sounddevice as sd
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    def cb(indata, frames, t, status):
        loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=int(sr*block_ms/1000), callback=cb):
        while True:
            yield await q.get()

def play(audio: np.ndarray, sr: int):
    import sounddevice as sd
    sd.play(audio, sr); sd.wait()
