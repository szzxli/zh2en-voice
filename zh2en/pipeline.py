"""三级流水线：VAD/ASR -> MT(流式按句) -> TTS(克隆音色)，asyncio 队列 + 线程池并发。"""
import asyncio, time, logging
log = logging.getLogger("zh2en")

class Pipeline:
    def __init__(self, cfg, asr, mt, tts, vad, sink):
        self.cfg, self.asr, self.mt, self.tts, self.vad, self.sink = cfg, asr, mt, tts, vad, sink
        self.q_seg, self.q_sent = asyncio.Queue(8), asyncio.Queue(32)

    async def feed(self, chunk):
        for seg in self.vad.accept(chunk):
            await self.q_seg.put((time.perf_counter(), seg))

    async def _asr_mt(self):
        loop = asyncio.get_running_loop()
        while True:
            t0, seg = await self.q_seg.get()
            zh = await asyncio.to_thread(self.asr.transcribe, seg)
            t1 = time.perf_counter()
            if not zh: continue
            log.info("ASR %.0fms | %s", (t1 - t0) * 1e3, zh)
            def run():
                for s in self.mt.translate_sentences(zh):
                    loop.call_soon_threadsafe(self.q_sent.put_nowait, (t0, s))
            await asyncio.to_thread(run)

    async def _tts(self):
        while True:
            t0, en = await self.q_sent.get()
            first = True
            gen = self.tts.synth(en)
            while True:
                chunk = await asyncio.to_thread(next, gen, None)
                if chunk is None: break
                if first:
                    log.info("首包 E2E %.0fms | %s", (time.perf_counter() - t0) * 1e3, en); first = False
                await self.sink(chunk, self.tts.sr, en)

    def start(self):
        return [asyncio.create_task(self._asr_mt()), asyncio.create_task(self._tts())]
