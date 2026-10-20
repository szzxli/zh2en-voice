"""VAD 分段：silero(sherpa-onnx 封装) 或能量阈值(调试)。输出完整语音段 float32@16k。"""
import numpy as np

class EnergyVAD:
    def __init__(self, sr=16000, min_silence_ms=300, max_segment_s=6.0, thresh=0.01):
        self.sr, self.thresh = sr, thresh
        self.min_sil = int(sr * min_silence_ms / 1000)
        self.max_len = int(sr * max_segment_s)
        self.buf, self.sil, self.active = [], 0, False

    def accept(self, chunk: np.ndarray):
        """返回已结束的语音段列表"""
        out = []
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        if rms > self.thresh:
            self.active, self.sil = True, 0
            self.buf.append(chunk)
        elif self.active:
            self.buf.append(chunk); self.sil += len(chunk)
            if self.sil >= self.min_sil:
                out.append(np.concatenate(self.buf)); self.buf, self.active = [], False
        if self.active and sum(map(len, self.buf)) >= self.max_len:
            out.append(np.concatenate(self.buf)); self.buf, self.sil = [], 0
        return out

class SileroVAD:
    def __init__(self, model, sr=16000, min_silence_ms=300, max_segment_s=6.0):
        import sherpa_onnx
        c = sherpa_onnx.VadModelConfig()
        c.silero_vad.model = model
        c.silero_vad.min_silence_duration = min_silence_ms / 1000
        c.silero_vad.max_speech_duration = max_segment_s
        c.sample_rate = sr
        self.vad = sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=30)
        self.win = c.silero_vad.window_size
        self.pending = np.zeros(0, np.float32)

    def accept(self, chunk):
        self.pending = np.concatenate([self.pending, chunk])
        while len(self.pending) >= self.win:
            self.vad.accept_waveform(self.pending[:self.win]); self.pending = self.pending[self.win:]
        out = []
        while not self.vad.empty():
            out.append(np.array(self.vad.front.samples, np.float32)); self.vad.pop()
        return out

def build_vad(cfg):
    v, sr = cfg["vad"], cfg["audio"]["sample_rate"]
    if v["backend"] == "silero":
        return SileroVAD(v["model"], sr, v["min_silence_ms"], v["max_segment_s"])
    return EnergyVAD(sr, v["min_silence_ms"], v["max_segment_s"])
