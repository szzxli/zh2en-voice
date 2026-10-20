"""中文 ASR 后端。transcribe(seg)->str（段级，VAD 切分后的流式识别）。"""
import numpy as np

class MockASR:
    def transcribe(self, seg, sr=16000):
        return "今天天气真好，我们去公园散步吧！"

class SherpaParaformerASR:
    """sherpa-onnx 流式 Paraformer（CPU, RTF≈0.05~0.1）。"""
    def __init__(self, model_dir, num_threads=4):
        import sherpa_onnx, os
        j = lambda n: os.path.join(model_dir, n)
        self.rec = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=j("tokens.txt"), encoder=j("encoder.int8.onnx"), decoder=j("decoder.int8.onnx"),
            num_threads=num_threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")

    def transcribe(self, seg, sr=16000):
        s = self.rec.create_stream()
        s.accept_waveform(sr, seg)
        s.accept_waveform(sr, np.zeros(int(sr * 0.5), np.float32))  # 尾部填充
        s.input_finished()
        while self.rec.is_ready(s):
            self.rec.decode_stream(s)
        return self.rec.get_result(s).strip()

class FunASRStreaming:
    """FunASR paraformer-zh-streaming + ct-punc（CPU 可跑，依赖 torch）。"""
    def __init__(self):
        from funasr import AutoModel
        self.m = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", disable_update=True)
    def transcribe(self, seg, sr=16000):
        r = self.m.generate(input=seg)
        return r[0]["text"] if r else ""

def build_asr(cfg):
    a = cfg["asr"]
    if a["backend"] == "sherpa_onnx":
        return SherpaParaformerASR(a["model_dir"], a["num_threads"])
    if a["backend"] == "funasr":
        return FunASRStreaming()
    return MockASR()
