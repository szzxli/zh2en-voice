"""英文 TTS —— 必须使用用户克隆音色（跨语种：中文参考音频 -> 英文语音）。
预设音色仅在 tts.debug_preset=true 时用于调试，绝不作为默认输出。"""
import json, numpy as np
from pathlib import Path

class VoiceProfile:
    """enroll_voice.py 产物：voices/<name>/{ref.wav, ref.txt, profile.json, samples/*}"""
    def __init__(self, d):
        self.dir = Path(d)
        meta = self.dir / "profile.json"
        if not meta.exists():
            raise FileNotFoundError(f"未找到声音档案 {meta}，请先运行: python scripts/enroll_voice.py --name me")
        self.meta = json.loads(meta.read_text(encoding="utf-8"))
        self.ref_wav = str(self.dir / "ref.wav")
        self.ref_text = (self.dir / "ref.txt").read_text(encoding="utf-8").strip()

class CosyVoice2TTS:
    """默认。CosyVoice2-0.5B（Apache-2.0），零样本跨语种克隆 + 双向流式，首包 ~150ms(GPU)。"""
    def __init__(self, cfg, profile: VoiceProfile):
        import sys, types, importlib.util
        if importlib.util.find_spec("pyworld") is None:  # pyworld 仅训练数据处理用，推理不需要；无编译器时打桩
            sys.modules["pyworld"] = types.ModuleType("pyworld")
        sys.path += [cfg["repo_dir"], cfg["repo_dir"] + "/third_party/Matcha-TTS"]
        import torch
        if cfg.get("num_threads"):
            torch.set_num_threads(int(cfg["num_threads"]))
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.m = CosyVoice2(cfg["model_dir"], load_jit=False, load_trt=False, fp16=False)
        self.sr = self.m.sample_rate
        self.stream_mode = cfg.get("stream", True)
        self.set_profile(profile)

    def set_profile(self, profile: VoiceProfile):
        """预计算说话人特征（跨语种：不用参考转写，避免中文韵律带入英文）"""
        self.prompt = profile.ref_wav
        self.spk_id = "user"
        self.m.add_zero_shot_spk("", self.prompt, self.spk_id)

    def synth(self, text):
        for out in self.m.inference_cross_lingual(text, self.prompt, zero_shot_spk_id=self.spk_id, stream=self.stream_mode):
            yield out["tts_speech"].numpy().reshape(-1).astype(np.float32)

class OpenVoiceV2TTS:
    """CPU 备选。OpenVoice V2（MIT）：MeloTTS 英文生成 + Tone Color Converter 转为用户音色。
    最终输出为用户音色（音色转换），韵律来自基础模型；克隆相似度略低于 CosyVoice2。"""
    def __init__(self, cfg, profile: VoiceProfile):
        import torch
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS
        d = cfg["ckpt_dir"]
        self.conv = ToneColorConverter(f"{d}/converter/config.json", device="cpu")
        self.conv.load_ckpt(f"{d}/converter/checkpoint.pth")
        self.tgt_se, _ = se_extractor.get_se(profile.ref_wav, self.conv, vad=True)
        self.base = TTS(language="EN", device="cpu")
        self.spk = self.base.hps.data.spk2id["EN-US"]
        self.src_se = torch.load(f"{d}/base_speakers/ses/en-us.pth", map_location="cpu")
        self.sr = self.conv.hps.data.sampling_rate
    def synth(self, text):
        import tempfile, soundfile as sf
        with tempfile.TemporaryDirectory() as t:
            self.base.tts_to_file(text, self.spk, f"{t}/a.wav", speed=1.0, quiet=True)
            self.conv.convert(audio_src_path=f"{t}/a.wav", src_se=self.src_se, tgt_se=self.tgt_se, output_path=f"{t}/b.wav")
            a, _ = sf.read(f"{t}/b.wav", dtype="float32")
        yield a

class F5TTS:
    """可选。F5-TTS：代码 MIT，但官方预训练权重 CC-BY-NC-4.0（非商用）；跨语种英文需 ref_text。"""
    def __init__(self, cfg, profile):
        from f5_tts.api import F5TTS as _F
        self.m, self.p, self.sr = _F(), profile, 24000
    def synth(self, text):
        wav, sr, _ = self.m.infer(ref_file=self.p.ref_wav, ref_text=self.p.ref_text, gen_text=text)
        self.sr = sr; yield np.asarray(wav, np.float32)

class MockCloneTTS:
    """冒烟测试用：仍要求存在声音档案，输出静音占位。"""
    def __init__(self, cfg, profile):
        self.sr, self.profile = 24000, profile
    def synth(self, text):
        yield np.zeros(int(self.sr * 0.06 * len(text.split())), np.float32)

def build_tts(cfg, profile=None):
    t = cfg["tts"]
    if t.get("debug_preset"):
        raise RuntimeError("debug_preset 仅供本地调试，请自行接入 Kokoro/Piper；正式输出必须为克隆音色。")
    profile = VoiceProfile(t["voice_profile"]) if profile is None else profile
    cls = {"cosyvoice2": CosyVoice2TTS, "openvoice_v2": OpenVoiceV2TTS, "f5_tts": F5TTS}.get(t["backend"], MockCloneTTS)
    sub = t.get(t["backend"], {})
    obj = cls(sub, profile)
    if not hasattr(obj, "set_profile"):  # 其它后端：换档案时重建
        obj.set_profile = lambda p, _c=cls, _s=sub, _o=obj: _o.__dict__.update(_c(_s, p).__dict__)
    return obj
