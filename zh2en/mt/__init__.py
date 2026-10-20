"""LLM 中译英：保留语气情感，流式输出并按句切分。"""
import re
from typing import Iterator

SYSTEM_PROMPT = """You are a real-time simultaneous interpreter from Mandarin Chinese to spoken English.
Rules:
1. Translate faithfully, as natural spoken English a native speaker would SAY aloud.
2. Preserve the speaker's tone and emotion: excitement, anger, hesitation, sarcasm, politeness level, and intensity.
   Keep interjections and fillers as English equivalents (e.g. 哇->Wow, 嗯->Hmm, 唉->Sigh..., 哈哈->Haha).
3. Keep punctuation that carries emotion (!, ?, ..., —). Do not add explanations, notes, quotes or the Chinese source.
4. ASR text may lack punctuation or contain homophone errors; infer the intended meaning from context.
5. Output ONLY the English translation."""

SENT_END = re.compile(r"([.!?…]+[\"')\]]?)(\s|$)")

def split_sentences(stream: Iterator[str], min_chars=12) -> Iterator[str]:
    """把 token 流按句末标点切成句子，尽早送入 TTS。"""
    buf = ""
    for tok in stream:
        buf += tok
        while True:
            m = SENT_END.search(buf)
            if not m or m.end(1) < min_chars:
                break
            yield buf[:m.end(1)].strip(); buf = buf[m.end():]
    if buf.strip():
        yield buf.strip()

class BaseMT:
    def __init__(self, cfg):
        self.cfg, self.history = cfg, []
    def messages(self, zh):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for z, e in self.history[-self.cfg.get("context_turns", 3):]:
            msgs += [{"role": "user", "content": z}, {"role": "assistant", "content": e}]
        return msgs + [{"role": "user", "content": zh}]
    def stream(self, zh) -> Iterator[str]:
        raise NotImplementedError
    def translate_sentences(self, zh) -> Iterator[str]:
        full = []
        for s in split_sentences(self.stream(zh)):
            full.append(s); yield s
        self.history.append((zh, " ".join(full)))

class MockMT(BaseMT):
    def stream(self, zh):
        for w in "What a lovely day! Let's go for a walk in the park.".split(" "):
            yield w + " "

class LlamaCppMT(BaseMT):
    """CPU：llama-cpp-python + Qwen2.5-1.5B/3B-Instruct GGUF Q4_K_M。"""
    def __init__(self, cfg):
        super().__init__(cfg)
        from llama_cpp import Llama
        c = cfg["llama_cpp"]
        self.llm = Llama(model_path=c["model_path"], n_ctx=c["n_ctx"], n_threads=c["n_threads"], verbose=False)
    def stream(self, zh):
        for ch in self.llm.create_chat_completion(self.messages(zh), temperature=self.cfg["temperature"], stream=True, max_tokens=256):
            d = ch["choices"][0]["delta"].get("content")
            if d: yield d

class OpenAICompatMT(BaseMT):
    """GPU：vLLM 部署 Qwen2.5-7B-Instruct（OpenAI 兼容接口）；也可接 Ollama。"""
    def __init__(self, cfg):
        super().__init__(cfg)
        from openai import OpenAI
        c = cfg["openai_compat"]
        self.client, self.model = OpenAI(base_url=c["base_url"], api_key=c["api_key"]), c["model"]
    def stream(self, zh):
        r = self.client.chat.completions.create(model=self.model, messages=self.messages(zh),
                                                temperature=self.cfg["temperature"], stream=True, max_tokens=256)
        for ch in r:
            if ch.choices and ch.choices[0].delta.content:
                yield ch.choices[0].delta.content

def build_mt(cfg):
    m = cfg["mt"]
    return {"llama_cpp": LlamaCppMT, "openai_compat": OpenAICompatMT}.get(m["backend"], MockMT)(m)
