#!/usr/bin/env bash
# 一键安装 demo：Python 3.10 虚拟环境 + 依赖 + 模型（ASR / VAD / Qwen2.5-1.5B GGUF / CosyVoice2-0.5B，共约 7GB）
# macOS / Linux：bash scripts/setup_demo.sh
# Windows：用 WSL2（Ubuntu）运行本脚本；或参见 README 里的「Windows 原生」步骤
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-3.10}; VENV=${VENV:-.venv310}
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
case "$(uname -s)" in
  Darwin) command -v brew >/dev/null && brew list portaudio >/dev/null 2>&1 || brew install portaudio || true ;;
  Linux)  ldconfig -p 2>/dev/null | grep -q libportaudio || echo "⚠ 本机麦克风需要：sudo apt install libportaudio2（只用 --input-wav / 网页版可忽略）" ;;
esac
uv venv -p "$PY" "$VENV"
export VIRTUAL_ENV="$PWD/$VENV"; export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install "setuptools<70" wheel
uv pip install --no-build-isolation openai-whisper==20231117
uv pip install --index-strategy unsafe-best-match -r requirements-cosyvoice-cpu.txt \
  pyyaml sounddevice scipy aiohttp sherpa-onnx openai noisereduce pytest huggingface_hub pillow
# llama-cpp-python：优先预编译 CPU 轮子（macOS 上会自动编译 Metal 版）
uv pip install --index-strategy unsafe-best-match llama-cpp-python --only-binary=:all: \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu || uv pip install llama-cpp-python
bash scripts/download_models.sh vad
[ -f models/sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt ] || bash scripts/download_models.sh asr
[ -f models/qwen2.5-1.5b-instruct-q4_k_m.gguf ] || bash scripts/download_models.sh llm-1.5b
[ -f models/CosyVoice2-0.5B/llm.pt ] || bash scripts/download_models.sh cosyvoice2
echo
echo "✅ 安装完成。运行："
echo "  $VENV/bin/python -m zh2en.demo                       # 麦克风"
echo "  $VENV/bin/python -m zh2en.demo --input-wav third_party/CosyVoice/asset/zero_shot_prompt.wav --prompt-text '希望你以后能够做的比我还好呦。' --no-play"
echo "  $VENV/bin/python -m zh2en --source web                # 浏览器 http://localhost:8765/"
