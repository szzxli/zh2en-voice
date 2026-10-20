#!/usr/bin/env bash
# 下载模型。用法: bash scripts/download_models.sh [asr|vad|llm-1.5b|llm-3b|cosyvoice2|openvoice|all]
set -e; cd "$(dirname "$0")/.."; mkdir -p models third_party
GH=https://github.com/k2-fsa/sherpa-onnx/releases/download
what=${1:-all}
has(){ [[ $what == all || $what == $1 ]]; }
if has vad; then curl -L -o models/silero_vad.onnx $GH/asr-models/silero_vad.onnx; fi
if has asr; then
  curl -L -o /tmp/asr.tar.bz2 $GH/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2 && tar xjf /tmp/asr.tar.bz2 -C models
fi
if has llm-1.5b; then python -c "from huggingface_hub import hf_hub_download as d; d('Qwen/Qwen2.5-1.5B-Instruct-GGUF','qwen2.5-1.5b-instruct-q4_k_m.gguf',local_dir='models')"; fi
if has llm-3b; then python -c "from huggingface_hub import hf_hub_download as d; d('Qwen/Qwen2.5-3B-Instruct-GGUF','qwen2.5-3b-instruct-q4_k_m.gguf',local_dir='models')"; fi
if has cosyvoice2; then
  [ -d third_party/CosyVoice ] || git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
  python -c "from huggingface_hub import snapshot_download as s; s('FunAudioLLM/CosyVoice2-0.5B',local_dir='models/CosyVoice2-0.5B')"
fi
if has openvoice; then
  curl -L -o /tmp/ov2.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
  unzip -o /tmp/ov2.zip -d models && mv models/checkpoints_v2 models/openvoice_v2
fi
echo "done"
