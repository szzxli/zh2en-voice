# zh2en-voice：实时中文语音 → 英文语音（用你自己的声音）

麦克风（本机或手机浏览器）收中文 → 流式 ASR → LLM 翻译成英文（保留语气和情感）→ **用你本人的克隆音色**念出英文。目标端到端延迟 1–2 秒。

> ⚠️ 硬性设计：TTS 只输出用户克隆音色。没有声音档案时程序直接报错，不会退回到预设音色。Kokoro/Piper 这类预设音色只能在本地调试时自己接入（`tts.debug_preset`），不进默认流程。

## 🚀 快速体验 Demo（推荐先跑这个）
```bash
bash scripts/setup_demo.sh            # 一键：Python3.10 环境 + 依赖 + 模型（约 7GB，CPU 即可）
.venv310/bin/python -m zh2en.demo     # 显示中文提示 → 回车录音（停顿 0.8s 自动结束）→ 识别 → 译英 → 用你的克隆音色播放
```
- 第一次运行没有声音档案时，**这次朗读的录音同时作为克隆参考**（因为朗读的是提示句，文本已知）。之后会一直用 `voices/me/`。想换参考就删掉它，或者用 `scripts/enroll_voice.py` 多录几段。
- 任何情况下都不会退回到预设音色：TTS 只接受 `voices/<name>/` 里的声音档案。
- 结果会打印识别文本、英文和每个环节的延迟；音频保存为 `out.wav`，结果明细保存为 `out.json`。
- 没有麦克风时用 wav 文件代替：`--input-wav x.wav --prompt-text "x.wav 的原文" --no-play`；加 `--repeat 2` 可以看到热启动后的延迟。

**各平台说明**
- **macOS**：`brew install portaudio`（脚本会自动装）。Apple Silicon 上 torch 走 CPU；llama-cpp-python 会自动编译 Metal 版。
- **Linux**：`sudo apt install libportaudio2`（只用 `--input-wav` 或网页版时不需要）。
- **Windows**：推荐用 WSL2 Ubuntu 运行脚本，网页版通过 `http://localhost:8765` 访问。如果用原生 Windows：装 Python 3.10 和 uv，在 PowerShell 里依次执行 `uv venv -p 3.10 .venv310`、`.venv310\Scripts\activate`，再执行 `setup_demo.sh` 里的 pip 命令。模型可以用 Git Bash 运行 `download_models.sh` 下载。

### 实测结果（本仓库开发机：8 核 x86 CPU，15GB 内存，无 GPU；输入为 CosyVoice 自带的 Apache-2.0 样本 `asset/zero_shot_prompt.wav`）
| 项 | 结果 |
|---|---|
| 识别 | 希望你以后能够做的比我还好（原文“…还好呦。”，句尾的“呦”丢了，流式 Paraformer 也不输出标点） |
| 英文 | I hope you do even better than I do. |
| ASR（sherpa-onnx Paraformer int8） | 191–409 ms |
| 翻译首句（Qwen2.5-1.5B Q4_K_M，llama.cpp） | 冷启动 1316 ms / 热启动 537 ms |
| TTS 首包（CosyVoice2-0.5B，CPU fp32） | 5.6–8.8 s（RTF 2.9–4.9） |
| 端到端首包 | 热启动约 6.5 s（冷启动 8–10 s） |
| 模型加载 | ASR+LLM 3.4 s，CosyVoice2 16.5 s |

结论：**ASR 和翻译在 CPU 上已经能达到 1 秒级；瓶颈是 CosyVoice2 在 CPU 上的合成速度，要做到 1–2 秒需要 GPU**（首包约 150–300 ms）。纯 CPU 设备可以改用 OpenVoice V2（`tts.backend: openvoice_v2`，估计每句 0.5–1 s，克隆相似度较低，这一项还没实测）。

## 📱 手机使用（PWA，可添加到主屏）
网页版就是一个 PWA，包含 manifest、图标和 Service Worker，可以添加到主屏，全屏打开。页面流程：显示中文提示 → 点按录音（停顿后自动结束）→ 显示识别文本和英文 → 播放克隆音，同时显示各环节延迟。另有一个「实时模式」标签页，使用 WebSocket 流式翻译。

浏览器只允许 **HTTPS**（或 localhost）页面使用麦克风，以下三种方式任选其一：

**方式 A：cloudflared / ngrok 隧道（最省事，iOS 和 Android 都能用）**
```bash
.venv310/bin/python -m zh2en --source web            # 本机 http://localhost:8765
cloudflared tunnel --url http://localhost:8765        # 输出 https://xxxx.trycloudflare.com
# 或 ngrok http 8765                                   # 输出 https://xxxx.ngrok-free.app
```
手机打开隧道给出的 https 地址，允许麦克风即可。注意：音频会经过第三方隧道，介意的话用方式 B。

**方式 B：局域网 + 自签证书**
```bash
bash scripts/make_cert.sh 192.168.x.x                  # 生成含该 IP 的 cert.pem / key.pem
.venv310/bin/python -m zh2en --source web --cert cert.pem --key key.pem
```
- **Android Chrome**：打开 `https://<电脑局域网IP>:8765`，依次点「高级 → 继续前往」。
- **iOS Safari**：iOS 不允许直接跳过证书警告去使用麦克风，需要先信任证书：
  1. 把 `cert.pem` 通过 AirDrop 或邮件发到 iPhone，打开后，在「设置 → 已下载描述文件」里安装；
  2. 进入「设置 → 通用 → 关于本机 → 证书信任设置」，打开 zh2en-voice 的完全信任开关；
  3. 用 Safari 打开 `https://<电脑IP>:8765`。生成证书时传入的 IP 必须和访问地址一致；也可以改用 `mkcert <电脑IP>` 生成证书。
  
**方式 C**：电脑本机浏览器直接打开 `http://localhost:8765`（localhost 不需要 HTTPS）。

**添加到主屏**：iOS 用 Safari 的「分享 → 添加到主屏幕」；Android 用 Chrome 菜单里的「安装应用 / 添加到主屏幕」。
提示：iOS 必须先点一下页面按钮才会出声音（浏览器的自动播放限制）；建议戴耳机，避免英文播放声被麦克风收进去。

## 架构
```
mic/WebSocket ─► Silero VAD 分段 ─► ASR(sherpa-onnx 流式 Paraformer) ─► LLM 流式翻译 ─► 按句切分 ─► 克隆 TTS(流式) ─► 播放
                  (q_seg)                                               (q_sent)
```
用 asyncio 队列加线程池组成两级流水线：第 N 句在做 TTS 时，第 N+1 段已经在做 ASR/翻译。LLM 只要吐出一个完整句子（按 `.!?…` 切）就马上送进 TTS，不用等整段翻译完。

## 目录结构
```
zh2en-voice/
├── config.yaml              # 全部配置（后端切换）
├── requirements.txt / requirements-tts.txt / requirements-cosyvoice-cpu.txt
├── zh2en/
│   ├── __main__.py          # 入口：python -m zh2en [--source local|web]
│   ├── demo.py              # 单句可测试 demo：python -m zh2en.demo [--input-wav]
│   ├── enroll.py            # 声音档案核心（校验/裁剪/保存）
│   ├── config.py
│   ├── pipeline.py          # asyncio 两级并发流水线 + 延迟日志
│   ├── server.py            # aiohttp WebSocket 服务（手机收音/回放）
│   ├── audio/{vad.py,io.py} # Silero/能量 VAD；sounddevice 收放音
│   ├── asr/__init__.py      # sherpa-onnx Paraformer | FunASR | mock
│   ├── mt/__init__.py       # 翻译 prompt、按句切分、llama.cpp | OpenAI 兼容(vLLM/Ollama) | mock
│   └── tts/__init__.py      # VoiceProfile + CosyVoice2 | OpenVoice V2 | F5-TTS | mock
├── scripts/
│   ├── enroll_voice.py      # 声音样本采集与校验（必做）
│   ├── setup_demo.sh        # 一键安装 demo（依赖+模型）
│   ├── download_models.sh   # 模型下载
│   ├── make_cert.sh         # 生成 HTTPS 自签名证书（手机收音用）
│   └── bench.py             # 各环节离线计时
├── web/                     # PWA：index.html、manifest.webmanifest、sw.js、icons/
├── tests/test_smoke.py
└── voices/ models/ third_party/  (git 忽略)
```

## 安装
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# llama-cpp-python 编译失败时用预编译 CPU 轮子：
pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
# Linux 收音还需要：sudo apt install libportaudio2

bash scripts/download_models.sh vad        # 0.6MB
bash scripts/download_models.sh asr        # ~1GB（压缩包，含 int8 模型）
bash scripts/download_models.sh llm-3b     # ~2GB（或 llm-1.5b ~1GB）
bash scripts/download_models.sh cosyvoice2 # 克隆 CosyVoice 仓库 + 0.5B 权重 ~2GB
pip install -r requirements-cosyvoice-cpu.txt   # CPU 版（Python 3.10，去掉了 deepspeed/tensorrt）；GPU 用 third_party/CosyVoice/requirements.txt
# CPU 备选：bash scripts/download_models.sh openvoice，再按 requirements-tts.txt 装 OpenVoice/MeloTTS
```
GPU 翻译：`vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 4096`（需要 ≥16GB 显存的 NVIDIA GPU），然后把 `mt.backend` 改成 `openai_compat`。Ollama：`ollama pull qwen2.5:3b`，`base_url: http://localhost:11434/v1`，`model: qwen2.5:3b`。

## 第一步：录入你的声音（必须）
```bash
python scripts/enroll_voice.py --name me --n 4 --denoise
# 已有录音：python scripts/enroll_voice.py --name me --from-dir my_wavs/   # 同名 .txt 放转写文本
```
- 脚本会给你几句带不同情绪的中文朗读稿（平静、激动、疲惫、疑问），逐段录音。
- 自动检查：采样率 ≥16k，单段时长 3–15s，峰值音量，削波，信噪比 ≥20dB。不合格会提示重录。
- 自动处理：去掉首尾静音，可选 noisereduce 稳态降噪（降得太狠会损伤音色，在安静环境下录可以不开）。
- 输出 `voices/me/`：`samples/*.wav|txt`、`ref.wav`（自动选 SNR 最高、时长接近 8s 的一段）、`ref.txt`、`profile.json`。
- CosyVoice2 用 `inference_cross_lingual` 模式：只用中文参考音色，不用中文转写，避免英文带上中文 prompt 的韵律。F5-TTS 会用到 `ref.txt`。

## 运行
```bash
python -m zh2en                       # 本机麦克风 + 扬声器（建议戴耳机，防止回授）
bash scripts/make_cert.sh
python -m zh2en --source web --cert cert.pem --key key.pem
# 手机和电脑连同一个 Wi-Fi，手机浏览器打开 https://<电脑IP>:8765/ ，信任证书后点「开始」
python scripts/bench.py sample_zh.wav # 各环节计时
pytest -q tests                       # 冒烟测试（不需要模型）
```

## 克隆 TTS 选型
| 方案 | 跨语种零样本克隆 | 流式 | 许可证 | CPU 首包/RTF | GPU 首包 | 结论 |
|---|---|---|---|---|---|---|
| **CosyVoice2-0.5B** | ✅ 中文参考→英文，相似度高 | ✅ 双向流式 | Apache-2.0（代码+权重） | 首包 ~1.5–3s，RTF ~1–2（8 核） | **~150–300ms** | **默认**：质量、流式、许可证都适合开源 |
| OpenVoice V2 | ✅ 音色转换（MeloTTS 生成+音色迁移） | ✗（按句） | MIT（V2 可商用） | **~0.5–1s/句**，RTF ~0.3 | ~0.2s | **CPU 备选**：快，但相似度和情感比 CosyVoice2 差 |
| F5-TTS | ✅ 需要 ref_text | ✗（按句/分块） | 代码 MIT，**权重 CC-BY-NC**（不可商用） | RTF ~3–5，不实用 | ~0.3–0.6s | 可选，开源分发时要说明非商用 |
| XTTS-v2 | ✅ 17 种语言 | ✅ | **CPML，非商用** | RTF ~1.5–3 | ~200ms | 不默认，只在文档里提一句；开源项目不要捆绑它的权重 |
| Kokoro / Piper | ✗ 只有预设音色 | ✅ | Apache / MIT | 很快 | — | **只做调试**，不符合"本人音色"要求 |

## 延迟估算（单句约 10 个汉字，说完到听到第一个英文音节）
| 环节 | CPU（8 核，无 GPU） | GPU（RTX 4090 级） |
|---|---|---|
| VAD 尾静音判定 | 300ms（可调 200–500） | 300ms |
| ASR（sherpa-onnx 流式 Paraformer int8） | 50–150ms | 50–150ms（CPU 已足够） |
| 翻译首句（CPU：Qwen2.5-1.5B/3B Q4；GPU：vLLM 7B） | 1.5B：250–500ms；3B：500–900ms | 100–200ms |
| TTS 首包（CPU：OpenVoice V2；GPU：CosyVoice2 流式） | OpenVoice 500–1000ms（估计）；CosyVoice2 **实测 5.6–8.8s** | 150–300ms |
| **端到端首包** | **≈1.1–2.3s**（1.5B+OpenVoice） | **≈0.6–1.0s** |

实测结果见上文的「实测结果」表；其余数字是根据公开基准的估算，以 `bench.py` 和 `zh2en.demo` 的实测为准。
降低延迟的办法：缩短 `min_silence_ms`；CPU 上用 1.5B 模型；翻译 `max_tokens` 设小；佩戴耳机后可以放开 AEC。

## 许可证
本项目代码：MIT。模型各自遵循原许可证（见上表）。分发时不要打包模型权重，由用户用脚本自行下载。
