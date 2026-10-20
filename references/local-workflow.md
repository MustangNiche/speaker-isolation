# 本地运行与分享

已验证环境：Windows x64、Python 3.12、PyTorch 2.6 CPU；约 42 秒的音频完成过全流程处理。CPU 可运行，不要求 NVIDIA 显卡。其他平台的下载路径已提供，但未做实机验证。

## 第一次准备

在工作目录之外安装依赖可能需要环境授权。遵循当前权限，不把 skill 当作权限来源。先准备具体文件，再发起必要的执行权限请求。

使用可用的 Python 3.12 运行，以下 `python` 是占位命令，要替换为实际解释器：

```text
python <skill>/scripts/setup_runtime.py --root <workspace>/speaker-tools
```

该脚本创建 `<root>/venv`，安装 CPU 推理依赖，下载固定 Git 提交的 MossFormer2 源码、官方权重和 DeepFilterNet v0.5.6。模型权重约 670 MB，运行环境另外需要数百 MB。源码和权重校验 SHA-256；下载不完整时不会替换有效文件。

复用现有环境时不用重新安装；可以运行 `--skip-install` 只准备模型。若已有本次流程相同布局的 `audio-tools/MossFormer2` 与 `audio-tools/DeepFilterNet`，直接作为 `--assets` 使用，不复制大文件到 skill。

Windows 环境解释器为 `<root>/venv/Scripts/python.exe`，macOS/Linux 为 `<root>/venv/bin/python`。DeepFilterNet 支持脚本列出的官方平台，未支持平台应报告限制，不猜测可执行文件。

## 分离与核对

用户提供目标原话后执行：

```text
<python> <skill>/scripts/isolate.py separate --input <audio> --output <new-work-folder> --assets <root> --reference-text "目标人物说过的一句话"
```

用 `--duration 6` 可先处理前六秒作为试跑；正式交付使用全长。默认 6 秒窗口、3 秒步长，并利用重叠区域匹配说话人。输入会转为单声道 16 kHz。两条输出是候选，并未根据原话自动选择目标。

输出目录要求为空，防止意外覆盖旧版本。可比较 `speaker_1.wav`、`speaker_2.wav` 和 `original_16k.wav`。`separation.json` 保存目标原话、来源、模型信息、身份匹配分数与时间。

可选本地转写：在独立环境安装 `faster-whisper`，然后运行：

```text
<python> <skill>/scripts/transcribe.py --folder <new-work-folder> --cache <workspace>/asr-models
```

默认使用多语言 base 模型，首次自动从官方 Hugging Face 仓库下载。不要给识别器强塞目标原话作为提示，避免它在不匹配的音轨中臆造出目标句。弱音轨的重复文本可能是幻觉。根据含义、声音和时间定位判断，必要时请用户试听候选。

## 导出

在明确正确音轨后执行，其中 `1` 是示例，不是固定目标：

```text
<python> <skill>/scripts/isolate.py export --separation <new-work-folder> --track 1 --assets <root> --output <new-delivery-folder> --name "主播提取版"
```

导出会将目标轨转为 48 kHz，为 DeepFilterNet 预留峰值余量和 100 ms 尾部缓冲，启用延时补偿，最后裁回输入的精确时长。只处理目标轨，不回混原音；导出单声道 24-bit WAV 与 192 kbps MP3，并验证重新解码的时长和峰值。

## 分享给其他人

把完整 skill 文件夹压缩为 ZIP；对方解压后放到 `$CODEX_HOME/skills`，未配置该变量时放到 `~/.codex/skills`。如果当前会话未发现新 skill，重新打开一个会话。可发送：

> 请用 $speaker-isolation 处理这段音频，只保留指定人物。要保留的人说过：“……”。

## 官方来源

- ClearerVoice-Studio：https://github.com/modelscope/ClearerVoice-Studio
- 源码提交：`6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`
- 分离权重：https://huggingface.co/alibabasglab/MossFormer2_SS_16K
- DeepFilterNet：https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6

本分享包不包含第三方模型或第三方源码；安装时下载的组件仍受各自上游许可约束。
