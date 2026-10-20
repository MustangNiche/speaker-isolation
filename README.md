# 指定说话人提取 · Speaker Isolation

一个可复用的 Codex skill：从多人混合录音中提取指定主播、念稿人或发言人的原声，再清理残留底噪。

**开始前，先打出需要保留的人在录音里说过的一句话。**

例如：

> 请只保留念“今天我们来聊聊如何给孩子选择合适的书”的那个人。

Skill 用这句话核对候选音轨；如果知道大致时间，也可以一并提供。文字用于辨认目标人物，不是声纹模型的输入。

## 如何安装

下载本仓库 ZIP 并解压，将包含 `SKILL.md` 的文件夹命名为 `speaker-isolation`，放到 Codex 的 skills 目录：

- Windows：`%USERPROFILE%\.codex\skills\speaker-isolation`
- macOS / Linux：`~/.codex/skills/speaker-isolation`
- 如设置了 `CODEX_HOME`：使用该目录下的 `skills/speaker-isolation`。

重新开启会话后，发送：

```text
请用 $speaker-isolation 处理这个音频，只保留指定人物。
要保留的人说过：“……”。
音频位置：……
```

如果没有提供目标人物的原话，skill 会先提醒你补充。

## 处理方式

1. 使用 **MossFormer2_SS_16K** 将混合说话声分离为两条候选音轨。
2. 用用户提供的原话、时间位置及试听或转写核对目标人物。
3. 只对选中的音轨运行 **DeepFilterNet** 清理底噪。
4. 导出 MP3 和 WAV，检查时长、峰值和文件完整性。

处理在本地运行，不需要上传录音。首次使用需要联网下载依赖和模型；仓库不包含录音、转写、模型权重或虚拟环境。

## 环境与使用说明

已验证 Windows x64、Python 3.12、PyTorch 2.6 CPU。无需独立显卡。分离模型约 670 MB，运行环境另外需要数百 MB。

完整安装、命令行流程和故障边界见 [本地运行说明](references/local-workflow.md)。供 Codex 使用的入口是 [SKILL.md](SKILL.md)。

原流程已处理过约 42 秒的录音；通用分享脚本另外通过了 9 秒样片的分离、说话人匹配、转写与导出验证。安装器已验证现有模型的校验和复用；其他操作系统尚未实机验证。

## 能力边界

- 这是双说话人、单声道、16 kHz 分离模型；导出 48 kHz 不会恢复原始高频。
- 三人以上、相似音色或严重重叠说话可能仍有串音或身份切换，需要核对候选音轨。
- 不保证绝对零残留，也不会自行用合成声音替换主播原声。
- 语音转写只作核对辅助，识别结果可能出错，不能代替试听。

## 上游项目

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- [MossFormer2_SS_16K 官方权重](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

本仓库提供 skill 说明与处理脚本；第三方源码、模型与程序由安装器从官方来源下载，分别遵循上游许可。
