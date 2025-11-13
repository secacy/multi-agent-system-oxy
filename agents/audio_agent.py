"""
AudioAgent (L2 - 音频分析专家)

这是"听觉"专家，负责转录和理解音频文件中的内容。

- 工具集：
  - audio_tools (transcribe, identify_music)
- 核心功能：
  - 音频转录 (ASR)：将语音/歌词转换为文本
  - 音频指纹/识别：听歌识曲功能
  - 音频提示工程：将高级指令转换为工具调用
- 复现要求：
  - 将 task_id 传递给工具，确保日志链条完整
"""

import os
from pathlib import Path
from oxygent import oxy


def create_audio_agent(llm_model: str = "default_llm") -> oxy.ReActAgent:
    """
    创建 AudioAgent 实例
    
    Args:
        llm_model: 使用的 LLM 模型名称
        
    Returns:
        配置好的 AudioAgent 实例
    """
    # 读取 audio.prompt 文件
    prompt_file = Path(__file__).parent.parent / "prompts" / "multimodal-sub" / "audio.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            audio_prompt = f.read()
    else:
        print(f"⚠️  警告: 未找到 audio.prompt 文件，路径: {prompt_file}")
        audio_prompt = "你是一个 audio_agent，负责音频分析。"
    
    # 创建 AudioAgent
    agent = oxy.ReActAgent(
        name="audio_agent",
        desc=(
            "Audio analysis specialist (L2). Expert in audio transcription (ASR) and "
            "music identification. Can transcribe speech/lyrics to text and identify songs. "
            "Uses transcribe and identify_music tools from audio_tools."
        ),
        tools=["audio_tools"],
        llm_model=llm_model,
        prompt=audio_prompt,
        max_react_rounds=5,  # 音频分析通常不需要多轮
        timeout=180,  # 音频处理可能较耗时
        trust_mode=True,  # 直接返回工具的原始输出
    )
    
    return agent

