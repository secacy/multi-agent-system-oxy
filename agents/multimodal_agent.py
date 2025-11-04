"""
MultimodalAgent (多模态分析智能体)

这是"感知"专家，负责理解所有非文本的像素、声音和扫描图像。

- 工具集：
  - 图像识别 (CV) 模型 (`analyze_image`)。
  - 音视频分析工具 (`analyze_video`, `analyze_audio`)。
  - 光学字符识别 (OCR) (`ocr_image_or_pdf`)。
- 专业技能：
  - 图像分析：如 `556a96c3...`（识别京东吉祥物）。
  - 音视频分析：如 `e0d8203d...`（分析 `买iphone_副本.mp4`）或 `fddfe2fc...`（识别`大满贯.mp3`的歌词）。
  - OCR：处理扫描版PDF（如 `22eb45c7...` 识别"人寿"字样 ）或图片中的文字（如 `0b2fa816...` 查看订单状态 ）。
- 复现要求：
  - 必须将分析的关键证据（如视频截图、OCR结果）以 `task_id` 命名，保存到 `local_es_data/` 目录。
"""

import os
from pathlib import Path
from oxygent import oxy


def create_multimodal_agent(llm_model: str = "default_llm") -> oxy.ReActAgent:
    """
    创建 MultimodalAgent 实例
    
    Args:
        llm_model: 使用的 LLM 模型名称
        
    Returns:
        配置好的 MultimodalAgent 实例
    """
    # 读取 multimodal.prompt 文件
    prompt_file = Path(__file__).parent.parent / "prompts" / "multimodal.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            multimodal_prompt = f.read()
    else:
        print(f"⚠️  警告: 未找到 multimodal.prompt 文件，路径: {prompt_file}")
        multimodal_prompt = "你是一个 multimodal_agent，负责多模态内容分析。"
    
    # 创建 MultimodalAgent
    agent = oxy.ReActAgent(
        name="multimodal_agent",
        desc=(
            "Multimodal analysis specialist. Expert in image recognition, video analysis, "
            "audio transcription, and OCR. Can analyze .png, .jpg, .mp4, .mp3 files and "
            "perform optical character recognition on images and PDFs. "
            "Uses analyze_image, analyze_video, analyze_audio, and ocr_image_or_pdf tools."
        ),
        tools=["multimodal_tools"],
        llm_model=llm_model,
        prompt=multimodal_prompt,
        max_react_rounds=10,  # 最大 ReAct 轮数
        timeout=300,  # 超时时间（秒）
        multimedia_supported=True,  # 支持多模态
        trust_mode=True,  # Agent 直接返回工具的原始输出
    )
    
    return agent

