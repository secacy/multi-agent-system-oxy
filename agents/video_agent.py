"""
VideoAgent (L2 - 视频分析专家)

这是"时序分析"专家。它负责理解视频内容，尤其是将用户的查询与特定时间点或时间段内发生的事件相关联。

- 工具集：
  - video_tools (get_video_metadata, detect_objects_in_video, track_user_events, 
                 find_content_in_video, analyze_video_flow)
- 核心功能：
  - 时间戳/事件精确定位
  - 关键帧内容分析（CV + OCR）
  - 用户行为与流程追踪
  - 时序逻辑理解
  - 音视频元数据检索
- 复现要求：
  - 必须将分析所依据的关键帧和分析结果，以 task_id 命名，保存到 local_es_data/ 目录
"""

import os
from pathlib import Path
from oxygent import oxy


def create_video_agent(llm_model: str = "default_llm") -> oxy.ReActAgent:
    """
    创建 VideoAgent 实例
    
    Args:
        llm_model: 使用的 LLM 模型名称
        
    Returns:
        配置好的 VideoAgent 实例
    """
    # 读取 video.prompt 文件
    prompt_file = Path(__file__).parent.parent / "prompts" / "multimodal-sub" / "video.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            video_prompt = f.read()
    else:
        print(f"⚠️  警告: 未找到 video.prompt 文件，路径: {prompt_file}")
        video_prompt = "你是一个 video_agent，负责视频分析。"
    
    # 创建 VideoAgent
    agent = oxy.ReActAgent(
        name="video_agent",
        desc=(
            "Video analysis specialist (L2). Expert in temporal-event correlation, analyzing video content "
            "at specific timestamps. Can detect objects (CV), search text (OCR/ASR), track user interactions, "
            "and analyze complex workflows. Uses video_tools including get_video_metadata, detect_objects_in_video, "
            "track_user_events, find_content_in_video, and analyze_video_flow."
        ),
        tools=["video_tools"],
        llm_model=llm_model,
        prompt=video_prompt,
        max_react_rounds=10,  # 视频分析可能需要多步骤
        timeout=300,  # 视频处理较耗时
        trust_mode=True,  # 直接返回工具的原始输出
    )
    
    return agent

