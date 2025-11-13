"""
MultimodalAgent (L1 - 多模态管理器)

这是一个管理型智能体。它不直接分析文件，只负责将其下属的专业子智能体暴露给 OrchestratorAgent。

- 工具集 (Sub-Agents)：
  - image_agent (图像分析专家) - 待实现
  - video_agent (视频分析专家) ✅
  - audio_agent (音频分析专家) ✅
  - ocr_agent (光学识别专家) - 待实现
- 专业技能：
  - 任务路由 (Task Routing)：OrchestratorAgent 会将所有多模态任务发送给它。
    它的唯一职责是分析 file_name（如 .mp4）和指令，然后将任务精确地转发给正确的 L2 子智能体。
- 复现要求：
  - 在分派任务时，必须将 OrchestratorAgent 传来的 task_id 完整地传递给 L2 子智能体。
"""

import os
from pathlib import Path
from oxygent import oxy


def create_multimodal_agent(
    llm_model: str = "default_llm",
    sub_agents: list = None
) -> oxy.ReActAgent:
    """
    创建 MultimodalAgent 实例（L1 路由型管理智能体）
    
    Args:
        llm_model: 使用的 LLM 模型名称
        sub_agents: L2 子智能体列表
        
    Returns:
        配置好的 MultimodalAgent 实例
    """
    # 默认子智能体列表
    if sub_agents is None:
        sub_agents = [
            "audio_agent",
            "video_agent",  # ✅ 已实现
            # "image_agent",  # 待实现
            # "ocr_agent",    # 待实现
        ]
    
    # 读取 multimodal.prompt 文件
    prompt_file = Path(__file__).parent.parent / "prompts" / "multimodal.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            multimodal_prompt = f.read()
    else:
        print(f"⚠️  警告: 未找到 multimodal.prompt 文件，路径: {prompt_file}")
        multimodal_prompt = "你是一个 multimodal_agent (L1)，负责路由多模态任务到子智能体。"
    
    # 创建 MultimodalAgent（L1 管理器）
    agent = oxy.ReActAgent(
        name="multimodal_agent",
        desc=(
            "Multimodal router agent (L1). Routes multimodal tasks to specialized L2 agents. "
            "Analyzes file_name (.mp3, .mp4, .jpg, .pdf) and forwards tasks to "
            "audio_agent (for .mp3/.wav), video_agent (for .mp4/.avi), "
            "image_agent (for .jpg/.png), or ocr_agent (for text extraction)."
        ),
        sub_agents=sub_agents,  # L2 子智能体
        tools=[],  # L1 不直接使用工具，只路由
        llm_model=llm_model,
        prompt=multimodal_prompt,
        max_react_rounds=5,  # 路由不需要多轮
        timeout=300,
        multimedia_supported=True,
        trust_mode=True,  # 直接返回子智能体的输出
    )
    
    return agent

