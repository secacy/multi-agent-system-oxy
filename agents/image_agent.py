"""
Image Agent (L2 - 图像分析管理者)

负责分析视觉任务意图并调度正确的 L3 视觉专家。

核心职责：
1. 接收来自 multimodal_agent 的高级指令
2. 分析视觉意图（开放式问题、实体识别、区域提取、区域比较）
3. 选择并调用正确的 L3 专家（analyze_general_agent, identify_entities_agent 等）
4. 返回分析结果或能力不匹配错误

架构设计：
- L2 (ImageAgent)：使用文本 LLM 作为"大脑"，负责任务路由和决策
- L3 (专家 Agent)：使用 VLM 作为"大脑"，负责实际的视觉分析
"""

import os
from pathlib import Path
from oxygent import oxy


def load_image_prompt():
    """加载 image agent 的 prompt"""
    prompt_file = Path(__file__).parent.parent / "prompts" / "multimodal-sub" / "image.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    
    # 如果文件不存在，返回默认 prompt
    return """你是一个专门负责**分析视觉任务**并**调用正确工具**的专家智能体。"""


def create_image_agent(llm_model: str = "default_llm"):
    """
    创建 Image Agent (L2 - 管理者)
    
    架构说明：
    - 使用文本 LLM（default_llm）作为"大脑"
    - 负责分析任务意图，选择合适的 L3 专家
    - 通过 sub_agents 调用 L3 视觉专家（他们使用 VLM）
    
    Args:
        llm_model: 使用的 LLM 模型名称（应该是文本 LLM）
        
    Returns:
        配置好的 Image Agent
    """
    
    # 加载 prompt
    prompt = load_image_prompt()
    
    # 创建 ReActAgent（L2 管理者）
    agent = oxy.ReActAgent(
        name="image_agent",
        desc="图像分析管理者（L2）。负责分析视觉任务意图，并调度合适的 L3 视觉专家完成任务。",
        desc_for_llm="Image analysis manager (L2). Analyzes visual task intent and dispatches appropriate L3 vision experts to complete tasks.",
        
        # 子智能体配置（L3 专家）
        sub_agents=[
            "analyze_general_agent",          # 通用图像分析
            "identify_entities_agent",        # 实体识别
            "analyze_visual_regions_agent",   # 视觉区域分析
            "compare_regions_agent",          # 区域比较
        ],
        
        # LLM 配置（使用文本 LLM）
        llm_model=llm_model,
        
        # Prompt 配置
        prompt=prompt,
        
        # Agent 配置
        category="agent",
        class_name="ReActAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=120,  # 给 L3 专家足够的时间
        retries=3,
        delay=1,
        
        # 注意：L2 不需要多模态支持，因为它只负责路由
        # 真正的图像分析由 L3 专家（使用 VLM）完成
        is_multimodal_supported=False,
        
        # 并发配置
        semaphore=2,
    )
    
    return agent

