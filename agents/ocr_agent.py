"""
OCR Agent (L2 - 文本提取管理者)

负责分析 OCR 任务意图并调度正确的 L3 OCR 专家。

核心职责：
1. 接收来自 multimodal_agent 的高级指令
2. 分析 OCR 任务意图（通用文本提取、K-V提取、文本定位）
3. 选择并调用正确的 L3 专家
4. 返回提取结果或能力不匹配错误

架构设计：
- L2 (OcrAgent)：使用文本 LLM 作为"大脑"，负责任务路由和决策
- L3 (OCR 专家)：使用 VLM 作为"大脑"，负责实际的文本提取
"""

import os
from pathlib import Path
from oxygent import oxy


def load_ocr_prompt():
    """加载 ocr agent 的 prompt"""
    prompt_file = Path(__file__).parent.parent / "prompts" / "multimodal-sub" / "ocr.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    
    # 如果文件不存在，返回默认 prompt
    return """你是一个专门负责**分析 OCR 任务**并**调度正确的 L3 OCR 专家**的管理型智能体。"""


def create_ocr_agent(llm_model: str = "default_llm"):
    """
    创建 OCR Agent (L2 - 管理者)
    
    架构说明：
    - 使用文本 LLM（default_llm）作为"大脑"
    - 负责分析任务意图，选择合适的 L3 OCR 专家
    - 通过 sub_agents 调用 L3 OCR 专家（他们使用 VLM）
    
    Args:
        llm_model: 使用的 LLM 模型名称（应该是文本 LLM）
        
    Returns:
        配置好的 OCR Agent
    """
    
    # 加载 prompt
    prompt = load_ocr_prompt()
    
    # 创建 ReActAgent（L2 管理者）
    agent = oxy.ReActAgent(
        name="ocr_agent",
        desc="OCR 管理者（L2）。负责分析 OCR 任务意图，并调度合适的 L3 OCR 专家完成文本提取任务。支持快速路径（Python）和VLM降级路径。",
        desc_for_llm="OCR manager (L2). Analyzes OCR task intent and dispatches appropriate L3 OCR experts. Supports fast path (Python) and VLM fallback.",
        
        # 工具配置（L3-A: Python 工具）
        tools=["pdf_tools"],  # 包含 py_pdf_extractor_agent 和 pdf_to_base64_images
        
        # 子智能体配置（L3-B 和 L3-C）
        sub_agents=[
            # L3-B: VLM 智能体（降级路径）
            "vlm_extract_text_agent",              # VLM 通用文本提取
            "vlm_extract_structured_data_agent",   # VLM 结构化数据提取（K-V）
            "vlm_find_text_coordinates_agent",     # VLM 文本定位
            # L3-C: 文本分析智能体（快速路径第2步）
            "text_analyzer_agent",                 # 文本 LLM 分析器
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
        timeout=180,  # 给 L3 专家足够的时间（增加了，因为有多步骤）
        retries=3,
        delay=1,
        max_react_rounds=10,  # 允许多轮 React（因为需要执行多步计划）
        
        # 注意：L2 不需要多模态支持，因为它只负责路由
        # 真正的 OCR 处理由 L3 专家（使用 VLM）完成
        is_multimodal_supported=False,
        
        # 并发配置
        semaphore=2,
    )
    
    return agent

