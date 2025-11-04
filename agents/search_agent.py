"""
SearchAgent (搜索与情报智能体)

这是"信息检索"专家，负责从Web或本地非结构化文本文档中提取信息。

- 工具集：
  - 网页浏览器（`search_google`, `open_url`, `find_and_click` 等）。
  - PDF文本解析器 (`extract_pdf_text`)。
- 专业技能：
  - 复杂浏览：执行多步网页操作，如 `f0321476...`（点击"历史版本"）或 `93c98572...`（点击"问大家"）。
  - 文档查询：解析文本型PDF 16并回答问题，如 `d88d3b99...`（查询违约条例）或 `54ffde35...` (`help_1756089021301.pdf`) 。
- 复现要求：
  - 必须将检索到的原始证据（如网页HTML、PDF提取文本）以 `task_id` 命名，保存到 `local_es_data/` 目录 。
"""

import os
from pathlib import Path
from oxygent import oxy


def create_search_agent(llm_model: str = "default_llm") -> oxy.ReActAgent:
    """
    创建 SearchAgent 实例（占位符实现）
    
    Args:
        llm_model: 使用的 LLM 模型名称
        
    Returns:
        配置好的 SearchAgent 实例
    """
    # 读取 search.prompt 文件（如果存在）
    prompt_file = Path(__file__).parent.parent / "prompts" / "search.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            search_prompt = f.read()
    else:
        search_prompt = """
你是一个 SearchAgent，专门负责网页搜索和信息检索。

**当前状态**: 这是一个占位符实现。实际的搜索功能尚未完全实现。

你应该返回一个说明，告诉用户 SearchAgent 功能正在开发中。
"""
    
    # 创建 SearchAgent（占位符）
    agent = oxy.ReActAgent(
        name="search_agent",
        desc=(
            "搜索与情报智能体。专门负责网页浏览、信息检索和PDF文本提取。"
            "【注意】当前为占位符实现，完整功能开发中。"
        ),
        tools=[],  # 暂时没有工具
        llm_model=llm_model,
        prompt=search_prompt,
        max_react_rounds=5,
        timeout=300,
    )
    
    return agent

