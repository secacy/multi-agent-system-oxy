"""
SearchAgent (搜索与情报智能体)

这是"信息检索"专家，负责从Web中提取信息。

- 工具集：
  - 网页浏览器（search, open_url, click_element 等）。
- 专业技能：
  - 复杂浏览：执行多步网页操作，如点击"历史版本"、点击"问大家"等。
  - 开放式信息检索：回答需要外部知识的"事实型"问题。
  - 特定网站信息提取：访问并解析特定网站（如京东官网、京东健康、GitHub）。
  - 跨域多步推理：分解复杂任务为多个网络步骤，在不同网站间跳转收集信息。
- 复现要求：
  - 必须将检索到的原始证据（如网页HTML、内容）以 task_id 命名，保存到 local_es_data/ 目录。
"""

import os
from pathlib import Path
from oxygent import oxy

# 导入 search_tools
from tools.search_toolkit import search_tools


def create_search_agent(llm_model: str = "default_llm") -> oxy.ReActAgent:
    """
    创建 SearchAgent 实例
    
    Args:
        llm_model: 使用的 LLM 模型名称
        
    Returns:
        配置好的 SearchAgent 实例
    """
    # 读取 search.prompt 文件
    prompt_file = Path(__file__).parent.parent / "prompts" / "search.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            search_prompt = f.read()
    else:
        print(f"⚠️ 警告: 未找到 search.prompt 文件，路径: {prompt_file}")
        search_prompt = """
你是一个 SearchAgent，专门负责网页搜索和信息检索。

**工具**:
- search: 执行互联网搜索
- open_url: 打开网页
- find_in_current_page: 在当前页面搜索文本
- click_element: 点击页面元素
- scroll_page: 滚动页面
- get_image_url: 获取图片URL

**职责**:
1. 接收 orchestrator_agent 的指令
2. 制定多步骤执行计划
3. 使用工具执行计划
4. 提取并返回最终信息

**重要**:
- 必须将 task_id 传递给每个工具调用
- 所有原始数据会自动保存到 local_es_data/
"""
    # 创建 SearchAgent
    agent = oxy.ReActAgent(
        name="search_agent",
        desc=(
            "搜索与情报智能体。专门负责网页浏览、信息检索和互联网搜索。"
            "能够执行复杂的多步骤网页交互，从互联网中提取准确信息。"
        ),
        tools=["search_tools"],  # 注册工具包
        llm_model=llm_model,
        prompt=search_prompt,
        max_react_rounds=10,  # 允许多轮ReAct循环
        timeout=300,  # 5分钟超时
    )
    
    return agent
