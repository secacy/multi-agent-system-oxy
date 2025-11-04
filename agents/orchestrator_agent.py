"""
OrchestratorAgent (主控智能体 - 大脑)

这是系统的唯一协调者。它不执行任何工具，只负责思考、规划、分派和交付。

- 核心职责：

  1. 分析与规划 (Analyze & Plan)：
     - 接收：`task_id`, `query`, `level`, `file_name`。
     - 分析：利用 `level` 字段判断任务复杂度（例如 Level 3 意味着多步骤）。
     - 规划：根据验证集 `steps` 的经验，结合下述的"核心路由逻辑"，动态生成多步骤执行计划。
  2. 调度与分派 (Dispatch)：
     - 根据计划，一次一步，将子任务分派给最合适的专家智能体。
     - 分派时必须传递完整的上下文，包括 `task_id`（用于日志记录）和上一步的结果。
  3. 常识推理 (In-built Reasoning)：
     - 对于不需要工具的简单常识（例如 `task_id: daec84d5-c69a-4172-8540-6bfe70683bba` 牛奶和水的问题）或实体链接，`OrchestratorAgent` 亲自解决，不分派给任何专家。
  4. 综合与交付 (Synthesize & Deliver)：
     - 这是它的专属职责。
     - 收集所有专家的执行结果后，`OrchestratorAgent` 负责执行计划的最后一步：根据 `query` 中的严格约束（例如 "仅输出数值"），对原始信息进行"规范化处理"。
"""

import os
from pathlib import Path
from oxygent import oxy
from oxygent.schemas import LLMResponse, LLMState
import json
import yaml
import xml.etree.ElementTree as ET



def create_orchestrator_agent(
    llm_model: str = "default_llm",
    sub_agents: list = None
) -> oxy.ReActAgent:
    """
    创建 OrchestratorAgent 实例
    
    Args:
        llm_model: 使用的 LLM 模型名称
        sub_agents: 子智能体列表，默认为 ["search_agent", "multimodal_agent", "code_agent"]
        
    Returns:
        配置好的 OrchestratorAgent 实例
    """
    if sub_agents is None:
        sub_agents = ["search_agent", "multimodal_agent", "code_agent"]
    
    # 读取 orchestrator.prompt 文件
    prompt_file = Path(__file__).parent.parent / "prompts" / "orchestrator.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            orchestrator_prompt = f.read()
    else:
        print(f"⚠️  警告: 未找到 orchestrator.prompt 文件，路径: {prompt_file}")
        orchestrator_prompt = "你是一个 OrchestratorAgent，负责协调各个子智能体完成任务。"
    
    # 创建 OrchestratorAgent
    agent = oxy.ReActAgent(
        name="orchestrator_agent",
        is_master=True,
        desc=(
            "主控智能体，负责任务分析、规划、调度和结果交付。"
            "协调 SearchAgent、MultimodalAgent 和 CodeAgent 完成复杂任务。"
        ),
        sub_agents=sub_agents,
        llm_model=llm_model,
        prompt=orchestrator_prompt,
        max_react_rounds=15,  # 最大 ReAct 轮数（可能需要多次交互）
        timeout=600,  # 超时时间（秒）
        multimedia_supported=True,  # 支持多模态
    )
    
    return agent


