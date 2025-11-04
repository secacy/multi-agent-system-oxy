"""
CodeAgent (代码、逻辑与数据智能体)

这是"精确计算与结构化数据"专家，负责所有需要代码、算法或数据分析的任务。

- 工具集：
  - Python 3 解释器 (`python_interpreter`)：这是它的唯一工具。它通过生成代码来调用 `os`, `glob`, `math`, `pandas`, `openpyxl`, `python-pptx` 等库。
- 专业技能 ：
  - 数学计算：如 `c192f0c4...`（计算达到光速的天数）。
  - 文件系统操作：通过 `import os, glob` 来执行，如 `cdf7066e...`（创建目录/文件）或 `798d58a0...`（统计log文件）。
  - 精确逻辑推理：通过算法解决LLM无法"猜"对的谜题，如 `52ca290b...`（毒酒问题, `2^8=256`）29或 `5775255e...`（天平测假币）。
  - 结构化数据处理 (新！)：
    - `.xlsx`：如 `d8b695a6...`（读取库存表和成本表）。
    - `.pptx`：如 `036c3dd6...`（比较两个PPT的页码）。
    - `.parquet`：如 `dc247edf...`（读取`gsm-8k`数据集）。
    - `.txt` (作为数据源)：如 `d8b695a6...`（读取`8-transfer_costs.txt`）。
- 复现要求：
  - 必须将用于解决问题的完整Python代码和代码的执行输出，以 `task_id` 命名，保存到 `local_es_data/` 目录 。
"""

import os
from pathlib import Path
from oxygent import oxy


def create_code_agent(llm_model: str = "default_llm") -> oxy.ReActAgent:
    """
    创建 CodeAgent 实例
    
    Args:
        llm_model: 使用的 LLM 模型名称
        
    Returns:
        配置好的 CodeAgent 实例
    """
    # 读取 code.prompt 文件
    prompt_file = Path(__file__).parent.parent / "prompts" / "code.prompt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            code_prompt = f.read()
    else:
        print(f"⚠️  警告: 未找到 code.prompt 文件，路径: {prompt_file}")
        code_prompt = "你是一个 code agent，负责生成和执行Python代码。"
    
    # 创建 CodeAgent
    agent = oxy.ReActAgent(
        name="code_agent",
        desc=(
            "A specialist agent for code execution, mathematical calculations, "
            "file system operations, and structured data processing. "
            "Expert in handling Excel (.xlsx), PowerPoint (.pptx), Parquet (.parquet) files, "
            "and performing complex logical reasoning through Python code. "
            "Uses python_interpreter tool to execute Python 3 code."
        ),
        tools=["code_tools"],
        llm_model=llm_model,
        prompt=code_prompt,
        max_react_rounds=10,  # 最大 ReAct 轮数
        timeout=300,  # 超时时间（秒）
        trust_mode=True,  # Agent 直接返回工具的原始输出,不进行额外处理
    )
    
    return agent



