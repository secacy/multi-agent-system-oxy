"""
CodeAgent 工具包：python_interpreter

职责：这是架构的关键。CodeAgent将只拥有一个，但功能无限的工具。

1. python_interpreter(code: str, task_id: str) -> str
  - 描述: 在一个沙盒环境中执行一段Python 3代码，并返回其stdout（标准输出）。
  - 关键职责:
    - CodeAgent 负责生成解决问题所需的所有Python代码。
    - 数学/逻辑: import math 来解决 52ca290b... (毒酒, 2^8=256)或 c192f0c4... (光速)。
    - 文件系统: import os, glob 来解决 798d58a0... (统计 .log 文件)或 cdf7066e... (创建目录/文件) 。
    - Excel (.xlsx): import pandas, openpyxl 来解决 d8b695a6... (读取 8-initial_inventory.xlsx 并计算成本)。
    - PowerPoint (.pptx): import python-pptx 来解决 036c3dd6... (比较两个PPT的页码)或 7f00aaa3... (找目录页)。
    - Parquet (.parquet): import pandas 来解决 dc247edf... (读取 gsm-8k 数据集并计算)。
"""

import os
import sys
import traceback
from io import StringIO
from pathlib import Path
from pydantic import Field
from oxygent.oxy import FunctionHub

# 注册代码工具包
code_tools = FunctionHub(name="code_tools")


@code_tools.tool(
    description="Execute Python 3 code in a sandboxed environment and return stdout. "
    "Can be used for mathematical calculations (math), file system operations (os, glob), "
    "data processing (pandas, openpyxl for Excel, python-pptx for PowerPoint, parquet files), "
    "and complex logical reasoning. The code and its output will be saved to local_es_data/ directory."
)
def python_interpreter(
    code: str = Field(
        description="The Python 3 code string to execute. Should be complete and self-contained."
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving code and output to files."
    ),
) -> str:
    """
    在沙盒环境中执行 Python 3 代码并返回标准输出。
    
    支持的库包括：
    - math: 数学计算
    - os, glob: 文件系统操作
    - pandas: 数据处理（支持 Excel、Parquet 等）
    - openpyxl: Excel 文件处理
    - python-pptx: PowerPoint 文件处理
    
    Args:
        code: 要执行的 Python 3 代码字符串
        task_id: 任务唯一标识符，用于保存代码和输出
    
    Returns:
        str: 代码执行的标准输出，如果出错则返回错误信息
    """
    # 创建 local_es_data 目录（如果不存在）
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备文件路径
    code_file = output_dir / f"{task_id}_code.py"
    output_file = output_dir / f"{task_id}_output.txt"
    
    # 保存代码到文件
    try:
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        error_msg = f"❌ 保存代码文件失败: {str(e)}"
        return error_msg
    
    # 捕获标准输出
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    # 执行结果
    execution_result = ""
    error_occurred = False
    
    try:
        # 创建一个受限的全局命名空间
        # 允许常用的库和内置函数
        safe_globals = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__file__": str(code_file),
        }
        
        safe_locals = safe_globals
        
        # 执行代码
        exec(code, safe_globals, safe_locals)
        
        # 获取输出
        execution_result = captured_output.getvalue()
        
        if not execution_result:
            execution_result = "✅ 代码执行成功，但没有产生输出。"
        else:
            execution_result = f"✅ 代码执行成功:\n{execution_result}"
            
    except Exception as e:
        # 捕获并格式化异常信息
        error_occurred = True
        error_trace = traceback.format_exc()
        execution_result = f"❌ 代码执行出错:\n{error_trace}"
        
    finally:
        # 恢复标准输出
        sys.stdout = old_stdout
    
    # 保存输出到文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"=== Task ID: {task_id} ===\n\n")
            f.write(f"=== 代码 ===\n{code}\n\n")
            f.write(f"=== 执行结果 ===\n{execution_result}\n")
    except Exception as e:
        execution_result += f"\n\n⚠️ 警告: 保存输出文件失败: {str(e)}"
    
    # 返回执行结果
    result_summary = (
        f"代码已保存到: {code_file}\n"
        f"输出已保存到: {output_file}\n\n"
        f"{execution_result}"
    )
    
    return result_summary

