"""
CodeAgent 集成示例

展示如何在 OxyGent 框架中集成和使用 code_toolkit
"""

import os
import asyncio
from oxygent import MAS, Config, oxy

# 导入 code_tools
from code_toolkit import code_tools


# 配置默认 LLM
Config.set_agent_llm_model("default_llm")


# 定义 oxy_space
oxy_space = [
    # LLM 配置
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
        timeout=240,
    ),
    
    # 注册代码工具包
    code_tools,
    
    # CodeAgent - 代码执行专家
    oxy.ReActAgent(
        name="code_agent",
        desc="A specialist agent for code execution, mathematical calculations, "
             "file system operations, and structured data processing. "
             "Can handle Excel (.xlsx), PowerPoint (.pptx), Parquet (.parquet) files, "
             "and perform complex logical reasoning through Python code.",
        tools=["code_tools"],
        llm_model="default_llm",
        additional_prompt=(
            "Important instructions for CodeAgent:\n"
            "1. Always generate complete, self-contained Python code\n"
            "2. Use print() statements to output results\n"
            "3. Import necessary libraries at the beginning\n"
            "4. Handle potential errors gracefully\n"
            "5. For data files, use relative paths from project root\n"
            "6. Provide clear explanations of your approach\n\n"
            "Supported libraries:\n"
            "- math: mathematical calculations\n"
            "- os, glob: file system operations\n"
            "- pandas, openpyxl: Excel and data processing\n"
            "- python-pptx: PowerPoint processing\n"
            "- Standard Python libraries\n\n"
            "Always include the task_id when calling python_interpreter."
        ),
    ),
    
    # Master Agent - 总控制器
    oxy.ReActAgent(
        is_master=True,
        name="master_agent",
        desc="Master coordinator that analyzes tasks and delegates to appropriate specialist agents.",
        sub_agents=["code_agent"],
        llm_model="default_llm",
        additional_prompt=(
            "Task routing guidelines:\n\n"
            "Delegate to code_agent for:\n"
            "1. Mathematical calculations and logical reasoning\n"
            "   - Complex formulas, combinatorial problems\n"
            "   - Problems requiring precise computation\n"
            "2. File system operations\n"
            "   - Creating, reading, listing files\n"
            "   - Directory operations\n"
            "3. Structured data processing\n"
            "   - Excel files (.xlsx) - use pandas/openpyxl\n"
            "   - PowerPoint files (.pptx) - use python-pptx\n"
            "   - Parquet files (.parquet) - use pandas\n"
            "   - CSV/text data files\n"
            "4. Algorithmic problems\n"
            "   - Logic puzzles (poisoned wine, fake coin, etc.)\n"
            "   - Optimization problems\n"
            "   - Data structure manipulations\n\n"
            "Always extract the task_id from the user query and pass it to code_agent.\n"
            "The task_id is typically provided in the format: 'task_id: xxxx' or as part of the context."
        ),
    ),
]


async def test_basic_math():
    """测试：基础数学计算"""
    print("\n" + "=" * 70)
    print("测试 1: 毒酒问题 (task_id: 52ca290b)")
    print("=" * 70)
    
    query = """
task_id: 52ca290b
query: 有256瓶酒，其中有一瓶有毒。一只老鼠喝了有毒的酒会在24小时内死亡。
请问最少需要多少只老鼠才能在24小时内找出那瓶有毒的酒？
"""
    
    async with MAS(oxy_space=oxy_space) as mas:
        result = await mas.run(query=query)
        print("\n结果:")
        print(result.output)


async def test_light_speed():
    """测试：光速计算"""
    print("\n" + "=" * 70)
    print("测试 2: 光速计算 (task_id: c192f0c4)")
    print("=" * 70)
    
    query = """
task_id: c192f0c4
query: 如果一个物体以10米/秒²的恒定加速度加速，需要多少天才能达到光速（299,792,458米/秒）？
"""
    
    async with MAS(oxy_space=oxy_space) as mas:
        result = await mas.run(query=query)
        print("\n结果:")
        print(result.output)


async def test_file_operations():
    """测试：文件系统操作"""
    print("\n" + "=" * 70)
    print("测试 3: 统计日志文件 (task_id: 798d58a0)")
    print("=" * 70)
    
    query = """
task_id: 798d58a0
query: 请统计 data/ 目录及其子目录中所有 .log 文件的数量，并列出每个文件的大小。
"""
    
    async with MAS(oxy_space=oxy_space) as mas:
        result = await mas.run(query=query)
        print("\n结果:")
        print(result.output)


async def test_fake_coin():
    """测试：算法问题 - 天平称假币"""
    print("\n" + "=" * 70)
    print("测试 4: 天平称假币问题 (task_id: 5775255e)")
    print("=" * 70)
    
    query = """
task_id: 5775255e
query: 有12个硬币，其中有一个是假币（重量不同，但不知道是轻还是重）。
只有一个天平，请问最少需要称几次才能找出假币并确定它是轻是重？
请用Python实现算法并解释原理。
"""
    
    async with MAS(oxy_space=oxy_space) as mas:
        result = await mas.run(query=query)
        print("\n结果:")
        print(result.output)


async def interactive_mode():
    """交互模式：启动 Web 服务"""
    print("\n" + "=" * 70)
    print("启动交互式 Web 服务")
    print("=" * 70)
    print("\n访问 http://localhost:8082 与 CodeAgent 交互")
    print("按 Ctrl+C 停止服务\n")
    
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            port=8082,
            first_query="你好！我是 CodeAgent，擅长执行 Python 代码来解决数学、逻辑和数据处理问题。"
        )


async def main():
    """主函数 - 选择运行模式"""
    print("\n" + "=" * 70)
    print("CodeAgent 集成示例")
    print("=" * 70)
    print("\n选择运行模式:")
    print("  1. 运行所有测试")
    print("  2. 启动交互式 Web 服务")
    print("  0. 退出")
    
    # 为了演示，这里直接运行所有测试
    # 在实际使用中，可以根据用户输入选择模式
    
    mode = input("\n请选择 (0-2): ").strip()
    
    if mode == "1":
        # 运行所有测试
        await test_basic_math()
        await test_light_speed()
        await test_file_operations()
        await test_fake_coin()
        
        print("\n" + "=" * 70)
        print("所有测试完成！")
        print("=" * 70)
        print("\n📁 查看 local_es_data/ 目录了解详细的代码和输出")
        
    elif mode == "2":
        # 启动交互式服务
        await interactive_mode()
        
    else:
        print("\n再见！")


if __name__ == "__main__":
    """
    使用方法:
    
    1. 确保已设置环境变量:
       export DEFAULT_LLM_API_KEY="your_api_key"
       export DEFAULT_LLM_BASE_URL="your_base_url"
       export DEFAULT_LLM_MODEL_NAME="your_model_name"
    
    2. 安装依赖:
       pip install oxygent pandas openpyxl python-pptx pyarrow pydantic
    
    3. 运行:
       python code_agent_example.py
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n程序已停止。")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

