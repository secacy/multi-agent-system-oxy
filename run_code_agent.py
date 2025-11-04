"""
完整的 CodeAgent 运行示例

展示如何集成和使用 CodeAgent 来解决各种任务
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oxygent import MAS, Config, oxy

# 导入 code_tools 和 code_agent
try:
    # 尝试从 tools.py 目录导入
    spec_tools = __import__('importlib.util').util.spec_from_file_location(
        "code_toolkit",
        project_root / "tools.py" / "code_toolkit.py"
    )
    code_toolkit = __import__('importlib.util').util.module_from_spec(spec_tools)
    spec_tools.loader.exec_module(code_toolkit)
    code_tools = code_toolkit.code_tools
    
    # 导入 code_agent
    from agents.code_agent import create_code_agent
    
    print("✅ 成功导入 code_tools 和 code_agent")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


# 配置默认 LLM
Config.set_agent_llm_model("default_llm")


def create_oxy_space():
    """创建 oxy_space 配置"""
    
    # 检查环境变量
    required_env_vars = [
        "DEFAULT_LLM_API_KEY",
        "DEFAULT_LLM_BASE_URL", 
        "DEFAULT_LLM_MODEL_NAME"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠️  缺少环境变量: {', '.join(missing_vars)}")
        print("\n请设置以下环境变量:")
        print("  export DEFAULT_LLM_API_KEY='your_api_key'")
        print("  export DEFAULT_LLM_BASE_URL='your_base_url'")
        print("  export DEFAULT_LLM_MODEL_NAME='your_model_name'")
        return None
    
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
        
        # CodeAgent
        create_code_agent(llm_model="default_llm"),
        
        # Master Agent（如果需要协调多个 agent）
        oxy.ReActAgent(
            is_master=True,
            name="master_agent",
            desc="主控智能体，负责协调和分派任务",
            sub_agents=["code_agent"],
            llm_model="default_llm",
        ),
    ]
    
    return oxy_space


async def test_math_calculation():
    """测试 1: 数学计算 - 毒酒问题"""
    print("\n" + "=" * 70)
    print("测试 1: 毒酒问题 (52ca290b)")
    print("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    query = """
task_id: 52ca290b
问题: 有256瓶酒，其中有一瓶有毒。一只老鼠喝了有毒的酒会在24小时内死亡。
请问最少需要多少只老鼠才能在24小时内找出那瓶有毒的酒？请只输出数字。
"""
    
    try:
        async with MAS(oxy_space=oxy_space) as mas:
            result = await mas.run(query=query)
            print("\n结果:")
            print(result.output)
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")


async def test_light_speed():
    """测试 2: 光速计算"""
    print("\n" + "=" * 70)
    print("测试 2: 光速计算 (c192f0c4)")
    print("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    query = """
task_id: c192f0c4
问题: 如果一个物体以10米/秒²的恒定加速度加速，需要多少天才能达到光速（299,792,458米/秒）？
请计算并只输出天数的数值（保留两位小数）。
"""
    
    try:
        async with MAS(oxy_space=oxy_space) as mas:
            result = await mas.run(query=query)
            print("\n结果:")
            print(result.output)
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")


async def test_file_operations():
    """测试 3: 文件系统操作"""
    print("\n" + "=" * 70)
    print("测试 3: 文件系统操作 (798d58a0)")
    print("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    # 先创建一些测试文件
    test_dir = project_root / "data" / "test_logs"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = [
        test_dir / "app.log",
        test_dir / "error.log",
        test_dir / "access.log",
    ]
    
    for file in test_files:
        file.write_text(f"测试日志内容 - {file.name}\n" * 10)
    
    print(f"\n已创建测试文件在: {test_dir}")
    
    query = f"""
task_id: 798d58a0
问题: 请统计 {test_dir} 目录中所有 .log 文件的数量。只输出数字。
"""
    
    try:
        async with MAS(oxy_space=oxy_space) as mas:
            result = await mas.run(query=query)
            print("\n结果:")
            print(result.output)
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")


async def test_fake_coin():
    """测试 4: 算法问题 - 天平称假币"""
    print("\n" + "=" * 70)
    print("测试 4: 天平称假币问题 (5775255e)")
    print("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    query = """
task_id: 5775255e
问题: 有12个硬币，其中有一个是假币（重量不同）。只有一个天平，请问最少需要称几次才能找出假币？
请只输出需要称重的次数（数字）。
"""
    
    try:
        async with MAS(oxy_space=oxy_space) as mas:
            result = await mas.run(query=query)
            print("\n结果:")
            print(result.output)
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")


async def interactive_mode():
    """交互模式：启动 Web 服务"""
    print("\n" + "=" * 70)
    print("启动交互式 Web 服务")
    print("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    print("\n🌐 访问 http://localhost:8082 与 CodeAgent 交互")
    print("按 Ctrl+C 停止服务\n")
    
    try:
        async with MAS(oxy_space=oxy_space) as mas:
            await mas.start_web_service(
                port=8082,
                first_query=(
                    "你好！我是 CodeAgent，专门负责执行 Python 代码来解决各种问题。\n\n"
                    "我擅长:\n"
                    "• 数学计算和逻辑推理\n"
                    "• 文件系统操作\n"
                    "• 数据处理（Excel、PowerPoint、Parquet等）\n\n"
                    "请告诉我你需要解决什么问题？"
                )
            )
    except KeyboardInterrupt:
        print("\n\n服务已停止。")
    except Exception as e:
        print(f"\n❌ 服务出错: {e}")


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 70)
    print("CodeAgent 测试系统")
    print("=" * 70)
    print("\n选择测试模式:")
    print("  1. 数学计算 - 毒酒问题")
    print("  2. 物理计算 - 光速问题")
    print("  3. 文件操作 - 统计日志文件")
    print("  4. 算法问题 - 天平称假币")
    print("  5. 运行所有测试")
    print("  6. 启动交互式 Web 服务")
    print("  0. 退出")
    print()


async def main():
    """主函数"""
    
    # 检查环境
    print("\n" + "=" * 70)
    print("CodeAgent 运行环境检查")
    print("=" * 70)
    
    # 检查必要的目录
    local_es_data = project_root / "local_es_data"
    local_es_data.mkdir(exist_ok=True)
    print(f"✅ local_es_data 目录: {local_es_data}")
    
    # 检查 code.prompt
    code_prompt = project_root / "prompts" / "code.prompt"
    if code_prompt.exists():
        print(f"✅ code.prompt 文件: {code_prompt}")
    else:
        print(f"⚠️  code.prompt 文件不存在: {code_prompt}")
        print("   将使用默认 prompt")
    
    # 检查环境变量
    env_vars = ["DEFAULT_LLM_API_KEY", "DEFAULT_LLM_BASE_URL", "DEFAULT_LLM_MODEL_NAME"]
    missing = [v for v in env_vars if not os.getenv(v)]
    
    if missing:
        print(f"\n⚠️  缺少环境变量: {', '.join(missing)}")
        print("\n提示: 设置环境变量后再运行测试")
        print("或者直接运行演示模式（不需要 LLM）")
        print("\n按任意键继续...")
        input()
    else:
        print("✅ 环境变量已配置")
    
    # 主循环
    while True:
        print_menu()
        
        try:
            choice = input("请选择 (0-6): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n再见！")
            break
        
        if choice == "0":
            print("\n再见！")
            break
        elif choice == "1":
            await test_math_calculation()
        elif choice == "2":
            await test_light_speed()
        elif choice == "3":
            await test_file_operations()
        elif choice == "4":
            await test_fake_coin()
        elif choice == "5":
            print("\n运行所有测试...")
            await test_math_calculation()
            await test_light_speed()
            await test_file_operations()
            await test_fake_coin()
            print("\n" + "=" * 70)
            print("所有测试完成！")
            print("=" * 70)
            print(f"\n📁 查看 {local_es_data} 目录了解详细的代码和输出")
        elif choice == "6":
            await interactive_mode()
        else:
            print("\n❌ 无效选项，请重新选择。")
        
        if choice in ["1", "2", "3", "4", "5"]:
            print("\n按 Enter 键继续...")
            input()


if __name__ == "__main__":
    """
    使用方法:
    
    1. 设置环境变量（用于测试 LLM 集成）:
       export DEFAULT_LLM_API_KEY="your_api_key"
       export DEFAULT_LLM_BASE_URL="your_base_url"
       export DEFAULT_LLM_MODEL_NAME="your_model_name"
    
    2. 安装依赖:
       pip install oxygent pandas openpyxl python-pptx pyarrow pydantic
    
    3. 运行:
       python run_code_agent.py
    """
    
    print("\n" + "=" * 70)
    print("CodeAgent 完整示例")
    print("=" * 70)
    print("\n这个示例展示了如何使用 CodeAgent 解决各种问题")
    print("包括数学计算、文件操作、算法问题等")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n程序已停止。")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()

