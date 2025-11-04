"""
京东多智能体挑战赛 - 主运行脚本

这个脚本负责：
1. 加载所有智能体（CodeAgent、SearchAgent、MultimodalAgent、OrchestratorAgent）
2. 读取验证集数据
3. 执行任务并保存结果
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oxygent import MAS, Config, oxy

# 导入 code_tools
try:
    spec_tools = __import__('importlib.util').util.spec_from_file_location(
        "code_toolkit",
        project_root / "tools.py" / "code_toolkit.py"
    )
    code_toolkit = __import__('importlib.util').util.module_from_spec(spec_tools)
    spec_tools.loader.exec_module(code_toolkit)
    code_tools = code_toolkit.code_tools
    print("✅ 成功导入 code_tools")
except Exception as e:
    print(f"❌ 导入 code_tools 失败: {e}")
    code_tools = None

# 导入 multimodal_tools
try:
    spec_mm = __import__('importlib.util').util.spec_from_file_location(
        "multimodal_toolkit",
        project_root / "tools.py" / "multimodal_toolkit.py"
    )
    multimodal_toolkit = __import__('importlib.util').util.module_from_spec(spec_mm)
    spec_mm.loader.exec_module(multimodal_toolkit)
    multimodal_tools = multimodal_toolkit.multimodal_tools
    print("✅ 成功导入 multimodal_tools")
except Exception as e:
    print(f"❌ 导入 multimodal_tools 失败: {e}")
    multimodal_tools = None

# 导入所有智能体
try:
    from agents.code_agent import create_code_agent
    from agents.search_agent import create_search_agent
    from agents.multimodal_agent import create_multimodal_agent
    from agents.orchestrator_agent import create_orchestrator_agent
    print("✅ 成功导入所有智能体")
except Exception as e:
    print(f"❌ 导入智能体失败: {e}")
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
        print(f"\n⚠️  缺少环境变量: {', '.join(missing_vars)}")
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
    ]
    
    # 添加工具包（如果成功导入）
    if code_tools:
        oxy_space.append(code_tools)
    if multimodal_tools:
        oxy_space.append(multimodal_tools)
    
    # 添加所有专家智能体
    oxy_space.extend([
        create_code_agent(llm_model="default_llm"),
        create_search_agent(llm_model="default_llm"),
        create_multimodal_agent(llm_model="default_llm"),
    ])
    
    # 添加 OrchestratorAgent（主控）
    oxy_space.append(
        create_orchestrator_agent(
            llm_model="default_llm",
            sub_agents=["code_agent", "search_agent", "multimodal_agent"]
        )
    )
    
    return oxy_space


def load_validation_set(file_path: str = "data/validation_set.jsonl"):
    """加载验证集数据"""
    data_file = project_root / file_path
    
    if not data_file.exists():
        print(f"❌ 验证集文件不存在: {data_file}")
        return []
    
    tasks = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f"✅ 加载了 {len(tasks)} 个任务")
    return tasks


async def run_single_task(mas, task):
    """运行单个任务"""
    task_id = task.get("task_id", "unknown")
    query = task.get("query", "")
    level = task.get("level", "")
    file_name = task.get("file_name", "")
    
    print(f"\n{'='*70}")
    print(f"任务 ID: {task_id}")
    print(f"级别: {level}")
    print(f"问题: {query[:100]}..." if len(query) > 100 else f"问题: {query}")
    if file_name:
        print(f"文件: {file_name}")
    print(f"{'='*70}")
    
    # 构建输入
    task_input = f"""
task_id: {task_id}
query: {query}
level: {level}
file_name: {file_name}
"""
    
    try:
        result = await mas.call(  
            callee="orchestrator_agent",   
            arguments={"query": task_input}  
        )  
        answer = result.output if hasattr(result, 'output') else str(result)
        
        print(f"\n✅ 结果: {answer}")
        
        return {
            "task_id": task_id,
            "answer": answer
        }
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "task_id": task_id,
            "answer": f"ERROR: {str(e)}"
        }


async def run_validation_set(limit: int = None):
    """运行验证集任务"""
    print("\n" + "=" * 70)
    print("运行验证集任务")
    print("=" * 70)
    
    # 加载任务
    tasks = load_validation_set()
    if not tasks:
        print("❌ 没有任务可运行")
        return
    
    # 限制任务数量（用于测试）
    if limit:
        tasks = tasks[:limit]
        print(f"\n⚠️  限制运行前 {limit} 个任务")
    
    # 创建 oxy_space
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    # 运行任务
    results = []
    
    async with MAS(oxy_space=oxy_space) as mas:
        for i, task in enumerate(tasks, 1):
            print(f"\n\n{'#'*70}")
            print(f"进度: {i}/{len(tasks)}")
            print(f"{'#'*70}")
            
            result = await run_single_task(mas, task)
            results.append(result)
            
            # 保存中间结果
            save_results(results, f"outputs/results_partial_{i}.json")
    
    # 保存最终结果
    save_results(results, "outputs/results_final.json")
    
    print("\n" + "=" * 70)
    print("验证集任务完成！")
    print("=" * 70)
    print(f"总任务数: {len(results)}")
    print(f"结果已保存到: outputs/results_final.json")


def save_results(results, filename):
    """保存结果到文件"""
    output_file = project_root / filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def run_interactive_mode(port: int = 8082):
    """交互模式：启动 Web 服务"""
    print("\n" + "=" * 70)
    print("启动交互式 Web 服务")
    print("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    print(f"\n🌐 访问 http://localhost:{port} 与系统交互")
    print("按 Ctrl+C 停止服务\n")
    
    try:
        async with MAS(oxy_space=oxy_space) as mas:
            await mas.start_web_service(
                port=port,
                first_query=(
                    "你好！我是京东多智能体挑战赛系统。\n\n"
                    "我包含以下智能体:\n"
                    "• CodeAgent - 代码执行和数据处理\n"
                    "• SearchAgent - 网页搜索和信息检索\n"
                    "• MultimodalAgent - 多模态内容分析\n"
                    "• OrchestratorAgent - 任务协调和规划\n\n"
                    "请告诉我你需要解决什么问题？"
                )
            )
    except KeyboardInterrupt:
        print("\n\n服务已停止。")
    except Exception as e:
        print(f"\n❌ 服务出错: {e}")


async def run_test_task():
    """运行单个测试任务"""
    print("\n" + "=" * 70)
    print("运行测试任务")
    print("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    # 测试任务：毒酒问题
    test_task = {
        "task_id": "test_001",
        "query": "有256瓶酒，其中有一瓶有毒。一只老鼠喝了有毒的酒会在24小时内死亡。请问最少需要多少只老鼠才能在24小时内找出那瓶有毒的酒？只输出数字。",
        "level": "2",
        "file_name": ""
    }
    
    async with MAS(oxy_space=oxy_space) as mas:
        result = await run_single_task(mas, test_task)
        print(f"\n最终结果: {result}")


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 70)
    print("京东多智能体挑战赛 - 主程序")
    print("=" * 70)
    print("\n选择运行模式:")
    print("  1. 运行单个测试任务")
    print("  2. 运行验证集（前 3 个任务）")
    print("  3. 运行验证集（前 10 个任务）")
    print("  4. 运行完整验证集（全部任务）")
    print("  5. 启动交互式 Web 服务")
    print("  0. 退出")
    print()


async def main():
    """主函数"""
    
    # 环境检查
    print("\n" + "=" * 70)
    print("环境检查")
    print("=" * 70)
    
    # 检查目录
    local_es_data = project_root / "local_es_data"
    local_es_data.mkdir(exist_ok=True)
    print(f"✅ local_es_data 目录: {local_es_data}")
    
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    print(f"✅ outputs 目录: {outputs_dir}")
    
    # 检查环境变量
    env_vars = ["DEFAULT_LLM_API_KEY", "DEFAULT_LLM_BASE_URL", "DEFAULT_LLM_MODEL_NAME"]
    missing = [v for v in env_vars if not os.getenv(v)]
    
    if missing:
        print(f"\n⚠️  缺少环境变量: {', '.join(missing)}")
        print("\n提示: 设置环境变量后再运行")
        print("\n按 Enter 键继续...")
        input()
    else:
        print("✅ 环境变量已配置")
    
    # 主循环
    while True:
        print_menu()
        
        try:
            choice = input("请选择 (0-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n再见！")
            break
        
        if choice == "0":
            print("\n再见！")
            break
        elif choice == "1":
            await run_test_task()
        elif choice == "2":
            await run_validation_set(limit=3)
        elif choice == "3":
            await run_validation_set(limit=10)
        elif choice == "4":
            await run_validation_set()
        elif choice == "5":
            await run_interactive_mode()
        else:
            print("\n❌ 无效选项，请重新选择。")
        
        if choice in ["1", "2", "3", "4"]:
            print("\n按 Enter 键继续...")
            input()


if __name__ == "__main__":
    """
    使用方法:
    
    1. 设置环境变量:
       export DEFAULT_LLM_API_KEY="your_api_key"
       export DEFAULT_LLM_BASE_URL="your_base_url"
       export DEFAULT_LLM_MODEL_NAME="your_model_name"
    
    2. 安装依赖:
       pip install oxygent pandas openpyxl python-pptx pyarrow pydantic
    
    3. 运行:
       python run.py
    """
    
    print("\n" + "=" * 70)
    print("京东多智能体挑战赛 - 主程序")
    print("=" * 70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n程序已停止。")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()
