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
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入日志配置
from logger_config import setup_logger

# 初始化日志系统
logger = setup_logger(
    name="oxy-mas",
    log_dir=str(project_root / "logs"),
    level=logging.INFO
)

from oxygent import MAS, Config, oxy

# 导入 code_tools
try:
    spec_tools = __import__('importlib.util').util.spec_from_file_location(
        "code_toolkit",
        project_root / "tools" / "code_toolkit.py"
    )
    code_toolkit = __import__('importlib.util').util.module_from_spec(spec_tools)
    spec_tools.loader.exec_module(code_toolkit)
    code_tools = code_toolkit.code_tools
    logger.info("成功导入 code_tools")
except Exception as e:
    logger.error(f"导入 code_tools 失败: {e}")
    code_tools = None


# 导入 audio_tools
try:
    spec_audio = __import__('importlib.util').util.spec_from_file_location(
        "audio_toolkit",
        project_root / "tools" / "audio_toolkit.py"
    )
    audio_toolkit = __import__('importlib.util').util.module_from_spec(spec_audio)
    spec_audio.loader.exec_module(audio_toolkit)
    audio_tools = audio_toolkit.audio_tools
    logger.info("成功导入 audio_tools")
except Exception as e:
    logger.error(f"导入 audio_tools 失败: {e}")
    audio_tools = None

# 导入 pdf_tools
try:
    spec_pdf = __import__('importlib.util').util.spec_from_file_location(
        "pdf_toolkit",
        project_root / "tools" / "pdf_toolkit.py"
    )
    pdf_toolkit = __import__('importlib.util').util.module_from_spec(spec_pdf)
    spec_pdf.loader.exec_module(pdf_toolkit)
    pdf_tools = pdf_toolkit.pdf_tools
    logger.info("成功导入 pdf_tools")
except Exception as e:
    logger.error(f"导入 pdf_tools 失败: {e}")
    pdf_tools = None

# 导入 search_tools
try:
    spec_search = __import__('importlib.util').util.spec_from_file_location(
        "search_toolkit",
        project_root / "tools" / "search_toolkit.py"
    )
    search_toolkit = __import__('importlib.util').util.module_from_spec(spec_search)
    spec_search.loader.exec_module(search_toolkit)
    search_tools = search_toolkit.search_tools
    logger.info("成功导入 search_tools")
except Exception as e:
    logger.error(f"导入 search_tools 失败: {e}")
    search_tools = None

# 导入 video_tools
try:
    spec_video = __import__('importlib.util').util.spec_from_file_location(
        "video_toolkit",
        project_root / "tools" / "video_toolkit.py"
    )
    video_toolkit = __import__('importlib.util').util.module_from_spec(spec_video)
    spec_video.loader.exec_module(video_toolkit)
    video_tools = video_toolkit.video_tools
    logger.info("成功导入 video_tools")
except Exception as e:
    logger.error(f"导入 video_tools 失败: {e}")
    video_tools = None

# 导入所有智能体
try:
    from agents.code_agent import create_code_agent
    from agents.search_agent import create_search_agent
    from agents.audio_agent import create_audio_agent
    from agents.video_agent import create_video_agent
    from agents.image_agent import create_image_agent
    from agents.image_l3_agents import create_all_image_l3_agents
    from agents.ocr_agent import create_ocr_agent
    from agents.ocr_l3_agents import create_all_ocr_l3_agents
    from agents.multimodal_agent import create_multimodal_agent
    from agents.orchestrator_agent import create_orchestrator_agent
    logger.info("成功导入所有智能体")
except Exception as e:
    logger.error(f"导入智能体失败: {e}")
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
        logger.warning(f"缺少环境变量: {', '.join(missing_vars)}")
        logger.info("请设置以下环境变量:")
        logger.info("  export DEFAULT_LLM_API_KEY='your_api_key'")
        logger.info("  export DEFAULT_LLM_BASE_URL='your_base_url'")
        logger.info("  export DEFAULT_LLM_MODEL_NAME='your_model_name'")
        logger.info("如果需要图像分析功能，还需要设置 VLM 环境变量:")
        logger.info("  export DEFAULT_VLM_API_KEY='your_vlm_api_key'")
        logger.info("  export DEFAULT_VLM_BASE_URL='your_vlm_base_url'")
        logger.info("  export DEFAULT_VLM_MODEL_NAME='your_vlm_model_name'")
        return None
    
    oxy_space = [
        # ==================== LLM 配置 ====================
        
        # 1. 文本 LLM（用于大多数 Agent 的推理和决策）
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
    
    # 2. VLM（视觉语言模型，用于 L3 图像分析专家）
    if os.getenv("DEFAULT_VLM_API_KEY"):
        oxy_space.append(
            oxy.HttpLLM(
                name="default_vlm",
                api_key=os.getenv("DEFAULT_VLM_API_KEY"),
                base_url=os.getenv("DEFAULT_VLM_BASE_URL"),
                model_name=os.getenv("DEFAULT_VLM_MODEL_NAME"),
                llm_params={"temperature": 0.6, "max_tokens": 2048},
                is_multimodal_supported=True,  # 启用多模态支持
                max_pixels=10000000,  # 图像最大像素
                is_convert_url_to_base64=True,  # 将URL转换为base64
                semaphore=4,
                timeout=240,
            )
        )
        logger.info("已配置 VLM（视觉语言模型）")
    else:
        logger.warning("未配置 VLM，图像分析功能将不可用")
    
    # ==================== 工具包 ====================
    
    if code_tools:
        oxy_space.append(code_tools)
    if audio_tools:
        oxy_space.append(audio_tools)
    if pdf_tools:
        oxy_space.append(pdf_tools)
    if search_tools:
        oxy_space.append(search_tools)
        logger.info("已注册 search_tools")
    if video_tools:
        oxy_space.append(video_tools)
        logger.info("已注册 video_tools")
    
    # ==================== L3 专家智能体 ====================
    
    # L3 图像分析专家（使用 VLM）
    if os.getenv("DEFAULT_VLM_API_KEY"):
        oxy_space.extend(create_all_image_l3_agents(llm_model="default_vlm"))
        logger.info("已注册 L3 图像分析专家（4个）")
    
    # L3 OCR 专家（使用 VLM 和文本 LLM）
    if os.getenv("DEFAULT_VLM_API_KEY"):
        oxy_space.extend(create_all_ocr_l3_agents(llm_model_vlm="default_vlm", llm_model_text="default_llm"))
        logger.info("已注册 L3 OCR 专家（4个：3个VLM + 1个文本分析）")
    
    # ==================== L2 专家智能体 ====================
    
    oxy_space.extend([
        create_audio_agent(llm_model="default_llm"),  # L2: 音频专家
        create_video_agent(llm_model="default_llm"),  # L2: 视频专家
    ])
    
    # L2 图像管理者（使用文本 LLM）
    if os.getenv("DEFAULT_VLM_API_KEY"):
        oxy_space.append(create_image_agent(llm_model="default_llm"))
        logger.info("已注册 L2 图像管理者")
    
    # L2 OCR 管理者（使用文本 LLM）
    if os.getenv("DEFAULT_VLM_API_KEY"):
        oxy_space.append(create_ocr_agent(llm_model="default_llm"))
        logger.info("已注册 L2 OCR 管理者")
    
    # ==================== L1 专家智能体 ====================
    
    # 添加 L1 专家智能体
    oxy_space.extend([
        create_code_agent(llm_model="default_llm"),
        create_search_agent(llm_model="default_llm"),
        create_multimodal_agent(llm_model="default_llm"),  # L1: 多模态路由器
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
        logger.error(f"验证集文件不存在: {data_file}")
        return []
    
    tasks = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    logger.info(f"加载了 {len(tasks)} 个任务")
    return tasks


async def run_single_task(mas, task):
    """运行单个任务"""
    task_id = task.get("task_id", "unknown")
    query = task.get("query", "")
    level = task.get("level", "")
    file_name = task.get("file_name", "")
    
    logger.info("=" * 70)
    logger.info(f"任务 ID: {task_id}")
    logger.info(f"级别: {level}")
    logger.info(f"问题: {query[:100]}..." if len(query) > 100 else f"问题: {query}")
    if file_name:
        logger.info(f"文件: {file_name}")
    logger.info("=" * 70)
    
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
        
        logger.info(f"结果: {answer}")
        
        return {
            "task_id": task_id,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"异常详情: {traceback.format_exc()}")
        
        return {
            "task_id": task_id,
            "answer": f"ERROR: {str(e)}"
        }


async def run_validation_set(limit: int = None):
    """运行验证集任务"""
    logger.info("=" * 70)
    logger.info("运行验证集任务")
    logger.info("=" * 70)
    
    # 加载任务
    tasks = load_validation_set()
    if not tasks:
        logger.error("没有任务可运行")
        return
    
    # 限制任务数量（用于测试）
    if limit:
        tasks = tasks[:limit]
        logger.warning(f"限制运行前 {limit} 个任务")
    
    # 创建 oxy_space
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    # 运行任务
    results = []
    
    async with MAS(oxy_space=oxy_space) as mas:
        for i, task in enumerate(tasks, 1):
            logger.info("#" * 70)
            logger.info(f"进度: {i}/{len(tasks)}")
            logger.info("#" * 70)
            
            result = await run_single_task(mas, task)
            results.append(result)
            
            # 保存中间结果
            save_results(results, f"outputs/results_partial_{i}.json")
    
    # 保存最终结果
    save_results(results, "outputs/results_final.json")
    
    logger.info("=" * 70)
    logger.info("验证集任务完成！")
    logger.info("=" * 70)
    logger.info(f"总任务数: {len(results)}")
    logger.info(f"结果已保存到: outputs/results_final.json")


def save_results(results, filename):
    """保存结果到文件"""
    output_file = project_root / filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def run_interactive_mode(port: int = 8088):
    """交互模式：启动 Web 服务"""
    logger.info("=" * 70)
    logger.info("启动交互式 Web 服务")
    logger.info("=" * 70)
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    logger.info(f"访问 http://localhost:{port} 与系统交互")
    logger.info("按 Ctrl+C 停止服务")
    
    try:
        async with MAS(oxy_space=oxy_space) as mas:
            await mas.start_web_service(
                port=port,
                first_query=(
                    ""
                )
            )
    except KeyboardInterrupt:
        logger.info("服务已停止")
    except Exception as e:
        logger.error(f"服务出错: {e}")


async def run_test_task():
    """运行单个测试任务"""
    logger.info("=" * 70)
    logger.info("运行测试任务")
    logger.info("=" * 70)
    
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
        logger.info(f"最终结果: {result}")


async def run_search_test():
    """测试 SearchAgent 功能"""
    logger.info("=" * 70)
    logger.info("测试 SearchAgent")
    logger.info("=" * 70)
    
    # 检查 Serper API Key
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        logger.warning("未设置 SERPER_API_KEY，搜索功能将不可用")
        logger.info("请运行: export SERPER_API_KEY='your_api_key'")
        logger.info("注册地址: https://serper.dev/")
        return
    
    oxy_space = create_oxy_space()
    if not oxy_space:
        return
    
    # 测试任务：搜索并打开网页
    test_task = {
        "task_id": "search_test_001",
        "query": "搜索 'Python 官方网站'，并访问第一个结果，告诉我页面上是否有 'Download' 相关内容。",
        "level": "2",
        "file_name": ""
    }
    
    async with MAS(oxy_space=oxy_space) as mas:
        result = await run_single_task(mas, test_task)
        logger.info(f"最终结果: {result}")
        
    # 清理浏览器资源
    try:
        from tools.search_toolkit import cleanup_browser
        await cleanup_browser()
        logger.info("浏览器资源已清理")
    except Exception as e:
        logger.warning(f"清理浏览器失败: {e}")


def print_menu():
    """打印菜单"""
    logger.info("=" * 70)
    logger.info("京东多智能体挑战赛 - 主程序")
    logger.info("=" * 70)
    logger.info("选择运行模式:")
    logger.info("  1. 运行单个测试任务（CodeAgent - 毒酒问题）")
    logger.info("  2. 测试 SearchAgent（需要 SERPER_API_KEY）")
    logger.info("  3. 运行验证集（前 3 个任务）")
    logger.info("  4. 运行验证集（前 10 个任务）")
    logger.info("  5. 运行完整验证集（全部任务）")
    logger.info("  6. 启动交互式 Web 服务")
    logger.info("  0. 退出")


async def main():
    """主函数"""
    
    # 环境检查
    logger.info("=" * 70)
    logger.info("环境检查")
    logger.info("=" * 70)
    
    # 检查目录
    local_es_data = project_root / "local_es_data"
    local_es_data.mkdir(exist_ok=True)
    logger.info(f"local_es_data 目录: {local_es_data}")
    
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    logger.info(f"outputs 目录: {outputs_dir}")
    
    # 检查环境变量
    env_vars = ["DEFAULT_LLM_API_KEY", "DEFAULT_LLM_BASE_URL", "DEFAULT_LLM_MODEL_NAME"]
    missing = [v for v in env_vars if not os.getenv(v)]
    
    if missing:
        logger.warning(f"缺少环境变量: {', '.join(missing)}")
        logger.info("提示: 设置环境变量后再运行")
        print("\n按 Enter 键继续...")
        input()
    else:
        logger.info("环境变量已配置")
    
    # 检查 SearchAgent 相关环境变量
    if not os.getenv("SERPER_API_KEY"):
        logger.warning("未设置 SERPER_API_KEY，SearchAgent 的搜索功能将不可用")
        logger.info("如需使用搜索功能，请运行: export SERPER_API_KEY='your_api_key'")
        logger.info("注册地址: https://serper.dev/")
    else:
        logger.info("SERPER_API_KEY 已配置")
    
    # 主循环
    while True:
        print_menu()
        
        try:
            choice = input("请选择 (0-6): ").strip()
        except (KeyboardInterrupt, EOFError):
            logger.info("再见！")
            break
        
        if choice == "0":
            logger.info("再见！")
            break
        elif choice == "1":
            await run_test_task()
        elif choice == "2":
            await run_search_test()
        elif choice == "3":
            await run_validation_set(limit=3)
        elif choice == "4":
            await run_validation_set(limit=10)
        elif choice == "5":
            await run_validation_set()
        elif choice == "6":
            await run_interactive_mode()
        else:
            logger.warning("无效选项，请重新选择")
        
        if choice in ["1", "2", "3", "4", "5"]:
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
    
    logger.info("=" * 70)
    logger.info("京东多智能体挑战赛 - 主程序")
    logger.info("=" * 70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序已停止")
    except Exception as e:
        logger.error(f"程序出错: {e}")
        import traceback
        logger.error(f"异常详情: {traceback.format_exc()}")
