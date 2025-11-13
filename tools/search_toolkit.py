"""
SearchAgent 工具包：基于会话的搜索与浏览 (V5 - 认证版)

职责：提供互联网搜索和网页浏览功能，支持复杂的多步骤网页交互和登录状态。

工具列表：

=== 核心搜索 (Core Search) ===
1. search(query: str, task_id: str) -> List[SearchResult]
   - 执行互联网搜索，使用 Serper API
   - 返回标题、URL、摘要

=== 会话管理 (Session Management) ===
2. open_url(url: str, task_id: str, auth_context: Optional[str] = None) -> str
   - 打开网页并返回简化内容（使用 Playwright）
   - 为 task_id 创建或重置持久浏览器会话
   - 【新功能】支持通过 auth_context 加载登录状态
   - 支持 PDF 文件自动提取

3. close_browser_session(task_id: str) -> str
   - 关闭并清理指定 task_id 的浏览器会话
   - Agent 必须在任务结束时调用此工具释放资源

=== 状态与导航 (State & Navigation) ===
4. get_current_url(task_id: str) -> str
   - 获取浏览器地址栏的当前 URL
   - 用于确认位置和构造完整路径
   
5. go_back(task_id: str) -> str
   - 后退到上一页（模拟浏览器后退按钮）
   - 用于错误恢复和撤销错误点击

6. scroll_page(direction: str, task_id: str) -> str
   - 滚动页面以加载更多内容
   - 触发无限滚动和懒加载

=== 页面交互 (Page Interaction) ===
7. click_element(text_on_element: str, role: str, task_id: str) -> str
   - 点击包含指定文本的元素（支持 link/button 精确区分）
   - 自动处理动态内容加载（AJAX/XHR）
   
8. type_text_in_element(text_to_type: str, element_label_or_placeholder: str, task_id: str) -> str
   - 在输入框中输入文本
   - 支持通过 label 或 placeholder 定位元素

9. press_key(key: str, task_id: str) -> str
   - 模拟按键（如 "Enter" 提交搜索）
   - 自动等待新页面加载

=== 页面检查 (Page Inspection) ===
10. find_in_page(query: str, task_id: str) -> List[str]
    - 在当前页面中搜索文本（模拟 Ctrl+F）
    - 返回匹配的文本片段及上下文

11. list_interactive_elements(task_id: str) -> List[Dict]
    - 列出所有可交互元素（链接和按钮）的结构化信息
    - 返回 text、role、info 字段

=== 特殊功能 (Special Features) ===
12. get_image_url(alt_text_query: str, task_id: str) -> str
    - 根据 alt_text 查找图片并返回 URL
    - SearchAgent 定位图片，MultimodalAgent 分析图片

13. query_pdf_url(url: str, query: str, task_id: str) -> str
    - 【新】使用 RAG 技术查询 PDF 文档
    - 只返回最相关的 3 个文本片段（避免上下文窗口爆炸）
    - 必须用此工具代替 open_url 处理 PDF
    - 适用于 ESG 报告、研究论文、产品手册等大型文档
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional
from pydantic import Field
from oxygent.oxy import FunctionHub
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
import requests

# 导入模块化组件
from tools.search_toolkit_sub.browser_manager import session_manager
from tools.search_toolkit_sub.html_utils import (
    get_clean_page_content,
    save_page_content,
    save_search_results
)

# 注册搜索工具包
search_tools = FunctionHub(name="search_tools")

# 【全局】预加载嵌入模型（避免每次调用 query_pdf_url 时重新加载）
_embedder_model = None

# 【全局】PDF 嵌入缓存（核心性能优化）
# 结构: {url: {"embeddings": tensor, "pages": list, "timestamp": float}}
_pdf_embedding_cache = {}

# 【配置】PDF 处理护栏
MAX_PAGES_TO_PROCESS = 150  # 最大页数限制，防止处理超大 PDF

def _get_embedder():
    """
    延迟加载并缓存嵌入模型
    【性能优化】自动检测并使用 GPU（如果可用）
    """
    global _embedder_model
    if _embedder_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # 检测 GPU 可用性
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🔄 首次加载嵌入模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
            print(f"🖥️  使用设备: {device.upper()}" + (" 🚀 [GPU加速]" if device == 'cuda' else " [CPU模式]"))
            
            _embedder_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
            print("✅ 嵌入模型加载完成")
        except ImportError:
            raise ImportError(
                "缺少 'sentence-transformers' 库。请安装: pip install sentence-transformers"
            )
    return _embedder_model


# ==================== A. 外部搜索工具 (Stateless Function) ====================

@search_tools.tool(
    description="Search the internet for information using a search engine. "
    "Returns a list of search results with titles, URLs, and snippets. "
    "Use this tool when you need to find information about a topic, "
    "locate specific websites, or gather general knowledge."
)
async def search(
    query: str = Field(
        description="The search query string (e.g., '京东大事记', 'Python tutorial')"
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving results."
    ),
) -> str:
    """
    执行互联网搜索（使用 Serper API）
    
    Args:
        query: 搜索查询字符串
        task_id: 任务唯一标识符
    
    Returns:
        str: JSON 格式的搜索结果列表
    """
    print(f"🔍 搜索: {query}")
    
    # 检查 Serper API Key
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        error_msg = (
            "⚠️ Serper API 未配置\n\n"
            "请设置环境变量:\n"
            "export SERPER_API_KEY='your_api_key'\n\n"
            "注册地址: https://serper.dev/"
        )
        return error_msg
    
    try:
        # 调用 Serper API
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "q": query,
            "num": 10,  # 返回 10 条结果
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # 解析结果
        search_results = []
        
        # 有机搜索结果
        for item in result.get("organic", []):
            search_results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        
        # 保存结果
        save_search_results(task_id, query, search_results)
        
        print(f"✅ 搜索完成，找到 {len(search_results)} 条结果")
        
        # 返回格式化的结果
        return json.dumps(search_results, ensure_ascii=False, indent=2)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ 搜索失败: {str(e)}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ 搜索出错: {str(e)}"
        print(error_msg)
        return error_msg


# ==================== B. 会话管理 (Session Management) ====================

@search_tools.tool(
    description="Open a web page and return its simplified content. "
    "Creates or resets a persistent browser session for this task_id. "
    "Uses a headless browser (Playwright) to handle JavaScript-rendered pages. "
    "【NEW】Optional auth_context parameter (e.g., 'jd.com') automatically loads matching login state. "
    "【IMPORTANT】If the URL is a PDF, this tool will guide you to use 'query_pdf_url' instead (to avoid context window explosion). "
    "Returns clean, readable text content (converted to Markdown) suitable for LLM processing. "
    "Essential for accessing dynamic websites and authenticated pages."
)
async def open_url(
    url: str = Field(
        description="The URL to open (e.g., 'https://item.jd.com/7307091.html' or 'https://example.com/document.pdf')"
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Creates a persistent browser session for this task_id."
    ),
    auth_context: Optional[str] = Field(
        default=None,
        description="Optional authentication context key (e.g., 'jd.com', 'zhihu.com'). "
        "If provided, loads the corresponding login state (cookies + localStorage). "
        "If omitted or the file doesn't exist, opens as a guest. "
        "Use this when accessing pages that require login (e.g., 'My Orders', 'Shopping Cart')."
    ),
) -> str:
    """
    打开网页或 PDF 并返回简化内容
    
    【架构核心】支持通过 auth_context 加载登录状态
    
    Args:
        url: 要打开的URL（支持网页和PDF）
        task_id: 任务唯一标识符
        auth_context: 认证上下文密钥（例如 "jd.com"）
    
    Returns:
        str: 简化后的页面内容（Markdown格式）或 PDF 文本
    """
    print(f"🌐 打开URL: {url}")
    if auth_context:
        print(f"🔐 使用认证上下文: {auth_context}")
    
    # 检测是否是 PDF 链接
    is_pdf = url.lower().endswith('.pdf') or '.pdf?' in url.lower()
    
    # 【关键改进】如果是 PDF，引导 Agent 使用 query_pdf_url 工具
    if is_pdf:
        return (
            "⚠️ 检测到 PDF 文件。\n\n"
            "为了避免上下文窗口爆炸，请使用 'query_pdf_url' 工具来查询此 PDF，"
            "而不是使用 'open_url'。\n\n"
            f"建议操作：\n"
            f"- 工具: query_pdf_url\n"
            f"- url: {url}\n"
            f"- query: [将您的原始问题作为查询参数]\n"
            f"- task_id: {task_id}\n\n"
            f"示例：如果您的任务是'找到京东健康的ESG政策数量'，则 query 参数应为 "
            f"'京东健康中提到的ESG政策总共有几个？'"
        )
    
    # 否则使用 Playwright 处理普通网页
    try:
        # 【核心升级】创建新会话，支持 auth_context
        page = await session_manager.create_session(task_id, url, auth_context)
        
        # 获取页面内容
        content = await get_clean_page_content(page)
        html = await page.content()
        
        # 更新会话缓存
        session_manager.update_content(task_id, content, html)
        
        # 保存内容
        save_page_content(task_id, url, content, html)
        
        print(f"✅ 页面已打开，内容长度: {len(content)} 字符")
        
        return content
        
    except PlaywrightTimeoutError:
        error_msg = f"❌ 打开页面超时: {url}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ 打开页面失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


@search_tools.tool(
    description="Close and cleanup the browser session for this task_id to release resources. "
    "CRITICAL: Agent MUST call this tool at the end of every task before returning the final answer. "
    "This ensures proper resource cleanup and prevents memory leaks."
)
async def close_browser_session(
    task_id: str = Field(
        description="Unique identifier for this task. The browser session associated with this task_id will be closed."
    ),
) -> str:
    """
    关闭并清理浏览器会话
    
    Args:
        task_id: 任务唯一标识符
    
    Returns:
        str: 确认消息
    """
    print(f"🔒 关闭会话: {task_id[:8]}...")
    return await session_manager.close_session(task_id)


# ==================== C. 状态与导航 (State & Navigation) ====================

@search_tools.tool(
    description="Get the current URL from the browser's address bar. "
    "Essential for confirming your location after navigation or redirects, "
    "and for constructing full URLs by appending file names to the current path. "
    "Use this when you need to know exactly where you are in the website."
)
async def get_current_url(
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    获取当前浏览器的 URL
    
    Args:
        task_id: 任务唯一标识符
    
    Returns:
        str: 当前页面的完整 URL
    """
    print(f"📍 获取当前URL")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        current_url = page.url
        print(f"✅ 当前URL: {current_url}")
        
        return current_url
        
    except Exception as e:
        error_msg = f"❌ 获取URL失败: {str(e)}"
        print(error_msg)
        return error_msg


@search_tools.tool(
    description="Go back to the previous page (simulates browser's back button). "
    "Critical for error recovery when you click a wrong link or enter a dead end. "
    "Returns the content of the previous page. "
    "Use this instead of calling open_url again when you need to undo navigation."
)
async def go_back(
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    后退到上一页
    
    Args:
        task_id: 任务唯一标识符
    
    Returns:
        str: 后退后页面的简化内容
    """
    print(f"⬅️ 后退到上一页")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        # 执行后退操作
        await page.go_back(wait_until="networkidle", timeout=30000)
        
        # 等待页面加载
        await asyncio.sleep(2)
        
        # 获取页面内容
        content = await get_clean_page_content(page)
        html = await page.content()
        
        # 更新会话缓存
        session_manager.update_content(task_id, content, html)
        
        # 保存内容
        current_url = page.url
        save_page_content(task_id, current_url, content, html)
        
        print(f"✅ 后退成功，当前URL: {current_url}")
        print(f"✅ 页面内容长度: {len(content)} 字符")
        
        return content
        
    except Exception as e:
        error_msg = f"❌ 后退失败: {str(e)}"
        print(error_msg)
        return error_msg


@search_tools.tool(
    description="Scroll the current page up or down to load more content. "
    "Essential for infinite-scroll pages, lazy-loaded content, or long pages. "
    "Use this when you need to load older comments, more news items, or additional GitHub issues. "
    "Waits 3 seconds after scrolling for JS rendering. "
    "Returns the updated page content after scrolling."
)
async def scroll_page(
    direction: str = Field(
        default="down",
        description="Direction to scroll: 'down' or 'up'"
    ),
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    滚动页面
    
    Args:
        direction: 滚动方向 ('down' 或 'up')
        task_id: 任务唯一标识符
    
    Returns:
        str: 滚动后的页面内容
    """
    print(f"📜 滚动页面: {direction}")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        # 执行滚动
        if direction.lower() == "down":
            # 滚动到底部
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            # 滚动到顶部
            await page.evaluate("window.scrollTo(0, 0)")
        
        # 【关键】等待新内容加载（为 JS 渲染提供时间）
        await asyncio.sleep(3)
        
        # 获取更新后的内容
        content = await get_clean_page_content(page)
        html = await page.content()
        
        # 更新会话缓存
        session_manager.update_content(task_id, content, html)
        
        # 保存内容
        current_url = page.url
        save_page_content(task_id, current_url, content, html)
        
        print(f"✅ 滚动成功，新内容长度: {len(content)} 字符")
        
        return content
        
    except Exception as e:
        error_msg = f"❌ 滚动失败: {str(e)}"
        print(error_msg)
        return error_msg


# ==================== D. 页面交互 (Page Interaction) ====================

@search_tools.tool(
    description="Click an element (link or button) on the current page by its text content AND role. "
    "This tool has PRECISE VISION - it can distinguish between elements with the same text but different roles. "
    "For example, it can click the 'main' link (folder) instead of the 'main' button (branch selector). "
    "CRITICAL FIX: Automatically handles dynamic content (AJAX/XHR) by waiting for network idle or 3 seconds. "
    "Use the exact 'text' and 'role' values from list_interactive_elements output. "
    "Role must be either 'link' or 'button'. Returns the new page content after clicking."
)
async def click_element(
    text_on_element: str = Field(
        description="The visible text on the element to click (e.g., '商品评论', 'main', 'Files and versions')"
    ),
    role: str = Field(
        description="The role of the element: 'link' or 'button'. Get this from list_interactive_elements output."
    ),
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    通过文本和角色精确点击页面元素
    
    Args:
        text_on_element: 元素上的文本
        role: 元素的角色（'link' 或 'button'）
        task_id: 任务唯一标识符
    
    Returns:
        str: 点击后的新页面内容
    """
    print(f"👆 点击元素: '{text_on_element}' (role={role})")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        # 验证 role 参数
        if role not in ['link', 'button']:
            return f"❌ 错误: role 必须是 'link' 或 'button'，但得到了 '{role}'"
        
        clicked = False
        
        # 策略1: 使用 Playwright 的 get_by_role (最精确)
        try:
            element = page.get_by_role(role, name=text_on_element, exact=True)
            await element.click(timeout=5000)
            clicked = True
            print(f"✅ 使用精确 role+name 匹配成功")
        except Exception as e:
            print(f"⚠️ 精确 role+name 匹配失败: {e}")
        
        # 策略2: 使用 get_by_role 但不要求精确匹配
        if not clicked:
            try:
                element = page.get_by_role(role, name=text_on_element)
                await element.first.click(timeout=5000)
                clicked = True
                print(f"✅ 使用模糊 role+name 匹配成功")
            except Exception as e:
                print(f"⚠️ 模糊 role+name 匹配失败: {e}")
        
        # 策略3: 使用 CSS 选择器 + 文本匹配
        if not clicked:
            try:
                if role == 'link':
                    # 查找所有 <a> 标签
                    links = await page.query_selector_all("a")
                    for link in links:
                        link_text = await link.inner_text()
                        if link_text.strip() == text_on_element:
                            await link.click(timeout=5000)
                            clicked = True
                            print(f"✅ 使用 <a> 标签匹配成功")
                            break
                elif role == 'button':
                    # 查找所有 <button> 标签和 role="button" 的元素
                    buttons = await page.query_selector_all("button, [role='button']")
                    for button in buttons:
                        button_text = await button.inner_text()
                        if button_text.strip() == text_on_element:
                            await button.click(timeout=5000)
                            clicked = True
                            print(f"✅ 使用 button 标签匹配成功")
                            break
            except Exception as e:
                print(f"⚠️ CSS 选择器匹配失败: {e}")
        
        # 策略4: 回退到旧的通用策略（仅当前面都失败时）
        if not clicked:
            try:
                print(f"⚠️ 尝试回退到通用文本匹配")
                element = page.get_by_text(text_on_element, exact=True)
                await element.click(timeout=5000)
                clicked = True
                print(f"✅ 使用通用文本匹配成功（但可能点击了错误的元素）")
            except:
                pass
        
        if not clicked:
            return f"❌ 未找到匹配的元素: text='{text_on_element}', role='{role}'\n建议：先调用 list_interactive_elements 确认元素存在"
        
        # 【关键修复】智能等待：处理动态内容加载
        try:
            # 尝试等待页面导航（如果是跳转链接）
            await page.wait_for_load_state('networkidle', timeout=5000)
            print("✅ 检测到页面导航")
        except PlaywrightTimeoutError:
            # 超时说明是动态内容（AJAX/XHR），固定等待 3 秒
            print("⚠️ 未检测到页面导航，可能是动态内容，等待 3 秒...")
            await asyncio.sleep(3)
        
        # 获取新页面内容
        content = await get_clean_page_content(page)
        html = await page.content()
        
        # 更新会话缓存
        session_manager.update_content(task_id, content, html)
        
        # 保存内容
        current_url = page.url
        save_page_content(task_id, current_url, content, html)
        
        print(f"✅ 点击成功，新页面内容长度: {len(content)} 字符")
        
        return content
        
    except Exception as e:
        error_msg = f"❌ 点击失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


@search_tools.tool(
    description="Type text into an input field on the current page. "
    "Identifies the input field by its label or placeholder text. "
    "Essential for using search boxes, filling forms, or entering queries. "
    "After typing, you typically need to call press_key('Enter') to submit."
)
async def type_text_in_element(
    text_to_type: str = Field(
        description="The text to type into the input field (e.g., 'numpy', 'gsm8k')"
    ),
    element_label_or_placeholder: str = Field(
        description="The label or placeholder text of the input field (e.g., 'Search issues', 'Username')"
    ),
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    在输入框中输入文本
    
    Args:
        text_to_type: 要输入的文本
        element_label_or_placeholder: 输入框的 label 或 placeholder
        task_id: 任务唯一标识符
    
    Returns:
        str: 当前页面的内容（输入通常不刷新页面）
    """
    print(f"⌨️ 输入文本: '{text_to_type}' 到 '{element_label_or_placeholder}'")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        target = None
        
        # 策略1: 通过 placeholder 定位
        try:
            target = page.get_by_placeholder(element_label_or_placeholder).first
            await target.fill(text_to_type, timeout=5000)
            print(f"✅ 使用 placeholder 定位成功")
        except Exception as e:
            print(f"⚠️ placeholder 定位失败: {e}")
        
        # 策略2: 通过 label 定位
        if not target:
            try:
                target = page.get_by_label(element_label_or_placeholder).first
                await target.fill(text_to_type, timeout=5000)
                print(f"✅ 使用 label 定位成功")
            except Exception as e:
                print(f"⚠️ label 定位失败: {e}")
                return f"❌ 未找到匹配的输入框: label/placeholder='{element_label_or_placeholder}'"
        
        # 等待一下（某些网站会有输入延迟）
        await asyncio.sleep(1)
        
        # 获取当前页面内容
        content = await get_clean_page_content(page)
        html = await page.content()
        
        # 更新会话缓存
        session_manager.update_content(task_id, content, html)
        
        print(f"✅ 输入成功")
        
        return content
        
    except Exception as e:
        error_msg = f"❌ 输入失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


@search_tools.tool(
    description="Press a keyboard key (e.g., 'Enter') on the current page. "
    "Essential for submitting search queries after typing text. "
    "Automatically waits for new page to load after pressing Enter. "
    "Returns the new page content after key press."
)
async def press_key(
    key: str = Field(
        description="The key to press (e.g., 'Enter', 'Tab', 'Escape')"
    ),
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    模拟按键
    
    Args:
        key: 要按的键（如 'Enter', 'Tab', 'Escape'）
        task_id: 任务唯一标识符
    
    Returns:
        str: 按键后的新页面内容
    """
    print(f"⌨️ 按键: {key}")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        # 按键
        await page.keyboard.press(key)
        
        # 【关键】按 Enter 通常会触发导航，必须等待
        if key.lower() == "enter":
            try:
                await page.wait_for_load_state('networkidle', timeout=15000)
                print("✅ 检测到页面导航")
            except PlaywrightTimeoutError:
                # 超时则固定等待
                print("⚠️ 等待超时，固定等待 3 秒...")
                await asyncio.sleep(3)
        else:
            # 其他按键等待短暂时间
            await asyncio.sleep(1)
        
        # 获取新页面内容
        content = await get_clean_page_content(page)
        html = await page.content()
        
        # 更新会话缓存
        session_manager.update_content(task_id, content, html)
        
        # 保存内容
        current_url = page.url
        save_page_content(task_id, current_url, content, html)
        
        print(f"✅ 按键成功，新页面内容长度: {len(content)} 字符")
        
        return content
        
    except Exception as e:
        error_msg = f"❌ 按键失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


# ==================== E. 页面检查 (Page Inspection) ====================

@search_tools.tool(
    description="Search for specific text in the currently opened page. "
    "Simulates browser's Ctrl+F functionality. Returns matching text snippets with context. "
    "Searches in the clean Markdown content, which preserves key-value relationships (e.g., '噪音: 40db'). "
    "Use this tool to find specific information on the current page after opening it with open_url."
)
async def find_in_page(
    query: str = Field(
        description="The text to search for in the current page (e.g., '商品评论', '2023-03-26', 'numpy')"
    ),
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    在当前页面中搜索文本
    
    Args:
        query: 要搜索的文本
        task_id: 任务唯一标识符
    
    Returns:
        str: JSON格式的匹配结果列表
    """
    print(f"🔍 在当前页面搜索: {query}")
    
    content = session_manager.get_content(task_id)
    
    if not content:
        return f"❌ 错误: 会话 {task_id[:8]}... 没有内容。请先使用 open_url 打开一个页面。"
    
    try:
        # 在内容中搜索
        lines = content.split('\n')
        matches = []
        
        for i, line in enumerate(lines):
            if query.lower() in line.lower():
                # 获取上下文（前后各2行）
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context_lines = lines[start:end]
                
                matches.append({
                    "line_number": i + 1,
                    "matched_text": line.strip(),
                    "context": "\n".join(context_lines)
                })
        
        print(f"✅ 找到 {len(matches)} 个匹配")
        
        if not matches:
            return f"未找到匹配 '{query}' 的内容"
        
        # 限制返回数量（避免过长）
        if len(matches) > 20:
            matches = matches[:20]
            result = json.dumps(matches, ensure_ascii=False, indent=2)
            result += f"\n\n(结果已截断，仅显示前20个匹配)"
            return result
        
        return json.dumps(matches, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = f"❌ 搜索失败: {str(e)}"
        print(error_msg)
        return error_msg


@search_tools.tool(
    description="List all interactive elements (links and buttons) on the current page with structured information. "
    "This is your 'eyes' with COLOR VISION - it can distinguish between links and buttons with the same text. "
    "Returns a list of objects with 'text', 'role', and 'info' fields. "
    "Use this tool FIRST before clicking to see what's actually available and identify the correct element. "
    "Essential for eliminating blind clicking and avoiding confusion between elements with identical text."
)
async def list_interactive_elements(
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    列出当前页面上所有可交互元素的结构化信息
    
    Args:
        task_id: 任务唯一标识符
    
    Returns:
        str: JSON格式的可交互元素列表，每个元素包含 text、role、info
    """
    print(f"👀 列出所有可交互元素（结构化）")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        # 获取所有链接和按钮（结构化）
        interactive_elements = []
        
        # 策略1: 获取所有链接 (a 标签)
        links = await page.query_selector_all("a")
        for link in links:
            try:
                text = await link.inner_text()
                text = text.strip()
                if not text or len(text) == 0 or len(text) > 200:
                    continue
                
                # 获取额外信息
                href = await link.get_attribute("href") or ""
                title = await link.get_attribute("title") or ""
                aria_label = await link.get_attribute("aria-label") or ""
                
                # 推断元素的用途
                info = ""
                if "folder" in href.lower() or "tree" in href.lower():
                    info = "Folder or directory"
                elif "file" in href.lower() or any(ext in href.lower() for ext in ['.parquet', '.json', '.csv', '.txt']):
                    info = "File"
                elif title:
                    info = title
                elif aria_label:
                    info = aria_label
                
                interactive_elements.append({
                    "text": text,
                    "role": "link",
                    "info": info
                })
            except:
                continue
        
        # 策略2: 获取所有按钮 (button 标签)
        buttons = await page.query_selector_all("button")
        for button in buttons:
            try:
                text = await button.inner_text()
                text = text.strip()
                if not text or len(text) == 0 or len(text) > 200:
                    continue
                
                # 获取额外信息
                title = await button.get_attribute("title") or ""
                aria_label = await button.get_attribute("aria-label") or ""
                button_type = await button.get_attribute("type") or ""
                
                # 推断元素的用途
                info = ""
                if "branch" in aria_label.lower() or "selector" in aria_label.lower():
                    info = "Branch selector"
                elif title:
                    info = title
                elif aria_label:
                    info = aria_label
                elif button_type:
                    info = f"Button ({button_type})"
                
                interactive_elements.append({
                    "text": text,
                    "role": "button",
                    "info": info
                })
            except:
                continue
        
        # 策略3: 获取具有 role="button" 的元素
        role_buttons = await page.query_selector_all("[role='button']")
        for btn in role_buttons:
            try:
                # 跳过已经是 button 标签的元素（避免重复）
                tag_name = await btn.evaluate("element => element.tagName")
                if tag_name.lower() == "button":
                    continue
                
                text = await btn.inner_text()
                text = text.strip()
                if not text or len(text) == 0 or len(text) > 200:
                    continue
                
                aria_label = await btn.get_attribute("aria-label") or ""
                title = await btn.get_attribute("title") or ""
                
                info = aria_label or title or "Interactive element"
                
                interactive_elements.append({
                    "text": text,
                    "role": "button",
                    "info": info
                })
            except:
                continue
        
        print(f"✅ 找到 {len(interactive_elements)} 个可交互元素")
        
        # 返回结构化列表（不去重，因为同名元素的 role 可能不同）
        if not interactive_elements:
            return "⚠️ 当前页面没有找到可交互元素"
        
        # 限制返回数量（避免过长）
        if len(interactive_elements) > 100:
            interactive_elements = interactive_elements[:100]
            result = json.dumps(interactive_elements, ensure_ascii=False, indent=2)
            result += "\n\n(列表已截断，仅显示前100个元素)"
            return result
        
        return json.dumps(interactive_elements, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = f"❌ 列出元素失败: {str(e)}"
        print(error_msg)
        return error_msg


@search_tools.tool(
    description="Find an image on the current page by its alt text or nearby text, and return its source URL. "
    "SearchAgent doesn't analyze images, but it can locate and extract image URLs. "
    "The URL can then be passed to multimodal_agent for image analysis. "
    "Use this when you need to find a specific image based on its description or context."
)
async def get_image_url(
    alt_text_query: str = Field(
        description="The alt text or nearby text to identify the image (e.g., '产品图片', 'logo')"
    ),
    task_id: str = Field(
        description="Unique identifier for this task."
    ),
) -> str:
    """
    根据 alt_text 查找图片并返回 URL
    
    Args:
        alt_text_query: 图片的 alt 文本或相邻文本
        task_id: 任务唯一标识符
    
    Returns:
        str: 图片URL
    """
    print(f"🖼️ 查找图片: {alt_text_query}")
    
    try:
        page = await session_manager.get_session(task_id)
        
        if not page:
            return f"❌ 错误: 会话 {task_id[:8]}... 不存在。请先使用 open_url 创建会话。"
        
        # 查找图片
        images = []
        
        # 策略1: 通过 alt 属性查找
        try:
            img_elements = await page.query_selector_all(f"img[alt*='{alt_text_query}']")
            for img in img_elements:
                src = await img.get_attribute("src")
                if src:
                    images.append(src)
        except:
            pass
        
        # 策略2: 查找所有图片，通过周围文本判断
        if not images:
            try:
                all_imgs = await page.query_selector_all("img")
                for img in all_imgs[:20]:  # 限制数量
                    alt = await img.get_attribute("alt") or ""
                    title = await img.get_attribute("title") or ""
                    src = await img.get_attribute("src") or ""
                    
                    if alt_text_query.lower() in alt.lower() or alt_text_query.lower() in title.lower():
                        if src:
                            images.append(src)
            except:
                pass
        
        if not images:
            return f"❌ 未找到匹配 '{alt_text_query}' 的图片"
        
        # 返回第一个匹配的图片URL
        image_url = images[0]
        
        # 处理相对URL
        if image_url.startswith("//"):
            image_url = "https:" + image_url
        elif image_url.startswith("/"):
            current_url = page.url
            from urllib.parse import urlparse
            parsed = urlparse(current_url)
            image_url = f"{parsed.scheme}://{parsed.netloc}{image_url}"
        
        print(f"✅ 找到图片: {image_url}")
        
        # 保存结果
        output_dir = Path("local_es_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"{task_id}_image_url.txt"
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(f"=== Task ID: {task_id} ===\n\n")
                f.write(f"=== 查询 ===\n{alt_text_query}\n\n")
                f.write(f"=== 图片URL ===\n{image_url}\n")
        except Exception as e:
            print(f"⚠️ 保存结果失败: {e}")
        
        return image_url
        
    except Exception as e:
        error_msg = f"❌ 查找图片失败: {str(e)}"
        print(error_msg)
        return error_msg


@search_tools.tool(
    description="Download a PDF from a URL, process it, and answer a specific question based on its content using semantic search. "
    "This is a RAG (Retrieval-Augmented Generation) tool that extracts ONLY the relevant snippets from the PDF, not the full text. "
    "CRITICAL: Use this instead of 'open_url' when you find a PDF document (e.g., ESG reports, research papers, manuals). "
    "Returns the top 3 most relevant text snippets with page numbers and similarity scores. "
    "Essential for efficiently extracting specific information from large documents without context window explosion."
)
async def query_pdf_url(
    url: str = Field(
        description="The URL of the PDF document (e.g., 'https://example.com/report-2024.pdf')"
    ),
    query: str = Field(
        description="The specific question to answer based on the PDF content. "
        "Pass your ORIGINAL TASK QUESTION here (e.g., 'How many ESG policies does JD Health mention?', "
        "'What is the noise level of the product?'). "
        "The tool will find the most relevant pages automatically."
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving results."
    ),
) -> str:
    """
    从 PDF URL 下载、处理并查询内容（轻量级 RAG - Retrieval-Augmented Generation）
    
    【核心优势】
    - 只返回相关片段（top 3），而不是全量文本
    - 避免上下文窗口爆炸
    - 降低 Token 成本
    - 提高准确性（减少噪声）
    
    【性能优化】（V2 版本）
    - ✅ 添加护栏：最大 150 页限制
    - ✅ 智能缓存：同一 PDF 的后续查询为毫秒级
    - ✅ GPU 加速：自动使用 GPU（如果可用）
    
    Args:
        url: PDF 文件的 URL
        query: 要查询的具体问题（使用原始任务问题）
        task_id: 任务唯一标识符
    
    Returns:
        str: 前 3 个最相关的文本片段（包含页码和相似度分数）
    """
    print(f"📄 正在查询 PDF: {url}")
    print(f"🔍 查询问题: {query}")
    
    try:
        # ===== 【性能优化 1】检查缓存 =====
        global _pdf_embedding_cache
        
        if url in _pdf_embedding_cache:
            print("⚡️ 缓存命中！从缓存加载嵌入（跳过下载、提取、嵌入步骤）...")
            corpus_embeddings = _pdf_embedding_cache[url]["embeddings"]
            extracted_pages = _pdf_embedding_cache[url]["pages"]
            print(f"✅ 从缓存加载了 {len(extracted_pages)} 页的嵌入向量")
        else:
            print("🐌 缓存未命中，执行完整 PDF 处理...")
            
            # ===== 步骤 1: 下载 PDF =====
            print(f"📥 下载 PDF...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 保存 PDF 文件（可选）
            output_dir = Path("local_es_data")
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf_file = output_dir / f"{task_id}_queried.pdf"
            
            with open(pdf_file, "wb") as f:
                f.write(response.content)
            
            print(f"✅ PDF 已下载到: {pdf_file}")
            
            # ===== 步骤 2: 文本提取（按页） =====
            print(f"📖 提取 PDF 文本...")
            
            try:
                import pdfplumber
                from io import BytesIO
            except ImportError:
                error_msg = (
                    "❌ 缺少 'pdfplumber' 库\n"
                    "请安装: pip install pdfplumber"
                )
                print(error_msg)
                return error_msg
            
            extracted_pages = []
            
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                # ===== 【护栏 1】检查页数限制 =====
                total_pages = len(pdf.pages)
                
                if total_pages > MAX_PAGES_TO_PROCESS:
                    error_msg = (
                        f"⚠️ PDF 处理中止：文件有 {total_pages} 页，"
                        f"超过了 {MAX_PAGES_TO_PROCESS} 页的限制。\n\n"
                        f"建议：\n"
                        f"1. 尝试更精确的搜索，找到更小的文档\n"
                        f"2. 如果必须处理此文档，请联系管理员提高页数限制\n"
                        f"3. 或者下载文档并使用本地工具处理"
                    )
                    print(error_msg)
                    return error_msg
                
                print(f"📄 PDF 总页数: {total_pages} (在限制 {MAX_PAGES_TO_PROCESS} 页以内)")
                
                # 提取文本
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_pages.append({
                            "page": i,
                            "content": text.strip()
                        })
            
            if not extracted_pages:
                return "⚠️ PDF 文本提取为空，可能是扫描版或图片 PDF。建议使用 multimodal_agent 处理此文档。"
            
            print(f"✅ 提取了 {len(extracted_pages)} 页文本")
            
            # ===== 步骤 3: 向量嵌入（核心 RAG 逻辑） =====
            print(f"🔍 生成向量嵌入...")
            
            try:
                from sentence_transformers import util
                import time
                
                # 获取预加载的嵌入模型（支持 GPU 加速）
                embedder = _get_embedder()
                
                # a. 嵌入 PDF 页面内容（这是最耗时的步骤）
                start_time = time.time()
                corpus_embeddings = embedder.encode(
                    [page['content'] for page in extracted_pages], 
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embedding_time = time.time() - start_time
                
                print(f"✅ 嵌入完成，耗时 {embedding_time:.2f} 秒 ({len(extracted_pages)} 页)")
                
                # ===== 【性能优化 2】存储到缓存 =====
                _pdf_embedding_cache[url] = {
                    "embeddings": corpus_embeddings,
                    "pages": extracted_pages,
                    "timestamp": time.time()
                }
                print(f"💾 已将嵌入缓存到内存（缓存大小: {len(_pdf_embedding_cache)} 个 PDF）")
                
            except ImportError:
                error_msg = (
                    "❌ 缺少 'sentence-transformers' 库\n"
                    "请安装: pip install sentence-transformers"
                )
                print(error_msg)
                return error_msg
        
        # ===== 步骤 4: 执行语义搜索（每次查询都需要） =====
        print(f"🔎 执行语义搜索...")
        
        try:
            from sentence_transformers import util
            
            # 获取嵌入模型
            embedder = _get_embedder()
            
            # b. 嵌入查询（快速操作）
            query_embedding = embedder.encode(
                query, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # c. 执行相似度搜索（返回 top 3）
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
            
        except ImportError:
            error_msg = (
                "❌ 缺少 'sentence-transformers' 库\n"
                "请安装: pip install sentence-transformers"
            )
            print(error_msg)
            return error_msg
        
        # ===== 步骤 5: 组装并返回相关片段 =====
        relevant_snippets = []
        
        for hit in hits[0]:  # hits[0] 是第一个查询的结果
            page_data = extracted_pages[hit['corpus_id']]
            relevant_snippets.append(
                f"--- (相关片段 - 来自第 {page_data['page']} 页, 相似度: {hit['score']:.2f}) ---\n"
                f"{page_data['content']}\n"
            )
        
        final_response = (
            f"根据PDF文档 '{url}' 对查询 '{query}' 的分析，找到以下高相关性片段：\n\n"
            + "\n".join(relevant_snippets)
        )
        
        # 保存查询结果
        output_dir = Path("local_es_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"{task_id}_pdf_query_result.txt"
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(f"=== Task ID: {task_id} ===\n\n")
                f.write(f"=== PDF URL ===\n{url}\n\n")
                f.write(f"=== 查询 ===\n{query}\n\n")
                f.write(f"=== 结果 ===\n{final_response}\n")
        except Exception as e:
            print(f"⚠️ 保存结果失败: {e}")
        
        print(f"✅ PDF 查询完成，返回 {len(hits[0])} 个相关片段。")
        
        return final_response
        
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ 下载 PDF 失败: {str(e)}"
        print(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"❌ 处理 PDF 查询时出错: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


# ==================== F. 缓存管理工具 (Cache Management Utilities) ====================

def get_pdf_cache_stats() -> dict:
    """
    获取 PDF 嵌入缓存的统计信息
    
    Returns:
        dict: 包含缓存大小、URL列表等信息的字典
    """
    global _pdf_embedding_cache
    
    return {
        "cache_size": len(_pdf_embedding_cache),
        "cached_urls": list(_pdf_embedding_cache.keys()),
        "total_pages_cached": sum(
            len(cache_data["pages"]) 
            for cache_data in _pdf_embedding_cache.values()
        )
    }


def clear_pdf_cache(url: Optional[str] = None) -> str:
    """
    清除 PDF 嵌入缓存
    
    Args:
        url: 可选，指定要清除的 PDF URL。如果为 None，则清除所有缓存
    
    Returns:
        str: 操作结果消息
    """
    global _pdf_embedding_cache
    
    if url is None:
        # 清除所有缓存
        cache_size = len(_pdf_embedding_cache)
        _pdf_embedding_cache.clear()
        return f"✅ 已清除所有 PDF 缓存（共 {cache_size} 个）"
    else:
        # 清除指定 URL 的缓存
        if url in _pdf_embedding_cache:
            del _pdf_embedding_cache[url]
            return f"✅ 已清除 PDF 缓存: {url}"
        else:
            return f"⚠️ 缓存中不存在此 URL: {url}"
