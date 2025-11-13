"""
HTML 处理工具模块

职责：提供 HTML 到 Markdown 转换、页面内容提取等辅助功能
"""

import json
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
from playwright.async_api import Page


def html_to_markdown(html: str) -> str:
    """
    将 HTML 转换为简化的 Markdown 格式
    移除脚本、样式等噪音，提取主要文本内容
    
    关键改进：尽量保持键值对的紧密关系（如 "噪音: 40db"）
    
    Args:
        html: 原始 HTML 字符串
    
    Returns:
        str: 简化后的 Markdown 格式内容
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # 移除不需要的标签
    for tag in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'path']):
        tag.decompose()
    
    # 提取文本
    lines = []
    
    # 标题
    for i in range(1, 7):
        for heading in soup.find_all(f'h{i}'):
            text = heading.get_text(strip=True)
            if text:
                lines.append(f"{'#' * i} {text}\n")
    
    # 段落和文本（保持结构）
    for elem in soup.find_all(['p', 'div', 'li', 'td', 'th', 'span', 'a', 'label']):
        text = elem.get_text(strip=True)
        if text and len(text) > 1:  # 过滤空白和单字符
            lines.append(text)
    
    # 去重但保持顺序
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen and len(line) > 2:
            seen.add(line)
            unique_lines.append(line)
    
    content = "\n".join(unique_lines)
    
    # 限制长度（避免内容过长）
    if len(content) > 50000:
        content = content[:50000] + "\n\n... (内容过长，已截断)"
    
    return content


async def get_clean_page_content(page: Page) -> str:
    """
    获取页面的干净 Markdown 内容
    
    Args:
        page: Playwright 页面实例
    
    Returns:
        str: 干净的 Markdown 格式内容
    """
    html = await page.content()
    content = html_to_markdown(html)
    return content


def save_page_content(task_id: str, url: str, content: str, html: str = ""):
    """
    保存页面内容到 local_es_data/
    
    Args:
        task_id: 任务ID
        url: 页面URL
        content: 简化后的内容（Markdown）
        html: 原始HTML（可选）
    """
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存简化内容
    content_file = output_dir / f"{task_id}_page_content.txt"
    try:
        with open(content_file, "w", encoding="utf-8") as f:
            f.write(f"=== Task ID: {task_id} ===\n\n")
            f.write(f"=== URL ===\n{url}\n\n")
            f.write(f"=== 页面内容 ===\n{content}\n")
    except Exception as e:
        print(f"⚠️ 保存页面内容失败: {e}")
    
    # 保存原始HTML（可选）
    if html:
        html_file = output_dir / f"{task_id}_page_html.html"
        try:
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            print(f"⚠️ 保存HTML失败: {e}")


def save_search_results(task_id: str, query: str, results: List[Dict]):
    """
    保存搜索结果到 local_es_data/
    
    Args:
        task_id: 任务ID
        query: 搜索查询
        results: 搜索结果列表
    """
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"{task_id}_search_results.json"
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "query": query,
                "results": results
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存搜索结果失败: {e}")

