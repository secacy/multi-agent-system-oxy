"""
浏览器会话管理器模块

职责：管理持久化的浏览器会话，支持认证状态加载
"""

import os
import asyncio
import threading
from typing import Dict, Optional
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError


class BrowserSessionManager:
    """
    全局浏览器会话管理器（V5 - 认证版）
    
    职责：
    1. 维护一个线程安全的字典 Dict[task_id, SessionState]
    2. 根据 task_id 创建、检索和销毁 Playwright 浏览器会话
    3. 【新功能】支持通过 auth_context 加载预先保存的登录状态
    4. 每个会话包含独立的 BrowserContext 和 Page 实例
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self._playwright = None
        self._browser: Optional[Browser] = None
        
        # 【核心升级】认证文件映射表
        self.auth_files_map = {
            "jd.com": "auth/jd_auth_state.json",
            "zhihu.com": "auth/zhihu_auth_state.json",
            # 可以在此预先填充所有可用的登录状态
        }
    
    async def _ensure_playwright(self):
        """确保 Playwright 已初始化"""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        return self._playwright
    
    async def _ensure_browser(self):
        """确保浏览器已启动"""
        if self._browser is None:
            playwright = await self._ensure_playwright()
            self._browser = await playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
        return self._browser
    
    async def create_session(
        self, 
        task_id: str, 
        url: str, 
        auth_context: Optional[str] = None
    ) -> Page:
        """
        创建或重置指定 task_id 的浏览器会话
        
        【架构核心】支持通过 auth_context 加载登录状态
        
        Args:
            task_id: 任务唯一标识符
            url: 要打开的 URL
            auth_context: 认证上下文密钥（例如 "jd.com"），用于加载对应的登录状态
        
        Returns:
            Page: Playwright 页面实例
        """
        with self.lock:
            # 如果会话已存在，先关闭旧的
            if task_id in self.sessions:
                print(f"⚠️ 会话 {task_id[:8]}... 已存在，将重置")
        
        # 关闭旧会话（如果存在）
        await self.close_session(task_id)
        
        # 确保浏览器已启动
        browser = await self._ensure_browser()
        
        # 【关键升级】创建带有认证的上下文 (Context)
        storage_state_path = None
        if auth_context:
            storage_state_path = self.auth_files_map.get(auth_context)
        
        context: BrowserContext
        if storage_state_path and os.path.exists(storage_state_path):
            # 如果提供了 auth_context 且文件存在，则加载它
            print(f"🔐 为 task_id '{task_id[:8]}...' 加载 '{storage_state_path}' 认证状态")
            context = await browser.new_context(storage_state=storage_state_path)
        else:
            # 否则，创建一个"游客"上下文
            if auth_context:
                print(f"⚠️ 认证文件不存在: {storage_state_path}，将以游客身份访问")
            context = await browser.new_context()
            print(f"👤 为 task_id '{task_id[:8]}...' 创建游客会话")
        
        # 创建页面 (Page)
        page = await context.new_page()
        page.set_default_timeout(30000)  # 30秒超时
        
        # 导航到 URL
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            # 等待页面加载
            await asyncio.sleep(2)  # 额外等待动态内容加载
            print(f"✅ 页面已打开: {url}")
        except PlaywrightTimeoutError:
            print(f"⚠️ 页面加载超时: {url}")
        except Exception as e:
            print(f"⚠️ 导航失败: {e}")
        
        # 存储会话（注意：需要同时存储 context 和 page）
        with self.lock:
            self.sessions[task_id] = {
                "context": context,
                "page": page,
                "current_content": "",
                "current_html": ""
            }
        
        print(f"✅ 会话 {task_id[:8]}... 已创建")
        return page
    
    async def get_session(self, task_id: str) -> Optional[Page]:
        """
        获取指定 task_id 的浏览器会话页面
        
        Args:
            task_id: 任务唯一标识符
        
        Returns:
            Page: Playwright 页面实例，如果不存在返回 None
        """
        with self.lock:
            session = self.sessions.get(task_id)
            if session is None:
                return None
            return session["page"]
    
    async def close_session(self, task_id: str) -> str:
        """
        关闭并清理指定 task_id 的浏览器会话
        
        【架构核心】关闭整个 BrowserContext（包括所有页面和登录状态）
        
        Args:
            task_id: 任务唯一标识符
        
        Returns:
            str: 确认消息
        """
        with self.lock:
            session = self.sessions.pop(task_id, None)
        
        if session is None:
            return f"⚠️ 会话 {task_id[:8]}... 不存在，无需关闭"
        
        try:
            page = session.get("page")
            context = session.get("context")
            
            # 先关闭页面
            if page:
                await page.close()
            
            # 再关闭上下文（会同时关闭所有页面）
            if context:
                await context.close()
            
            print(f"✅ 会话 {task_id[:8]}... 已关闭")
            return f"会话 {task_id[:8]}... 已成功关闭。"
        except Exception as e:
            print(f"⚠️ 关闭会话时出错: {e}")
            return f"会话 {task_id[:8]}... 关闭时出错: {str(e)}"
    
    def update_content(self, task_id: str, content: str, html: str = ""):
        """更新会话的内容缓存"""
        with self.lock:
            session = self.sessions.get(task_id)
            if session:
                session["current_content"] = content
                session["current_html"] = html
    
    def get_content(self, task_id: str) -> str:
        """获取会话的内容缓存"""
        with self.lock:
            session = self.sessions.get(task_id)
            if session:
                return session["current_content"]
            return ""
    
    async def cleanup_all(self):
        """关闭所有会话"""
        task_ids = list(self.sessions.keys())
        for task_id in task_ids:
            await self.close_session(task_id)
        
        # 关闭浏览器
        if self._browser:
            await self._browser.close()
            self._browser = None
        
        # 停止 Playwright
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


# 全局会话管理器实例
session_manager = BrowserSessionManager()


# ==================== 清理函数 ====================

async def cleanup_all_sessions():
    """清理所有浏览器会话（用于程序退出时）"""
    await session_manager.cleanup_all()

