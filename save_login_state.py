#!/usr/bin/env python3
"""
登录状态保存工具 (Authentication State Saver)

用途：
    这是一个一次性交互式脚本，用于生成登录凭证文件。
    它会启动一个有头浏览器，等待用户手动登录指定网站，
    然后自动保存登录状态（Cookies + LocalStorage）到文件。

使用方法：
    python save_login_state.py

工作流程：
    1. 选择要登录的网站（如 jd.com, zhihu.com）
    2. 启动有头浏览器
    3. 用户手动完成登录
    4. 用户在控制台按 Enter 确认登录完成
    5. 自动保存登录状态到 auth/ 目录
    6. 关闭浏览器

注意事项：
    - 此脚本只需运行一次（或当登录凭证过期时重新运行）
    - 生成的凭证文件会被 search_toolkit 的 open_url 工具使用
    - 请勿将凭证文件提交到 git（已在 .gitignore 中配置）
"""

import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright


# 网站配置：可以根据需要添加更多网站
SUPPORTED_SITES = {
    "1": {
        "name": "京东 (jd.com)",
        "url": "https://passport.jd.com/new/login.aspx",
        "auth_key": "jd.com",
        "output_file": "auth/jd_auth_state.json",
        "instructions": "请在浏览器中登录京东账号（使用账号密码或扫码）"
    },
    "2": {
        "name": "知乎 (zhihu.com)",
        "url": "https://www.zhihu.com/signin",
        "auth_key": "zhihu.com",
        "output_file": "auth/zhihu_auth_state.json",
        "instructions": "请在浏览器中登录知乎账号"
    },
    "3": {
        "name": "自定义网站",
        "url": None,  # 将由用户输入
        "auth_key": None,
        "output_file": None,
        "instructions": "请在浏览器中完成登录"
    }
}


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🔐 登录状态保存工具 (Authentication State Saver)")
    print("=" * 60)
    print()


def select_site():
    """选择要登录的网站"""
    print("请选择要登录的网站：")
    print()
    for key, site in SUPPORTED_SITES.items():
        print(f"  [{key}] {site['name']}")
    print()
    
    while True:
        choice = input("请输入选项编号 (1/2/3): ").strip()
        if choice in SUPPORTED_SITES:
            return choice
        print("⚠️ 无效选项，请重新输入")


def get_custom_site_info():
    """获取自定义网站信息"""
    print()
    print("自定义网站配置：")
    print()
    
    url = input("请输入登录页面的 URL (如 https://example.com/login): ").strip()
    if not url.startswith("http"):
        print("⚠️ URL 必须以 http:// 或 https:// 开头")
        sys.exit(1)
    
    auth_key = input("请输入认证密钥 (如 example.com): ").strip()
    if not auth_key:
        print("⚠️ 认证密钥不能为空")
        sys.exit(1)
    
    output_file = f"auth/{auth_key}_auth_state.json"
    
    return {
        "name": f"自定义网站 ({auth_key})",
        "url": url,
        "auth_key": auth_key,
        "output_file": output_file,
        "instructions": "请在浏览器中完成登录"
    }


async def save_login_state_interactive(site_config):
    """
    交互式保存登录状态
    
    Args:
        site_config: 网站配置字典
    """
    print()
    print("=" * 60)
    print(f"📋 网站: {site_config['name']}")
    print(f"📋 URL: {site_config['url']}")
    print(f"📋 输出文件: {site_config['output_file']}")
    print("=" * 60)
    print()
    
    # 确保 auth 目录存在
    Path("auth").mkdir(parents=True, exist_ok=True)
    
    print("🚀 正在启动浏览器...")
    print()
    
    async with async_playwright() as p:
        # 启动有头浏览器
        browser = await p.chromium.launch(
            headless=False,  # 有头模式
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        # 创建浏览器上下文
        context = await browser.new_context()
        
        # 创建新页面
        page = await context.new_page()
        
        # 导航到登录页面
        print(f"📄 正在打开登录页面: {site_config['url']}")
        await page.goto(site_config['url'])
        
        print()
        print("=" * 60)
        print(f"✅ 浏览器已启动！")
        print()
        print(f"📝 {site_config['instructions']}")
        print()
        print("⚠️  完成登录后，请回到此控制台按 Enter 键继续...")
        print("=" * 60)
        print()
        
        # 等待用户按 Enter
        input(">>> 按 Enter 键继续（确认已完成登录）...")
        
        print()
        print("💾 正在保存登录状态...")
        
        # 保存登录状态
        output_path = site_config['output_file']
        await context.storage_state(path=output_path)
        
        print(f"✅ 登录状态已保存到: {output_path}")
        print()
        
        # 关闭浏览器
        print("🔒 正在关闭浏览器...")
        await browser.close()
        
        print()
        print("=" * 60)
        print("🎉 完成！")
        print()
        print(f"📋 认证密钥: {site_config['auth_key']}")
        print(f"📋 凭证文件: {output_path}")
        print()
        print("💡 使用方法：")
        print(f"   在调用 open_url 工具时，传入 auth_context=\"{site_config['auth_key']}\"")
        print()
        print("   示例：")
        print(f"   open_url(")
        print(f"       url=\"https://...\",")
        print(f"       task_id=\"...\",")
        print(f"       auth_context=\"{site_config['auth_key']}\"")
        print(f"   )")
        print()
        print("⚠️  注意：请勿将凭证文件提交到 git 仓库")
        print("=" * 60)


def main():
    """主函数"""
    print_banner()
    
    # 选择网站
    choice = select_site()
    site_config = SUPPORTED_SITES[choice]
    
    # 如果是自定义网站，获取配置信息
    if choice == "3":
        site_config = get_custom_site_info()
    
    # 运行交互式保存流程
    try:
        asyncio.run(save_login_state_interactive(site_config))
    except KeyboardInterrupt:
        print()
        print("⚠️ 用户取消操作")
        sys.exit(0)
    except Exception as e:
        print()
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

