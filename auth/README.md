# 认证状态目录 (Authentication State Directory)

此目录用于存储网站的登录凭证文件，供 `search_toolkit` 的 `open_url` 工具使用。

## 文件说明

### 生成的凭证文件（由 `save_login_state.py` 创建）

- `jd_auth_state.json` - 京东登录状态
- `zhihu_auth_state.json` - 知乎登录状态
- `{site}_auth_state.json` - 其他网站的登录状态

### 凭证文件内容

每个凭证文件都是 Playwright 的 `storage_state` 格式，包含：
- **Cookies**: 会话 cookies 和持久化 cookies
- **LocalStorage**: 本地存储数据
- **SessionStorage**: 会话存储数据（可选）

## 使用方法

### 1. 生成凭证文件

运行 `save_login_state.py` 脚本：

```bash
python save_login_state.py
```

按照提示选择网站并完成登录。

### 2. 在 SearchAgent 中使用

在调用 `open_url` 工具时，传入 `auth_context` 参数：

```python
# 以京东登录状态访问页面
open_url(
    url="https://order.jd.com/center/list.action",
    task_id="...",
    auth_context="jd.com"
)

# 以知乎登录状态访问页面
open_url(
    url="https://www.zhihu.com/settings/account",
    task_id="...",
    auth_context="zhihu.com"
)
```

### 3. 可用的认证密钥 (auth_context)

认证密钥在 `tools/search_toolkit/browser_manager.py` 中的 `auth_files_map` 定义：

- `"jd.com"` → `auth/jd_auth_state.json`
- `"zhihu.com"` → `auth/zhihu_auth_state.json`

## 安全注意事项

⚠️ **重要：请勿将凭证文件提交到 git 仓库**

- 凭证文件包含敏感的登录信息
- 已通过 `.gitignore` 忽略所有 `*.json` 文件
- 请妥善保管凭证文件，避免泄露

## 凭证过期

如果凭证过期（通常是几周到几个月后），重新运行 `save_login_state.py` 生成新的凭证文件即可。

## 故障排查

### 问题：登录状态无效

**症状**：使用 `auth_context` 后仍然显示未登录状态

**解决方案**：
1. 检查凭证文件是否存在：`ls -la auth/`
2. 检查凭证文件是否过期：重新运行 `save_login_state.py`
3. 检查网站是否更新了安全策略：可能需要额外的验证步骤

### 问题：浏览器启动失败

**症状**：`save_login_state.py` 无法启动浏览器

**解决方案**：
```bash
# 安装 Playwright 浏览器
playwright install chromium

# 或者使用 pip
pip install playwright
playwright install
```

