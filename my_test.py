# check_ip.py
import httpx
import os

# 打印出 Python 脚本“看到”的环境变量 (用于调试)
print(f"HTTPS_PROXY env var: {os.environ.get('HTTPS_PROXY')}")

try:
    # httpx 会自动使用 HTTPS_PROXY 环境变量
    client = httpx.Client(http2=True)
    resp = client.get("https://api.ipify.org")

    print(f"\n[测试成功] 我的公网 IP 是: {resp.text}")

except Exception as e:
    print(f"\n[测试失败] 请求出错: {e}")

from openai import OpenAI

client = OpenAI(
    api_key="API KEY VALUE",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

models = client.models.list()
for model in models:
    print(model.id)
