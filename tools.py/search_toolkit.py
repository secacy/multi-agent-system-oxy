"""
职责：网页浏览、信息检索和纯文本PDF提取。

1. `search_google(query: str, task_id: str) -> List[SearchResult]`
   - 描述: 搜索互联网以获取与查询相关的URL和内容摘要。
   - 关键职责:
     - 处理所有开放式搜索请求。
     - `data.jsonl` 示例: `f09fb5f6...` ("《感动中国2023年度人物盛典》-萧凯恩的官方视频"), `c027d65e...` ("地狱厨神：异国寻味的第三季第十集"), `92dd6c46...` ("京东健康中提到的ESG政策")。
2. `open_url_and_read_text(url: str, task_id: str) -> str`
   - 描述: 打开一个指定的URL并提取其所有可读的文本内容。
   - 关键职责:
     - 处理所有 `query` 中直接给定的`https://item.jd.com/...`链接。
     - `data.jsonl` 示例: `7549b8f3...` ("商品中文品牌名是什么？"), `6d00be10...` ("商品详情中的烹饪建议是什么")。
3. `find_and_click(element_description: str, task_id: str) -> bool`
   - 描述: 在当前打开的网页上，根据文本描述（例如 "问大家" 或 "历史版本"）查找并点击一个链接或按钮。
   - 关键职责:
     - 实现`SearchAgent`的多步骤浏览能力。
     - `data.jsonl` 示例: `a93a0bc8...` (需要点击 "问大家"), `5c605328...` (需要点击 "大事记", "百度百科", "参考资料")。
4. `get_current_page_content(task_id: str) -> str`
   - 描述: 在执行`find_and_click`后，获取更新后的页面文本内容。
   - 关键职责:
     - 配合`find_and_click`完成多步骤浏览。
5. `extract_pdf_text(file_path: str, task_id: str) -> str`
   - 描述: 提取一个基于文本的PDF文件中的所有文本内容。
   - 关键职责:
     - 处理 `data.jsonl` 中的 `help_... .pdf` 文件。
     - `data.jsonl` 示例: `54ffde35...` (`help_1756089021301.pdf`) `8609ba08...` (`help_1756091590654.pdf`), `e3d19eca...` (`help_1756092800998.pdf`) 。
     - (注：扫描件PDF由`MultimodalAgent`的OCR工具处理，例如 `22eb45c7...` )。
"""