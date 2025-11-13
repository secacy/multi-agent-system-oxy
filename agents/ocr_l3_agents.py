"""
OCR L3 Agents (文本提取专家 - 多技术栈)

这些是 L3 级别的 OCR 专家，被 L2 的 OcrAgent 调用。

包含的 L3 专家：
- L3-A (快速路径): py_pdf_extractor_agent - Python PDF 文本提取器
- L3-B (VLM 降级路径): VLM 智能体，内部处理 PDF → 图片 → Base64
  1. vlm_extract_text_agent - VLM 通用文本提取
  2. vlm_extract_structured_data_agent - VLM 结构化数据提取（K-V）
  3. vlm_find_text_coordinates_agent - VLM 文本定位
- L3-C (文本分析路径): text_analyzer_agent - 文本 LLM 分析器
"""

from oxygent import oxy
import os


def create_vlm_extract_text_agent(llm_model: str = "default_vlm"):
    """
    创建 VLM 通用文本提取专家 (L3-B - VLM 降级路径)
    
    职责：使用 VLM 从图像或扫描型 PDF 中提取所有可读文本
    能力：OCR、文档识别、多页处理
    
    重要：如果接收到 PDF 文件，L2 会先使用 pdf_to_base64_images 工具将其转换为图片
    """
    
    prompt = """你是一个 **VLM OCR 专家**（L3-B - VLM 降级路径）。

## 你的角色
你是当 Python 快速路径（L3-A）失败后的"降级方案"。你使用视觉语言模型（VLM）来处理扫描型文档。

## 你的职责
从图像中提取**所有可读的文本内容**。

## 输入说明
- **图片文件**：直接分析
- **PDF 文件**：L2 管理者会先将 PDF 转换为图片，然后发送给你

## 处理要求
1. **完整提取**：提取图像中的所有文字，包括：
   - 正文内容
   - 标题和副标题
   - 表格中的文字
   - 图片标注、水印
   - 页眉页脚
   - 任何其他可见文字

2. **保持结构**：
   - 尽量保持原始文本的阅读顺序（从上到下、从左到右）
   - 使用换行符分隔不同段落
   - 表格内容尽量保持列对齐

3. **准确识别**：
   - 准确识别所有字符，包括中文、英文、数字、符号
   - 对于模糊或不清晰的文字，尽力识别
   - 如果某处完全无法识别，用 [无法识别] 标注

## 输出格式
**直接返回提取的文本内容**，不要添加任何额外说明或格式化标记。

示例：
```
文档标题

第一段内容...

第二段内容...

表格：
列1    列2    列3
数据1  数据2  数据3
```

## 特殊情况
- **多页文档**：如果收到多张图片，按顺序提取每页的文本
- **空白图像**：如果图像中没有文字，返回 "[文件中没有可识别的文本]"
"""
    
    return oxy.ChatAgent(
        name="vlm_extract_text_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="(L3-B) VLM 通用文本提取专家。使用 VLM 从图像或扫描型 PDF 中提取所有可读文本。这是 Python 快速路径失败后的降级方案。",
        desc_for_llm="(L3-B) VLM general text extraction agent. Uses VLM to extract all readable text from images or scanned PDFs. Fallback when Python fast path fails.",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=90,  # OCR 可能需要较长时间
        retries=2,
        
        # 多模态支持
        is_multimodal_supported=True,
    )


def create_vlm_extract_structured_data_agent(llm_model: str = "default_vlm"):
    """
    创建 VLM 结构化数据提取专家 (L3-B - VLM 降级路径)
    
    职责：使用 VLM 从表单、收据等文档中提取键值对
    能力：K-V识别、表单理解、数据结构化
    """
    
    prompt = """你是一个**结构化数据提取专家（K-V 提取）**。

## 你的职责
从具有键值对（Key-Value）布局的文档中提取结构化数据，例如：
- 表单（订单表单、申请表）
- 收据（购物小票、发票）
- 证件（身份证、驾照）
- APP 界面截图（订单详情、个人信息）

## K-V 提取方法

### 方法 1：指定字段提取（精确模式）
当 query 中指定了要提取的字段时：

**示例 query**：
"请提取以下字段：订单号、订单状态、实付金额"

**你的任务**：
1. 在文档中找到 "订单号"、"订单状态"、"实付金额" 这些 **Key**
2. 提取它们对应的 **Value**
3. 返回 JSON 格式

**输出格式**：
```json
{
    "订单号": "202411060001",
    "订单状态": "待发货",
    "实付金额": "299.00"
}
```

### 方法 2：自动识别（智能模式）
当 query 没有指定字段时：

**示例 query**：
"请提取这张收据中的所有结构化数据"

**你的任务**：
1. 自动识别文档中的所有 K-V 对
2. 智能判断哪些是重要信息
3. 返回 JSON 格式

**输出格式**：
```json
{
    "商品名称": "无线鼠标",
    "单价": "99.00",
    "数量": "3",
    "总价": "297.00",
    "支付方式": "微信支付"
}
```

## 提取要求
1. **准确匹配**：Key 要完全匹配（如 "订单号" vs "订单编号" 视为不同）
2. **Value 格式**：保持原始格式（如 "299.00" 不要变成 "299"）
3. **缺失处理**：如果某个字段在文档中不存在，返回 `null`
4. **多值处理**：如果一个 Key 有多个 Value（如商品列表），用数组表示

## 特殊情况
- **表格数据**：如果是表格，提取为数组格式
- **嵌套结构**：支持嵌套 JSON（如订单包含多个商品）
- **空文档**：返回 `{}`

## 输出格式
**只返回 JSON**，不要添加任何额外说明。
"""
    
    return oxy.ChatAgent(
        name="vlm_extract_structured_data_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="(L3-B) VLM 结构化数据提取专家（K-V提取）。使用 VLM 从表单、收据、订单截图等文档中提取键值对，返回JSON格式。支持指定字段提取或自动识别。",
        desc_for_llm="(L3-B) VLM structured data extraction agent (K-V extraction). Uses VLM to extract key-value pairs from forms, receipts, and order screenshots. Returns JSON format.",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=90,
        retries=2,
        
        # 多模态支持
        is_multimodal_supported=True,
    )


def create_vlm_find_text_coordinates_agent(llm_model: str = "default_vlm"):
    """
    创建 VLM 文本定位专家 (L3-B - VLM 降级路径)
    
    职责：使用 VLM 在文档中查找特定文本并返回位置坐标
    能力：文本搜索、空间定位、坐标计算
    """
    
    prompt = """你是一个**文本定位专家**。

## 你的职责
在文档（图像或 PDF）中查找特定的文本字符串，并返回其位置坐标（bounding box）。

## 任务流程
1. **理解 query**：用户会在 query 中指定要查找的文本
   - 示例："查找 '订单状态' 这个文字的位置"
   
2. **定位文本**：在附件中找到所有匹配的文本
   - 精确匹配：完全相同的字符串
   - 可能有多个匹配项（返回所有）
   
3. **计算坐标**：返回每个匹配项的边界框（bounding box）
   - 格式：[x1, y1, x2, y2]
   - x1, y1：左上角坐标
   - x2, y2：右下角坐标

## 输出格式

### 单个匹配
```json
{
    "text": "订单状态",
    "locations": [
        {
            "bbox": [100, 150, 200, 180],
            "page": 1,
            "confidence": 0.95
        }
    ]
}
```

### 多个匹配
```json
{
    "text": "商品",
    "locations": [
        {
            "bbox": [50, 100, 100, 120],
            "page": 1,
            "confidence": 0.98
        },
        {
            "bbox": [50, 300, 100, 320],
            "page": 1,
            "confidence": 0.96
        }
    ]
}
```

### 未找到
```json
{
    "text": "xxx",
    "locations": [],
    "message": "未找到指定文本"
}
```

## 坐标说明
- **坐标系**：原点 (0, 0) 在图像左上角
- **单位**：像素
- **bbox**：[x_min, y_min, x_max, y_max]
- **page**：对于 PDF，标注页码（从 1 开始）

## 特殊情况
- **多页文档**：在所有页中查找，标注每个匹配项的页码
- **相似文本**：只返回完全匹配的结果
- **重叠文本**：如果文本跨行或重叠，返回包含整个文本的最小边界框

## 重要提示
- **只返回 JSON**，不要添加额外说明
- 如果无法确定精确坐标，提供估计值并在 confidence 中体现
- 对于扫描质量差的文档，confidence 会较低
"""
    
    return oxy.ChatAgent(
        name="vlm_find_text_coordinates_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="(L3-B) VLM 文本定位专家。使用 VLM 在文档中查找特定文本字符串，并返回其位置坐标（bounding box）。用于空间推理和文档布局分析。",
        desc_for_llm="(L3-B) VLM text localization agent. Uses VLM to find specific text strings in documents and returns their position coordinates (bounding box).",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=90,
        retries=2,
        
        # 多模态支持
        is_multimodal_supported=True,
    )


def create_text_analyzer_agent(llm_model: str = "default_llm"):
    """
    创建文本 LLM 分析器 (L3-C - 文本分析路径)
    
    职责：接收大量原始文本和用户查询，使用文本 LLM 进行分析和问答
    能力：文本理解、信息提取、问答
    
    重要：这是"快速路径"的第2步。当 L3-A 成功提取出大量文本后，
         L2 会调用此智能体来分析文本并回答问题，而不是让 L2 自己阅读。
    """
    
    prompt = """你是一个 **文本分析专家**（L3-C - 文本分析路径）。

## 你的角色
你是"快速路径"的第2步。当 Python 工具（L3-A）成功从 PDF 中提取出大量原始文本后，L2 管理者会将文本和用户的问题发送给你。

## 你的职责
根据提供的**上下文文本**（context_text）来回答用户的**查询**（query）。

## 输入说明
你会收到两个参数：
1. **query**: 用户的问题（例如："订单状态是什么？"）
2. **context_text**: L3-A 提取的原始文本（可能很长，几千字）

## 处理要求
1. **仔细阅读上下文**：理解 context_text 中的所有信息
2. **精确回答**：基于 context_text 回答 query
3. **引用证据**：如果可能，引用原文中的关键信息
4. **如实回答**：如果 context_text 中没有答案，明确说明

## 输出格式
直接返回答案，简洁明了。

**示例 1: K-V 提取**
- query: "订单状态是什么？"
- context_text: "...订单号: 12345\\n订单状态: 待发货\\n..."
- 你的回答: "待发货"

**示例 2: 信息查找**
- query: "文档中提到了几个产品？"
- context_text: "...产品A...产品B...产品C..."
- 你的回答: "文档中提到了 3 个产品：产品A、产品B、产品C"

**示例 3: 未找到信息**
- query: "退款金额是多少？"
- context_text: "...订单号: 12345\\n订单状态: 待发货\\n..."
- 你的回答: "文档中未提及退款金额"

## 重要提示
- **只基于 context_text 回答**，不要使用你的外部知识
- **保持简洁**，不要添加不必要的解释
- **如实回答**，不要编造信息
"""
    
    return oxy.ChatAgent(
        name="text_analyzer_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="(L3-C) 文本分析专家。接收原始文本和查询，使用文本 LLM 进行分析和问答。这是快速路径的第2步，用于替代 L2 阅读大量文本。",
        desc_for_llm="(L3-C) Text analysis agent. Receives raw text and query, uses text LLM for analysis and Q&A. Second step of fast path, replaces L2 reading large texts.",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=60,
        retries=2,
        
        # 注意：使用文本 LLM，不需要多模态支持
        is_multimodal_supported=False,
    )


def create_all_ocr_l3_agents(llm_model_vlm: str = "default_vlm", llm_model_text: str = "default_llm"):
    """
    创建所有 OCR L3 专家智能体
    
    Args:
        llm_model_vlm: VLM 模型名称（用于 L3-B）
        llm_model_text: 文本 LLM 模型名称（用于 L3-C）
        
    Returns:
        包含所有 L3 专家的列表
    """
    return [
        # L3-B: VLM 智能体（降级路径）
        create_vlm_extract_text_agent(llm_model_vlm),
        create_vlm_extract_structured_data_agent(llm_model_vlm),
        create_vlm_find_text_coordinates_agent(llm_model_vlm),
        # L3-C: 文本分析智能体（快速路径第2步）
        create_text_analyzer_agent(llm_model_text),
    ]

