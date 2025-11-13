"""
PDF 工具包 (L3-A: 快速路径 + PDF 转图片辅助)

提供两类功能：
1. L3-A 快速路径：使用 PyMuPDF 进行快速 PDF 文本提取
2. PDF 转图片辅助：将 PDF 转换为 Base64 编码的图片（供 VLM 使用）

关键特性：
- 不使用任何 LLM
- 快速路径仅处理文本型 PDF
- PDF → 图片转换用于 VLM 降级路径
"""

import os
import base64
from io import BytesIO
from pathlib import Path
from pydantic import Field
from oxygent.oxy import FunctionHub

# 注册 PDF 工具包
pdf_tools = FunctionHub(name="pdf_tools")


@pdf_tools.tool(
    description="(L3-A - Python Fast Path) Attempts to extract raw text from a text-based PDF file using PyMuPDF. "
    "If successful, returns the full extracted text. "
    "If the PDF is scanned, empty, or text extraction fails, returns '[ERROR_NO_TEXT_FOUND]'. "
    "This is the preferred 'fast path' for PDF text extraction before falling back to VLM."
)
def py_pdf_extractor_agent(
    file_path: str = Field(
        description="Path to the PDF file (e.g., './data/document.pdf')"
    ),
    task_id: str = Field(
        default="",
        description="Optional task ID for logging and tracking"
    ),
) -> str:
    """
    L3-A: Python PDF 文本提取器（快速路径）
    
    使用 PyMuPDF 尝试从 PDF 中提取原始文本。
    
    Args:
        file_path: PDF 文件路径
        task_id: 任务 ID（用于日志记录）
    
    Returns:
        提取的原始文本，或 '[ERROR_NO_TEXT_FOUND]' 如果失败
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[ERROR_NO_TEXT_FOUND] PyMuPDF (fitz) 未安装。请运行: pip install PyMuPDF"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(file_path):
        return f"[ERROR_NO_TEXT_FOUND] 文件不存在: {file_path}"
    
    try:
        raw_text = ""
        page_texts = []
        
        # 打开 PDF 并提取文本
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                if page_text.strip():
                    page_texts.append(f"=== 第 {page_num} 页 ===\n{page_text}")
                raw_text += page_text
        
        # 启发式检查：判断是否提取到了有效文本
        # 如果文本太短（< 50 字符）或为空，认为是扫描型 PDF
        if not raw_text or len(raw_text.strip()) < 50:
            return "[ERROR_NO_TEXT_FOUND]"
        
        # 成功提取，返回格式化的文本
        result = []
        result.append(f"=== PDF 文档: {os.path.basename(file_path)} ===")
        result.append(f"总页数: {total_pages}")
        result.append(f"提取字符数: {len(raw_text)}")
        result.append("")
        result.extend(page_texts)
        
        return "\n".join(result)
        
    except Exception as e:
        # 任何异常都返回错误标记
        return f"[ERROR_NO_TEXT_FOUND] PyMuPDF 处理失败: {str(e)}"


@pdf_tools.tool(
    description="Convert PDF pages to temporary image files. Used internally by VLM agents for PDF processing. "
    "Returns paths to temporary image files (NOT Base64 strings to avoid context overflow)."
)
def pdf_to_base64_images(
    file_path: str = Field(
        description="Path to the PDF file (e.g., './data/document.pdf')"
    ),
    max_pages: int = Field(
        default=10,
        description="Maximum number of pages to convert (to avoid memory issues)"
    ),
    dpi: int = Field(
        default=150,
        description="DPI for rendering PDF pages (higher = better quality but larger size)"
    ),
) -> str:
    """
    将 PDF 每一页转换为临时图片文件
    
    ⚠️ 重要变更：此工具现在返回图片文件路径，而不是 Base64 字符串
    原因：Base64 字符串太大，会导致 ReActAgent 的上下文溢出（400 错误）
    
    这个工具供 VLM 智能体使用，用于处理扫描型 PDF。
    
    Args:
        file_path: PDF 文件路径
        max_pages: 最多转换的页数
        dpi: 渲染 DPI（150 适合大多数 OCR 任务）
    
    Returns:
        JSON字符串，包含每页的图片文件路径（临时文件）
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import tempfile
        import json
    except ImportError as e:
        return f"❌ 错误: 缺少依赖库 {e}。请安装 PyMuPDF 和 Pillow"
    
    if not os.path.exists(file_path):
        return f"❌ 错误: 文件不存在: {file_path}"
    
    try:
        result = {
            "file_name": os.path.basename(file_path),
            "total_pages": 0,
            "converted_pages": 0,
            "image_paths": []  # 改为返回路径而不是 Base64
        }
        
        # 创建临时目录
        temp_dir = Path(tempfile.gettempdir()) / "oxy_pdf_temp"
        temp_dir.mkdir(exist_ok=True)
        
        with fitz.open(file_path) as doc:
            result["total_pages"] = len(doc)
            pages_to_convert = min(len(doc), max_pages)
            
            for page_num in range(pages_to_convert):
                page = doc[page_num]
                
                # 将页面渲染为图片 (pix)
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # 缩放矩阵
                pix = page.get_pixmap(matrix=mat)
                
                # 转换为 PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # 保存为临时文件（而不是转换为 Base64）
                base_name = Path(file_path).stem
                temp_image_path = temp_dir / f"{base_name}_page{page_num + 1}.png"
                img.save(temp_image_path, format="PNG")
                
                result["image_paths"].append({
                    "page": page_num + 1,
                    "path": str(temp_image_path),  # 返回文件路径
                    "width": pix.width,
                    "height": pix.height
                })
            
            result["converted_pages"] = pages_to_convert
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import traceback
        return f"❌ 错误: PDF 转图片失败\n{traceback.format_exc()}"


@pdf_tools.tool(
    description="Get basic information about a PDF file (number of pages, metadata, etc.)"
)
def get_pdf_info(
    file_path: str = Field(
        description="Path to the PDF file (e.g., './data/document.pdf')"
    ),
) -> str:
    """
    获取 PDF 文件的基本信息
    
    Args:
        file_path: PDF 文件路径
    
    Returns:
        PDF 文件信息字符串
    """
    try:
        import fitz
    except ImportError:
        return "❌ 错误: PyMuPDF (fitz) 未安装"
    
    if not os.path.exists(file_path):
        return f"❌ 错误: 文件不存在: {file_path}"
    
    try:
        with fitz.open(file_path) as doc:
            info = []
            info.append(f"=== PDF 文件信息 ===")
            info.append(f"文件名: {os.path.basename(file_path)}")
            info.append(f"总页数: {len(doc)}")
            
            # 尝试获取元数据
            if doc.metadata:
                info.append("\n元数据:")
                for key, value in doc.metadata.items():
                    if value:
                        info.append(f"  {key}: {value}")
            
            return "\n".join(info)
            
    except Exception as e:
        return f"❌ 错误: {str(e)}"

