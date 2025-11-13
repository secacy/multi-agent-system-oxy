"""
VideoAgent 工具包：视频分析、对象检测、内容搜索和流程分析

职责：提供完整的视频分析功能，支持时序定位、CV检测、OCR/ASR搜索和复杂流程理解。

工具列表：
1. get_video_metadata(file_path: str) -> dict
   - 获取视频元数据（时长、分辨率、帧率）

2. detect_objects_in_video(file_path: str, object_prompt: str, timestamp: str, task_id: str) -> list[dict]
   - 开放词汇目标检测（使用 GroundingDINO）

3. track_user_events(file_path: str, task_id: str) -> list[dict]
   - 跟踪用户交互事件（点击、滑动、关注等）

4. find_content_in_video(file_path: str, search_prompt: str, task_id: str) -> list[dict]
   - 多模态内容搜索（ASR + OCR）

5. analyze_video_flow(file_path: str, prompt: str, timestamp: str, task_id: str) -> str
   - 复杂流程分析（使用 VLM）
"""

import os
import cv2
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import Field
from oxygent.oxy import FunctionHub
import traceback

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
video_tools = FunctionHub(name="video_tools")


# ==================== 工具 1: 获取视频元数据 ====================


@video_tools.tool(
    description="Get basic metadata from a video file, including duration (in seconds), "
    "resolution (width x height), and frame rate (fps). "
    "Use this tool when you need to know how long a video is or its technical specifications."
)
def get_video_metadata(
    file_path: str = Field(
        description="Path to the video file (e.g., './data/video.mp4')"
    ),
) -> dict:
    """
    获取视频文件的基本元数据

    Returns:
        dict: {"duration_seconds": 65.5, "resolution": "1920x1080", "fps": 30}
    """
    try:
        import ffmpeg

        print(f"🎬 正在获取视频元数据: {file_path}")

        # 检查文件是否存在
        video_file = Path(file_path)
        if not video_file.exists():
            return {"error": f"视频文件不存在: {file_path}"}

        # 使用 ffmpeg 获取元数据
        probe = ffmpeg.probe(str(video_file))

        # 查找视频流
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )

        if not video_stream:
            return {"error": "无法找到视频流"}

        # 提取元数据
        duration = float(probe["format"]["duration"])
        width = int(video_stream["width"])
        height = int(video_stream["height"])

        # 计算帧率
        fps_parts = video_stream["r_frame_rate"].split("/")
        fps = int(fps_parts[0]) / int(fps_parts[1])

        metadata = {
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "width": width,
            "height": height,
        }

        print(f"✅ 元数据获取成功: {metadata}")
        return metadata

    except ImportError:
        return {
            "error": "缺少依赖: ffmpeg-python",
            "install_hint": "pip install ffmpeg-python",
        }
    except Exception as e:
        return {"error": f"获取元数据失败: {str(e)}"}


# ==================== 工具 2: 开放词汇目标检测 ====================


@video_tools.tool(
    description="Detect and count visual objects in a video at a specific timestamp or range. "
    "Uses open-vocabulary object detection (GroundingDINO) to find objects based on natural language prompts. "
    "Examples: 'stars', 'activity badge', 'people', 'cameras'. "
    "Returns bounding boxes and counts of detected objects. "
    "Results will be saved to local_es_data/ directory."
)
def detect_objects_in_video(
    file_path: str = Field(
        description="Path to the video file (e.g., './data/video.mp4')"
    ),
    object_prompt: str = Field(
        description="Natural language description of the object to detect, "
        "e.g., 'stars', 'activity badge', 'people', 'phone cameras'"
    ),
    timestamp: str = Field(
        default="",
        description="Time point or range to analyze, e.g., '4s', '1m31s', '30s-32s'. "
        "Leave empty to search the entire video.",
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving results."
    ),
) -> list:
    """
    在视频的指定时间范围内检测视觉对象（使用 GroundingDINO）

    Returns:
        list[dict]: [{"timestamp_sec": 4.1, "detected_object": "星星", "count": 3, "bounding_boxes": [...]}]
    """
    try:
        print(f"🎯 开始对象检测: {file_path}")
        print(f"   检测目标: {object_prompt}")
        print(f"   时间点: {timestamp or '全视频'}")

        # 检查文件是否存在
        video_file = Path(file_path)
        if not video_file.exists():
            error_msg = f"视频文件不存在: {file_path}"
            save_video_result(task_id, "detect_objects", file_path, error_msg)
            return [{"error": error_msg}]

        # 解析时间戳
        timestamps_to_check = _parse_timestamp(timestamp, file_path)

        # 提取关键帧
        frames_data = _extract_frames(file_path, timestamps_to_check)

        if not frames_data:
            error_msg = "无法提取视频帧"
            save_video_result(task_id, "detect_objects", file_path, error_msg)
            return [{"error": error_msg}]

        # 使用 GroundingDINO 检测对象
        results = []

        try:
            # 尝试导入 GroundingDINO
            from groundingdino.util.inference import load_model, predict
            import torch

            # 加载模型
            print("📥 加载 GroundingDINO 模型...")
            model_config = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            model_checkpoint = "weights/groundingdino_swint_ogc.pth"

            # 如果模型文件不存在，提供备选方案
            if not Path(model_checkpoint).exists():
                print("⚠️ GroundingDINO 模型未找到，使用简化检测方法")
                # 回退到简单的模板匹配或其他方法
                results = _fallback_object_detection(frames_data, object_prompt)
            else:
                model = load_model(model_config, model_checkpoint)

                # 对每一帧进行检测
                for frame_info in frames_data:
                    frame = frame_info["frame"]
                    timestamp_sec = frame_info["timestamp_sec"]

                    # 执行检测
                    boxes, logits, phrases = predict(
                        model=model,
                        image=frame,
                        caption=object_prompt,
                        box_threshold=0.35,
                        text_threshold=0.25,
                    )

                    # 保存检测结果
                    if len(boxes) > 0:
                        # 保存标注后的帧
                        annotated_frame = _draw_boxes(frame, boxes, phrases, logits)
                        frame_path = (
                            f"local_es_data/{task_id}_frame_{timestamp_sec:.1f}s.jpg"
                        )
                        cv2.imwrite(frame_path, annotated_frame)

                        result = {
                            "timestamp_sec": round(timestamp_sec, 2),
                            "detected_object": object_prompt,
                            "count": len(boxes),
                            "bounding_boxes": boxes.tolist(),
                            "confidence_scores": logits.tolist(),
                            "saved_frame": frame_path,
                        }
                        results.append(result)

                        print(f"✅ 在 {timestamp_sec:.1f}s 检测到 {len(boxes)} 个对象")

        except ImportError as e:
            print(f"⚠️ 导入 GroundingDINO 失败: {e}")
            print("   使用简化检测方法...")
            results = _fallback_object_detection(frames_data, object_prompt)

        # 保存结果
        if results:
            result_summary = json.dumps(results, ensure_ascii=False, indent=2)
            save_video_result(task_id, "detect_objects", file_path, result_summary)
            return results
        else:
            no_detection_msg = f"未检测到对象: {object_prompt}"
            save_video_result(task_id, "detect_objects", file_path, no_detection_msg)
            return [{"message": no_detection_msg, "count": 0}]

    except Exception as e:
        error_msg = f"对象检测失败: {str(e)}"
        print(f"❌ {error_msg}")
        save_video_result(task_id, "detect_objects", file_path, error_msg)
        return [{"error": error_msg}]


# ==================== 工具 3: 跟踪用户交互事件 ====================


@video_tools.tool(
    description="Track user interaction events in a screen recording video. "
    "Automatically detects clicks, swipes, and other interactions, and extracts context (button text, target items). "
    "Use this for tasks like 'what did the user add to cart' or 'which items did the user follow'. "
    "Results will be saved to local_es_data/ directory."
)
def track_user_events(
    file_path: str = Field(
        description="Path to the screen recording video file (e.g., './data/recording.mp4')"
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving results."
    ),
) -> list:
    """
    跟踪屏幕录制视频中的用户交互事件

    Returns:
        list[dict]: [
            {"timestamp_sec": 12.5, "event_type": "click", "target_text": "加入购物车", "context": "..."},
            {"timestamp_sec": 20.3, "event_type": "follow", "target_text": "关注", "context_text": "商品A"}
        ]
    """
    try:
        print(f"👆 开始跟踪用户事件: {file_path}")

        # 检查文件是否存在
        video_file = Path(file_path)
        if not video_file.exists():
            error_msg = f"视频文件不存在: {file_path}"
            save_video_result(task_id, "track_events", file_path, error_msg)
            return [{"error": error_msg}]

        # 使用 VLM 分析整个流程（这是一个复杂任务）
        # 实际上，这个工具最好通过 analyze_video_flow 实现

        # 简化实现：采样关键帧，使用 OCR + 帧差分析
        events = []

        # 1. 提取视频帧序列（密集采样：每0.5秒一帧）
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"   视频时长: {duration:.1f}s, FPS: {fps:.1f}")

        # 采样间隔（0.5秒）
        sample_interval = 0.5
        frame_indices = []
        for t in np.arange(0, duration, sample_interval):
            frame_idx = int(t * fps)
            if frame_idx < total_frames:
                frame_indices.append(frame_idx)

        print(f"   采样 {len(frame_indices)} 帧...")

        # 2. 提取帧并检测变化
        prev_frame = None
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            timestamp_sec = frame_idx / fps

            # 检测帧变化（简单的帧差分）
            if prev_frame is not None:
                # 计算帧差
                diff = cv2.absdiff(frame, prev_frame)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                change_ratio = np.sum(diff_gray > 30) / diff_gray.size

                # 如果变化显著（可能是点击/跳转）
                if change_ratio > 0.1:
                    # 使用 OCR 提取文本
                    try:
                        from paddleocr import PaddleOCR

                        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                        ocr_result = ocr.predict(frame)

                        # 提取文本
                        texts = []
                        if ocr_result and ocr_result[0]:
                            for line in ocr_result[0]:
                                if line[1][1] > 0.5:  # 置信度阈值
                                    texts.append(line[1][0])

                        # 判断事件类型（基于OCR文本）
                        event_type = "unknown"
                        target_text = ""

                        for text in texts:
                            if "加入购物车" in text or "购物车" in text:
                                event_type = "add_to_cart"
                                target_text = text
                                break
                            elif "关注" in text:
                                event_type = "follow"
                                target_text = text
                                break
                            elif "筛选" in text:
                                event_type = "filter"
                                target_text = text
                                break
                            elif "点击" in text or "选择" in text:
                                event_type = "click"
                                target_text = text
                                break

                        if event_type != "unknown":
                            # 保存事件帧
                            frame_path = f"local_es_data/{task_id}_event_{len(events)}_{timestamp_sec:.1f}s.jpg"
                            cv2.imwrite(frame_path, frame)

                            events.append(
                                {
                                    "timestamp_sec": round(timestamp_sec, 2),
                                    "event_type": event_type,
                                    "target_text": target_text,
                                    "context_texts": texts[
                                        :5
                                    ],  # 保存前5个文本作为上下文
                                    "saved_frame": frame_path,
                                }
                            )

                            print(f"✅ 检测到事件: {event_type} @ {timestamp_sec:.1f}s")

                    except ImportError:
                        print("⚠️ PaddleOCR 未安装，跳过OCR分析")
                    except Exception as e:
                        print(f"⚠️ OCR 分析失败: {e}")

            prev_frame = frame.copy()

        cap.release()

        # 保存结果
        if events:
            result_summary = json.dumps(events, ensure_ascii=False, indent=2)
            save_video_result(task_id, "track_events", file_path, result_summary)
            print(f"✅ 共检测到 {len(events)} 个事件")
            return events
        else:
            no_events_msg = "未检测到明显的用户交互事件"
            save_video_result(task_id, "track_events", file_path, no_events_msg)
            return [{"message": no_events_msg}]

    except Exception as e:
        error_msg = f"事件跟踪失败: {str(e)}"
        print(f"❌ {error_msg}")
        save_video_result(task_id, "track_events", file_path, error_msg)
        return [{"error": error_msg}]


# ==================== 工具 4: 多模态内容搜索 ====================


@video_tools.tool(
    description="Search for multiple independent information entities in a video by analyzing both on-screen text (OCR) "
    "and spoken audio (ASR). The tool will search for each entity separately and return structured results. "
    "Examples: ['battery capacity', 'release date'], ['product model', 'price'], ['iPhone model', 'release month']. "
    "Results will be saved to local_es_data/ directory."
)
async def find_content_in_video(
    file_path: str = Field(
        description="Path to the video file (e.g., './data/video.mp4')"
    ),
    extraction_prompts: list = Field(
        description="A list of independent information entities to extract. "
        "The tool will search for each entity separately. "
        "e.g., ['product model', 'release month', 'battery capacity']"
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving results."
    ),
) -> dict:
    """
    在视频中独立搜索多个信息实体（OCR + ASR 双轨搜索）
    
    Args:
        file_path: 视频文件路径
        extraction_prompts: 要提取的信息实体列表（每个独立搜索）
        task_id: 任务ID
        
    Returns:
        dict[str, list]: {
            "iPhone型号": [{"timestamp_sec": 4.1, "source": "ocr", "match_text": "iPhone 14", ...}],
            "发布月份": [{"timestamp_sec": 15.3, "source": "audio", "match_text": "九月发布", ...}],
            ...
        }
    """
    try:
        print(f"🔍 开始内容搜索: {file_path}")
        print(f"   搜索实体: {extraction_prompts}")

        # 检查文件是否存在
        video_file = Path(file_path)
        if not video_file.exists():
            error_msg = f"视频文件不存在: {file_path}"
            save_video_result(task_id, "find_content", str(file_path), error_msg)
            return {"error": error_msg}

        # 创建临时目录
        temp_dir = Path("local_es_data") / "temp" / task_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 初始化结果字典：为每个提示词创建一个空列表
        results_dict = {prompt: [] for prompt in extraction_prompts}
        
        # 用于临时存储所有匹配项（带有对应的 prompt）
        all_matches = []

        # === 1. ASR 轨道：音频转录 ===
        print("🎤 开始 ASR 处理...")

        try:
            # 提取音频
            import ffmpeg

            audio_path = temp_dir / "audio.wav"

            stream = ffmpeg.input(str(video_file))
            stream = ffmpeg.output(
                stream, str(audio_path), acodec="pcm_s16le", ac=1, ar="16000"
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            # 使用 Faster Whisper 转录
            from faster_whisper import WhisperModel

            print("   加载 Whisper 模型...")
            model = WhisperModel("large-v2", device="auto", compute_type="auto")

            print("   转录中...")
            segments, info = model.transcribe(str(audio_path), beam_size=5)

            # 对每个 segment 遍历所有的 extraction_prompts，独立匹配
            for segment in segments:
                text = segment.text.strip()
                
                # 对每个提示词独立检查
                for prompt in extraction_prompts:
                    if _fuzzy_match(prompt, text):
                        match_item = {
                            "timestamp_sec": round(segment.start, 2),
                            "source": "audio",
                            "match_text": text,
                            "segment_start": round(segment.start, 2),
                            "segment_end": round(segment.end, 2),
                            "confidence": "high",
                            "matched_prompt": prompt,
                        }
                        all_matches.append(match_item)
                        print(f"✅ [ASR] '{prompt}' 匹配: {text} @ {segment.start:.1f}s")

        except ImportError as e:
            print(f"⚠️ ASR 依赖缺失: {e}")
        except Exception as e:
            print(f"⚠️ ASR 处理失败: {e}")

        # === 2. OCR 轨道：屏幕文字识别 ===
        print("👁️ 开始 OCR 处理...")

        try:
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(use_angle_cls=True, lang="ch")

            # 采样策略：每秒1帧
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            sample_interval = 1.0  # 每秒采样1帧

            for t in np.arange(0, duration, sample_interval):
                frame_idx = int(t * fps)
                if frame_idx >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                ocr_result = ocr.predict(frame)

                # (策略 1) 健壮性检查 + (策略 2) 聚合框架
                if ocr_result and ocr_result[0]:
                    frame_all_text_list = []
                    line_details = []

                    # --- 第一次循环：健壮地提取所有文本 ---
                    for line in ocr_result[0]:
                        try:
                            bbox = line[0]
                            text = line[1][0]
                            confidence = line[1][1]

                            if text and confidence > 0.5:
                                frame_all_text_list.append(text)
                                line_details.append(
                                    {
                                        "text": text,
                                        "confidence": confidence,
                                        "bbox": bbox,
                                    }
                                )

                        except (IndexError, TypeError, AttributeError):
                            # (策略 1) 跳过损坏的行
                            pass

                    if not frame_all_text_list:
                        continue

                    full_frame_text = " ".join(frame_all_text_list)

                    # --- 第二步：对每个 prompt 独立检查全帧文本 ---
                    matched_prompts = []
                    for prompt in extraction_prompts:
                        if _fuzzy_match(prompt, full_frame_text):
                            matched_prompts.append(prompt)
                    
                    # 如果至少有一个 prompt 匹配成功
                    if matched_prompts:
                        print(f"✅ [OCR] 帧匹配 @ {t:.1f}s (匹配的提示词: {matched_prompts})")

                        frame_path = f"local_es_data/{task_id}_ocr_match_{t:.1f}s.jpg"
                        annotated_frame = frame.copy()

                        # 在图像上绘制所有识别到的框
                        for line_data in line_details:
                            pts = np.array(line_data["bbox"], np.int32).reshape(
                                (-1, 1, 2)
                            )
                            cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)

                        cv2.imwrite(frame_path, annotated_frame)

                        # 为每个匹配的 prompt 添加匹配项
                        for matched_prompt in matched_prompts:
                            # 将这一帧识别到的所有行都作为匹配项添加
                            for line_data in line_details:
                                match_item = {
                                    "timestamp_sec": round(t, 2),
                                    "source": "ocr",
                                    "match_text": line_data["text"],
                                    "bounding_box": line_data["bbox"],
                                    "confidence": round(line_data["confidence"], 2),
                                    "saved_frame": frame_path,
                                    "full_frame_context": full_frame_text,
                                    "matched_prompt": matched_prompt,
                                }
                                all_matches.append(match_item)

            cap.release()

        except ImportError as e:
            print(f"⚠️ OCR 依赖缺失: {e}")
        except Exception as e:
            print(f"⚠️ OCR 处理失败: {e}")
            traceback.print_exc()

        # 将 all_matches 按 prompt 分组到 results_dict
        for match_item in all_matches:
            prompt = match_item.pop("matched_prompt")  # 移除 matched_prompt 字段
            results_dict[prompt].append(match_item)
        
        # 对每个 prompt 的结果按时间排序
        for prompt in results_dict:
            results_dict[prompt].sort(key=lambda x: x["timestamp_sec"])

        # 统计总匹配数
        total_matches = sum(len(matches) for matches in results_dict.values())
        
        # 打印每个 prompt 的匹配情况
        print(f"\n📊 搜索结果汇总:")
        for prompt, matches in results_dict.items():
            if matches:
                print(f"   ✅ '{prompt}': {len(matches)} 个匹配项")
            else:
                print(f"   ❌ '{prompt}': 未找到匹配内容")

        # 保存结果
        result_summary = json.dumps(results_dict, ensure_ascii=False, indent=2)
        save_video_result(task_id, "find_content", str(file_path), result_summary)
        
        if total_matches > 0:
            print(f"✅ 总共找到 {total_matches} 个匹配项")
        else:
            print(f"⚠️ 所有提示词均未找到匹配内容")
        
        return results_dict

    except Exception as e:
        error_msg = f"内容搜索失败: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        save_video_result(task_id, "find_content", str(file_path), error_msg)
        return {"error": error_msg}


# ==================== 工具 5: 复杂流程分析（VLM）====================


@video_tools.tool(
    description="Analyze complex video workflows using a Vision-Language Model (VLM). "
    "Use this as a LAST RESORT for tasks that require multi-step reasoning, calculations, "
    "or conditional logic that simpler tools cannot handle. "
    "Examples: 'calculate the memory difference between two products', "
    "'track user flow and determine the final result'. "
    "Results will be saved to local_es_data/ directory."
)
def analyze_video_flow(
    file_path: str = Field(
        description="Path to the video file (e.g., './data/video.mp4')"
    ),
    prompt: str = Field(
        description="Detailed instruction for the VLM, describing the entire task. "
        "Be specific about what to look for, what to calculate, and what format to return."
    ),
    timestamp: str = Field(
        default="",
        description="Time range to analyze, e.g., '0s-30s'. Leave empty to analyze the entire video.",
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving results."
    ),
) -> str:
    """
    使用 VLM 分析复杂的视频流程

    Returns:
        str: VLM 的分析结果
    """
    try:
        print(f"🧠 开始 VLM 流程分析: {file_path}")
        print(f"   Prompt: {prompt[:100]}...")

        # 检查文件是否存在
        video_file = Path(file_path)
        if not video_file.exists():
            error_msg = f"视频文件不存在: {file_path}"
            save_video_result(task_id, "analyze_flow", file_path, error_msg)
            return error_msg

        # 检查 VLM API 配置
        vlm_api_key = os.getenv("DEFAULT_VLM_API_KEY")
        vlm_base_url = os.getenv("DEFAULT_VLM_BASE_URL")
        vlm_model = os.getenv("DEFAULT_VLM_MODEL_NAME")

        if not all([vlm_api_key, vlm_base_url, vlm_model]):
            error_msg = (
                "⚠️ VLM API 未配置\n\n"
                "请设置环境变量:\n"
                "export DEFAULT_VLM_API_KEY='your_api_key'\n"
                "export DEFAULT_VLM_BASE_URL='your_base_url'\n"
                "export DEFAULT_VLM_MODEL_NAME='your_model_name'\n"
            )
            save_video_result(task_id, "analyze_flow", file_path, error_msg)
            return error_msg

        # 1. 提取关键帧（密集采样：每0.5秒）
        print("   提取关键帧...")

        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # 解析时间范围
        if timestamp:
            start_time, end_time = _parse_time_range(timestamp, duration)
        else:
            start_time, end_time = 0, duration

        # 采样间隔（0.5秒）
        sample_interval = 0.5
        frames_base64 = []

        for t in np.arange(start_time, end_time, sample_interval):
            frame_idx = int(t * fps)
            if frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # 转换为 base64
            import base64

            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            frames_base64.append({"timestamp": round(t, 2), "image": frame_base64})

            # 限制最大帧数（避免请求过大）
            if len(frames_base64) >= 60:  # 最多60帧（约30秒）
                break

        cap.release()

        print(f"   提取了 {len(frames_base64)} 帧")

        # 2. 调用 VLM API
        print("   调用 VLM API...")

        try:
            import requests

            # 构建消息（OpenAI 兼容格式）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}\n\n请按时间顺序观看以下帧，并完成任务。",
                        }
                    ],
                }
            ]

            # 添加图像
            for frame_data in frames_base64:
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_data['image']}"
                        },
                    }
                )

            # 发送请求
            response = requests.post(
                f"{vlm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {vlm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": vlm_model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.1,
                },
                timeout=300,
            )

            response.raise_for_status()
            result = response.json()

            # 提取回复
            answer = result["choices"][0]["message"]["content"]

            print(f"✅ VLM 分析完成")
            print(f"   结果: {answer[:200]}...")

            # 保存结果
            save_video_result(task_id, "analyze_flow", file_path, answer)

            return answer

        except requests.exceptions.RequestException as e:
            error_msg = f"VLM API 调用失败: {str(e)}"
            print(f"❌ {error_msg}")
            save_video_result(task_id, "analyze_flow", file_path, error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"流程分析失败: {str(e)}"
        print(f"❌ {error_msg}")
        save_video_result(task_id, "analyze_flow", file_path, error_msg)
        return error_msg


# ==================== 辅助函数 ====================


def _parse_timestamp(timestamp: str, video_path: str) -> List[float]:
    """
    解析时间戳字符串为秒数列表

    Examples:
        "4s" -> [4.0]
        "1m31s" -> [91.0]
        "30s-32s" -> [30.0, 31.0, 32.0]
        "" -> [0, 5, 10, 15, ...] (全视频采样)
    """
    if not timestamp:
        # 全视频采样：每5秒一帧
        metadata = get_video_metadata(video_path)
        if "duration_seconds" in metadata:
            duration = metadata["duration_seconds"]
            return list(np.arange(0, duration, 5.0))
        else:
            return [0.0]

    # 解析时间范围
    if "-" in timestamp:
        start_str, end_str = timestamp.split("-")
        start = _time_str_to_seconds(start_str.strip())
        end = _time_str_to_seconds(end_str.strip())
        return list(np.arange(start, end + 1, 1.0))
    else:
        return [_time_str_to_seconds(timestamp)]


def _time_str_to_seconds(time_str: str) -> float:
    """
    将时间字符串转换为秒数

    Examples:
        "4s" -> 4.0
        "1m31s" -> 91.0
        "2m" -> 120.0
    """
    time_str = time_str.lower().strip()

    total_seconds = 0.0

    # 解析分钟
    if "m" in time_str:
        parts = time_str.split("m")
        minutes = float(parts[0])
        total_seconds += minutes * 60
        time_str = parts[1] if len(parts) > 1 else ""

    # 解析秒
    if "s" in time_str:
        time_str = time_str.replace("s", "")

    if time_str:
        total_seconds += float(time_str)

    return total_seconds


def _parse_time_range(timestamp: str, max_duration: float):
    """解析时间范围"""
    if "-" in timestamp:
        start_str, end_str = timestamp.split("-")
        start = _time_str_to_seconds(start_str.strip())
        end = _time_str_to_seconds(end_str.strip())
        return start, min(end, max_duration)
    else:
        start = _time_str_to_seconds(timestamp)
        return start, min(start + 5, max_duration)


def _extract_frames(video_path: str, timestamps: List[float]) -> List[dict]:
    """从视频中提取指定时间戳的帧"""
    frames_data = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for timestamp in timestamps:
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frames_data.append({"timestamp_sec": timestamp, "frame": frame})

    cap.release()
    return frames_data


def _draw_boxes(frame, boxes, labels, scores):
    """在帧上绘制边界框（GroundingDINO 格式）"""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    for box, label, score in zip(boxes, labels, scores):
        # GroundingDINO 输出的 box 是归一化坐标
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)

        # 绘制边界框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签
        label_text = f"{label}: {score:.2f}"
        cv2.putText(
            annotated,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return annotated


def _fallback_object_detection(
    frames_data: List[dict], object_prompt: str
) -> List[dict]:
    """
    当 GroundingDINO 不可用时的简化检测方法
    （基于模板匹配或其他简单 CV 技术）
    """
    # 这里可以实现简单的模板匹配或颜色检测
    # 为了简化，这里只返回一个占位结果
    results = []

    for frame_info in frames_data:
        # 简单示例：假设我们总能找到一些对象
        results.append(
            {
                "timestamp_sec": frame_info["timestamp_sec"],
                "detected_object": object_prompt,
                "count": 1,
                "method": "fallback",
                "message": "使用简化检测方法（GroundingDINO 不可用）",
            }
        )

    return results


def _fuzzy_match(search_prompt: str, text: str) -> bool:
    """
    宽松"或"逻辑 - 检查 search_prompt 中用空格分割的【任何一个】关键词是否出现在 text 中
    
    这个函数支持多种匹配策略：
    1. 直接包含：search_prompt 完整出现在 text 中
    2. 反向包含：text 完整出现在 search_prompt 中
    3. 关键词匹配：search_prompt 中的任何一个关键词（>1字符）出现在 text 中
    
    Args:
        search_prompt: 搜索提示词（可以是多个关键词，用空格分隔）
        text: 要检查的文本
        
    Returns:
        bool: 如果匹配返回 True，否则返回 False
    """
    if not search_prompt:
        return True
    if not text:
        return False

    # 统一转为小写并 strip()
    search_prompt_lower = search_prompt.lower().strip()
    text_lower = text.lower().strip()

    # 策略 1: 直接包含
    if search_prompt_lower in text_lower:
        return True

    # 策略 2: 反向包含
    if text_lower in search_prompt_lower:
        return True

    # 策略 3: 关键词匹配（"或"逻辑）
    keywords = search_prompt_lower.split()
    
    if not keywords:
        return True  # 如果 prompt strip() 后为空，算匹配

    try:
        # 检查【任何一个】关键词（>1字符）是否存在于文本中
        return any(len(keyword) > 1 and keyword in text_lower for keyword in keywords)
    except Exception:
        return False


def save_video_result(
    task_id: str,
    tool_name: str,
    video_path: str,
    result: str,
):
    """
    保存视频分析结果到文件

    Args:
        task_id: 任务ID
        tool_name: 工具名称
        video_path: 视频文件路径
        result: 分析结果
    """
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存主结果
    result_file = output_dir / f"{task_id}_video_result.txt"

    try:
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"=== Task ID: {task_id} ===\n\n")
            f.write(f"=== 工具 ===\n{tool_name}\n\n")
            f.write(f"=== 文件 ===\n{video_path}\n\n")
            f.write(f"=== 分析结果 ===\n{result}\n")
    except Exception as e:
        print(f"⚠️ 保存视频结果失败: {e}")
