"""
AudioAgent 工具包：音频转录、音乐识别和人声分离

职责：提供高级音频分析功能，支持 ASR 转录、音乐识别和人声分离。

1. separate_vocals(audio_path: str, task_id: str) -> str
   - 描述: 将音频中的人声与伴奏分离，生成只包含人声的音频文件
   - 实现: 使用 Demucs 模型 (mdx_extra)
   - 参数:
     - audio_path: 音频文件路径（.mp3, .wav等）
     - task_id: 任务ID，用于保存结果
   - 返回: 分离后的人声文件路径

2. transcribe(audio_path: str, language_hint: str, task_id: str, use_vocals_separation: bool) -> str
   - 描述: 将音频转录为文本，使用 faster-whisper 模型
   - 增强: 支持自动人声分离功能（提升歌词转录准确度）
   - 参数:
     - audio_path: 音频文件路径（.mp3, .wav等）
     - language_hint: 语言提示（如 'zh', 'en'），可选
     - task_id: 任务ID，用于保存结果
     - use_vocals_separation: 是否先分离人声再转录（默认False）
   - 返回: 转录的文本

3. identify_music(audio_path: str, task_id: str) -> str
   - 描述: 识别音频中的音乐（听歌识曲），使用 ACRCloud API
   - 参数:
     - audio_path: 音频文件路径
     - task_id: 任务ID，用于保存结果
   - 返回: 歌曲信息（JSON格式）或识别失败信息
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional
from pydantic import Field
from oxygent.oxy import FunctionHub

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
audio_tools = FunctionHub(name="audio_tools")

@audio_tools.tool(
    description="Separate vocals (human voice) from accompaniment (instrumentals) in an audio file. Supports mp3 and wav format. Use this tool when you need to isolate a singer's voice from a song before transcription. This significantly improves lyric transcription accuracy. Returns the file path to the separated vocals-only audio file."
)
def separate_vocals(
    audio_path: str = Field(
        description="Path to the input audio file (e.g., './data/song.mp3')"
    ),
    input_file_type: str = Field(
        description="Separates vocals (human voice) from the accompaniment (instrumentals) in an audio file. Use this tool when a user wants to isolate a singer's voice from a song or remove the music. Input is the path to the original audio file. Returns a string message containing the file path to the new vocals-only audio file.",
        default="mp3",
    ),
    task_id: str = Field(
        description="Unique identifier for this task. Used for saving results."
    ),
) -> str:
    """
    用于将输入音频文件中的人声与伴奏分离开来，并得到分离后的人声的音频文件路径。
    """
    # 尝试导入 demucs
    import demucs.separate
    
    print(f"🎵 开始分离人声: {audio_path}")

    demucs.separate.main(
        [
            f"--{input_file_type}",
            "--two-stems",
            "vocals",
            "-n",
            "mdx_extra",
            audio_path,
            "-o",
            "local_es_data/separated",
        ]
    )
    
    # Demucs 输出结构: separated/mdx_extra/<filename>/vocals.mp3
    vocals_path = f"local_es_data/separated/mdx_extra/{os.path.basename(audio_path).split('.')[0]}/vocals.mp3"
    
    # 复制到 local_es_data 目录（方便管理）
    final_vocals_path = f"local_es_data/{task_id}_vocals.mp3"
    shutil.copy2(vocals_path, final_vocals_path)
    save_audio_result(
        task_id, "separate_vocals", audio_path,
        f"分离成功，人声文件路径: {final_vocals_path}"
    )
    
    return str(final_vocals_path)


@audio_tools.tool(
    description="Transcribe audio file to text using automatic speech recognition (ASR). "
    "Supports multiple languages and can handle music lyrics, speech, etc. "
    "**Enhanced Feature**: Can automatically separate vocals before transcription to improve lyric accuracy. "
    "Set 'use_vocals_separation=True' when transcribing songs. "
    "Results will be saved to local_es_data/ directory."
)
async def transcribe(
    audio_path: str = Field(
        description="Path to the audio file (e.g., './data/audio.mp3')"
    ),
    language_hint: str = Field(
        default="",
        description="Optional language hint (e.g., 'zh', 'en', 'zh,en'). Leave empty for auto-detection.",
    ),
    input_file_type: str = Field(
        description="Separates vocals (human voice) from the accompaniment (instrumentals) in an audio file. Use this tool when a user wants to isolate a singer's voice from a song or remove the music. Input is the path to the original audio file. Returns a string message containing the file path to the new vocals-only audio file.",
        default="mp3",
    ),
    task_id: str = Field(
        default="",
        description="Unique identifier for this task. Used for saving results.",
    ),
    use_vocals_separation: bool = Field(
        default=False,
        description="Whether to separate vocals before transcription. "
        "Set to True when transcribing song lyrics to improve accuracy.",
    ),
) -> str:
    """
    使用 faster-whisper 转录音频为文本。
    支持自动人声分离功能（提升歌词转录准确度）。
    """
    # 创建 local_es_data 目录
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否存在
    audio_file = Path(audio_path)
    if not audio_file.exists():
        error_msg = f"❌ 音频文件不存在: {audio_path}"
        if task_id:
            save_audio_result(task_id, "transcribe", audio_path, error_msg)
        return error_msg
    
    # 如果启用人声分离，先分离人声
    original_audio_path = audio_path
    if use_vocals_separation:
        print("🎤 检测到音乐场景，启用人声分离...")
        vocals_path = await separate_vocals(audio_path, input_file_type, task_id)
        print(f"✅ 人声分离成功，使用人声文件进行转录: {vocals_path}")
        audio_file = Path(vocals_path)
    
    
    from faster_whisper import WhisperModel
    
    print(f"🎵 开始转录音频: {audio_file}")
    print("📥 加载 Whisper 模型（large-v2）...")
    model = WhisperModel("large-v2", device="auto", compute_type="auto")
    
    # 判断是否多语言
    multilingual = False
    if language_hint and ',' in language_hint:
        multilingual = True
    
    # 转录音频
    print("🔄 正在转录...")
    segments, info = model.transcribe(
        str(audio_file),
        # vad_filter=True,  # 使用 VAD 过滤静音
        multilingual=multilingual,  # 对每个片段执行语言检测
        beam_size=5,
    )
    
    # 收集转录结果
    transcript_lines = []
    full_text = []
    
    for segment in segments:
        text = segment.text.strip()
        full_text.append(text)
        
        # 带时间戳的转录
        transcript_lines.append(
            f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {text}"
        )
    
    # 完整文本
    complete_text = " ".join(full_text)
    
    # 结果
    result = {
        "language": info.language,
        "language_probability": round(info.language_probability, 2),
        "duration": round(info.duration, 2),
        "transcript": complete_text,
        "segments_count": len(transcript_lines),
        "vocals_separated": use_vocals_separation,
        "original_file": original_audio_path,
        "transcribed_file": str(audio_file),
    }
    
    # 保存结果
    if task_id:
        save_audio_result(
            task_id, "transcribe", original_audio_path,
            json.dumps(result, ensure_ascii=False, indent=2),
            complete_text,
            "\n".join(transcript_lines)
        )
    
    print(f"✅ 转录完成，共 {len(full_text)} 个片段")
    
    # 返回完整文本
    return complete_text


@audio_tools.tool(
    description="Identify music in an audio file (like Shazam). "
    "Returns song metadata (title, artist, album) if successful. "
    "Results will be saved to local_es_data/ directory."
)
def identify_music(
    audio_path: str = Field(
        description="Path to the audio file (e.g., './data/music.mp3')"
    ),
    task_id: str = Field(
        default="",
        description="Unique identifier for this task. Used for saving results."
    ),
) -> str:
    """
    识别音频中的音乐（听歌识曲）
    
    Args:
        audio_path: 音频文件路径
        task_id: 任务唯一标识符
    
    Returns:
        str: 歌曲信息（JSON格式）或识别失败信息
    """
    # 创建 local_es_data 目录
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否存在
    audio_file = Path(audio_path)
    if not audio_file.exists():
        error_msg = f"❌ 音频文件不存在: {audio_path}"
        if task_id:
            save_audio_result(task_id, "identify_music", audio_path, error_msg)
        return error_msg
    
    try:
        # 尝试使用 ACRCloud API
        import base64
        import hmac
        import hashlib
        import time
        import requests
        
        print(f"🎵 开始识别音乐: {audio_path}")
        
        # ACRCloud 配置（需要环境变量）
        access_key = os.getenv("ACRCLOUD_ACCESS_KEY")
        access_secret = os.getenv("ACRCLOUD_ACCESS_SECRET")
        host = os.getenv("ACRCLOUD_HOST", "identify-ap-southeast-1.acrcloud.com")
        
        if not access_key or not access_secret:
            error_msg = (
                "⚠️ ACRCloud API 未配置\n\n"
                "请设置环境变量:\n"
                "export ACRCLOUD_ACCESS_KEY='your_access_key'\n"
                "export ACRCLOUD_ACCESS_SECRET='your_access_secret'\n"
                "export ACRCLOUD_HOST='your_host'  # 可选\n\n"
                "注册地址: https://www.acrcloud.com/"
            )
            if task_id:
                save_audio_result(task_id, "identify_music", audio_path, error_msg)
            return error_msg
        
        # 读取音频文件
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        # 准备请求
        http_method = "POST"
        http_uri = "/v1/identify"
        data_type = "audio"
        signature_version = "1"
        timestamp = str(int(time.time()))
        
        # 生成签名
        string_to_sign = f"{http_method}\n{http_uri}\n{access_key}\n{data_type}\n{signature_version}\n{timestamp}"
        signature = base64.b64encode(
            hmac.new(
                access_secret.encode('utf-8'),
                string_to_sign.encode('utf-8'),
                hashlib.sha1
            ).digest()
        ).decode('utf-8')
        
        # 构建请求
        files = {'sample': audio_data}
        data = {
            'access_key': access_key,
            'sample_bytes': len(audio_data),
            'timestamp': timestamp,
            'signature': signature,
            'data_type': data_type,
            'signature_version': signature_version
        }

        # 发送请求
        url = f"https://{host}{http_uri}"
        response = requests.post(url, files=files, data=data, timeout=10)
        response.encoding = 'utf-8'
        result = response.json()
        
        # 解析结果
        if result.get('status', {}).get('code') == 0:
            # 识别成功
            metadata = result.get('metadata', {})
            music_list = metadata.get('music', [])
            
            if music_list:
                song = music_list[0]  # 取第一个匹配结果
                
                song_info = {
                    "title": song.get('title', 'Unknown'),
                    "artist": ", ".join([a['name'] for a in song.get('artists', [])]),
                    "album": song.get('album', {}).get('name', 'Unknown'),
                    "release_date": song.get('release_date', 'Unknown'),
                    "score": song.get('score', 0),
                }
                
                output = f"""✅ 音乐识别成功

文件: {audio_path}

歌曲信息:
- 标题: {song_info['title']}
- 艺术家: {song_info['artist']}
- 专辑: {song_info['album']}
- 发行日期: {song_info['release_date']}
- 匹配分数: {song_info['score']}
"""
                
                # 保存结果
                if task_id:
                    save_audio_result(
                        task_id, "identify_music", audio_path,
                        json.dumps(song_info, ensure_ascii=False, indent=2)
                    )
                
                print(f"✅ 识别成功: {song_info['title']} - {song_info['artist']}")
                
                return json.dumps(song_info, ensure_ascii=False)
            else:
                error_msg = "⚠️ 未识别到音乐，可能是语音或噪音"
                if task_id:
                    save_audio_result(task_id, "identify_music", audio_path, error_msg)
                return error_msg
        else:
            error_msg = f"❌ 识别失败: {result.get('status', {}).get('msg', 'Unknown error')}"
            if task_id:
                save_audio_result(task_id, "identify_music", audio_path, error_msg)
            return error_msg
            
    except ImportError as e:
        error_msg = f"❌ 缺少依赖: {str(e)}\n请安装: pip install requests"
        if task_id:
            save_audio_result(task_id, "identify_music", audio_path, error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"❌ 识别失败: {str(e)}"
        if task_id:
            save_audio_result(task_id, "identify_music", audio_path, error_msg)
        return error_msg


# ==================== 辅助函数 ====================

def format_timestamp(seconds: float) -> str:
    """格式化时间戳为 MM:SS 格式"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def save_audio_result(
    task_id: str,
    tool_name: str,
    audio_path: str,
    result: str,
    full_text: str = "",
    segments: str = ""
):
    """
    保存音频分析结果到文件
    
    Args:
        task_id: 任务ID
        tool_name: 工具名称
        audio_path: 音频文件路径
        result: 分析结果
        full_text: 完整转录文本（仅转录时使用）
        segments: 带时间戳的分段（仅转录时使用）
    """
    output_dir = Path("local_es_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存主结果
    result_file = output_dir / f"{task_id}_audio_result.txt"
    
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"=== Task ID: {task_id} ===\n\n")
            f.write(f"=== 工具 ===\n{tool_name}\n\n")
            f.write(f"=== 文件 ===\n{audio_path}\n\n")
            f.write(f"=== 分析结果 ===\n{result}\n")
        
        # 如果是转录，额外保存纯文本
        if tool_name == "transcribe" and full_text:
            text_file = output_dir / f"{task_id}_transcript.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            # 保存带时间戳的版本
            if segments:
                segments_file = output_dir / f"{task_id}_transcript_segments.txt"
                with open(segments_file, "w", encoding="utf-8") as f:
                    f.write(segments)
                    
    except Exception as e:
        print(f"⚠️ 保存音频结果失败: {e}")
