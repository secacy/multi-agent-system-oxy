"""
日志配置模块

提供统一的日志配置，支持：
1. 同时输出到控制台和文件
2. 自动创建日志目录
3. 按日期和时间命名日志文件
4. 彩色终端输出
5. 详细的日志格式
6. 自动解码 bytes 类型的日志消息
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime


class FileDecodingFormatter(logging.Formatter):
    """
    自定义文件格式化器，自动解码 bytes 类型的日志消息.
    """

    def format(self, record):
        # 1. 检查 'msg' (原始消息) 是否为 bytes
        if isinstance(record.msg, bytes):
            try:
                # 尝试用 utf-8 解码，替换无法解码的字符
                record.msg = record.msg.decode("utf-8", errors="replace")
            except Exception:
                pass  # 解码失败，保持原样（repr会处理）

        # 2. 检查 'args' (格式化参数) 是否包含 bytes
        if isinstance(record.args, tuple):
            new_args = []
            for arg in record.args:
                if isinstance(arg, bytes):
                    try:
                        new_args.append(arg.decode("utf-8", errors="replace"))
                    except Exception:
                        new_args.append(arg)  # 失败则添加原始字节
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

        # 3. 调用父类的 format 方法
        return super().format(record)


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于终端输出），并支持解码"""

    # ANSI 颜色代码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # --- 添加解码逻辑 ---
        # 1. 检查 'msg' (原始消息) 是否为 bytes
        if isinstance(record.msg, bytes):
            try:
                record.msg = record.msg.decode("utf-8", errors="replace")
            except Exception:
                pass

        # 2. 检查 'args' (格式化参数) 是否包含 bytes
        if isinstance(record.args, tuple):
            new_args = []
            for arg in record.args:
                if isinstance(arg, bytes):
                    try:
                        new_args.append(arg.decode("utf-8", errors="replace"))
                    except Exception:
                        new_args.append(arg)
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        # --- 解码逻辑结束 ---

        # 保存原始 levelname
        original_levelname = record.levelname

        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        # 调用父类 format
        output = super().format(record)

        # 恢复 record
        record.levelname = original_levelname

        return output


def setup_logger(
    name: str = "oxy-mas",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    capture_oxygent: bool = True,
) -> logging.Logger:
    """
    配置日志系统

    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        level: 控制台的日志级别 (注：文件级别在下面单独设置)
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        capture_oxygent: 是否捕获oxygent框架日志

    Returns:
        配置好的日志记录器
    """

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 根记录器级别设为DEBUG，以捕获所有

    # 清除已有处理器，避免重复
    if root_logger.handlers:
        root_logger.handlers.clear()

    # 获取我们自己的 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 日志格式 (带解码功能)
    file_format = FileDecodingFormatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_format = ColoredFormatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)  # 控制台级别使用传入的 level

        if sys.stdout.isatty():
            console_handler.setFormatter(console_format)
        else:
            console_handler.setFormatter(file_format)

        root_logger.addHandler(console_handler)

    # 2. 文件处理器
    if file_output:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"oxy_mas_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

        # --- V V V 这里是你的要求 V V V ---
        file_handler.setLevel(logging.INFO)  # 文件记录 INFO 及以上级别的日志
        # --- ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ---

        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

        # 这条 info 消息现在可以被 file_handler 记录
        logger.info(f"日志文件创建: {log_file} (级别: INFO)")

    # 3. 配置oxygent框架的日志
    if capture_oxygent:
        oxygent_loggers = [
            "oxygent",
            "oxygent.mas",
            "oxygent.agent",
            "oxygent.oxy",
            "oxygent.oxy.llms",
            "oxygent.oxy.base_oxy",
        ]

        for logger_name in oxygent_loggers:
            ox_logger = logging.getLogger(logger_name)
            ox_logger.setLevel(logging.DEBUG)  # 保持 DEBUG 以便捕获
            ox_logger.propagate = True  # 确保日志传播到 root logger

    return logger


def get_logger(name: str = "oxy-mas") -> logging.Logger:
    """
    获取已配置的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    return logging.getLogger(name)


# 为了兼容现有的print语句，创建一个日志代理类
class LoggerProxy:
    """日志代理类，用于替代print函数"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.original_print = print

    def __call__(self, *args, **kwargs):
        """模拟print函数的行为，同时记录到日志"""
        message = " ".join(str(arg) for arg in args)

        # 根据消息内容判断日志级别
        if "❌" in message or "ERROR" in message or "失败" in message:
            self.logger.error(message)
        elif "⚠️" in message or "WARNING" in message or "警告" in message:
            self.logger.warning(message)
        elif "✅" in message or "成功" in message:
            self.logger.info(message)
        elif "=" * 70 in message or "#" * 70 in message:
            # 分隔符，使用debug级别
            self.logger.debug(message)
        else:
            self.logger.info(message)

        # 注：由于logger已经配置了console handler，这里不需要再print


def enable_logging(log_dir: str = "logs", level: int = logging.INFO):
    """
    启用日志系统（简化版接口）

    Args:
        log_dir: 日志目录
        level: 控制台的日志级别

    Returns:
        日志记录器
    """
    logger = setup_logger(log_dir=log_dir, level=level)
    return logger


if __name__ == "__main__":
    # 测试日志系统
    logger = setup_logger(log_dir="logs", level=logging.DEBUG)
    
    logger.debug("这是一条调试消息")
    logger.info("✅ 这是一条信息消息")
    logger.warning("⚠️ 这是一条警告消息")
    logger.error("❌ 这是一条错误消息")
    logger.critical("这是一条严重错误消息")
    
    print("\n日志系统测试完成！")

