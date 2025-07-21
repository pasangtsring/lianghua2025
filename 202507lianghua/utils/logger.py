"""
日志系统模块
功能：多级日志记录、文件轮转、Telegram通知、性能监控、交易日志
作者：Trading System Team
创建时间：2025-01-28
"""

import logging
import logging.handlers
import os
import json
import time
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor

# 自定义日志级别
class LogLevel(Enum):
    """日志级别枚举"""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    TRADE = 25    # 交易专用级别
    PERFORMANCE = 15  # 性能监控级别

@dataclass
class LogMessage:
    """日志消息数据结构"""
    timestamp: float
    level: str
    logger_name: str
    message: str
    extra_data: Dict[str, Any] = field(default_factory=dict)
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "level": self.level,
            "logger": self.logger_name,
            "message": self.message,
            "module": self.module,
            "function": self.function,
            "line": self.line_number,
            "thread": self.thread_id,
            "extra": self.extra_data
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

class TelegramLogHandler(logging.Handler):
    """Telegram日志处理器"""
    
    def __init__(self, bot_token: str, chat_id: int, level: int = logging.ERROR):
        super().__init__(level)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.session = None
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.message_queue = asyncio.Queue() if asyncio.get_event_loop().is_running() else None
        
        # 消息限制
        self.max_message_length = 4096
        self.rate_limit = 30  # 每分钟最多发送30条消息
        self.message_count = 0
        self.last_reset = time.time()
        
        # 错误级别映射
        self.level_emojis = {
            'CRITICAL': '🚨',
            'ERROR': '❌',
            'WARNING': '⚠️',
            'INFO': 'ℹ️',
            'DEBUG': '🔍',
            'TRADE': '💹',
            'PERFORMANCE': '📊'
        }
    
    def emit(self, record: logging.LogRecord) -> None:
        """发送日志消息"""
        try:
            # 检查速率限制
            if not self._check_rate_limit():
                return
            
            # 格式化消息
            message = self._format_message(record)
            
            # 异步发送
            if self.loop and self.loop.is_running():
                asyncio.create_task(self._send_message_async(message))
            else:
                self.executor.submit(self._send_message_sync, message)
                
        except Exception as e:
            self.handleError(record)
    
    def _check_rate_limit(self) -> bool:
        """检查速率限制"""
        current_time = time.time()
        
        # 每分钟重置计数器
        if current_time - self.last_reset > 60:
            self.message_count = 0
            self.last_reset = current_time
        
        # 检查是否超过限制
        if self.message_count >= self.rate_limit:
            return False
        
        self.message_count += 1
        return True
    
    def _format_message(self, record: logging.LogRecord) -> str:
        """格式化消息"""
        emoji = self.level_emojis.get(record.levelname, '📝')
        
        # 基础信息
        message = f"{emoji} *{record.levelname}* - {record.name}\n"
        message += f"🕒 {datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # 消息内容
        if len(record.getMessage()) > 1000:
            message += f"💬 {record.getMessage()[:1000]}...\n"
        else:
            message += f"💬 {record.getMessage()}\n"
        
        # 位置信息
        if hasattr(record, 'pathname'):
            filename = os.path.basename(record.pathname)
            message += f"📍 {filename}:{record.lineno} in {record.funcName}\n"
        
        # 额外数据
        if hasattr(record, 'extra_data') and record.extra_data:
            extra_str = json.dumps(record.extra_data, ensure_ascii=False, indent=2)
            if len(extra_str) < 500:
                message += f"📊 Data: ```json\n{extra_str}\n```\n"
        
        # 异常信息
        if record.exc_info:
            exc_text = ''.join(traceback.format_exception(*record.exc_info))
            if len(exc_text) < 1000:
                message += f"🔥 Exception: ```\n{exc_text}\n```"
        
        # 截断消息
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length-100] + "\n... (truncated)"
        
        return message
    
    async def _send_message_async(self, message: str) -> None:
        """异步发送消息"""
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        print(f"Telegram发送失败: {response.status}")
                        
        except Exception as e:
            print(f"Telegram异步发送错误: {e}")
    
    def _send_message_sync(self, message: str) -> None:
        """同步发送消息"""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                print(f"Telegram发送失败: {response.status_code}")
                
        except Exception as e:
            print(f"Telegram同步发送错误: {e}")

class JsonFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 创建日志消息对象
        log_msg = LogMessage(
            timestamp=record.created,
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=getattr(record, 'module', ''),
            function=getattr(record, 'funcName', ''),
            line_number=getattr(record, 'lineno', 0),
            thread_id=record.thread,
            extra_data=getattr(record, 'extra_data', {})
        )
        
        # 添加异常信息
        if record.exc_info:
            log_msg.extra_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return log_msg.to_json()

class PerformanceLogHandler(logging.Handler):
    """性能监控日志处理器"""
    
    def __init__(self, metrics_file: str = "logs/performance.json"):
        super().__init__()
        self.metrics_file = metrics_file
        self.metrics_buffer = []
        self.buffer_size = 100
        self.last_flush = time.time()
        self.flush_interval = 60  # 60秒刷新一次
        
        # 创建目录
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    def emit(self, record: logging.LogRecord) -> None:
        """处理性能日志"""
        try:
            if record.levelno == LogLevel.PERFORMANCE.value:
                metric_data = {
                    'timestamp': record.created,
                    'metric': record.getMessage(),
                    'data': getattr(record, 'extra_data', {})
                }
                
                self.metrics_buffer.append(metric_data)
                
                # 检查是否需要刷新
                if (len(self.metrics_buffer) >= self.buffer_size or 
                    time.time() - self.last_flush > self.flush_interval):
                    self._flush_metrics()
                    
        except Exception as e:
            self.handleError(record)
    
    def _flush_metrics(self) -> None:
        """刷新性能指标到文件"""
        try:
            if not self.metrics_buffer:
                return
            
            # 读取现有数据
            existing_data = []
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            existing_data.append(json.loads(line))
            
            # 合并新数据
            all_data = existing_data + self.metrics_buffer
            
            # 保留最近1000条记录
            if len(all_data) > 1000:
                all_data = all_data[-1000:]
            
            # 写入文件
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 清空缓冲区
            self.metrics_buffer.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            print(f"性能指标刷新失败: {e}")

class TradingLogger:
    """交易日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建专门的交易日志记录器
        self.trade_logger = logging.getLogger("trading")
        self.trade_logger.setLevel(LogLevel.TRADE.value)
        
        # 设置交易日志文件处理器
        trade_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trades.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        trade_handler.setFormatter(JsonFormatter())
        self.trade_logger.addHandler(trade_handler)
        
        # 防止重复日志
        self.trade_logger.propagate = False
    
    def log_signal(self, symbol: str, signal_type: str, data: Dict[str, Any]) -> None:
        """记录交易信号"""
        extra_data = {
            'type': 'signal',
            'symbol': symbol,
            'signal_type': signal_type,
            'data': data
        }
        
        self.trade_logger.log(
            LogLevel.TRADE.value,
            f"Signal Generated: {signal_type} for {symbol}",
            extra={'extra_data': extra_data}
        )
    
    def log_order(self, symbol: str, order_type: str, data: Dict[str, Any]) -> None:
        """记录订单"""
        extra_data = {
            'type': 'order',
            'symbol': symbol,
            'order_type': order_type,
            'data': data
        }
        
        self.trade_logger.log(
            LogLevel.TRADE.value,
            f"Order {order_type}: {symbol}",
            extra={'extra_data': extra_data}
        )
    
    def log_position(self, symbol: str, action: str, data: Dict[str, Any]) -> None:
        """记录持仓变化"""
        extra_data = {
            'type': 'position',
            'symbol': symbol,
            'action': action,
            'data': data
        }
        
        self.trade_logger.log(
            LogLevel.TRADE.value,
            f"Position {action}: {symbol}",
            extra={'extra_data': extra_data}
        )
    
    def log_performance(self, period: str, metrics: Dict[str, Any]) -> None:
        """记录性能指标"""
        extra_data = {
            'type': 'performance',
            'period': period,
            'metrics': metrics
        }
        
        self.trade_logger.log(
            LogLevel.TRADE.value,
            f"Performance Report: {period}",
            extra={'extra_data': extra_data}
        )

class LoggerManager:
    """日志管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loggers = {}
        self.handlers = {}
        self.trading_logger = None
        self.performance_handler = None
        self.telegram_handler = None
        
        # 注册自定义日志级别
        self._register_custom_levels()
        
        # 初始化日志系统
        self._setup_logging()
    
    def _register_custom_levels(self) -> None:
        """注册自定义日志级别"""
        # 注册TRADE级别
        logging.addLevelName(LogLevel.TRADE.value, "TRADE")
        def trade(self, message, *args, **kwargs):
            if self.isEnabledFor(LogLevel.TRADE.value):
                self._log(LogLevel.TRADE.value, message, args, **kwargs)
        logging.Logger.trade = trade
        
        # 注册PERFORMANCE级别
        logging.addLevelName(LogLevel.PERFORMANCE.value, "PERFORMANCE")
        def performance(self, message, *args, **kwargs):
            if self.isEnabledFor(LogLevel.PERFORMANCE.value):
                self._log(LogLevel.PERFORMANCE.value, message, args, **kwargs)
        logging.Logger.performance = performance
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        # 创建日志目录
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.get('log_level', 'INFO'))
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "system.log",
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
        
        # 错误文件处理器
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(error_handler)
        
        # 性能监控处理器
        self.performance_handler = PerformanceLogHandler(
            str(log_dir / "performance.json")
        )
        root_logger.addHandler(self.performance_handler)
        
        # Telegram处理器
        telegram_token = self.config.get('telegram_bot_token')
        telegram_chat_id = self.config.get('telegram_chat_id')
        
        if telegram_token and telegram_chat_id:
            self.telegram_handler = TelegramLogHandler(
                telegram_token, 
                int(telegram_chat_id),
                level=logging.WARNING
            )
            root_logger.addHandler(self.telegram_handler)
        
        # 交易日志记录器
        self.trading_logger = TradingLogger(str(log_dir))
        
        # 保存处理器引用
        self.handlers['console'] = console_handler
        self.handlers['file'] = file_handler
        self.handlers['error'] = error_handler
        self.handlers['performance'] = self.performance_handler
        self.handlers['telegram'] = self.telegram_handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取日志记录器"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            # 添加额外的上下文信息
            logger = logging.LoggerAdapter(logger, {})
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def log_trade_signal(self, symbol: str, signal_type: str, data: Dict[str, Any]) -> None:
        """记录交易信号"""
        if self.trading_logger:
            self.trading_logger.log_signal(symbol, signal_type, data)
    
    def log_trade_order(self, symbol: str, order_type: str, data: Dict[str, Any]) -> None:
        """记录交易订单"""
        if self.trading_logger:
            self.trading_logger.log_order(symbol, order_type, data)
    
    def log_position_change(self, symbol: str, action: str, data: Dict[str, Any]) -> None:
        """记录持仓变化"""
        if self.trading_logger:
            self.trading_logger.log_position(symbol, action, data)
    
    def log_performance_metrics(self, period: str, metrics: Dict[str, Any]) -> None:
        """记录性能指标"""
        if self.trading_logger:
            self.trading_logger.log_performance(period, metrics)
    
    def log_system_performance(self, metric_name: str, value: float, extra: Dict[str, Any] = None) -> None:
        """记录系统性能"""
        logger = self.get_logger('system.performance')
        logger.info(
            f"PERFORMANCE - {metric_name}: {value}",
            extra={'extra_data': {'metric': metric_name, 'value': value, 'extra': extra or {}}}
        )
    
    def set_log_level(self, level: Union[str, int]) -> None:
        """设置日志级别"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # 更新所有处理器的级别
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(max(level, logging.INFO))
    
    def close(self) -> None:
        """关闭日志系统"""
        # 刷新所有处理器
        for handler in self.handlers.values():
            if handler:
                handler.flush()
                handler.close()
        
        # 关闭交易日志记录器
        if self.trading_logger:
            for handler in self.trading_logger.trade_logger.handlers:
                handler.flush()
                handler.close()

# 全局日志管理器
logger_manager = None

def init_logger(config: Optional[Dict[str, Any]] = None) -> LoggerManager:
    """初始化日志系统"""
    global logger_manager
    if logger_manager is None:
        logger_manager = LoggerManager(config)
    return logger_manager

def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    if logger_manager is None:
        init_logger()
    return logger_manager.get_logger(name)

def log_trade_signal(symbol: str, signal_type: str, data: Dict[str, Any]) -> None:
    """记录交易信号"""
    if logger_manager:
        logger_manager.log_trade_signal(symbol, signal_type, data)

def log_trade_order(symbol: str, order_type: str, data: Dict[str, Any]) -> None:
    """记录交易订单"""
    if logger_manager:
        logger_manager.log_trade_order(symbol, order_type, data)

def log_position_change(symbol: str, action: str, data: Dict[str, Any]) -> None:
    """记录持仓变化"""
    if logger_manager:
        logger_manager.log_position_change(symbol, action, data)

def log_performance_metrics(period: str, metrics: Dict[str, Any]) -> None:
    """记录性能指标"""
    if logger_manager:
        logger_manager.log_performance_metrics(period, metrics)

def log_system_performance(metric_name: str, value: float, extra: Dict[str, Any] = None) -> None:
    """记录系统性能"""
    if logger_manager:
        logger_manager.log_system_performance(metric_name, value, extra)

# 性能监控装饰器
def performance_monitor(func):
    """性能监控装饰器"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            log_system_performance(
                f"function.{func.__name__}.execution_time",
                execution_time,
                {
                    'module': func.__module__,
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_system_performance(
                f"function.{func.__name__}.error_time",
                execution_time,
                {
                    'module': func.__module__,
                    'function': func.__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise
    
    return wrapper

if __name__ == "__main__":
    # 测试日志系统
    logger_manager = init_logger({
        'log_level': 'DEBUG',
        'log_dir': 'logs',
        'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
        'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID')
    })
    
    # 测试各种日志级别
    logger = get_logger('test')
    
    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    
    # 测试交易日志
    log_trade_signal("BTCUSDT", "LONG", {
        "price": 45000,
        "confidence": 0.85,
        "macd_divergence": True
    })
    
    # 测试性能日志
    log_system_performance("api_latency", 0.15, {"endpoint": "/api/v1/klines"})
    
    # 测试装饰器
    @performance_monitor
    def test_function():
        time.sleep(0.1)
        return "测试完成"
    
    result = test_function()
    print(f"测试结果: {result}")
    
    # 关闭日志系统
    logger_manager.close() 