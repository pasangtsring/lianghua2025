"""
WebSocket处理器模块
负责实时数据流处理、断线重连、性能优化
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import ssl
import traceback

from utils.logger import get_logger
from config.config_manager import ConfigManager

class ConnectionStatus(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class MessageType(Enum):
    """消息类型"""
    TICK = "tick"
    KLINE = "kline"
    DEPTH = "depth"
    TRADE = "trade"
    ACCOUNT = "account"
    ORDER = "order"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    type: MessageType
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    raw_data: str

@dataclass
class SubscriptionInfo:
    """订阅信息"""
    symbol: str
    stream: str
    callback: Callable
    is_active: bool = True
    last_message_time: datetime = None
    message_count: int = 0

class WebSocketHandler:
    """WebSocket处理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # WebSocket配置
        self.websocket_config = config_manager.config.trading  # 使用trading配置
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.reconnect_attempts = 5
        self.reconnect_delay = 5
        self.ping_interval = 20
        self.ping_timeout = 10
        self.max_message_size = 1024 * 1024  # 1MB
        
        # 连接状态
        self.status = ConnectionStatus.DISCONNECTED
        self.websocket = None
        self.last_ping_time = None
        self.last_pong_time = None
        
        # 订阅管理
        self.subscriptions: Dict[str, SubscriptionInfo] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # 性能统计
        self.stats = {
            'total_messages': 0,
            'messages_per_second': 0.0,
            'last_message_time': None,
            'connection_time': None,
            'reconnect_count': 0,
            'error_count': 0,
            'avg_latency': 0.0
        }
        
        # 消息缓冲区
        self.message_buffer: List[WebSocketMessage] = []
        self.buffer_size = 1000
        
        # 运行状态
        self.is_running = False
        self.should_stop = False
        
        self.logger.info("WebSocket处理器初始化完成")
    
    async def start(self):
        """启动WebSocket处理器"""
        try:
            self.is_running = True
            self.should_stop = False
            
            # 启动主连接任务
            asyncio.create_task(self.connect_loop())
            
            # 启动统计任务
            asyncio.create_task(self.stats_monitor())
            
            self.logger.info("WebSocket处理器已启动")
            
        except Exception as e:
            self.logger.error(f"启动WebSocket处理器失败: {e}")
            raise
    
    async def stop(self):
        """停止WebSocket处理器"""
        try:
            self.should_stop = True
            
            # 关闭连接
            if self.websocket:
                await self.websocket.close()
            
            self.status = ConnectionStatus.DISCONNECTED
            self.is_running = False
            
            self.logger.info("WebSocket处理器已停止")
            
        except Exception as e:
            self.logger.error(f"停止WebSocket处理器失败: {e}")
    
    async def connect_loop(self):
        """连接循环"""
        reconnect_count = 0
        
        while self.is_running and not self.should_stop:
            try:
                if self.status == ConnectionStatus.DISCONNECTED:
                    await self.connect()
                    reconnect_count = 0
                    
                elif self.status == ConnectionStatus.CONNECTED:
                    await self.handle_messages()
                    
                await asyncio.sleep(0.1)
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket连接已关闭")
                self.status = ConnectionStatus.DISCONNECTED
                reconnect_count += 1
                
                if reconnect_count < self.reconnect_attempts:
                    self.logger.info(f"尝试重连 ({reconnect_count}/{self.reconnect_attempts})")
                    await asyncio.sleep(self.reconnect_delay)
                    self.status = ConnectionStatus.RECONNECTING
                else:
                    self.logger.error("重连次数超过限制，停止重连")
                    self.should_stop = True
                    
            except Exception as e:
                self.logger.error(f"连接循环错误: {e}")
                self.status = ConnectionStatus.ERROR
                self.stats['error_count'] += 1
                await asyncio.sleep(1)
    
    async def connect(self):
        """建立WebSocket连接"""
        try:
            self.status = ConnectionStatus.CONNECTING
            self.logger.info(f"连接到 {self.base_url}")
            
            # SSL上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 建立连接
            self.websocket = await websockets.connect(
                self.base_url,
                ssl=ssl_context,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                max_size=self.max_message_size
            )
            
            self.status = ConnectionStatus.CONNECTED
            self.stats['connection_time'] = datetime.now()
            
            # 重新订阅
            await self.resubscribe()
            
            self.logger.info("WebSocket连接成功")
            
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
            self.status = ConnectionStatus.ERROR
            self.stats['error_count'] += 1
            raise
    
    async def handle_messages(self):
        """处理消息"""
        try:
            if not self.websocket:
                return
            
            # 接收消息
            message = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=30.0
            )
            
            # 解析消息
            await self.process_message(message)
            
        except asyncio.TimeoutError:
            self.logger.warning("消息接收超时")
        except websockets.exceptions.ConnectionClosed:
            raise
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
            self.stats['error_count'] += 1
    
    async def process_message(self, raw_message: str):
        """
        处理原始消息
        
        Args:
            raw_message: 原始消息字符串
        """
        try:
            # 解析JSON
            data = json.loads(raw_message)
            
            # 确定消息类型
            message_type = self.determine_message_type(data)
            
            # 创建消息对象
            ws_message = WebSocketMessage(
                type=message_type,
                symbol=self.extract_symbol(data),
                data=data,
                timestamp=datetime.now(),
                raw_data=raw_message
            )
            
            # 添加到缓冲区
            self.add_to_buffer(ws_message)
            
            # 更新统计
            self.update_stats(ws_message)
            
            # 调用处理器
            await self.route_message(ws_message)
            
        except json.JSONDecodeError:
            self.logger.error(f"JSON解析失败: {raw_message[:100]}...")
            self.stats['error_count'] += 1
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
            self.stats['error_count'] += 1
    
    def determine_message_type(self, data: Dict) -> MessageType:
        """
        确定消息类型
        
        Args:
            data: 消息数据
            
        Returns:
            消息类型
        """
        try:
            # 检查流名称
            if 'stream' in data:
                stream = data['stream']
                if '@ticker' in stream:
                    return MessageType.TICK
                elif '@kline' in stream:
                    return MessageType.KLINE
                elif '@depth' in stream:
                    return MessageType.DEPTH
                elif '@trade' in stream:
                    return MessageType.TRADE
            
            # 检查事件类型
            if 'e' in data:
                event_type = data['e']
                if event_type == 'outboundAccountPosition':
                    return MessageType.ACCOUNT
                elif event_type == 'executionReport':
                    return MessageType.ORDER
            
            # 检查错误
            if 'error' in data:
                return MessageType.ERROR
            
            return MessageType.HEARTBEAT
            
        except Exception as e:
            self.logger.error(f"确定消息类型失败: {e}")
            return MessageType.ERROR
    
    def extract_symbol(self, data: Dict) -> str:
        """
        提取交易品种
        
        Args:
            data: 消息数据
            
        Returns:
            交易品种
        """
        try:
            # 从流名称提取
            if 'stream' in data:
                stream = data['stream']
                if '@' in stream:
                    return stream.split('@')[0].upper()
            
            # 从数据中提取
            if 'data' in data:
                message_data = data['data']
                if 's' in message_data:
                    return message_data['s']
                elif 'S' in message_data:
                    return message_data['S']
            
            # 直接从根数据提取
            if 's' in data:
                return data['s']
            
            return "UNKNOWN"
            
        except Exception as e:
            self.logger.error(f"提取交易品种失败: {e}")
            return "UNKNOWN"
    
    def add_to_buffer(self, message: WebSocketMessage):
        """
        添加消息到缓冲区
        
        Args:
            message: WebSocket消息
        """
        try:
            self.message_buffer.append(message)
            
            # 保持缓冲区大小
            if len(self.message_buffer) > self.buffer_size:
                self.message_buffer = self.message_buffer[-self.buffer_size:]
                
        except Exception as e:
            self.logger.error(f"添加消息到缓冲区失败: {e}")
    
    def update_stats(self, message: WebSocketMessage):
        """
        更新统计信息
        
        Args:
            message: WebSocket消息
        """
        try:
            self.stats['total_messages'] += 1
            self.stats['last_message_time'] = message.timestamp
            
            # 计算延迟（如果有时间戳）
            if 'E' in message.data:
                server_time = message.data['E']
                latency = message.timestamp.timestamp() * 1000 - server_time
                
                # 更新平均延迟
                if self.stats['avg_latency'] == 0:
                    self.stats['avg_latency'] = latency
                else:
                    self.stats['avg_latency'] = (self.stats['avg_latency'] * 0.9 + latency * 0.1)
            
            # 更新订阅统计
            symbol = message.symbol
            if symbol in self.subscriptions:
                subscription = self.subscriptions[symbol]
                subscription.message_count += 1
                subscription.last_message_time = message.timestamp
                
        except Exception as e:
            self.logger.error(f"更新统计失败: {e}")
    
    async def route_message(self, message: WebSocketMessage):
        """
        路由消息到处理器
        
        Args:
            message: WebSocket消息
        """
        try:
            # 全局处理器
            if message.type in self.message_handlers:
                handler = self.message_handlers[message.type]
                await handler(message)
            
            # 订阅回调
            symbol = message.symbol
            if symbol in self.subscriptions:
                subscription = self.subscriptions[symbol]
                if subscription.is_active and subscription.callback:
                    await subscription.callback(message)
                    
        except Exception as e:
            self.logger.error(f"路由消息失败: {e}")
    
    async def subscribe(self, symbol: str, stream: str, callback: Callable):
        """
        订阅数据流
        
        Args:
            symbol: 交易品种
            stream: 数据流名称
            callback: 回调函数
        """
        try:
            # 创建订阅信息
            subscription = SubscriptionInfo(
                symbol=symbol,
                stream=stream,
                callback=callback
            )
            
            self.subscriptions[symbol] = subscription
            
            # 发送订阅消息
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@{stream}"],
                "id": int(time.time())
            }
            
            if self.websocket and self.status == ConnectionStatus.CONNECTED:
                await self.websocket.send(json.dumps(subscribe_message))
                self.logger.info(f"订阅成功: {symbol}@{stream}")
            else:
                self.logger.warning(f"连接未就绪，稍后重新订阅: {symbol}@{stream}")
                
        except Exception as e:
            self.logger.error(f"订阅失败: {e}")
    
    async def unsubscribe(self, symbol: str):
        """
        取消订阅
        
        Args:
            symbol: 交易品种
        """
        try:
            if symbol in self.subscriptions:
                subscription = self.subscriptions[symbol]
                subscription.is_active = False
                
                # 发送取消订阅消息
                unsubscribe_message = {
                    "method": "UNSUBSCRIBE",
                    "params": [f"{symbol.lower()}@{subscription.stream}"],
                    "id": int(time.time())
                }
                
                if self.websocket and self.status == ConnectionStatus.CONNECTED:
                    await self.websocket.send(json.dumps(unsubscribe_message))
                    self.logger.info(f"取消订阅成功: {symbol}")
                
                del self.subscriptions[symbol]
                
        except Exception as e:
            self.logger.error(f"取消订阅失败: {e}")
    
    async def resubscribe(self):
        """重新订阅所有活跃的流"""
        try:
            for symbol, subscription in self.subscriptions.items():
                if subscription.is_active:
                    subscribe_message = {
                        "method": "SUBSCRIBE",
                        "params": [f"{symbol.lower()}@{subscription.stream}"],
                        "id": int(time.time())
                    }
                    
                    await self.websocket.send(json.dumps(subscribe_message))
                    self.logger.info(f"重新订阅: {symbol}@{subscription.stream}")
                    
        except Exception as e:
            self.logger.error(f"重新订阅失败: {e}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """
        注册消息处理器
        
        Args:
            message_type: 消息类型
            handler: 处理器函数
        """
        try:
            self.message_handlers[message_type] = handler
            self.logger.info(f"注册消息处理器: {message_type.value}")
            
        except Exception as e:
            self.logger.error(f"注册消息处理器失败: {e}")
    
    def unregister_message_handler(self, message_type: MessageType):
        """
        取消注册消息处理器
        
        Args:
            message_type: 消息类型
        """
        try:
            if message_type in self.message_handlers:
                del self.message_handlers[message_type]
                self.logger.info(f"取消注册消息处理器: {message_type.value}")
                
        except Exception as e:
            self.logger.error(f"取消注册消息处理器失败: {e}")
    
    async def stats_monitor(self):
        """统计监控"""
        last_message_count = 0
        
        while self.is_running:
            try:
                # 计算每秒消息数
                current_count = self.stats['total_messages']
                self.stats['messages_per_second'] = current_count - last_message_count
                last_message_count = current_count
                
                # 检查连接健康状态
                if self.stats['last_message_time']:
                    time_since_last = datetime.now() - self.stats['last_message_time']
                    if time_since_last > timedelta(seconds=60):
                        self.logger.warning(f"长时间未收到消息: {time_since_last}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"统计监控失败: {e}")
                await asyncio.sleep(1)
    
    def get_message_buffer(self, message_type: MessageType = None, 
                          symbol: str = None, limit: int = 100) -> List[WebSocketMessage]:
        """
        获取消息缓冲区
        
        Args:
            message_type: 消息类型过滤
            symbol: 交易品种过滤
            limit: 限制数量
            
        Returns:
            消息列表
        """
        try:
            messages = self.message_buffer
            
            # 过滤消息类型
            if message_type:
                messages = [msg for msg in messages if msg.type == message_type]
            
            # 过滤交易品种
            if symbol:
                messages = [msg for msg in messages if msg.symbol == symbol]
            
            # 限制数量
            if limit > 0:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            self.logger.error(f"获取消息缓冲区失败: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'status': self.status.value,
            'total_messages': self.stats['total_messages'],
            'messages_per_second': self.stats['messages_per_second'],
            'last_message_time': self.stats['last_message_time'].isoformat() if self.stats['last_message_time'] else None,
            'connection_time': self.stats['connection_time'].isoformat() if self.stats['connection_time'] else None,
            'reconnect_count': self.stats['reconnect_count'],
            'error_count': self.stats['error_count'],
            'avg_latency': self.stats['avg_latency'],
            'active_subscriptions': len([s for s in self.subscriptions.values() if s.is_active]),
            'total_subscriptions': len(self.subscriptions)
        }
    
    def get_subscription_info(self) -> Dict:
        """获取订阅信息"""
        try:
            info = {}
            for symbol, subscription in self.subscriptions.items():
                info[symbol] = {
                    'stream': subscription.stream,
                    'is_active': subscription.is_active,
                    'message_count': subscription.message_count,
                    'last_message_time': subscription.last_message_time.isoformat() if subscription.last_message_time else None
                }
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取订阅信息失败: {e}")
            return {}
    
    def get_status(self) -> Dict:
        """获取处理器状态"""
        return {
            'is_running': self.is_running,
            'connection_status': self.status.value,
            'websocket_connected': self.websocket is not None and not self.websocket.closed,
            'active_subscriptions': len([s for s in self.subscriptions.values() if s.is_active]),
            'message_buffer_size': len(self.message_buffer),
            'total_messages': self.stats['total_messages'],
            'error_count': self.stats['error_count']
        } 