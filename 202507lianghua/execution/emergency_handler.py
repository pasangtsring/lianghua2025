"""
应急处理器模块
负责系统异常处理、紧急止损、风险控制
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

from utils.logger import get_logger
from config.config_manager import ConfigManager

class EmergencyLevel(Enum):
    """应急级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EmergencyType(Enum):
    """应急类型"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    POSITION_RISK = "position_risk"
    ACCOUNT_RISK = "account_risk"
    SYSTEM_ERROR = "system_error"
    MARKET_HALT = "market_halt"
    LIQUIDITY_ISSUE = "liquidity_issue"

class EmergencyAction(Enum):
    """应急动作"""
    STOP_TRADING = "stop_trading"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    REDUCE_POSITION_SIZE = "reduce_position_size"
    CANCEL_ALL_ORDERS = "cancel_all_orders"
    ENABLE_HEDGE = "enable_hedge"
    ALERT_ONLY = "alert_only"
    SYSTEM_SHUTDOWN = "system_shutdown"

@dataclass
class EmergencyEvent:
    """应急事件"""
    id: str
    type: EmergencyType
    level: EmergencyLevel
    description: str
    data: Dict[str, Any]
    timestamp: datetime
    is_resolved: bool = False
    resolution_time: Optional[datetime] = None
    actions_taken: List[EmergencyAction] = None
    
    def __post_init__(self):
        if self.actions_taken is None:
            self.actions_taken = []

@dataclass
class RiskThreshold:
    """风险阈值配置"""
    max_position_loss_pct: float = 5.0  # 单仓位最大亏损百分比
    max_total_loss_pct: float = 10.0    # 总体最大亏损百分比
    max_drawdown_pct: float = 15.0      # 最大回撤百分比
    max_position_size: float = 0.5      # 最大仓位大小
    price_change_threshold: float = 10.0 # 价格变化阈值百分比
    volume_spike_threshold: float = 5.0  # 成交量异常阈值倍数

class EmergencyHandler:
    """应急处理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 风险阈值配置
        self.risk_thresholds = RiskThreshold()
        
        # 应急事件管理
        self.active_events: Dict[str, EmergencyEvent] = {}
        self.event_history: List[EmergencyEvent] = []
        
        # 应急动作映射
        self.action_handlers: Dict[EmergencyAction, Callable] = {}
        
        # 监控状态
        self.is_monitoring = False
        self.emergency_mode = False
        self.last_check_time = datetime.now()
        
        # 统计数据
        self.stats = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_level': {},
            'actions_taken': {},
            'false_alarms': 0,
            'response_time_avg': 0.0
        }
        
        # 外部系统引用
        self.order_executor = None
        self.position_tracker = None
        self.websocket_handler = None
        
        self.logger.info("应急处理器初始化完成")
    
    def set_external_systems(self, order_executor=None, position_tracker=None, websocket_handler=None):
        """设置外部系统引用"""
        self.order_executor = order_executor
        self.position_tracker = position_tracker
        self.websocket_handler = websocket_handler
        self.logger.info("外部系统引用设置完成")
    
    async def start(self):
        """启动应急处理器"""
        try:
            self.is_monitoring = True
            
            # 注册默认动作处理器
            self.register_default_action_handlers()
            
            # 启动监控任务
            asyncio.create_task(self.monitor_system())
            asyncio.create_task(self.monitor_market())
            asyncio.create_task(self.monitor_positions())
            
            self.logger.info("应急处理器已启动")
            
        except Exception as e:
            self.logger.error(f"启动应急处理器失败: {e}")
            raise
    
    async def stop(self):
        """停止应急处理器"""
        try:
            self.is_monitoring = False
            
            # 处理所有活跃事件
            for event_id in list(self.active_events.keys()):
                await self.resolve_event(event_id)
            
            self.logger.info("应急处理器已停止")
            
        except Exception as e:
            self.logger.error(f"停止应急处理器失败: {e}")
    
    def register_default_action_handlers(self):
        """注册默认动作处理器"""
        try:
            self.action_handlers = {
                EmergencyAction.STOP_TRADING: self.handle_stop_trading,
                EmergencyAction.CLOSE_ALL_POSITIONS: self.handle_close_all_positions,
                EmergencyAction.REDUCE_POSITION_SIZE: self.handle_reduce_position_size,
                EmergencyAction.CANCEL_ALL_ORDERS: self.handle_cancel_all_orders,
                EmergencyAction.ENABLE_HEDGE: self.handle_enable_hedge,
                EmergencyAction.ALERT_ONLY: self.handle_alert_only,
                EmergencyAction.SYSTEM_SHUTDOWN: self.handle_system_shutdown
            }
            
            self.logger.info("默认动作处理器注册完成")
            
        except Exception as e:
            self.logger.error(f"注册默认动作处理器失败: {e}")
    
    async def trigger_emergency(self, event_type: EmergencyType, level: EmergencyLevel,
                              description: str, data: Dict[str, Any] = None) -> str:
        """
        触发应急事件
        
        Args:
            event_type: 事件类型
            level: 应急级别
            description: 事件描述
            data: 事件数据
            
        Returns:
            事件ID
        """
        try:
            event_id = f"emergency_{int(time.time() * 1000)}"
            
            # 创建应急事件
            event = EmergencyEvent(
                id=event_id,
                type=event_type,
                level=level,
                description=description,
                data=data or {},
                timestamp=datetime.now()
            )
            
            # 添加到活跃事件
            self.active_events[event_id] = event
            
            # 记录统计
            self.update_stats(event)
            
            # 执行应急响应
            await self.execute_emergency_response(event)
            
            self.logger.error(f"应急事件触发: {event_type.value} - {description}")
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"触发应急事件失败: {e}")
            return ""
    
    async def execute_emergency_response(self, event: EmergencyEvent):
        """
        执行应急响应
        
        Args:
            event: 应急事件
        """
        try:
            # 确定应急动作
            actions = self.determine_emergency_actions(event)
            
            # 执行应急动作
            for action in actions:
                try:
                    if action in self.action_handlers:
                        await self.action_handlers[action](event)
                        event.actions_taken.append(action)
                        self.logger.info(f"执行应急动作: {action.value}")
                    else:
                        self.logger.warning(f"未找到应急动作处理器: {action.value}")
                        
                except Exception as e:
                    self.logger.error(f"执行应急动作失败: {action.value} - {e}")
            
            # 发送通知
            await self.send_emergency_notification(event)
            
        except Exception as e:
            self.logger.error(f"执行应急响应失败: {e}")
    
    def determine_emergency_actions(self, event: EmergencyEvent) -> List[EmergencyAction]:
        """
        确定应急动作
        
        Args:
            event: 应急事件
            
        Returns:
            应急动作列表
        """
        try:
            actions = []
            
            # 根据事件级别确定动作
            if event.level == EmergencyLevel.CRITICAL:
                if event.type in [EmergencyType.SYSTEM_ERROR, EmergencyType.ACCOUNT_RISK]:
                    actions = [EmergencyAction.STOP_TRADING, EmergencyAction.CLOSE_ALL_POSITIONS]
                elif event.type == EmergencyType.MARKET_HALT:
                    actions = [EmergencyAction.CANCEL_ALL_ORDERS, EmergencyAction.ALERT_ONLY]
                else:
                    actions = [EmergencyAction.CLOSE_ALL_POSITIONS]
                    
            elif event.level == EmergencyLevel.HIGH:
                if event.type == EmergencyType.POSITION_RISK:
                    actions = [EmergencyAction.REDUCE_POSITION_SIZE]
                elif event.type in [EmergencyType.PRICE_SPIKE, EmergencyType.VOLUME_SPIKE]:
                    actions = [EmergencyAction.CANCEL_ALL_ORDERS, EmergencyAction.ENABLE_HEDGE]
                else:
                    actions = [EmergencyAction.CANCEL_ALL_ORDERS]
                    
            elif event.level == EmergencyLevel.MEDIUM:
                actions = [EmergencyAction.ALERT_ONLY]
                
            else:  # LOW
                actions = [EmergencyAction.ALERT_ONLY]
            
            return actions
            
        except Exception as e:
            self.logger.error(f"确定应急动作失败: {e}")
            return [EmergencyAction.ALERT_ONLY]
    
    async def monitor_system(self):
        """监控系统状态"""
        while self.is_monitoring:
            try:
                # 检查系统组件状态
                if self.order_executor:
                    executor_status = self.order_executor.get_status()
                    if not executor_status.get('is_running', False):
                        await self.trigger_emergency(
                            EmergencyType.SYSTEM_ERROR,
                            EmergencyLevel.HIGH,
                            "订单执行器停止运行",
                            {"component": "order_executor"}
                        )
                
                if self.websocket_handler:
                    ws_status = self.websocket_handler.get_status()
                    if ws_status.get('connection_status') == 'error':
                        await self.trigger_emergency(
                            EmergencyType.NETWORK_ERROR,
                            EmergencyLevel.MEDIUM,
                            "WebSocket连接错误",
                            {"component": "websocket_handler"}
                        )
                
                await asyncio.sleep(10)  # 10秒检查一次
                
            except Exception as e:
                self.logger.error(f"系统监控失败: {e}")
                await asyncio.sleep(10)
    
    async def monitor_market(self):
        """监控市场状态"""
        price_history = {}
        volume_history = {}
        
        while self.is_monitoring:
            try:
                # 这里应该从实际数据源获取市场数据
                # 现在使用模拟数据
                symbols = ['BTCUSDT', 'ETHUSDT']
                
                for symbol in symbols:
                    # 模拟获取价格和成交量数据
                    current_price = await self.get_market_price(symbol)
                    current_volume = await self.get_market_volume(symbol)
                    
                    if current_price and current_volume:
                        # 检查价格异常
                        if symbol in price_history:
                            last_price = price_history[symbol]
                            price_change_pct = abs(current_price - last_price) / last_price * 100
                            
                            if price_change_pct > self.risk_thresholds.price_change_threshold:
                                await self.trigger_emergency(
                                    EmergencyType.PRICE_SPIKE,
                                    EmergencyLevel.HIGH,
                                    f"{symbol}价格异常波动: {price_change_pct:.2f}%",
                                    {"symbol": symbol, "price_change": price_change_pct, "current_price": current_price}
                                )
                        
                        # 检查成交量异常
                        if symbol in volume_history:
                            last_volume = volume_history[symbol]
                            volume_ratio = current_volume / last_volume if last_volume > 0 else 1
                            
                            if volume_ratio > self.risk_thresholds.volume_spike_threshold:
                                await self.trigger_emergency(
                                    EmergencyType.VOLUME_SPIKE,
                                    EmergencyLevel.MEDIUM,
                                    f"{symbol}成交量异常: {volume_ratio:.2f}倍",
                                    {"symbol": symbol, "volume_ratio": volume_ratio, "current_volume": current_volume}
                                )
                        
                        # 更新历史数据
                        price_history[symbol] = current_price
                        volume_history[symbol] = current_volume
                
                await asyncio.sleep(5)  # 5秒检查一次
                
            except Exception as e:
                self.logger.error(f"市场监控失败: {e}")
                await asyncio.sleep(5)
    
    async def monitor_positions(self):
        """监控持仓风险"""
        while self.is_monitoring:
            try:
                if self.position_tracker:
                    positions = self.position_tracker.get_all_positions()
                    
                    # 检查单个持仓风险
                    for symbol, position in positions.items():
                        # 检查单仓位亏损
                        if position.roi < -self.risk_thresholds.max_position_loss_pct / 100:
                            await self.trigger_emergency(
                                EmergencyType.POSITION_RISK,
                                EmergencyLevel.HIGH,
                                f"{symbol}单仓位亏损超过阈值: {position.roi:.2%}",
                                {"symbol": symbol, "roi": position.roi, "unrealized_pnl": position.unrealized_pnl}
                            )
                    
                    # 检查总体风险
                    summary = self.position_tracker.get_position_summary()
                    total_pnl_pct = summary.total_unrealized_pnl / summary.total_margin_used if summary.total_margin_used > 0 else 0
                    
                    if total_pnl_pct < -self.risk_thresholds.max_total_loss_pct / 100:
                        await self.trigger_emergency(
                            EmergencyType.ACCOUNT_RISK,
                            EmergencyLevel.CRITICAL,
                            f"总体亏损超过阈值: {total_pnl_pct:.2%}",
                            {"total_pnl_pct": total_pnl_pct, "total_pnl": summary.total_unrealized_pnl}
                        )
                
                await asyncio.sleep(3)  # 3秒检查一次
                
            except Exception as e:
                self.logger.error(f"持仓监控失败: {e}")
                await asyncio.sleep(3)
    
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """获取市场价格（模拟）"""
        try:
            import random
            base_price = 50000.0 if symbol == 'BTCUSDT' else 3000.0
            change = random.uniform(-0.02, 0.02)  # ±2%
            return base_price * (1 + change)
        except Exception as e:
            self.logger.error(f"获取市场价格失败: {e}")
            return None
    
    async def get_market_volume(self, symbol: str) -> Optional[float]:
        """获取市场成交量（模拟）"""
        try:
            import random
            base_volume = 1000000.0
            change = random.uniform(0.5, 2.0)  # 0.5-2倍
            return base_volume * change
        except Exception as e:
            self.logger.error(f"获取市场成交量失败: {e}")
            return None
    
    # 应急动作处理器
    async def handle_stop_trading(self, event: EmergencyEvent):
        """停止交易"""
        try:
            self.emergency_mode = True
            self.logger.critical("执行应急动作: 停止交易")
        except Exception as e:
            self.logger.error(f"停止交易失败: {e}")
    
    async def handle_close_all_positions(self, event: EmergencyEvent):
        """关闭所有持仓"""
        try:
            if self.position_tracker and self.order_executor:
                positions = self.position_tracker.get_all_positions()
                for symbol, position in positions.items():
                    # 这里应该创建平仓订单
                    self.logger.critical(f"应急平仓: {symbol}")
            self.logger.critical("执行应急动作: 关闭所有持仓")
        except Exception as e:
            self.logger.error(f"关闭所有持仓失败: {e}")
    
    async def handle_reduce_position_size(self, event: EmergencyEvent):
        """减少仓位大小"""
        try:
            symbol = event.data.get('symbol')
            if symbol and self.position_tracker:
                position = self.position_tracker.get_position(symbol)
                if position:
                    # 减少50%仓位
                    reduce_size = position.size * 0.5
                    self.logger.warning(f"应急减仓: {symbol} 减少 {reduce_size}")
        except Exception as e:
            self.logger.error(f"减少仓位大小失败: {e}")
    
    async def handle_cancel_all_orders(self, event: EmergencyEvent):
        """取消所有订单"""
        try:
            if self.order_executor:
                cancelled_count = await self.order_executor.cancel_all_orders()
                self.logger.warning(f"应急取消订单: {cancelled_count} 个")
        except Exception as e:
            self.logger.error(f"取消所有订单失败: {e}")
    
    async def handle_enable_hedge(self, event: EmergencyEvent):
        """启用对冲"""
        try:
            self.logger.info("执行应急动作: 启用对冲")
            # 这里应该实现对冲逻辑
        except Exception as e:
            self.logger.error(f"启用对冲失败: {e}")
    
    async def handle_alert_only(self, event: EmergencyEvent):
        """仅发送警报"""
        try:
            self.logger.warning(f"应急警报: {event.description}")
        except Exception as e:
            self.logger.error(f"发送警报失败: {e}")
    
    async def handle_system_shutdown(self, event: EmergencyEvent):
        """系统关闭"""
        try:
            self.logger.critical("执行应急动作: 系统关闭")
            # 这里应该实现系统关闭逻辑
        except Exception as e:
            self.logger.error(f"系统关闭失败: {e}")
    
    async def send_emergency_notification(self, event: EmergencyEvent):
        """发送应急通知"""
        try:
            notification = {
                "type": "emergency",
                "event_id": event.id,
                "level": event.level.value,
                "description": event.description,
                "timestamp": event.timestamp.isoformat(),
                "actions": [action.value for action in event.actions_taken]
            }
            
            # 这里应该发送到Telegram或其他通知渠道
            self.logger.critical(f"应急通知: {json.dumps(notification, ensure_ascii=False)}")
            
        except Exception as e:
            self.logger.error(f"发送应急通知失败: {e}")
    
    async def resolve_event(self, event_id: str, resolution_notes: str = ""):
        """
        解决应急事件
        
        Args:
            event_id: 事件ID
            resolution_notes: 解决备注
        """
        try:
            if event_id in self.active_events:
                event = self.active_events[event_id]
                event.is_resolved = True
                event.resolution_time = datetime.now()
                
                # 移动到历史记录
                self.event_history.append(event)
                del self.active_events[event_id]
                
                self.logger.info(f"应急事件已解决: {event_id} - {resolution_notes}")
                
        except Exception as e:
            self.logger.error(f"解决应急事件失败: {e}")
    
    def update_stats(self, event: EmergencyEvent):
        """更新统计数据"""
        try:
            self.stats['total_events'] += 1
            
            # 按类型统计
            event_type = event.type.value
            if event_type not in self.stats['events_by_type']:
                self.stats['events_by_type'][event_type] = 0
            self.stats['events_by_type'][event_type] += 1
            
            # 按级别统计
            event_level = event.level.value
            if event_level not in self.stats['events_by_level']:
                self.stats['events_by_level'][event_level] = 0
            self.stats['events_by_level'][event_level] += 1
            
        except Exception as e:
            self.logger.error(f"更新统计数据失败: {e}")
    
    def get_active_events(self) -> Dict[str, EmergencyEvent]:
        """获取活跃事件"""
        return self.active_events.copy()
    
    def get_event_history(self, limit: int = 100) -> List[EmergencyEvent]:
        """获取事件历史"""
        return self.event_history[-limit:]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_events': self.stats['total_events'],
            'active_events': len(self.active_events),
            'events_by_type': self.stats['events_by_type'],
            'events_by_level': self.stats['events_by_level'],
            'emergency_mode': self.emergency_mode,
            'is_monitoring': self.is_monitoring,
            'last_check_time': self.last_check_time.isoformat()
        }
    
    def get_status(self) -> Dict:
        """获取处理器状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'emergency_mode': self.emergency_mode,
            'active_events_count': len(self.active_events),
            'total_events_processed': self.stats['total_events'],
            'last_check_time': self.last_check_time.isoformat()
        } 