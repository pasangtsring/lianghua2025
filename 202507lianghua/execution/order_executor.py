"""
订单执行器模块
负责异步下单、滚仓管理、止损止盈、挂单策略
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from decimal import Decimal

from utils.logger import get_logger
from config.config_manager import ConfigManager

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(Enum):
    """订单时效性"""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill

@dataclass
class Order:
    """订单数据类"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.client_order_id is None:
            self.client_order_id = f"order_{int(time.time() * 1000)}"

@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StopLossTakeProfit:
    """止损止盈配置"""
    symbol: str
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    trailing_stop_callback: Optional[float] = None
    enabled: bool = True

class OrderExecutor:
    """订单执行器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 交易配置
        self.trading_config = config_manager.config.trading
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # 订单管理
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.positions: Dict[str, PositionInfo] = {}
        self.stop_loss_orders: Dict[str, StopLossTakeProfit] = {}
        
        # 执行状态
        self.is_running = False
        self.last_heartbeat = datetime.now()
        
        # 性能统计
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'avg_execution_time': 0.0
        }
        
        self.logger.info("订单执行器初始化完成")
    
    async def start(self):
        """启动订单执行器"""
        try:
            self.is_running = True
            self.logger.info("订单执行器已启动")
            
            # 启动监控任务
            asyncio.create_task(self.monitor_orders())
            asyncio.create_task(self.monitor_positions())
            asyncio.create_task(self.heartbeat())
            
        except Exception as e:
            self.logger.error(f"启动订单执行器失败: {e}")
            raise
    
    async def stop(self):
        """停止订单执行器"""
        try:
            self.is_running = False
            
            # 取消所有挂单
            await self.cancel_all_orders()
            
            self.logger.info("订单执行器已停止")
            
        except Exception as e:
            self.logger.error(f"停止订单执行器失败: {e}")
    
    async def submit_order(self, order: Order) -> bool:
        """
        提交订单
        
        Args:
            order: 订单对象
            
        Returns:
            是否成功提交
        """
        try:
            start_time = time.time()
            
            # 订单验证
            if not self.validate_order(order):
                return False
            
            # 添加到活跃订单
            self.active_orders[order.id] = order
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            
            # 模拟订单提交（实际应该调用交易所API）
            success = await self.execute_order_simulation(order)
            
            if success:
                self.execution_stats['successful_orders'] += 1
                self.logger.info(f"订单提交成功: {order.id} {order.symbol} {order.side.value} {order.quantity}")
            else:
                self.execution_stats['failed_orders'] += 1
                order.status = OrderStatus.REJECTED
                order.error_message = "订单执行失败"
                self.logger.error(f"订单提交失败: {order.id}")
            
            # 更新统计
            execution_time = time.time() - start_time
            self.update_execution_stats(execution_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"提交订单失败: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return False
    
    async def execute_order_simulation(self, order: Order) -> bool:
        """
        模拟订单执行（实际应该调用交易所API）
        
        Args:
            order: 订单对象
            
        Returns:
            是否执行成功
        """
        try:
            # 模拟网络延迟
            await asyncio.sleep(0.1)
            
            # 模拟订单执行
            if order.type == OrderType.MARKET:
                # 市价单立即成交
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_price = order.price or 50000.0  # 模拟价格
                order.exchange_order_id = f"exchange_{int(time.time() * 1000)}"
                
                # 更新持仓
                await self.update_position(order)
                
                # 移到历史记录
                self.order_history.append(order)
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                    
                return True
                
            elif order.type == OrderType.LIMIT:
                # 限价单挂单
                order.status = OrderStatus.SUBMITTED
                order.exchange_order_id = f"exchange_{int(time.time() * 1000)}"
                return True
                
            else:
                # 其他类型订单
                order.status = OrderStatus.SUBMITTED
                order.exchange_order_id = f"exchange_{int(time.time() * 1000)}"
                return True
                
        except Exception as e:
            self.logger.error(f"模拟订单执行失败: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功取消
        """
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"订单不存在: {order_id}")
                return False
            
            order = self.active_orders[order_id]
            
            # 模拟取消订单
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            # 移到历史记录
            self.order_history.append(order)
            del self.active_orders[order_id]
            
            self.logger.info(f"订单已取消: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"取消订单失败: {e}")
            return False
    
    async def cancel_all_orders(self) -> int:
        """
        取消所有活跃订单
        
        Returns:
            取消的订单数量
        """
        try:
            cancelled_count = 0
            order_ids = list(self.active_orders.keys())
            
            for order_id in order_ids:
                if await self.cancel_order(order_id):
                    cancelled_count += 1
            
            self.logger.info(f"取消了 {cancelled_count} 个订单")
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"取消所有订单失败: {e}")
            return 0
    
    async def update_position(self, order: Order):
        """
        更新持仓信息
        
        Args:
            order: 已成交的订单
        """
        try:
            symbol = order.symbol
            
            if symbol not in self.positions:
                # 新建持仓
                self.positions[symbol] = PositionInfo(
                    symbol=symbol,
                    side=order.side.value,
                    size=order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity,
                    entry_price=order.average_price,
                    current_price=order.average_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    margin_used=order.filled_quantity * order.average_price
                )
            else:
                # 更新现有持仓
                position = self.positions[symbol]
                
                if order.side == OrderSide.BUY:
                    new_size = position.size + order.filled_quantity
                else:
                    new_size = position.size - order.filled_quantity
                
                if new_size == 0:
                    # 持仓清零
                    del self.positions[symbol]
                else:
                    # 更新持仓
                    if (position.size > 0 and new_size > 0) or (position.size < 0 and new_size < 0):
                        # 同方向加仓
                        total_cost = position.size * position.entry_price + order.filled_quantity * order.average_price
                        position.entry_price = total_cost / new_size
                    
                    position.size = new_size
                    position.current_price = order.average_price
                    position.timestamp = datetime.now()
            
            self.logger.info(f"持仓已更新: {symbol}")
            
        except Exception as e:
            self.logger.error(f"更新持仓失败: {e}")
    
    async def set_stop_loss_take_profit(self, symbol: str, stop_loss_price: Optional[float] = None,
                                       take_profit_price: Optional[float] = None,
                                       trailing_stop_distance: Optional[float] = None):
        """
        设置止损止盈
        
        Args:
            symbol: 交易品种
            stop_loss_price: 止损价格
            take_profit_price: 止盈价格
            trailing_stop_distance: 追踪止损距离
        """
        try:
            self.stop_loss_orders[symbol] = StopLossTakeProfit(
                symbol=symbol,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                trailing_stop_distance=trailing_stop_distance
            )
            
            self.logger.info(f"设置止损止盈: {symbol} 止损:{stop_loss_price} 止盈:{take_profit_price}")
            
        except Exception as e:
            self.logger.error(f"设置止损止盈失败: {e}")
    
    async def monitor_orders(self):
        """监控订单状态"""
        while self.is_running:
            try:
                # 检查挂单状态
                for order_id, order in list(self.active_orders.items()):
                    if order.status == OrderStatus.SUBMITTED:
                        # 模拟限价单成交检查
                        if order.type == OrderType.LIMIT:
                            # 这里应该检查市场价格是否触及限价
                            pass
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"监控订单失败: {e}")
                await asyncio.sleep(5)
    
    async def monitor_positions(self):
        """监控持仓和止损止盈"""
        while self.is_running:
            try:
                for symbol, position in self.positions.items():
                    # 更新持仓盈亏
                    current_price = await self.get_current_price(symbol)
                    if current_price:
                        position.current_price = current_price
                        
                        if position.size > 0:
                            position.unrealized_pnl = position.size * (current_price - position.entry_price)
                        else:
                            position.unrealized_pnl = abs(position.size) * (position.entry_price - current_price)
                    
                    # 检查止损止盈
                    if symbol in self.stop_loss_orders:
                        await self.check_stop_loss_take_profit(symbol, current_price)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"监控持仓失败: {e}")
                await asyncio.sleep(5)
    
    async def check_stop_loss_take_profit(self, symbol: str, current_price: float):
        """
        检查止损止盈触发
        
        Args:
            symbol: 交易品种
            current_price: 当前价格
        """
        try:
            if symbol not in self.positions or symbol not in self.stop_loss_orders:
                return
            
            position = self.positions[symbol]
            stop_order = self.stop_loss_orders[symbol]
            
            if not stop_order.enabled:
                return
            
            # 检查止损
            if stop_order.stop_loss_price:
                if (position.size > 0 and current_price <= stop_order.stop_loss_price) or \
                   (position.size < 0 and current_price >= stop_order.stop_loss_price):
                    await self.execute_stop_loss(symbol, current_price)
                    return
            
            # 检查止盈
            if stop_order.take_profit_price:
                if (position.size > 0 and current_price >= stop_order.take_profit_price) or \
                   (position.size < 0 and current_price <= stop_order.take_profit_price):
                    await self.execute_take_profit(symbol, current_price)
                    return
            
        except Exception as e:
            self.logger.error(f"检查止损止盈失败: {e}")
    
    async def execute_stop_loss(self, symbol: str, current_price: float):
        """执行止损"""
        try:
            position = self.positions[symbol]
            
            # 创建市价平仓订单
            order = Order(
                id=f"stop_loss_{symbol}_{int(time.time() * 1000)}",
                symbol=symbol,
                side=OrderSide.SELL if position.size > 0 else OrderSide.BUY,
                type=OrderType.MARKET,
                quantity=abs(position.size),
                price=current_price
            )
            
            await self.submit_order(order)
            
            # 禁用止损止盈
            if symbol in self.stop_loss_orders:
                self.stop_loss_orders[symbol].enabled = False
            
            self.logger.info(f"执行止损: {symbol} @ {current_price}")
            
        except Exception as e:
            self.logger.error(f"执行止损失败: {e}")
    
    async def execute_take_profit(self, symbol: str, current_price: float):
        """执行止盈"""
        try:
            position = self.positions[symbol]
            
            # 创建市价平仓订单
            order = Order(
                id=f"take_profit_{symbol}_{int(time.time() * 1000)}",
                symbol=symbol,
                side=OrderSide.SELL if position.size > 0 else OrderSide.BUY,
                type=OrderType.MARKET,
                quantity=abs(position.size),
                price=current_price
            )
            
            await self.submit_order(order)
            
            # 禁用止损止盈
            if symbol in self.stop_loss_orders:
                self.stop_loss_orders[symbol].enabled = False
            
            self.logger.info(f"执行止盈: {symbol} @ {current_price}")
            
        except Exception as e:
            self.logger.error(f"执行止盈失败: {e}")
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        获取当前价格（模拟）
        
        Args:
            symbol: 交易品种
            
        Returns:
            当前价格
        """
        try:
            # 模拟价格获取
            import random
            base_price = 50000.0
            price_change = random.uniform(-0.01, 0.01)
            return base_price * (1 + price_change)
            
        except Exception as e:
            self.logger.error(f"获取当前价格失败: {e}")
            return None
    
    async def heartbeat(self):
        """心跳检查"""
        while self.is_running:
            try:
                self.last_heartbeat = datetime.now()
                await asyncio.sleep(30)  # 30秒心跳
                
            except Exception as e:
                self.logger.error(f"心跳检查失败: {e}")
                await asyncio.sleep(30)
    
    def validate_order(self, order: Order) -> bool:
        """
        验证订单
        
        Args:
            order: 订单对象
            
        Returns:
            是否有效
        """
        try:
            # 基本验证
            if order.quantity <= 0:
                order.error_message = "订单数量必须大于0"
                return False
            
            if order.type == OrderType.LIMIT and order.price is None:
                order.error_message = "限价单必须指定价格"
                return False
            
            # 资金检查（简化）
            if order.side == OrderSide.BUY:
                required_margin = order.quantity * (order.price or 50000.0)
                # 这里应该检查实际可用资金
                if required_margin > 100000:  # 模拟资金限制
                    order.error_message = "资金不足"
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证订单失败: {e}")
            order.error_message = str(e)
            return False
    
    def update_execution_stats(self, execution_time: float):
        """更新执行统计"""
        self.execution_stats['total_orders'] += 1
        
        # 更新平均执行时间
        total_time = self.execution_stats['avg_execution_time'] * (self.execution_stats['total_orders'] - 1)
        self.execution_stats['avg_execution_time'] = (total_time + execution_time) / self.execution_stats['total_orders']
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """获取订单状态"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
            else:
                # 在历史记录中查找
                for order in self.order_history:
                    if order.id == order_id:
                        break
                else:
                    return None
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.type.value,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'average_price': order.average_price,
                'created_at': order.created_at.isoformat(),
                'updated_at': order.updated_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取订单状态失败: {e}")
            return None
    
    def get_positions(self) -> Dict[str, Dict]:
        """获取所有持仓"""
        try:
            positions = {}
            for symbol, position in self.positions.items():
                positions[symbol] = {
                    'symbol': position.symbol,
                    'side': position.side,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'margin_used': position.margin_used,
                    'timestamp': position.timestamp.isoformat()
                }
            
            return positions
            
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            return {}
    
    def get_execution_stats(self) -> Dict:
        """获取执行统计"""
        return self.execution_stats.copy()
    
    def get_status(self) -> Dict:
        """获取执行器状态"""
        return {
            'is_running': self.is_running,
            'active_orders': len(self.active_orders),
            'total_positions': len(self.positions),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'execution_stats': self.execution_stats
        } 