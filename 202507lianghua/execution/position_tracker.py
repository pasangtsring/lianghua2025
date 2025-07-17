"""
持仓跟踪器模块
负责实时跟踪持仓状态、盈亏计算、风险监控
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import json

from utils.logger import get_logger
from config.config_manager import ConfigManager

class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class PositionStatus(Enum):
    """持仓状态"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"

@dataclass
class TradeRecord:
    """交易记录"""
    id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    order_id: str
    
@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    commission_paid: float
    max_profit: float
    max_loss: float
    duration: timedelta
    roi: float  # 投资回报率
    status: PositionStatus = PositionStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    trade_records: List[TradeRecord] = field(default_factory=list)
    
    def __post_init__(self):
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """计算持仓指标"""
        if self.margin_used > 0:
            self.roi = self.unrealized_pnl / self.margin_used
        else:
            self.roi = 0.0
        
        self.duration = datetime.now() - self.created_at
        
        # 更新最大盈利和最大亏损
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl

@dataclass
class PositionSummary:
    """持仓汇总"""
    total_positions: int
    long_positions: int
    short_positions: int
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_margin_used: float
    total_commission: float
    average_roi: float
    win_rate: float
    profit_factor: float

class PositionTracker:
    """持仓跟踪器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 配置
        self.trading_config = config_manager.config.trading
        self.update_interval = 1.0  # 更新间隔（秒）
        
        # 持仓数据
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.trade_records: List[TradeRecord] = []
        
        # 运行状态
        self.is_running = False
        self.last_update = datetime.now()
        
        # 统计数据
        self.position_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'max_profit': 0.0,
            'average_hold_time': 0.0
        }
        
        # 价格数据缓存
        self.price_cache: Dict[str, float] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        self.logger.info("持仓跟踪器初始化完成")
    
    async def start(self):
        """启动持仓跟踪器"""
        try:
            self.is_running = True
            self.logger.info("持仓跟踪器已启动")
            
            # 启动监控任务
            asyncio.create_task(self.monitor_positions())
            asyncio.create_task(self.update_price_data())
            
        except Exception as e:
            self.logger.error(f"启动持仓跟踪器失败: {e}")
            raise
    
    async def stop(self):
        """停止持仓跟踪器"""
        try:
            self.is_running = False
            self.logger.info("持仓跟踪器已停止")
            
        except Exception as e:
            self.logger.error(f"停止持仓跟踪器失败: {e}")
    
    def add_trade(self, trade_record: TradeRecord):
        """
        添加交易记录
        
        Args:
            trade_record: 交易记录
        """
        try:
            self.trade_records.append(trade_record)
            symbol = trade_record.symbol
            
            if symbol not in self.positions:
                # 创建新持仓
                side = PositionSide.LONG if trade_record.side == 'buy' else PositionSide.SHORT
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    size=trade_record.quantity,
                    entry_price=trade_record.price,
                    current_price=trade_record.price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    margin_used=trade_record.quantity * trade_record.price,
                    commission_paid=trade_record.commission,
                    max_profit=0.0,
                    max_loss=0.0,
                    duration=timedelta(0),
                    roi=0.0,
                    trade_records=[trade_record]
                )
                self.logger.info(f"创建新持仓: {symbol} {side.value} {trade_record.quantity}")
                
            else:
                # 更新现有持仓
                position = self.positions[symbol]
                position.trade_records.append(trade_record)
                
                if trade_record.side == 'buy':
                    if position.side == PositionSide.LONG:
                        # 加仓
                        total_cost = position.size * position.entry_price + trade_record.quantity * trade_record.price
                        position.size += trade_record.quantity
                        position.entry_price = total_cost / position.size
                        position.margin_used += trade_record.quantity * trade_record.price
                        
                    elif position.side == PositionSide.SHORT:
                        # 平仓
                        close_quantity = min(position.size, trade_record.quantity)
                        realized_pnl = close_quantity * (position.entry_price - trade_record.price)
                        position.realized_pnl += realized_pnl
                        position.size -= close_quantity
                        
                        if position.size == 0:
                            position.status = PositionStatus.CLOSED
                            self.close_position(symbol)
                
                else:  # sell
                    if position.side == PositionSide.SHORT:
                        # 加仓
                        total_cost = position.size * position.entry_price + trade_record.quantity * trade_record.price
                        position.size += trade_record.quantity
                        position.entry_price = total_cost / position.size
                        position.margin_used += trade_record.quantity * trade_record.price
                        
                    elif position.side == PositionSide.LONG:
                        # 平仓
                        close_quantity = min(position.size, trade_record.quantity)
                        realized_pnl = close_quantity * (trade_record.price - position.entry_price)
                        position.realized_pnl += realized_pnl
                        position.size -= close_quantity
                        
                        if position.size == 0:
                            position.status = PositionStatus.CLOSED
                            self.close_position(symbol)
                
                position.commission_paid += trade_record.commission
                position.updated_at = datetime.now()
                
                self.logger.info(f"更新持仓: {symbol} 新数量: {position.size}")
            
            # 更新统计
            self.update_statistics()
            
        except Exception as e:
            self.logger.error(f"添加交易记录失败: {e}")
    
    def close_position(self, symbol: str):
        """
        关闭持仓
        
        Args:
            symbol: 交易品种
        """
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                position.status = PositionStatus.CLOSED
                position.updated_at = datetime.now()
                
                # 移动到已关闭持仓
                self.closed_positions.append(position)
                del self.positions[symbol]
                
                self.logger.info(f"持仓已关闭: {symbol} 盈亏: {position.realized_pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"关闭持仓失败: {e}")
    
    def update_position_price(self, symbol: str, current_price: float):
        """
        更新持仓价格
        
        Args:
            symbol: 交易品种
            current_price: 当前价格
        """
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = current_price
                
                # 计算未实现盈亏
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = position.size * (current_price - position.entry_price)
                elif position.side == PositionSide.SHORT:
                    position.unrealized_pnl = position.size * (position.entry_price - current_price)
                
                # 更新指标
                position.calculate_metrics()
                position.updated_at = datetime.now()
                
                # 更新价格历史
                self.price_history[symbol].append((datetime.now(), current_price))
                
                # 保持历史数据长度
                if len(self.price_history[symbol]) > 1000:
                    self.price_history[symbol] = self.price_history[symbol][-1000:]
                
        except Exception as e:
            self.logger.error(f"更新持仓价格失败: {e}")
    
    async def monitor_positions(self):
        """监控持仓状态"""
        while self.is_running:
            try:
                # 更新所有持仓的盈亏
                for symbol, position in self.positions.items():
                    if symbol in self.price_cache:
                        current_price = self.price_cache[symbol]
                        self.update_position_price(symbol, current_price)
                
                # 检查风险阈值
                await self.check_risk_thresholds()
                
                self.last_update = datetime.now()
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"监控持仓失败: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def update_price_data(self):
        """更新价格数据"""
        while self.is_running:
            try:
                # 模拟价格更新
                for symbol in self.positions.keys():
                    new_price = await self.get_market_price(symbol)
                    if new_price:
                        self.price_cache[symbol] = new_price
                
                await asyncio.sleep(0.5)  # 更频繁的价格更新
                
            except Exception as e:
                self.logger.error(f"更新价格数据失败: {e}")
                await asyncio.sleep(1)
    
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """
        获取市场价格（模拟）
        
        Args:
            symbol: 交易品种
            
        Returns:
            当前价格
        """
        try:
            # 模拟价格波动
            import random
            base_price = 50000.0
            if symbol in self.price_cache:
                base_price = self.price_cache[symbol]
            
            # 小幅波动
            change = random.uniform(-0.005, 0.005)  # ±0.5%
            new_price = base_price * (1 + change)
            
            return new_price
            
        except Exception as e:
            self.logger.error(f"获取市场价格失败: {e}")
            return None
    
    async def check_risk_thresholds(self):
        """检查风险阈值"""
        try:
            for symbol, position in self.positions.items():
                # 检查单个持仓风险
                if position.roi < -0.05:  # 5%亏损警告
                    self.logger.warning(f"持仓风险警告: {symbol} ROI: {position.roi:.2%}")
                
                # 检查持仓时间
                if position.duration.total_seconds() > 86400:  # 24小时
                    self.logger.info(f"长期持仓: {symbol} 持续时间: {position.duration}")
                
        except Exception as e:
            self.logger.error(f"检查风险阈值失败: {e}")
    
    def update_statistics(self):
        """更新统计数据"""
        try:
            # 计算基本统计
            total_trades = len(self.trade_records)
            winning_trades = sum(1 for p in self.closed_positions if p.realized_pnl > 0)
            losing_trades = sum(1 for p in self.closed_positions if p.realized_pnl < 0)
            
            total_pnl = sum(p.realized_pnl for p in self.closed_positions)
            total_pnl += sum(p.unrealized_pnl for p in self.positions.values())
            
            # 计算最大回撤
            max_drawdown = 0.0
            if self.closed_positions:
                cumulative_pnl = 0.0
                peak_pnl = 0.0
                for position in self.closed_positions:
                    cumulative_pnl += position.realized_pnl
                    if cumulative_pnl > peak_pnl:
                        peak_pnl = cumulative_pnl
                    
                    drawdown = peak_pnl - cumulative_pnl
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            # 计算平均持仓时间
            avg_hold_time = 0.0
            if self.closed_positions:
                total_time = sum(p.duration.total_seconds() for p in self.closed_positions)
                avg_hold_time = total_time / len(self.closed_positions)
            
            # 更新统计字典
            self.position_stats.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'average_hold_time': avg_hold_time
            })
            
        except Exception as e:
            self.logger.error(f"更新统计数据失败: {e}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取指定持仓
        
        Args:
            symbol: 交易品种
            
        Returns:
            持仓信息
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        return self.positions.copy()
    
    def get_position_summary(self) -> PositionSummary:
        """获取持仓汇总"""
        try:
            positions = list(self.positions.values())
            
            total_positions = len(positions)
            long_positions = sum(1 for p in positions if p.side == PositionSide.LONG)
            short_positions = sum(1 for p in positions if p.side == PositionSide.SHORT)
            
            total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            total_realized_pnl = sum(p.realized_pnl for p in self.closed_positions)
            total_margin_used = sum(p.margin_used for p in positions)
            total_commission = sum(p.commission_paid for p in positions)
            
            average_roi = 0.0
            if positions:
                average_roi = sum(p.roi for p in positions) / len(positions)
            
            win_rate = 0.0
            if self.closed_positions:
                winning_positions = sum(1 for p in self.closed_positions if p.realized_pnl > 0)
                win_rate = winning_positions / len(self.closed_positions)
            
            profit_factor = 0.0
            if self.closed_positions:
                total_profit = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0)
                total_loss = abs(sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0))
                if total_loss > 0:
                    profit_factor = total_profit / total_loss
            
            return PositionSummary(
                total_positions=total_positions,
                long_positions=long_positions,
                short_positions=short_positions,
                total_unrealized_pnl=total_unrealized_pnl,
                total_realized_pnl=total_realized_pnl,
                total_margin_used=total_margin_used,
                total_commission=total_commission,
                average_roi=average_roi,
                win_rate=win_rate,
                profit_factor=profit_factor
            )
            
        except Exception as e:
            self.logger.error(f"获取持仓汇总失败: {e}")
            return PositionSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_position_history(self, symbol: str) -> List[Dict]:
        """
        获取持仓历史
        
        Args:
            symbol: 交易品种
            
        Returns:
            持仓历史记录
        """
        try:
            history = []
            for position in self.closed_positions:
                if position.symbol == symbol:
                    history.append({
                        'symbol': position.symbol,
                        'side': position.side.value,
                        'size': position.size,
                        'entry_price': position.entry_price,
                        'realized_pnl': position.realized_pnl,
                        'roi': position.roi,
                        'duration': position.duration.total_seconds(),
                        'created_at': position.created_at.isoformat(),
                        'updated_at': position.updated_at.isoformat()
                    })
            
            return history
            
        except Exception as e:
            self.logger.error(f"获取持仓历史失败: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        try:
            summary = self.get_position_summary()
            
            return {
                'total_positions': summary.total_positions,
                'long_positions': summary.long_positions,
                'short_positions': summary.short_positions,
                'total_unrealized_pnl': summary.total_unrealized_pnl,
                'total_realized_pnl': summary.total_realized_pnl,
                'total_pnl': summary.total_unrealized_pnl + summary.total_realized_pnl,
                'total_margin_used': summary.total_margin_used,
                'total_commission': summary.total_commission,
                'average_roi': summary.average_roi,
                'win_rate': summary.win_rate,
                'profit_factor': summary.profit_factor,
                'max_drawdown': self.position_stats['max_drawdown'],
                'average_hold_time': self.position_stats['average_hold_time'],
                'last_update': self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取性能指标失败: {e}")
            return {}
    
    def get_status(self) -> Dict:
        """获取跟踪器状态"""
        return {
            'is_running': self.is_running,
            'total_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_trades': len(self.trade_records),
            'last_update': self.last_update.isoformat(),
            'monitored_symbols': list(self.positions.keys())
        } 