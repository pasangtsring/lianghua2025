"""
仓位管理模块
负责仓位大小计算和管理
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logger import get_logger
from config.config_manager import ConfigManager

@dataclass
class Position:
    """仓位信息数据类"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin_used: float
    timestamp: datetime

class PositionManager:
    """仓位管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        self.risk_config = config_manager.get_risk_config()
        
        # 仓位配置
        self.initial_capital = getattr(config_manager.get_trading_config(), 'initial_capital', 10000)
        self.max_position_size = getattr(self.risk_config, 'max_position_size', 0.5)
        self.risk_per_trade = getattr(self.risk_config, 'risk_per_trade', 0.02)
        
        # 当前仓位
        self.positions: Dict[str, Position] = {}
        
        self.logger.info("仓位管理器初始化完成")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float, 
                              current_equity: float) -> float:
        """
        计算仓位大小
        
        Args:
            symbol: 交易品种
            entry_price: 入场价格
            stop_loss_price: 止损价格
            current_equity: 当前权益
            
        Returns:
            仓位大小
        """
        try:
            # 计算风险金额
            risk_amount = current_equity * self.risk_per_trade
            
            # 计算每股风险
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit == 0:
                return 0.0
            
            # 计算理论仓位大小
            theoretical_size = risk_amount / risk_per_unit
            
            # 应用最大仓位限制
            max_size = current_equity * self.max_position_size / entry_price
            
            # 取最小值
            position_size = min(theoretical_size, max_size)
            
            self.logger.info(f"计算仓位大小: {symbol}, 理论:{theoretical_size:.4f}, 最大:{max_size:.4f}, 实际:{position_size:.4f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0.0
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float) -> bool:
        """
        添加仓位
        
        Args:
            symbol: 交易品种
            side: 买卖方向
            size: 仓位大小
            entry_price: 入场价格
            
        Returns:
            是否成功添加
        """
        try:
            position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                margin_used=size * entry_price,
                timestamp=datetime.now()
            )
            
            self.positions[symbol] = position
            self.logger.info(f"添加仓位: {symbol} {side} {size} @ {entry_price}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"添加仓位失败: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float) -> bool:
        """
        更新仓位信息
        
        Args:
            symbol: 交易品种
            current_price: 当前价格
            
        Returns:
            是否成功更新
        """
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            position.current_price = current_price
            
            # 计算未实现盈亏
            if position.side == 'long':
                position.unrealized_pnl = position.size * (current_price - position.entry_price)
            else:  # short
                position.unrealized_pnl = position.size * (position.entry_price - current_price)
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新仓位失败: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """
        关闭仓位
        
        Args:
            symbol: 交易品种
            exit_price: 出场价格
            
        Returns:
            实现盈亏
        """
        try:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            
            # 计算实现盈亏
            if position.side == 'long':
                realized_pnl = position.size * (exit_price - position.entry_price)
            else:  # short
                realized_pnl = position.size * (position.entry_price - exit_price)
            
            # 移除仓位
            del self.positions[symbol]
            
            self.logger.info(f"关闭仓位: {symbol} @ {exit_price}, 盈亏: {realized_pnl:.2f}")
            
            return realized_pnl
            
        except Exception as e:
            self.logger.error(f"关闭仓位失败: {e}")
            return None
    
    def get_total_exposure(self) -> float:
        """获取总敞口"""
        try:
            total_exposure = 0.0
            for position in self.positions.values():
                total_exposure += abs(position.size * position.current_price)
            
            return total_exposure
            
        except Exception as e:
            self.logger.error(f"获取总敞口失败: {e}")
            return 0.0
    
    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        try:
            total_pnl = 0.0
            for position in self.positions.values():
                total_pnl += position.unrealized_pnl
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"获取总盈亏失败: {e}")
            return 0.0
    
    def get_position_summary(self) -> Dict:
        """获取仓位汇总"""
        try:
            summary = {
                'total_positions': len(self.positions),
                'total_exposure': self.get_total_exposure(),
                'total_pnl': self.get_total_pnl(),
                'positions': []
            }
            
            for symbol, position in self.positions.items():
                summary['positions'].append({
                    'symbol': symbol,
                    'side': position.side,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'margin_used': position.margin_used
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取仓位汇总失败: {e}")
            return {} 