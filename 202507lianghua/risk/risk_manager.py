"""
风险管理模块
负责整体风险控制和管理
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..utils.logger import Logger
from ..config.config_manager import ConfigManager
from .position_manager import PositionManager
from .var_calculator import VarCalculator

@dataclass
class RiskMetrics:
    """风险指标数据类"""
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    total_pnl: float
    var_value: float
    position_risk: float
    leverage: float
    margin_usage: float

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = Logger(__name__)
        self.position_manager = PositionManager(config_manager)
        self.var_calculator = VarCalculator(config_manager)
        
        # 风险配置
        self.risk_config = config_manager.get_risk_config()
        self.max_daily_loss = self.risk_config.get('max_daily_loss', 0.05)
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.15)
        self.max_positions = self.risk_config.get('max_positions', 5)
        self.emergency_stop_loss = self.risk_config.get('emergency_stop_loss', 0.1)
        
        # 风险状态
        self.current_positions: Dict = {}
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.max_drawdown_seen: float = 0.0
        self.current_drawdown: float = 0.0
        self.emergency_stop_triggered: bool = False
        
        # 历史数据
        self.pnl_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[Tuple[datetime, float]] = []
        
        self.logger.info("风险管理器初始化完成")
    
    def check_pre_trade_risk(self, symbol: str, side: str, quantity: float, price: float) -> Tuple[bool, str]:
        """
        交易前风险检查
        
        Args:
            symbol: 交易品种
            side: 买卖方向 ('buy'/'sell')
            quantity: 交易数量
            price: 价格
            
        Returns:
            (是否允许交易, 风险信息)
        """
        try:
            # 检查每日亏损限制
            if self.daily_pnl <= -self.max_daily_loss:
                return False, f"每日亏损已达到限制：{self.daily_pnl:.2%}"
            
            # 检查最大回撤
            if self.current_drawdown >= self.max_drawdown:
                return False, f"最大回撤已达到限制：{self.current_drawdown:.2%}"
            
            # 检查仓位数量限制
            if len(self.current_positions) >= self.max_positions:
                return False, f"仓位数量已达到限制：{len(self.current_positions)}"
            
            # 检查紧急止损
            if self.emergency_stop_triggered:
                return False, "紧急止损已触发，禁止新交易"
            
            # 检查仓位大小
            position_value = quantity * price
            max_position_value = self.get_max_position_value()
            if position_value > max_position_value:
                return False, f"仓位大小超过限制：{position_value:.2f} > {max_position_value:.2f}"
            
            # 检查VaR限制
            var_value = self.var_calculator.calculate_var(symbol, quantity, price)
            if var_value > self.get_max_var_value():
                return False, f"VaR超过限制：{var_value:.2f}"
            
            return True, "风险检查通过"
            
        except Exception as e:
            self.logger.error(f"交易前风险检查失败: {e}")
            return False, f"风险检查失败：{e}"
    
    def update_position_risk(self, positions: Dict) -> None:
        """
        更新仓位风险
        
        Args:
            positions: 当前仓位信息
        """
        try:
            self.current_positions = positions
            
            # 更新每日PnL
            self.daily_pnl = self.calculate_daily_pnl()
            
            # 更新总PnL
            self.total_pnl = self.calculate_total_pnl()
            
            # 更新回撤
            self.update_drawdown()
            
            # 检查风险限制
            self.check_risk_limits()
            
        except Exception as e:
            self.logger.error(f"更新仓位风险失败: {e}")
    
    def calculate_daily_pnl(self) -> float:
        """计算每日PnL"""
        try:
            total_pnl = 0.0
            for position in self.current_positions.values():
                if position.get('update_time'):
                    # 只计算当日的PnL
                    update_time = datetime.fromtimestamp(position['update_time'] / 1000)
                    if update_time.date() == datetime.now().date():
                        total_pnl += position.get('unrealized_pnl', 0.0)
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"计算每日PnL失败: {e}")
            return 0.0
    
    def calculate_total_pnl(self) -> float:
        """计算总PnL"""
        try:
            total_pnl = 0.0
            for position in self.current_positions.values():
                total_pnl += position.get('unrealized_pnl', 0.0)
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"计算总PnL失败: {e}")
            return 0.0
    
    def update_drawdown(self) -> None:
        """更新回撤"""
        try:
            # 更新最大权益
            current_equity = self.get_current_equity()
            
            if hasattr(self, 'max_equity'):
                if current_equity > self.max_equity:
                    self.max_equity = current_equity
            else:
                self.max_equity = current_equity
            
            # 计算当前回撤
            if self.max_equity > 0:
                self.current_drawdown = (self.max_equity - current_equity) / self.max_equity
            else:
                self.current_drawdown = 0.0
            
            # 更新最大回撤
            if self.current_drawdown > self.max_drawdown_seen:
                self.max_drawdown_seen = self.current_drawdown
            
            # 记录历史
            self.drawdown_history.append((datetime.now(), self.current_drawdown))
            
        except Exception as e:
            self.logger.error(f"更新回撤失败: {e}")
    
    def check_risk_limits(self) -> None:
        """检查风险限制"""
        try:
            # 检查每日亏损限制
            if self.daily_pnl <= -self.max_daily_loss:
                self.logger.warning(f"每日亏损达到限制：{self.daily_pnl:.2%}")
                self.trigger_emergency_stop("每日亏损超限")
            
            # 检查最大回撤
            if self.current_drawdown >= self.max_drawdown:
                self.logger.warning(f"最大回撤达到限制：{self.current_drawdown:.2%}")
                self.trigger_emergency_stop("最大回撤超限")
            
            # 检查紧急止损
            if self.total_pnl <= -self.emergency_stop_loss:
                self.logger.warning(f"触发紧急止损：{self.total_pnl:.2%}")
                self.trigger_emergency_stop("紧急止损")
            
        except Exception as e:
            self.logger.error(f"检查风险限制失败: {e}")
    
    def trigger_emergency_stop(self, reason: str) -> None:
        """触发紧急止损"""
        try:
            self.emergency_stop_triggered = True
            self.logger.critical(f"触发紧急止损: {reason}")
            
            # 这里可以添加紧急止损逻辑
            # 例如：关闭所有仓位、发送告警通知等
            
        except Exception as e:
            self.logger.error(f"触发紧急止损失败: {e}")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """获取风险指标"""
        try:
            return RiskMetrics(
                current_drawdown=self.current_drawdown,
                max_drawdown=self.max_drawdown_seen,
                daily_pnl=self.daily_pnl,
                total_pnl=self.total_pnl,
                var_value=self.var_calculator.get_portfolio_var(),
                position_risk=self.calculate_position_risk(),
                leverage=self.calculate_leverage(),
                margin_usage=self.calculate_margin_usage()
            )
            
        except Exception as e:
            self.logger.error(f"获取风险指标失败: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_current_equity(self) -> float:
        """获取当前权益"""
        try:
            # 这里需要根据实际情况计算权益
            # 示例实现
            return 10000.0 + self.total_pnl
            
        except Exception as e:
            self.logger.error(f"获取当前权益失败: {e}")
            return 0.0
    
    def get_max_position_value(self) -> float:
        """获取最大仓位价值"""
        try:
            current_equity = self.get_current_equity()
            max_position_size = self.risk_config.get('max_position_size', 0.5)
            return current_equity * max_position_size
            
        except Exception as e:
            self.logger.error(f"获取最大仓位价值失败: {e}")
            return 0.0
    
    def get_max_var_value(self) -> float:
        """获取最大VaR值"""
        try:
            current_equity = self.get_current_equity()
            risk_per_trade = self.risk_config.get('risk_per_trade', 0.02)
            return current_equity * risk_per_trade
            
        except Exception as e:
            self.logger.error(f"获取最大VaR值失败: {e}")
            return 0.0
    
    def calculate_position_risk(self) -> float:
        """计算仓位风险"""
        try:
            if not self.current_positions:
                return 0.0
            
            total_risk = 0.0
            for position in self.current_positions.values():
                position_value = abs(float(position.get('position_amt', 0)))
                total_risk += position_value
            
            current_equity = self.get_current_equity()
            if current_equity > 0:
                return total_risk / current_equity
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"计算仓位风险失败: {e}")
            return 0.0
    
    def calculate_leverage(self) -> float:
        """计算杠杆率"""
        try:
            if not self.current_positions:
                return 0.0
            
            total_notional = 0.0
            for position in self.current_positions.values():
                notional = abs(float(position.get('notional', 0)))
                total_notional += notional
            
            current_equity = self.get_current_equity()
            if current_equity > 0:
                return total_notional / current_equity
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"计算杠杆率失败: {e}")
            return 0.0
    
    def calculate_margin_usage(self) -> float:
        """计算保证金使用率"""
        try:
            if not self.current_positions:
                return 0.0
            
            used_margin = 0.0
            for position in self.current_positions.values():
                margin = float(position.get('initial_margin', 0))
                used_margin += margin
            
            # 假设总保证金为当前权益
            total_margin = self.get_current_equity()
            if total_margin > 0:
                return used_margin / total_margin
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"计算保证金使用率失败: {e}")
            return 0.0
    
    def reset_daily_metrics(self) -> None:
        """重置每日指标"""
        try:
            self.daily_pnl = 0.0
            self.logger.info("每日指标已重置")
            
        except Exception as e:
            self.logger.error(f"重置每日指标失败: {e}")
    
    def reset_emergency_stop(self) -> None:
        """重置紧急止损状态"""
        try:
            self.emergency_stop_triggered = False
            self.logger.info("紧急止损状态已重置")
            
        except Exception as e:
            self.logger.error(f"重置紧急止损状态失败: {e}") 