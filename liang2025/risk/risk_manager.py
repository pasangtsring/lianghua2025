"""
风险管理模块
负责整体风险控制和管理
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import get_logger
from config.config_manager import ConfigManager
from .position_manager import PositionManager
from .var_calculator import VarCalculator

import asyncio

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
        self.logger = get_logger(__name__)
        self.position_manager = PositionManager(config_manager)
        self.var_calculator = VarCalculator(config_manager)
        
        # 风险配置
        self.risk_config = config_manager.get_risk_config()
        self.max_daily_loss = getattr(self.risk_config, 'max_daily_loss', 0.05)
        self.max_drawdown = getattr(self.risk_config, 'max_drawdown', 0.15)
        self.max_positions = getattr(self.risk_config, 'max_positions', 5)
        self.emergency_stop_loss = getattr(self.risk_config, 'emergency_stop_loss', 0.1)
        
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
    
    async def check_before_trade(self, signal: Dict[str, Any]) -> bool:
        """交易前风险检查"""
        try:
            # 检查是否触发应急停损
            if self.check_emergency_stop():
                self.logger.warning("应急停损触发，禁止交易")
                return False
            
            # 检查仓位数量
            if len(self.current_positions) >= self.max_positions:
                self.logger.warning(f"仓位数量已达上限: {len(self.current_positions)}")
                return False
            
            # 检查信号质量
            confidence = signal.get('confidence', 0)
            if confidence < 0.6:  # 最低置信度要求
                self.logger.warning(f"信号置信度过低: {confidence}")
                return False
            
            # 检查每笔交易风险
            quantity = signal.get('quantity', 0)
            price = signal.get('price', 0)
            trade_value = quantity * price
            
            if trade_value > self.get_max_position_value():
                self.logger.warning(f"交易价值过大: {trade_value}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"交易前风险检查失败: {e}")
            return False
    
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
            max_position_size = getattr(self.risk_config, 'max_position_size', 0.5)
            return current_equity * max_position_size
            
        except Exception as e:
            self.logger.error(f"获取最大仓位价值失败: {e}")
            return 0.0
    
    def get_max_var_value(self) -> float:
        """获取最大VaR值"""
        try:
            current_equity = self.get_current_equity()
            risk_per_trade = getattr(self.risk_config, 'risk_per_trade', 0.02)
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

# 在RiskManager类的末尾添加时间止损机制

class TimeBasedRiskManager:
    """
    基于时间的风险管理器 - 实现专家建议的时间止损机制
    """
    
    def __init__(self, config_manager: ConfigManager, position_manager = None):
        self.config = config_manager
        self.logger = get_logger(__name__)
        self.position_manager = position_manager
        
        # 获取时间止损配置
        self.risk_config = config_manager.get_risk_config()
        self.position_config = config_manager.get_position_management_config()
        
        # 时间止损参数
        self.time_stop_minutes = self.risk_config.time_stop_min  # [30, 60]
        self.reduce_after_minutes = getattr(self.position_config.time_based_management, 'reduce_after_minutes', 30)
        self.close_after_minutes = getattr(self.position_config.time_based_management, 'close_after_minutes', 60)
        self.reduce_ratio = getattr(self.position_config.time_based_management, 'reduce_ratio', 0.5)
        
        # 持仓时间跟踪
        self.position_start_times: Dict[str, datetime] = {}
        self.position_time_actions: Dict[str, List[str]] = {}  # 记录已执行的时间动作
        
        # 时间止损回调
        self.time_stop_callbacks: List[Callable] = []
        
        # 监控任务
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
        
        # 时间止损统计
        self.time_stop_stats = {
            'total_time_stops': 0,
            'partial_reduces': 0,
            'full_closes': 0,
            'avg_hold_time': 0.0
        }
        
        self.logger.info("时间风险管理器初始化完成")
    
    async def start_monitoring(self):
        """
        启动时间监控
        """
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self.monitor_position_times())
            
            self.logger.info("时间风险监控已启动")
            
        except Exception as e:
            self.logger.error(f"启动时间监控失败: {e}")
    
    async def stop_monitoring(self):
        """
        停止时间监控
        """
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                await self.monitoring_task
            
            self.logger.info("时间风险监控已停止")
            
        except Exception as e:
            self.logger.error(f"停止时间监控失败: {e}")
    
    def register_position_open(self, symbol: str, entry_time: datetime = None):
        """
        注册持仓开始时间
        
        Args:
            symbol: 交易品种
            entry_time: 入场时间（默认为当前时间）
        """
        try:
            if entry_time is None:
                entry_time = datetime.now()
            
            self.position_start_times[symbol] = entry_time
            self.position_time_actions[symbol] = []
            
            self.logger.info(f"注册持仓开始时间: {symbol} - {entry_time}")
            
        except Exception as e:
            self.logger.error(f"注册持仓时间失败: {e}")
    
    def register_position_close(self, symbol: str):
        """
        注册持仓结束
        
        Args:
            symbol: 交易品种
        """
        try:
            if symbol in self.position_start_times:
                # 计算持仓时间
                hold_time = datetime.now() - self.position_start_times[symbol]
                hold_minutes = hold_time.total_seconds() / 60
                
                # 更新统计
                self.update_hold_time_stats(hold_minutes)
                
                # 清理记录
                del self.position_start_times[symbol]
                if symbol in self.position_time_actions:
                    del self.position_time_actions[symbol]
                
                self.logger.info(f"持仓结束: {symbol}, 持仓时间: {hold_minutes:.1f}分钟")
            
        except Exception as e:
            self.logger.error(f"注册持仓结束失败: {e}")
    
    async def monitor_position_times(self):
        """
        监控持仓时间
        """
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                # 检查每个持仓的时间
                for symbol, start_time in self.position_start_times.items():
                    hold_time = current_time - start_time
                    hold_minutes = hold_time.total_seconds() / 60
                    
                    # 检查是否需要执行时间止损
                    await self.check_time_stop_conditions(symbol, hold_minutes)
                
                # 等待下次检查
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                self.logger.error(f"监控持仓时间失败: {e}")
                await asyncio.sleep(60)
    
    async def check_time_stop_conditions(self, symbol: str, hold_minutes: float):
        """
        检查时间止损条件
        
        Args:
            symbol: 交易品种
            hold_minutes: 持仓时间（分钟）
        """
        try:
            actions = self.position_time_actions.get(symbol, [])
            
            # 检查部分减仓条件
            if (hold_minutes >= self.reduce_after_minutes and 
                'partial_reduce' not in actions):
                
                await self.execute_time_based_action(symbol, 'partial_reduce', hold_minutes)
                actions.append('partial_reduce')
                self.position_time_actions[symbol] = actions
            
            # 检查完全平仓条件
            if (hold_minutes >= self.close_after_minutes and 
                'full_close' not in actions):
                
                await self.execute_time_based_action(symbol, 'full_close', hold_minutes)
                actions.append('full_close')
                self.position_time_actions[symbol] = actions
            
            # 检查自定义时间止损条件
            for stop_time in self.time_stop_minutes:
                action_key = f'time_stop_{stop_time}'
                if (hold_minutes >= stop_time and 
                    action_key not in actions):
                    
                    await self.execute_time_based_action(symbol, action_key, hold_minutes)
                    actions.append(action_key)
                    self.position_time_actions[symbol] = actions
            
        except Exception as e:
            self.logger.error(f"检查时间止损条件失败: {e}")
    
    async def execute_time_based_action(self, symbol: str, action_type: str, hold_minutes: float):
        """
        执行基于时间的动作
        
        Args:
            symbol: 交易品种
            action_type: 动作类型
            hold_minutes: 持仓时间（分钟）
        """
        try:
            self.logger.warning(f"执行时间止损动作: {symbol} - {action_type} (持仓{hold_minutes:.1f}分钟)")
            
            if action_type == 'partial_reduce':
                # 部分减仓
                await self.partial_reduce_position(symbol, self.reduce_ratio)
                self.time_stop_stats['partial_reduces'] += 1
                
            elif action_type == 'full_close':
                # 完全平仓
                await self.close_position_fully(symbol)
                self.time_stop_stats['full_closes'] += 1
                
            elif action_type.startswith('time_stop_'):
                # 自定义时间止损
                stop_minutes = int(action_type.split('_')[-1])
                await self.custom_time_stop(symbol, stop_minutes)
                self.time_stop_stats['total_time_stops'] += 1
            
            # 调用时间止损回调
            for callback in self.time_stop_callbacks:
                try:
                    await callback(symbol, action_type, hold_minutes)
                except Exception as e:
                    self.logger.error(f"时间止损回调失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"执行时间止损动作失败: {e}")
    
    async def partial_reduce_position(self, symbol: str, reduce_ratio: float):
        """
        部分减仓
        
        Args:
            symbol: 交易品种
            reduce_ratio: 减仓比例
        """
        try:
            if self.position_manager:
                # 获取当前持仓
                position = await self.position_manager.get_position(symbol)
                
                if position and abs(position['size']) > 0:
                    # 计算减仓数量
                    reduce_quantity = abs(position['size']) * reduce_ratio
                    
                    # 确定减仓方向
                    reduce_side = 'sell' if position['size'] > 0 else 'buy'
                    
                    # 执行减仓
                    await self.position_manager.reduce_position(
                        symbol, reduce_side, reduce_quantity, 
                        reason=f"时间止损-部分减仓({reduce_ratio:.1%})"
                    )
                    
                    self.logger.info(f"部分减仓执行: {symbol}, 减仓{reduce_ratio:.1%}")
                    
        except Exception as e:
            self.logger.error(f"部分减仓失败: {e}")
    
    async def close_position_fully(self, symbol: str):
        """
        完全平仓
        
        Args:
            symbol: 交易品种
        """
        try:
            if self.position_manager:
                # 获取当前持仓
                position = await self.position_manager.get_position(symbol)
                
                if position and abs(position['size']) > 0:
                    # 确定平仓方向
                    close_side = 'sell' if position['size'] > 0 else 'buy'
                    close_quantity = abs(position['size'])
                    
                    # 执行平仓
                    await self.position_manager.close_position(
                        symbol, close_side, close_quantity,
                        reason="时间止损-完全平仓"
                    )
                    
                    self.logger.warning(f"完全平仓执行: {symbol}")
                    
                    # 注册持仓结束
                    self.register_position_close(symbol)
                    
        except Exception as e:
            self.logger.error(f"完全平仓失败: {e}")
    
    async def custom_time_stop(self, symbol: str, stop_minutes: int):
        """
        自定义时间止损
        
        Args:
            symbol: 交易品种
            stop_minutes: 止损时间（分钟）
        """
        try:
            # 根据时间决定动作
            if stop_minutes <= 30:
                # 短时间：减仓50%
                await self.partial_reduce_position(symbol, 0.5)
            elif stop_minutes <= 60:
                # 中时间：减仓70%
                await self.partial_reduce_position(symbol, 0.7)
            else:
                # 长时间：完全平仓
                await self.close_position_fully(symbol)
            
            self.logger.warning(f"自定义时间止损执行: {symbol}, {stop_minutes}分钟")
            
        except Exception as e:
            self.logger.error(f"自定义时间止损失败: {e}")
    
    def add_time_stop_callback(self, callback: Callable):
        """
        添加时间止损回调
        
        Args:
            callback: 回调函数
        """
        self.time_stop_callbacks.append(callback)
    
    def get_position_hold_time(self, symbol: str) -> Optional[float]:
        """
        获取持仓时间
        
        Args:
            symbol: 交易品种
            
        Returns:
            持仓时间（分钟）或None
        """
        try:
            if symbol in self.position_start_times:
                hold_time = datetime.now() - self.position_start_times[symbol]
                return hold_time.total_seconds() / 60
            return None
            
        except Exception as e:
            self.logger.error(f"获取持仓时间失败: {e}")
            return None
    
    def get_all_position_hold_times(self) -> Dict[str, float]:
        """
        获取所有持仓时间
        
        Returns:
            持仓时间字典
        """
        try:
            hold_times = {}
            current_time = datetime.now()
            
            for symbol, start_time in self.position_start_times.items():
                hold_time = current_time - start_time
                hold_times[symbol] = hold_time.total_seconds() / 60
            
            return hold_times
            
        except Exception as e:
            self.logger.error(f"获取所有持仓时间失败: {e}")
            return {}
    
    def check_time_risk_status(self, symbol: str) -> Dict[str, Any]:
        """
        检查时间风险状态
        
        Args:
            symbol: 交易品种
            
        Returns:
            时间风险状态
        """
        try:
            hold_minutes = self.get_position_hold_time(symbol)
            
            if hold_minutes is None:
                return {'status': 'no_position', 'hold_time': 0}
            
            # 确定风险等级
            risk_level = 'low'
            next_action = None
            time_to_action = 0
            
            if hold_minutes >= self.close_after_minutes:
                risk_level = 'critical'
                next_action = 'full_close'
                time_to_action = 0
            elif hold_minutes >= self.reduce_after_minutes:
                risk_level = 'high'
                next_action = 'full_close'
                time_to_action = self.close_after_minutes - hold_minutes
            else:
                risk_level = 'medium' if hold_minutes >= self.reduce_after_minutes * 0.8 else 'low'
                next_action = 'partial_reduce'
                time_to_action = self.reduce_after_minutes - hold_minutes
            
            return {
                'status': 'active',
                'hold_time': hold_minutes,
                'risk_level': risk_level,
                'next_action': next_action,
                'time_to_action': max(0, time_to_action),
                'actions_taken': self.position_time_actions.get(symbol, [])
            }
            
        except Exception as e:
            self.logger.error(f"检查时间风险状态失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_hold_time_stats(self, hold_minutes: float):
        """
        更新持仓时间统计
        
        Args:
            hold_minutes: 持仓时间（分钟）
        """
        try:
            # 更新平均持仓时间
            current_avg = self.time_stop_stats['avg_hold_time']
            total_positions = (self.time_stop_stats['partial_reduces'] + 
                             self.time_stop_stats['full_closes'] + 
                             self.time_stop_stats['total_time_stops'])
            
            if total_positions > 0:
                self.time_stop_stats['avg_hold_time'] = (
                    (current_avg * (total_positions - 1) + hold_minutes) / total_positions
                )
            else:
                self.time_stop_stats['avg_hold_time'] = hold_minutes
                
        except Exception as e:
            self.logger.error(f"更新持仓时间统计失败: {e}")
    
    def get_time_stop_stats(self) -> Dict[str, Any]:
        """
        获取时间止损统计
        
        Returns:
            时间止损统计信息
        """
        try:
            return {
                'total_time_stops': self.time_stop_stats['total_time_stops'],
                'partial_reduces': self.time_stop_stats['partial_reduces'],
                'full_closes': self.time_stop_stats['full_closes'],
                'avg_hold_time': self.time_stop_stats['avg_hold_time'],
                'active_positions': len(self.position_start_times),
                'current_hold_times': self.get_all_position_hold_times(),
                'monitoring_active': self.monitoring_active,
                'config': {
                    'reduce_after_minutes': self.reduce_after_minutes,
                    'close_after_minutes': self.close_after_minutes,
                    'reduce_ratio': self.reduce_ratio,
                    'time_stop_minutes': self.time_stop_minutes
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取时间止损统计失败: {e}")
            return {}
    
    def reset_time_stop_stats(self):
        """
        重置时间止损统计
        """
        try:
            self.time_stop_stats = {
                'total_time_stops': 0,
                'partial_reduces': 0,
                'full_closes': 0,
                'avg_hold_time': 0.0
            }
            
            self.logger.info("时间止损统计已重置")
            
        except Exception as e:
            self.logger.error(f"重置时间止损统计失败: {e}")
    
    async def force_time_stop(self, symbol: str, action_type: str = 'full_close'):
        """
        强制执行时间止损
        
        Args:
            symbol: 交易品种
            action_type: 动作类型
        """
        try:
            hold_minutes = self.get_position_hold_time(symbol)
            
            if hold_minutes is not None:
                self.logger.warning(f"强制执行时间止损: {symbol} - {action_type}")
                await self.execute_time_based_action(symbol, action_type, hold_minutes)
            else:
                self.logger.warning(f"无法强制时间止损，持仓不存在: {symbol}")
                
        except Exception as e:
            self.logger.error(f"强制时间止损失败: {e}")
    
    def __del__(self):
        """
        析构函数
        """
        try:
            if self.monitoring_active:
                asyncio.create_task(self.stop_monitoring())
        except:
            pass 

    def calculate_risk_metrics(self) -> Optional[RiskMetrics]:
        """计算风险指标"""
        try:
            current_drawdown = self.calculate_current_drawdown()
            max_drawdown = self.get_max_drawdown()
            daily_pnl = self.daily_pnl
            total_pnl = self.total_pnl
            var_value = self.var_calculator.calculate_var([]) if hasattr(self, 'var_calculator') else 0.0
            position_risk = self.calculate_position_risk()
            leverage = self.calculate_current_leverage()
            margin_usage = self.calculate_margin_usage()
            
            return RiskMetrics(
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                daily_pnl=daily_pnl,
                total_pnl=total_pnl,
                var_value=var_value,
                position_risk=position_risk,
                leverage=leverage,
                margin_usage=margin_usage
            )
            
        except Exception as e:
            self.logger.error(f"计算风险指标失败: {e}")
            return None
    
    def check_emergency_stop(self) -> bool:
        """检查是否触发应急停损"""
        try:
            # 检查当前回撤是否超过阈值
            if self.calculate_current_drawdown() > self.max_drawdown:
                self.logger.warning(f"触发应急停损：回撤超过阈值")
                return True
            
            # 检查每日损失
            if self.daily_pnl < -self.max_daily_loss:
                self.logger.warning(f"触发应急停损：每日损失超过阈值")
                return True
            
            # 检查总损失
            if abs(self.total_pnl) > self.emergency_stop_loss:
                self.logger.warning(f"触发应急停损：总损失超过阈值")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"应急停损检查失败: {e}")
            return True  # 出现错误时保守处理 