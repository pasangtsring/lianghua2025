"""
VaR计算器模块
负责计算风险价值(Value at Risk)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from datetime import datetime, timedelta

from ..utils.logger import Logger
from ..config.config_manager import ConfigManager

class VarCalculator:
    """风险价值计算器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = Logger(__name__)
        
        # VaR配置
        self.risk_config = config_manager.get_risk_config()
        self.confidence_level = self.risk_config.get('var_confidence', 0.95)
        self.time_period = self.risk_config.get('var_period', 252)  # 交易日
        
        # 历史数据存储
        self.price_history: Dict[str, List[float]] = {}
        self.return_history: Dict[str, List[float]] = {}
        
        self.logger.info("VaR计算器初始化完成")
    
    def update_price_history(self, symbol: str, price: float, timestamp: datetime = None) -> None:
        """
        更新价格历史数据
        
        Args:
            symbol: 交易品种
            price: 价格
            timestamp: 时间戳
        """
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(price)
            
            # 保持历史数据长度
            if len(self.price_history[symbol]) > self.time_period:
                self.price_history[symbol] = self.price_history[symbol][-self.time_period:]
            
            # 计算收益率
            if len(self.price_history[symbol]) >= 2:
                returns = self.calculate_returns(symbol)
                self.return_history[symbol] = returns
            
        except Exception as e:
            self.logger.error(f"更新价格历史失败: {e}")
    
    def calculate_returns(self, symbol: str) -> List[float]:
        """
        计算收益率
        
        Args:
            symbol: 交易品种
            
        Returns:
            收益率列表
        """
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                return []
            
            prices = self.price_history[symbol]
            returns = []
            
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            return returns
            
        except Exception as e:
            self.logger.error(f"计算收益率失败: {e}")
            return []
    
    def calculate_var(self, symbol: str, position_size: float, position_value: float, 
                     method: str = 'historical') -> float:
        """
        计算VaR值
        
        Args:
            symbol: 交易品种
            position_size: 仓位大小
            position_value: 仓位价值
            method: 计算方法 ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR值
        """
        try:
            if method == 'historical':
                return self.calculate_historical_var(symbol, position_value)
            elif method == 'parametric':
                return self.calculate_parametric_var(symbol, position_value)
            elif method == 'monte_carlo':
                return self.calculate_monte_carlo_var(symbol, position_value)
            else:
                self.logger.warning(f"未知的VaR计算方法: {method}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"计算VaR失败: {e}")
            return 0.0
    
    def calculate_historical_var(self, symbol: str, position_value: float) -> float:
        """
        历史模拟法计算VaR
        
        Args:
            symbol: 交易品种
            position_value: 仓位价值
            
        Returns:
            历史VaR值
        """
        try:
            if symbol not in self.return_history or len(self.return_history[symbol]) < 30:
                return 0.0
            
            returns = np.array(self.return_history[symbol])
            
            # 计算分位数
            var_percentile = (1 - self.confidence_level) * 100
            var_return = np.percentile(returns, var_percentile)
            
            # 计算VaR值
            var_value = abs(var_return * position_value)
            
            return var_value
            
        except Exception as e:
            self.logger.error(f"计算历史VaR失败: {e}")
            return 0.0
    
    def calculate_parametric_var(self, symbol: str, position_value: float) -> float:
        """
        参数化方法计算VaR
        
        Args:
            symbol: 交易品种
            position_value: 仓位价值
            
        Returns:
            参数化VaR值
        """
        try:
            if symbol not in self.return_history or len(self.return_history[symbol]) < 30:
                return 0.0
            
            returns = np.array(self.return_history[symbol])
            
            # 计算收益率的均值和标准差
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # 计算Z值
            z_score = stats.norm.ppf(1 - self.confidence_level)
            
            # 计算VaR值
            var_return = mean_return + z_score * std_return
            var_value = abs(var_return * position_value)
            
            return var_value
            
        except Exception as e:
            self.logger.error(f"计算参数化VaR失败: {e}")
            return 0.0
    
    def calculate_monte_carlo_var(self, symbol: str, position_value: float, 
                                 num_simulations: int = 10000) -> float:
        """
        蒙特卡洛模拟法计算VaR
        
        Args:
            symbol: 交易品种
            position_value: 仓位价值
            num_simulations: 模拟次数
            
        Returns:
            蒙特卡洛VaR值
        """
        try:
            if symbol not in self.return_history or len(self.return_history[symbol]) < 30:
                return 0.0
            
            returns = np.array(self.return_history[symbol])
            
            # 计算收益率的均值和标准差
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # 蒙特卡洛模拟
            simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
            
            # 计算分位数
            var_percentile = (1 - self.confidence_level) * 100
            var_return = np.percentile(simulated_returns, var_percentile)
            
            # 计算VaR值
            var_value = abs(var_return * position_value)
            
            return var_value
            
        except Exception as e:
            self.logger.error(f"计算蒙特卡洛VaR失败: {e}")
            return 0.0
    
    def get_portfolio_var(self, positions: Dict = None) -> float:
        """
        计算投资组合VaR
        
        Args:
            positions: 仓位信息
            
        Returns:
            投资组合VaR值
        """
        try:
            if not positions:
                return 0.0
            
            total_var = 0.0
            
            for symbol, position in positions.items():
                position_value = abs(position.get('notional', 0))
                if position_value > 0:
                    var_value = self.calculate_var(symbol, 1, position_value)
                    total_var += var_value ** 2  # 假设相关性为0
            
            # 计算投资组合VaR（简化版本）
            portfolio_var = np.sqrt(total_var)
            
            return portfolio_var
            
        except Exception as e:
            self.logger.error(f"计算投资组合VaR失败: {e}")
            return 0.0
    
    def calculate_expected_shortfall(self, symbol: str, position_value: float) -> float:
        """
        计算预期不足额(Expected Shortfall)
        
        Args:
            symbol: 交易品种
            position_value: 仓位价值
            
        Returns:
            预期不足额
        """
        try:
            if symbol not in self.return_history or len(self.return_history[symbol]) < 30:
                return 0.0
            
            returns = np.array(self.return_history[symbol])
            
            # 计算VaR分位数
            var_percentile = (1 - self.confidence_level) * 100
            var_return = np.percentile(returns, var_percentile)
            
            # 计算超过VaR的损失的平均值
            tail_losses = returns[returns <= var_return]
            if len(tail_losses) > 0:
                expected_shortfall = abs(np.mean(tail_losses) * position_value)
            else:
                expected_shortfall = 0.0
            
            return expected_shortfall
            
        except Exception as e:
            self.logger.error(f"计算预期不足额失败: {e}")
            return 0.0
    
    def get_var_summary(self) -> Dict:
        """获取VaR汇总信息"""
        try:
            summary = {
                'confidence_level': self.confidence_level,
                'time_period': self.time_period,
                'symbols_tracked': list(self.price_history.keys()),
                'data_points': {}
            }
            
            for symbol in self.price_history:
                summary['data_points'][symbol] = len(self.price_history[symbol])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取VaR汇总失败: {e}")
            return {} 