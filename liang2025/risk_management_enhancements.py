#!/usr/bin/env python3
"""
风险管理增强模块
采纳大佬合理建议，但修复代码错误和过度复杂化问题
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

@dataclass
class RiskMetrics:
    """风险指标"""
    total_exposure: float
    max_exposure_pct: float
    position_count: int
    max_positions: int
    funding_rate: float
    hold_time_minutes: float
    add_count: int
    max_add: int

class EnhancedRiskManager:
    """增强的风险管理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 风险控制参数（采纳大佬的合理建议）
        self.max_add = config.get('max_add', 3)  # 最大加仓次数
        self.time_stop_min = config.get('time_stop_min', [30, 60])  # 时间止损分钟
        self.fee_thresh = config.get('fee_thresh', -0.0001)  # 资金费率阈值
        self.max_total_exposure = config.get('max_total_exposure', 0.3)  # 最大总敞口
        
        self.logger.info("增强风险管理器初始化完成")
    
    def should_add_position(self, position: Dict, profit_pct: float, balance: float) -> bool:
        """
        判断是否应该加仓（采纳大佬的合理建议）
        """
        try:
            # 检查加仓次数限制（采纳大佬建议）
            current_add_count = position.get('add_count', 0)
            if current_add_count >= self.max_add:
                self.logger.info(f"加仓次数已达上限 {self.max_add}")
                return False
            
            # 检查盈利阈值
            add_profit_thresh = self.config.get('add_profit_thresh', 0.02)
            if profit_pct < add_profit_thresh * 100:  # 转换为百分比
                return False
            
            # 检查总仓位风险（采纳大佬的风险控制建议）
            total_exposure = self.calculate_total_exposure(balance)
            if total_exposure > self.max_total_exposure:
                self.logger.warning(f"总敞口超限 {total_exposure:.1%} > {self.max_total_exposure:.1%}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"判断加仓失败: {e}")
            return False
    
    def should_reduce_position(self, position: Dict, funding_rate: float, 
                             hold_time_minutes: float) -> tuple[bool, float]:
        """
        判断是否应该减仓（采纳大佬的时间和费用止损建议）
        
        Returns:
            (should_reduce, reduce_ratio)
        """
        try:
            # 时间止损检查（采纳大佬建议）
            if hold_time_minutes > self.time_stop_min[1]:  # 长时间持仓
                self.logger.info(f"持仓时间过长 {hold_time_minutes:.0f}分钟，执行全部减仓")
                return True, 1.0  # 全部减仓
            
            elif hold_time_minutes > self.time_stop_min[0]:  # 中等时间持仓
                self.logger.info(f"持仓时间较长 {hold_time_minutes:.0f}分钟，执行部分减仓")
                return True, 0.5  # 减半
            
            # 资金费率止损检查（采纳大佬建议）
            if funding_rate < self.fee_thresh:
                self.logger.info(f"资金费率过低 {funding_rate:.6f}，执行减仓")
                return True, 0.7  # 减仓70%
            
            return False, 0.0
            
        except Exception as e:
            self.logger.error(f"判断减仓失败: {e}")
            return False, 0.0
    
    def calculate_total_exposure(self, balance: float) -> float:
        """计算总敞口比例（采纳大佬的风险控制理念）"""
        try:
            # 这里应该从实际持仓中计算，这是示例实现
            # 实际使用时需要传入持仓数据
            return 0.0  # 占位实现
            
        except Exception as e:
            self.logger.error(f"计算总敞口失败: {e}")
            return 0.0
    
    def validate_new_position(self, symbol: str, size: float, leverage: int, 
                            balance: float) -> bool:
        """
        验证新持仓是否符合风险控制要求
        """
        try:
            # 计算新持仓的市值
            estimated_value = size * leverage
            exposure_ratio = estimated_value / balance
            
            # 检查单笔风险（保持原有逻辑）
            max_single_exposure = self.config.get('risk_per_trade', 0.005)
            if exposure_ratio > max_single_exposure:
                self.logger.warning(f"单笔风险过大: {exposure_ratio:.3%} > {max_single_exposure:.3%}")
                return False
            
            # 检查杠杆限制（避免大佬建议的过高杠杆）
            max_leverage = self.config.get('max_leverage', 10)
            if leverage > max_leverage:
                self.logger.warning(f"杠杆过高: {leverage}x > {max_leverage}x")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证新持仓失败: {e}")
            return False
    
    def get_risk_metrics(self, positions: Dict, balance: float) -> RiskMetrics:
        """获取风险指标（采纳大佬的监控建议）"""
        try:
            total_exposure = 0.0
            position_count = len(positions)
            max_add_count = 0
            
            for symbol, position in positions.items():
                # 计算敞口
                position_value = abs(position.get('size', 0)) * position.get('leverage', 1)
                total_exposure += position_value
                
                # 统计最大加仓次数
                add_count = position.get('add_count', 0)
                max_add_count = max(max_add_count, add_count)
            
            exposure_pct = total_exposure / balance if balance > 0 else 0.0
            
            return RiskMetrics(
                total_exposure=total_exposure,
                max_exposure_pct=exposure_pct,
                position_count=position_count,
                max_positions=self.config.get('max_positions', 3),
                funding_rate=0.0,  # 需要实时获取
                hold_time_minutes=0.0,  # 需要计算
                add_count=max_add_count,
                max_add=self.max_add
            )
            
        except Exception as e:
            self.logger.error(f"获取风险指标失败: {e}")
            return RiskMetrics(0, 0, 0, 3, 0, 0, 0, 3)

class SafeCalculator:
    """安全计算工具（采纳大佬的错误处理建议）"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全除法（采纳大佬的除零保护建议）"""
        try:
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                return default
            
            result = numerator / denominator
            return result if not (np.isnan(result) or np.isinf(result)) else default
            
        except Exception:
            return default
    
    @staticmethod
    def safe_percentage(value: float, base: float, default: float = 0.0) -> float:
        """安全百分比计算"""
        try:
            if base == 0 or np.isnan(base) or np.isnan(value):
                return default
            
            result = (value / base - 1) * 100
            return result if not (np.isnan(result) or np.isinf(result)) else default
            
        except Exception:
            return default
    
    @staticmethod
    def validate_data_length(data: List, min_length: int, description: str = "数据") -> bool:
        """验证数据长度（采纳大佬建议）"""
        if len(data) < min_length:
            logging.warning(f"{description}长度不足: {len(data)} < {min_length}")
            return False
        return True
    
    @staticmethod
    def clean_nan_values(data: np.ndarray) -> np.ndarray:
        """清理NaN值（采纳大佬建议）"""
        try:
            if len(data) == 0:
                return data
            
            # 移除NaN值
            clean_data = data[~np.isnan(data)]
            
            if len(clean_data) == 0:
                logging.warning("清理后数据为空")
                return np.array([])
            
            return clean_data
            
        except Exception as e:
            logging.error(f"清理NaN值失败: {e}")
            return np.array([])

def create_enhanced_risk_manager(config: dict) -> EnhancedRiskManager:
    """创建增强风险管理器的工厂函数"""
    return EnhancedRiskManager(config)

# 使用示例
if __name__ == "__main__":
    config = {
        'max_add': 3,
        'time_stop_min': [30, 60],
        'fee_thresh': -0.0001,
        'max_total_exposure': 0.3,
        'risk_per_trade': 0.005,
        'max_leverage': 10,
        'max_positions': 3
    }
    
    risk_manager = create_enhanced_risk_manager(config)
    print("增强风险管理器创建成功") 