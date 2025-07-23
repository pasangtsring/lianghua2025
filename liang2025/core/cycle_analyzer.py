"""
周期分析模块
负责分析市场周期和阶段
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from utils.logger import get_logger
from config.config_manager import ConfigManager

class CyclePhase(Enum):
    """周期阶段枚举"""
    SPRING = "spring"  # 春季 - 积累期
    SUMMER = "summer"  # 夏季 - 上升期
    AUTUMN = "autumn"  # 秋季 - 分配期
    WINTER = "winter"  # 冬季 - 衰退期

@dataclass
class CycleAnalysis:
    """周期分析结果"""
    current_phase: CyclePhase
    phase_confidence: float
    phase_duration: int
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]

class CycleAnalyzer:
    """周期分析器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 周期分析配置
        self.lookback_period = 200
        self.trend_threshold = 0.02
        
        self.logger.info("周期分析器初始化完成")
    
    def analyze_cycle(self, kline_data: List[Dict]) -> CycleAnalysis:
        """
        分析市场周期
        
        Args:
            kline_data: K线数据
            
        Returns:
            周期分析结果
        """
        try:
            if len(kline_data) < self.lookback_period:
                return CycleAnalysis(
                    current_phase=CyclePhase.SPRING,
                    phase_confidence=0.0,
                    phase_duration=0,
                    trend_strength=0.0,
                    support_levels=[],
                    resistance_levels=[]
                )
            
            # 分析趋势强度
            trend_strength = self.calculate_trend_strength(kline_data)
            
            # 确定周期阶段
            current_phase = self.determine_phase(kline_data, trend_strength)
            
            # 计算置信度
            phase_confidence = self.calculate_phase_confidence(kline_data, current_phase)
            
            # 计算支撑阻力位
            support_levels = self.calculate_support_levels(kline_data)
            resistance_levels = self.calculate_resistance_levels(kline_data)
            
            return CycleAnalysis(
                current_phase=current_phase,
                phase_confidence=phase_confidence,
                phase_duration=self.calculate_phase_duration(kline_data),
                trend_strength=trend_strength,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )
            
        except Exception as e:
            self.logger.error(f"周期分析失败: {e}")
            return CycleAnalysis(
                current_phase=CyclePhase.SPRING,
                phase_confidence=0.0,
                phase_duration=0,
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[]
            )
    
    def calculate_trend_strength(self, kline_data: List[Dict]) -> float:
        """计算趋势强度（采纳大佬的错误处理建议，但修复代码问题）"""
        try:
            # 数据长度检查（采纳大佬建议）
            if len(kline_data) < 50:
                self.logger.warning("数据长度不足50，返回默认趋势强度0.0")
                return 0.0
            
            closes = [float(k['close']) for k in kline_data[-50:]]
            if len(closes) < 2:
                return 0.0
            
            # NaN检查（采纳大佬建议）
            closes_array = np.array(closes)
            if np.any(np.isnan(closes_array)):
                self.logger.warning("数据包含NaN值，进行清理")
                closes_array = closes_array[~np.isnan(closes_array)]
                if len(closes_array) < 2:
                    return 0.0
            
            # 简单线性回归计算趋势强度
            x = np.arange(len(closes_array))
            y = closes_array
            
            # 相关性计算（修复大佬代码中的错误处理）
            if len(x) > 1 and len(y) > 1:
                correlation = np.corrcoef(x, y)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"计算趋势强度失败: {e}")
            return 0.0
    
    def determine_phase(self, kline_data: List[Dict], trend_strength: float) -> CyclePhase:
        """确定周期阶段（改进逻辑，但避免大佬代码中的错误）"""
        try:
            # 数据长度检查（采纳大佬建议）
            if len(kline_data) < 20:
                self.logger.warning("数据长度不足20，返回默认阶段SPRING")
                return CyclePhase.SPRING
            
            # 获取最近价格和均线数据
            recent_closes = [float(k['close']) for k in kline_data[-50:]]
            current_price = recent_closes[-1]
            
            # 计算移动平均线（添加NaN检查）
            ma_50 = np.mean(recent_closes[-min(50, len(recent_closes)):])
            ma_20 = np.mean(recent_closes[-min(20, len(recent_closes)):])
            
            if np.isnan(ma_50) or np.isnan(ma_20):
                self.logger.warning("移动平均线计算包含NaN，返回默认阶段")
                return CyclePhase.SPRING
            
            # 改进的阶段判断逻辑（基于趋势强度和价格位置）
            price_above_ma50 = current_price > ma_50
            price_above_ma20 = current_price > ma_20
            
            if trend_strength > 0.6 and price_above_ma50 and price_above_ma20:
                return CyclePhase.SUMMER  # 强势上升
            elif trend_strength > 0.3 and (price_above_ma50 or price_above_ma20):
                return CyclePhase.SPRING  # 积累期
            elif trend_strength < -0.3 and not price_above_ma50:
                return CyclePhase.WINTER  # 下降期
            else:
                return CyclePhase.AUTUMN  # 调整期
                
        except Exception as e:
            self.logger.error(f"确定周期阶段失败: {e}")
            return CyclePhase.SPRING
    
    def calculate_phase_confidence(self, kline_data: List[Dict], phase: CyclePhase) -> float:
        """计算阶段置信度"""
        # 基础实现
        return 0.5
    
    def calculate_phase_duration(self, kline_data: List[Dict]) -> int:
        """计算阶段持续时间"""
        # 基础实现
        return 10
    
    def calculate_support_levels(self, kline_data: List[Dict]) -> List[float]:
        """计算支撑位"""
        # 基础实现
        return []
    
    def calculate_resistance_levels(self, kline_data: List[Dict]) -> List[float]:
        """计算阻力位"""
        # 基础实现
        return [] 