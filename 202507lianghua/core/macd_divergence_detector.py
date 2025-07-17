"""
MACD背离检测器模块
实现高精度的背离检测算法、动态阈值、峰值识别
严格遵循零简化原则，确保检测准确性和实用性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from enum import Enum
import math
from scipy import signal
from scipy.signal import find_peaks  # 添加这个导入
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

from config.config_manager import ConfigManager
from utils.logger import get_logger, performance_monitor
from core.technical_indicators import MACDResult, TechnicalIndicatorCalculator

# 设置高精度计算
getcontext().prec = 50

# 创建logger实例
logger = get_logger(__name__)


class DivergenceType(Enum):
    """背离类型枚举"""
    BULLISH_REGULAR = "bullish_regular"      # 常规看涨背离
    BEARISH_REGULAR = "bearish_regular"      # 常规看跌背离
    BULLISH_HIDDEN = "bullish_hidden"        # 隐藏看涨背离
    BEARISH_HIDDEN = "bearish_hidden"        # 隐藏看跌背离
    NO_DIVERGENCE = "no_divergence"          # 无背离


class SignalStrength(Enum):
    """信号强度枚举"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


@dataclass
class Peak:
    """峰值数据结构"""
    index: int
    value: float
    timestamp: datetime
    peak_type: str  # 'high' or 'low'
    confirmation_candles: int = 0
    strength: float = 0.0
    volume: float = 0.0
    
    def __post_init__(self):
        """计算峰值强度"""
        if self.confirmation_candles > 0:
            self.strength = min(1.0, self.confirmation_candles / 5.0)


@dataclass
class TrendLine:
    """趋势线数据结构"""
    points: List[Peak]
    slope: float
    intercept: float
    r_squared: float
    start_index: int
    end_index: int
    equation: str = ""
    
    def __post_init__(self):
        """生成趋势线方程"""
        self.equation = f"y = {self.slope:.6f}x + {self.intercept:.6f}"


@dataclass
class DivergencePattern:
    """背离模式数据结构"""
    divergence_type: DivergenceType
    price_trend: TrendLine
    macd_trend: TrendLine
    start_time: datetime
    end_time: datetime
    duration: timedelta
    signal_strength: SignalStrength
    confidence: float
    risk_reward_ratio: float
    expected_move: float
    supporting_indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算背离模式的持续时间"""
        self.duration = self.end_time - self.start_time


@dataclass
class MACDDivergenceSignal:
    """MACD背离信号数据结构"""
    symbol: str
    timeframe: str
    timestamp: datetime
    divergence_pattern: DivergencePattern
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_percentage: float
    expected_duration: timedelta
    invalidation_level: float
    notes: str = ""
    
    def __post_init__(self):
        """计算风险回报比"""
        if self.stop_loss != 0:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            self.risk_reward_ratio = reward / risk if risk > 0 else 0


class PeakDetector:
    """峰值检测器"""
    
    def __init__(self, config):
        self.config = config
        
        # 兼容新旧配置系统
        if hasattr(config, 'get_setting'):
            # 旧的ConfigManager系统
            self.min_peak_distance = config.get_setting('macd_divergence.min_peak_distance', 5)
            self.prominence_threshold = config.get_setting('macd_divergence.prominence_threshold', 0.1)
            self.confirmation_candles = config.get_setting('macd_divergence.confirmation_candles', 3)
        else:
            # 新的DivergenceDetectionConfig系统
            self.min_peak_distance = getattr(config, 'min_peak_distance', 5)
            self.prominence_threshold = getattr(config, 'prominence_multiplier', 0.1)
            self.confirmation_candles = getattr(config, 'min_confirmation_candles', 3)
    
    @performance_monitor
    def detect_peaks(self, data: List[float], volumes: List[float] = None) -> Tuple[List[Peak], List[Peak]]:
        """
        检测峰值和谷值
        
        Args:
            data: 价格或指标数据
            volumes: 成交量数据（可选）
            
        Returns:
            Tuple[List[Peak], List[Peak]]: (峰值列表, 谷值列表)
        """
        if len(data) < self.min_peak_distance * 2:
            return [], []
        
        data_array = np.array(data)
        volumes_array = np.array(volumes) if volumes else np.ones(len(data))
        
        # 检测峰值（高点）
        peaks_indices, peak_properties = signal.find_peaks(
            data_array,
            distance=self.min_peak_distance,
            prominence=self.prominence_threshold
        )
        
        # 检测谷值（低点）
        valleys_indices, valley_properties = signal.find_peaks(
            -data_array,
            distance=self.min_peak_distance,
            prominence=self.prominence_threshold
        )
        
        # 创建Peak对象列表
        peaks = []
        for idx in peaks_indices:
            if self._confirm_peak(data_array, idx, 'high'):
                peak = Peak(
                    index=idx,
                    value=data_array[idx],
                    timestamp=datetime.now(),
                    peak_type='high',
                    confirmation_candles=self.confirmation_candles,
                    volume=volumes_array[idx] if volumes else 0.0
                )
                peaks.append(peak)
        
        valleys = []
        for idx in valleys_indices:
            if self._confirm_peak(data_array, idx, 'low'):
                valley = Peak(
                    index=idx,
                    value=data_array[idx],
                    timestamp=datetime.now(),
                    peak_type='low',
                    confirmation_candles=self.confirmation_candles,
                    volume=volumes_array[idx] if volumes else 0.0
                )
                valleys.append(valley)
        
        return peaks, valleys
    
    def _confirm_peak(self, data: np.ndarray, peak_idx: int, peak_type: str) -> bool:
        """
        确认峰值有效性
        
        Args:
            data: 数据数组
            peak_idx: 峰值索引
            peak_type: 峰值类型 ('high' or 'low')
            
        Returns:
            bool: 是否为有效峰值
        """
        if peak_idx < self.confirmation_candles or peak_idx >= len(data) - self.confirmation_candles:
            return False
        
        peak_value = data[peak_idx]
        
        if peak_type == 'high':
            # 确认左侧和右侧都有足够的确认K线
            left_max = np.max(data[peak_idx - self.confirmation_candles:peak_idx])
            right_max = np.max(data[peak_idx + 1:peak_idx + self.confirmation_candles + 1])
            return peak_value > left_max and peak_value > right_max
        else:  # low
            left_min = np.min(data[peak_idx - self.confirmation_candles:peak_idx])
            right_min = np.min(data[peak_idx + 1:peak_idx + self.confirmation_candles + 1])
            return peak_value < left_min and peak_value < right_min
    
    @performance_monitor
    def filter_significant_peaks(self, peaks: List[Peak], min_significance: float = 0.02) -> List[Peak]:
        """
        过滤显著峰值
        
        Args:
            peaks: 峰值列表
            min_significance: 最小显著性阈值
            
        Returns:
            List[Peak]: 过滤后的峰值列表
        """
        if len(peaks) < 2:
            return peaks
        
        # 计算峰值之间的相对变化
        filtered_peaks = []
        for i, peak in enumerate(peaks):
            if i == 0:
                filtered_peaks.append(peak)
                continue
            
            prev_peak = filtered_peaks[-1]
            relative_change = abs(peak.value - prev_peak.value) / prev_peak.value
            
            if relative_change >= min_significance:
                filtered_peaks.append(peak)
        
        return filtered_peaks


class TrendLineAnalyzer:
    """趋势线分析器"""
    
    def __init__(self, config):
        self.config = config
        
        # 兼容新旧配置系统
        if hasattr(config, 'get_setting'):
            # 旧的ConfigManager系统
            self.min_r_squared = config.get_setting('macd_divergence.min_r_squared', 0.7)
            self.min_points = config.get_setting('macd_divergence.min_trend_points', 2)
        else:
            # 新的DivergenceDetectionConfig系统
            self.min_r_squared = getattr(config, 'trend_line_min_r_squared', 0.7)
            self.min_points = getattr(config, 'min_consecutive_count', 2)
    
    @performance_monitor
    def create_trend_line(self, peaks: List[Peak]) -> Optional[TrendLine]:
        """
        创建趋势线
        
        Args:
            peaks: 峰值列表
            
        Returns:
            Optional[TrendLine]: 趋势线对象或None
        """
        if len(peaks) < self.min_points:
            return None
        
        # 提取x和y坐标
        x_coords = [peak.index for peak in peaks]
        y_coords = [peak.value for peak in peaks]
        
        # 线性回归
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x_coords, y_coords)
            r_squared = r_value ** 2
            
            # 检查R²是否满足要求
            if r_squared < self.min_r_squared:
                return None
            
            trend_line = TrendLine(
                points=peaks,
                slope=slope,
                intercept=intercept,
                r_squared=r_squared,
                start_index=min(x_coords),
                end_index=max(x_coords)
            )
            
            return trend_line
            
        except Exception as e:
            logger.warning(f"趋势线创建失败: {e}")
            return None
    
    @performance_monitor
    def analyze_trend_divergence(self, price_trend: TrendLine, macd_trend: TrendLine) -> Tuple[DivergenceType, float]:
        """
        分析趋势背离
        
        Args:
            price_trend: 价格趋势线
            macd_trend: MACD趋势线
            
        Returns:
            Tuple[DivergenceType, float]: (背离类型, 背离强度)
        """
        price_slope = price_trend.slope
        macd_slope = macd_trend.slope
        
        # 计算背离强度
        slope_diff = abs(price_slope - macd_slope)
        max_slope = max(abs(price_slope), abs(macd_slope))
        divergence_strength = slope_diff / max_slope if max_slope > 0 else 0
        
        # 判断背离类型
        if price_slope > 0 and macd_slope < 0:
            return DivergenceType.BEARISH_REGULAR, divergence_strength
        elif price_slope < 0 and macd_slope > 0:
            return DivergenceType.BULLISH_REGULAR, divergence_strength
        elif price_slope > 0 and macd_slope > 0 and abs(price_slope) > abs(macd_slope) * 1.5:
            return DivergenceType.BEARISH_HIDDEN, divergence_strength * 0.7
        elif price_slope < 0 and macd_slope < 0 and abs(price_slope) > abs(macd_slope) * 1.5:
            return DivergenceType.BULLISH_HIDDEN, divergence_strength * 0.7
        else:
            return DivergenceType.NO_DIVERGENCE, 0.0


class DivergenceValidator:
    """背离验证器"""
    
    def __init__(self, config):
        self.config = config
        
        # 兼容新旧配置系统
        if hasattr(config, 'get_setting'):
            # 旧的ConfigManager系统
            self.min_divergence_strength = config.get_setting('macd_divergence.min_divergence_strength', 0.3)
            self.min_duration = config.get_setting('macd_divergence.min_duration_hours', 12)
            self.max_duration = config.get_setting('macd_divergence.max_duration_hours', 168)  # 7天
            self.min_r_squared = config.get_setting('macd_divergence.min_r_squared', 0.7)
        else:
            # 新的DivergenceDetectionConfig系统
            self.min_divergence_strength = getattr(config, 'min_divergence_gap', 0.3)
            self.min_duration = getattr(config, 'min_duration_hours', 12)
            self.max_duration = getattr(config, 'max_duration_hours', 168)  # 7天
            self.min_r_squared = getattr(config, 'trend_line_min_r_squared', 0.7)
    
    @performance_monitor
    def validate_divergence(self, pattern: DivergencePattern) -> Tuple[bool, float, List[str]]:
        """
        验证背离模式
        
        Args:
            pattern: 背离模式
            
        Returns:
            Tuple[bool, float, List[str]]: (是否有效, 置信度, 验证失败原因)
        """
        validation_issues = []
        confidence_score = 1.0
        
        # 1. 检查背离类型
        if pattern.divergence_type == DivergenceType.NO_DIVERGENCE:
            validation_issues.append("无背离模式")
            return False, 0.0, validation_issues
        
        # 2. 检查趋势线质量
        if pattern.price_trend.r_squared < self.min_r_squared:
            validation_issues.append(f"价格趋势线R²过低: {pattern.price_trend.r_squared:.3f}")
            confidence_score *= 0.8
        
        if pattern.macd_trend.r_squared < self.min_r_squared:
            validation_issues.append(f"MACD趋势线R²过低: {pattern.macd_trend.r_squared:.3f}")
            confidence_score *= 0.8
        
        # 3. 检查持续时间
        duration_hours = pattern.duration.total_seconds() / 3600
        if duration_hours < self.min_duration:
            validation_issues.append(f"持续时间过短: {duration_hours:.1f}小时")
            confidence_score *= 0.7
        elif duration_hours > self.max_duration:
            validation_issues.append(f"持续时间过长: {duration_hours:.1f}小时")
            confidence_score *= 0.8
        
        # 4. 检查峰值数量
        price_peaks = len(pattern.price_trend.points)
        macd_peaks = len(pattern.macd_trend.points)
        
        if price_peaks < 2 or macd_peaks < 2:
            validation_issues.append("峰值数量不足")
            confidence_score *= 0.6
        
        # 5. 检查背离强度
        if pattern.confidence < self.min_divergence_strength:
            validation_issues.append(f"背离强度不足: {pattern.confidence:.3f}")
            confidence_score *= 0.5
        
        # 6. 检查风险回报比
        if pattern.risk_reward_ratio < 1.0:
            validation_issues.append(f"风险回报比不佳: {pattern.risk_reward_ratio:.2f}")
            confidence_score *= 0.9
        
        # 7. 动态调整置信度
        # 根据背离类型调整
        if pattern.divergence_type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BEARISH_REGULAR]:
            confidence_score *= 1.0  # 常规背离权重正常
        else:
            confidence_score *= 0.8  # 隐藏背离权重降低
        
        # 根据R²值调整
        avg_r_squared = (pattern.price_trend.r_squared + pattern.macd_trend.r_squared) / 2
        confidence_score *= avg_r_squared
        
        # 最终验证
        is_valid = confidence_score >= 0.5 and len(validation_issues) == 0
        
        return is_valid, confidence_score, validation_issues
    
    @performance_monitor
    def calculate_signal_strength(self, confidence: float, risk_reward: float) -> SignalStrength:
        """
        计算信号强度
        
        Args:
            confidence: 置信度
            risk_reward: 风险回报比
            
        Returns:
            SignalStrength: 信号强度
        """
        # 综合评分
        score = (confidence * 0.6) + (min(risk_reward / 2.0, 1.0) * 0.4)
        
        if score >= 0.9:
            return SignalStrength.VERY_STRONG
        elif score >= 0.7:
            return SignalStrength.STRONG
        elif score >= 0.5:
            return SignalStrength.MODERATE
        elif score >= 0.3:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK


# 添加专家建议的优化配置参数
@dataclass
class DivergenceDetectionConfig:
    """背离检测配置参数"""
    lookback_period: int = 50           # 检测窗口（根据专家建议）
    min_peak_distance: int = 5          # 峰值最小间隔
    prominence_multiplier: float = 0.5   # 噪音过滤倍数
    min_divergence_gap: float = 0.1     # 最小背离强度阈值
    min_consecutive_count: int = 2      # 最小连续背离次数
    time_alignment_tolerance: int = 2   # 时间对齐容忍度
    max_time_window: int = 20          # 最大时间窗口
    
    # 原有配置保持兼容
    min_price_change: float = 0.01
    min_macd_change: float = 0.001
    min_confirmation_candles: int = 3
    max_peak_age: int = 100
    trend_line_min_r_squared: float = 0.7
    risk_reward_min_ratio: float = 1.5
    
    # 添加缺少的属性
    min_duration_hours: int = 12        # 最小持续时间（小时）
    max_duration_hours: int = 168       # 最大持续时间（小时）


class EnhancedPeakDetector:
    """增强型峰值检测器 - 结合专家建议和scipy.find_peaks"""
    
    def __init__(self, config: DivergenceDetectionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def detect_peaks_with_scipy(self, data: List[float], volumes: List[float] = None) -> Tuple[List[Peak], List[Peak]]:
        """使用scipy.find_peaks进行峰值检测"""
        if len(data) < self.config.min_peak_distance * 2:
            return [], []
        
        data_array = np.array(data)
        
        # 计算prominence阈值（根据专家建议）
        prominence_threshold = self.config.prominence_multiplier * np.std(data_array)
        
        # 检测峰值
        peak_indices, peak_properties = find_peaks(
            data_array, 
            distance=self.config.min_peak_distance,
            prominence=prominence_threshold
        )
        
        # 检测谷值
        valley_indices, valley_properties = find_peaks(
            -data_array, 
            distance=self.config.min_peak_distance,
            prominence=prominence_threshold
        )
        
        # 转换为Peak对象
        peaks = []
        for idx in peak_indices:
            peak = Peak(
                index=idx,
                value=data_array[idx],
                timestamp=datetime.now() - timedelta(hours=int(len(data) - idx)),
                peak_type='high',
                volume=volumes[idx] if volumes else 0.0
            )
            peaks.append(peak)
        
        valleys = []
        for idx in valley_indices:
            valley = Peak(
                index=idx,
                value=data_array[idx],
                timestamp=datetime.now() - timedelta(hours=int(len(data) - idx)),
                peak_type='low',
                volume=volumes[idx] if volumes else 0.0
            )
            valleys.append(valley)
        
        return peaks, valleys
    
    def filter_significant_peaks(self, peaks: List[Peak]) -> List[Peak]:
        """过滤显著峰值 - 增强版"""
        if len(peaks) < 2:
            return peaks
        
        # 按时间排序
        sorted_peaks = sorted(peaks, key=lambda p: p.index)
        
        # 计算动态阈值
        values = [p.value for p in sorted_peaks]
        value_std = np.std(values)
        threshold = value_std * self.config.prominence_multiplier
        
        filtered_peaks = []
        for i, peak in enumerate(sorted_peaks):
            # 检查是否与前一个峰值有足够差异
            if i == 0:
                filtered_peaks.append(peak)
            else:
                prev_peak = filtered_peaks[-1]
                value_diff = abs(peak.value - prev_peak.value)
                time_diff = abs(peak.index - prev_peak.index)
                
                if value_diff > threshold and time_diff >= self.config.min_peak_distance:
                    filtered_peaks.append(peak)
        
        return filtered_peaks


class ConsecutiveDivergenceDetector:
    """连续背离检测器 - 实现专家建议的连续背离检测"""
    
    def __init__(self, config: DivergenceDetectionConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def detect_consecutive_divergence(self, price_extrema: List[Peak], macd_extrema: List[Peak], 
                                    prices: List[float], macd_values: List[float], 
                                    is_bearish: bool = True) -> List[Dict]:
        """检测连续背离模式"""
        if len(price_extrema) < self.config.min_consecutive_count or len(macd_extrema) < self.config.min_consecutive_count:
            return []
        
        # 确保按时间排序
        price_extrema = sorted(price_extrema, key=lambda p: p.index)
        macd_extrema = sorted(macd_extrema, key=lambda p: p.index)
        
        consecutive_patterns = []
        
        # 滑动窗口检查连续背离
        for start_idx in range(len(price_extrema) - self.config.min_consecutive_count + 1):
            price_sequence = price_extrema[start_idx:start_idx + self.config.min_consecutive_count]
            
            # 为价格序列找到对应的MACD极值
            macd_sequence = []
            for price_peak in price_sequence:
                closest_macd = self._find_closest_peak_enhanced(macd_extrema, price_peak.index)
                if closest_macd:
                    macd_sequence.append(closest_macd)
            
            # 检查是否有足够的对应MACD峰值
            if len(macd_sequence) != len(price_sequence):
                continue
            
            # 验证连续背离
            divergence_count = 0
            total_strength = 0.0
            
            for i in range(1, len(price_sequence)):
                price_diff = price_sequence[i].value - price_sequence[i-1].value
                macd_diff = macd_sequence[i].value - macd_sequence[i-1].value
                
                # 背离条件检查
                if is_bearish:
                    # 看跌背离：价格上涨，MACD下降
                    if price_diff > 0 and macd_diff < 0:
                        strength = abs(macd_diff / price_diff) if price_diff != 0 else 0
                        if strength > self.config.min_divergence_gap:
                            divergence_count += 1
                            total_strength += strength
                else:
                    # 看涨背离：价格下跌，MACD上升
                    if price_diff < 0 and macd_diff > 0:
                        strength = abs(macd_diff / price_diff) if price_diff != 0 else 0
                        if strength > self.config.min_divergence_gap:
                            divergence_count += 1
                            total_strength += strength
            
            # 如果发现足够的连续背离
            if divergence_count >= self.config.min_consecutive_count - 1:
                avg_strength = total_strength / divergence_count if divergence_count > 0 else 0
                confidence = min((avg_strength / self.config.min_divergence_gap) * 0.5 + 0.5, 1.0)
                
                pattern = {
                    'type': 'bearish' if is_bearish else 'bullish',
                    'price_sequence': price_sequence,
                    'macd_sequence': macd_sequence,
                    'strength': avg_strength,
                    'confidence': confidence,
                    'divergence_count': divergence_count,
                    'start_time': price_sequence[0].timestamp,
                    'end_time': price_sequence[-1].timestamp
                }
                
                consecutive_patterns.append(pattern)
        
        return consecutive_patterns
    
    def _find_closest_peak_enhanced(self, peaks: List[Peak], target_index: int) -> Optional[Peak]:
        """增强版最近峰值查找"""
        if not peaks:
            return None
        
        min_distance = float('inf')
        closest_peak = None
        
        for peak in peaks:
            distance = abs(peak.index - target_index)
            if distance <= self.config.time_alignment_tolerance and distance < min_distance:
                min_distance = distance
                closest_peak = peak
        
        return closest_peak


# 现在更新主要的MACDDivergenceDetector类
class MACDDivergenceDetector:
    """
    MACD背离检测器 - 改进版
    实现专家建议的连续检测、强度过滤和动态阈值
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 获取配置参数
        self.macd_config = self.config.get_macd_divergence_config()
        self.lookback_period = self.macd_config.lookback_period
        self.min_peak_distance = self.macd_config.min_peak_distance
        self.prominence_threshold = self.macd_config.prominence_threshold
        self.confirmation_candles = self.macd_config.confirmation_candles
        self.min_r_squared = self.macd_config.min_r_squared
        self.min_trend_points = self.macd_config.min_trend_points
        self.min_divergence_strength = self.macd_config.min_divergence_strength
        self.min_duration = self.macd_config.min_duration_hours
        self.max_duration = self.macd_config.max_duration_hours
        self.dynamic_threshold = self.macd_config.dynamic_threshold
        self.min_significance = self.macd_config.min_significance
        
        # 新增连续检测参数
        self.continuous_detection = self.macd_config.continuous_detection
        self.consecutive_signals = self.macd_config.consecutive_signals
        self.prominence_mult = self.macd_config.prominence_mult
        self.strength_filter = self.macd_config.strength_filter
        
        # 初始化子组件
        self.peak_detector = PeakDetector(self.macd_config)
        self.trend_analyzer = TrendLineAnalyzer(self.macd_config)
        self.indicator_calc = TechnicalIndicatorCalculator(config_manager)
        
        # 用于连续检测的状态
        self.recent_divergences: List[DivergencePattern] = []
        self.signal_history: List[Dict] = []
        
        self.logger.info("MACD背离检测器初始化完成（改进版）")
    
    @performance_monitor
    def detect_divergence_enhanced(self, df: pd.DataFrame, symbol: str) -> Optional[DivergencePattern]:
        """
        增强型背离检测 - 实现专家建议的连续检测和强度过滤
        
        Args:
            df: K线数据
            symbol: 交易品种
            
        Returns:
            Optional[DivergencePattern]: 检测到的背离模式
        """
        try:
            if len(df) < self.lookback_period:
                return None
            
            # 1. 计算MACD指标
            macd_result = self.indicator_calc.calculate_macd(df)
            if not macd_result:
                return None
            
            # 2. 提取数据
            prices = df['close'].values
            macd_values = macd_result.macd.values
            signal_values = macd_result.signal.values
            histogram = macd_result.histogram.values
            
            # 3. 检测价格和MACD峰值
            price_peaks, price_valleys = self.peak_detector.detect_peaks(prices, df['volume'].values)
            macd_peaks, macd_valleys = self.peak_detector.detect_peaks(macd_values)
            
            # 4. 过滤显著峰值（增强版）
            price_peaks = self._filter_peaks_enhanced(price_peaks, prices)
            price_valleys = self._filter_valleys_enhanced(price_valleys, prices)
            macd_peaks = self._filter_peaks_enhanced(macd_peaks, macd_values)
            macd_valleys = self._filter_valleys_enhanced(macd_valleys, macd_values)
            
            # 5. 检测连续背离模式
            continuous_pattern = self._detect_continuous_divergence(
                price_peaks, price_valleys, macd_peaks, macd_valleys, 
                prices, macd_values, df
            )
            
            if continuous_pattern:
                # 6. 强度过滤
                if self._validate_signal_strength(continuous_pattern):
                    # 7. 记录信号历史
                    self._record_signal_history(continuous_pattern, symbol)
                    
                    self.logger.info(f"检测到连续背离模式: {symbol}, 类型: {continuous_pattern.divergence_type.value}")
                    return continuous_pattern
            
            return None
            
        except Exception as e:
            self.logger.error(f"增强型背离检测失败: {symbol} - {e}")
            return None
    
    def _filter_peaks_enhanced(self, peaks: List[Peak], data: np.ndarray) -> List[Peak]:
        """
        增强型峰值过滤 - 基于动态阈值和显著性
        
        Args:
            peaks: 原始峰值列表
            data: 数据数组
            
        Returns:
            List[Peak]: 过滤后的峰值列表
        """
        if len(peaks) < 2:
            return peaks
        
        # 计算动态阈值
        data_std = np.std(data)
        dynamic_threshold = data_std * self.prominence_mult
        
        # 过滤显著峰值
        filtered_peaks = []
        for i, peak in enumerate(peaks):
            if i == 0:
                filtered_peaks.append(peak)
                continue
            
            prev_peak = filtered_peaks[-1]
            
            # 检查价格差异
            price_diff = abs(peak.value - prev_peak.value)
            
            # 检查时间间隔
            time_diff = abs(peak.index - prev_peak.index)
            
            # 检查显著性
            if peak.peak_type == 'high':
                significance = price_diff / data_std
            else:
                significance = price_diff / data_std
            
            # 应用过滤条件
            if (price_diff > dynamic_threshold and 
                time_diff >= self.min_peak_distance and
                significance >= self.min_significance):
                
                filtered_peaks.append(peak)
        
        return filtered_peaks
    
    def _filter_valleys_enhanced(self, valleys: List[Peak], data: np.ndarray) -> List[Peak]:
        """
        增强型谷值过滤 - 基于动态阈值和显著性
        
        Args:
            valleys: 原始谷值列表
            data: 数据数组
            
        Returns:
            List[Peak]: 过滤后的谷值列表
        """
        return self._filter_peaks_enhanced(valleys, data)
    
    def _detect_continuous_divergence(self, price_peaks: List[Peak], price_valleys: List[Peak],
                                    macd_peaks: List[Peak], macd_valleys: List[Peak],
                                    prices: np.ndarray, macd_values: np.ndarray,
                                    df: pd.DataFrame) -> Optional[DivergencePattern]:
        """
        检测连续背离模式 - 核心改进算法
        
        Args:
            price_peaks: 价格峰值
            price_valleys: 价格谷值
            macd_peaks: MACD峰值
            macd_valleys: MACD谷值
            prices: 价格数组
            macd_values: MACD数组
            df: 原始数据框
            
        Returns:
            Optional[DivergencePattern]: 连续背离模式
        """
        # 检测看涨背离（价格创新低，MACD创新高）
        bullish_pattern = self._analyze_bullish_divergence_continuous(
            price_valleys, macd_valleys, prices, macd_values, df
        )
        
        # 检测看跌背离（价格创新高，MACD创新低）
        bearish_pattern = self._analyze_bearish_divergence_continuous(
            price_peaks, macd_peaks, prices, macd_values, df
        )
        
        # 选择最强的信号
        if bullish_pattern and bearish_pattern:
            return bullish_pattern if bullish_pattern.confidence > bearish_pattern.confidence else bearish_pattern
        elif bullish_pattern:
            return bullish_pattern
        elif bearish_pattern:
            return bearish_pattern
        else:
            return None
    
    def _analyze_bullish_divergence_continuous(self, price_valleys: List[Peak], macd_valleys: List[Peak],
                                             prices: np.ndarray, macd_values: np.ndarray,
                                             df: pd.DataFrame) -> Optional[DivergencePattern]:
        """
        分析连续看涨背离
        
        Args:
            price_valleys: 价格谷值
            macd_valleys: MACD谷值
            prices: 价格数组
            macd_values: MACD数组
            df: 原始数据框
            
        Returns:
            Optional[DivergencePattern]: 看涨背离模式
        """
        if len(price_valleys) < self.consecutive_signals or len(macd_valleys) < self.consecutive_signals:
            return None
        
        # 获取最近的谷值
        recent_price_valleys = price_valleys[-self.consecutive_signals:]
        recent_macd_valleys = macd_valleys[-self.consecutive_signals:]
        
        # 检查连续性
        consecutive_count = 0
        for i in range(1, len(recent_price_valleys)):
            price_valley_current = recent_price_valleys[i]
            price_valley_prev = recent_price_valleys[i-1]
            
            # 寻找对应的MACD谷值
            macd_valley_current = self._find_closest_valley(price_valley_current, recent_macd_valleys)
            macd_valley_prev = self._find_closest_valley(price_valley_prev, recent_macd_valleys)
            
            if macd_valley_current and macd_valley_prev:
                # 检查背离条件：价格新低，MACD新高
                if (price_valley_current.value < price_valley_prev.value and 
                    macd_valley_current.value > macd_valley_prev.value):
                    consecutive_count += 1
        
        # 要求至少有指定数量的连续背离
        if consecutive_count >= self.consecutive_signals - 1:
            # 创建趋势线
            price_trend = self.trend_analyzer.create_trend_line(recent_price_valleys)
            macd_trend = self.trend_analyzer.create_trend_line(recent_macd_valleys)
            
            if price_trend and macd_trend:
                # 计算背离强度
                divergence_strength = self._calculate_divergence_strength(
                    price_trend, macd_trend, DivergenceType.BULLISH_REGULAR
                )
                
                # 检查强度过滤
                if divergence_strength >= self.strength_filter:
                    return self._create_divergence_pattern(
                        DivergenceType.BULLISH_REGULAR,
                        price_trend, macd_trend, divergence_strength,
                        recent_price_valleys[0].timestamp, recent_price_valleys[-1].timestamp
                    )
        
        return None
    
    def _analyze_bearish_divergence_continuous(self, price_peaks: List[Peak], macd_peaks: List[Peak],
                                             prices: np.ndarray, macd_values: np.ndarray,
                                             df: pd.DataFrame) -> Optional[DivergencePattern]:
        """
        分析连续看跌背离
        
        Args:
            price_peaks: 价格峰值
            macd_peaks: MACD峰值
            prices: 价格数组
            macd_values: MACD数组
            df: 原始数据框
            
        Returns:
            Optional[DivergencePattern]: 看跌背离模式
        """
        if len(price_peaks) < self.consecutive_signals or len(macd_peaks) < self.consecutive_signals:
            return None
        
        # 获取最近的峰值
        recent_price_peaks = price_peaks[-self.consecutive_signals:]
        recent_macd_peaks = macd_peaks[-self.consecutive_signals:]
        
        # 检查连续性
        consecutive_count = 0
        for i in range(1, len(recent_price_peaks)):
            price_peak_current = recent_price_peaks[i]
            price_peak_prev = recent_price_peaks[i-1]
            
            # 寻找对应的MACD峰值
            macd_peak_current = self._find_closest_peak(price_peak_current, recent_macd_peaks)
            macd_peak_prev = self._find_closest_peak(price_peak_prev, recent_macd_peaks)
            
            if macd_peak_current and macd_peak_prev:
                # 检查背离条件：价格新高，MACD新低
                if (price_peak_current.value > price_peak_prev.value and 
                    macd_peak_current.value < macd_peak_prev.value):
                    consecutive_count += 1
        
        # 要求至少有指定数量的连续背离
        if consecutive_count >= self.consecutive_signals - 1:
            # 创建趋势线
            price_trend = self.trend_analyzer.create_trend_line(recent_price_peaks)
            macd_trend = self.trend_analyzer.create_trend_line(recent_macd_peaks)
            
            if price_trend and macd_trend:
                # 计算背离强度
                divergence_strength = self._calculate_divergence_strength(
                    price_trend, macd_trend, DivergenceType.BEARISH_REGULAR
                )
                
                # 检查强度过滤
                if divergence_strength >= self.strength_filter:
                    return self._create_divergence_pattern(
                        DivergenceType.BEARISH_REGULAR,
                        price_trend, macd_trend, divergence_strength,
                        recent_price_peaks[0].timestamp, recent_price_peaks[-1].timestamp
                    )
        
        return None
    
    def _find_closest_valley(self, price_valley: Peak, macd_valleys: List[Peak]) -> Optional[Peak]:
        """
        寻找最接近的MACD谷值
        
        Args:
            price_valley: 价格谷值
            macd_valleys: MACD谷值列表
            
        Returns:
            Optional[Peak]: 最接近的MACD谷值
        """
        if not macd_valleys:
            return None
        
        # 找到时间最接近的谷值
        min_distance = float('inf')
        closest_valley = None
        
        for macd_valley in macd_valleys:
            distance = abs(macd_valley.index - price_valley.index)
            if distance < min_distance:
                min_distance = distance
                closest_valley = macd_valley
        
        # 检查时间距离是否在合理范围内
        if min_distance <= self.min_peak_distance * 2:
            return closest_valley
        
        return None
    
    def _find_closest_peak(self, price_peak: Peak, macd_peaks: List[Peak]) -> Optional[Peak]:
        """
        寻找最接近的MACD峰值
        
        Args:
            price_peak: 价格峰值
            macd_peaks: MACD峰值列表
            
        Returns:
            Optional[Peak]: 最接近的MACD峰值
        """
        return self._find_closest_valley(price_peak, macd_peaks)
    
    def _calculate_divergence_strength(self, price_trend: TrendLine, macd_trend: TrendLine,
                                     divergence_type: DivergenceType) -> float:
        """
        计算背离强度 - 改进版
        
        Args:
            price_trend: 价格趋势线
            macd_trend: MACD趋势线
            divergence_type: 背离类型
            
        Returns:
            float: 背离强度 (0-1)
        """
        # 基础强度：基于斜率差异
        price_slope = abs(price_trend.slope)
        macd_slope = abs(macd_trend.slope)
        
        # 背离强度：斜率方向相反时强度更高
        if divergence_type == DivergenceType.BULLISH_REGULAR:
            # 看涨背离：价格下降，MACD上升
            if price_trend.slope < 0 and macd_trend.slope > 0:
                slope_strength = (price_slope + macd_slope) / 2
            else:
                slope_strength = 0.1
        elif divergence_type == DivergenceType.BEARISH_REGULAR:
            # 看跌背离：价格上升，MACD下降
            if price_trend.slope > 0 and macd_trend.slope < 0:
                slope_strength = (price_slope + macd_slope) / 2
            else:
                slope_strength = 0.1
        else:
            slope_strength = 0.1
        
        # R²强度：趋势线拟合度
        r_squared_strength = (price_trend.r_squared + macd_trend.r_squared) / 2
        
        # 时间强度：持续时间合理性
        time_strength = 1.0  # 暂时固定为1.0
        
        # 综合强度
        total_strength = (slope_strength * 0.4 + r_squared_strength * 0.4 + time_strength * 0.2)
        
        return min(1.0, total_strength)
    
    def _create_divergence_pattern(self, divergence_type: DivergenceType,
                                 price_trend: TrendLine, macd_trend: TrendLine,
                                 strength: float, start_time: datetime, end_time: datetime) -> DivergencePattern:
        """
        创建背离模式对象
        
        Args:
            divergence_type: 背离类型
            price_trend: 价格趋势线
            macd_trend: MACD趋势线
            strength: 背离强度
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DivergencePattern: 背离模式对象
        """
        # 计算信号强度
        signal_strength = self.calculate_signal_strength(strength, 2.0)  # 假设风险回报比为2.0
        
        # 计算预期变动
        expected_move = self._calculate_expected_move(price_trend, divergence_type)
        
        pattern = DivergencePattern(
            divergence_type=divergence_type,
            price_trend=price_trend,
            macd_trend=macd_trend,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            signal_strength=signal_strength,
            confidence=strength,
            risk_reward_ratio=2.0,
            expected_move=expected_move,
            supporting_indicators=["MACD", "连续背离"],
            metadata={
                "consecutive_signals": self.consecutive_signals,
                "strength_filter": self.strength_filter,
                "detection_method": "enhanced_continuous"
            }
        )
        
        return pattern
    
    def _calculate_expected_move(self, price_trend: TrendLine, divergence_type: DivergenceType) -> float:
        """
        计算预期价格变动
        
        Args:
            price_trend: 价格趋势线
            divergence_type: 背离类型
            
        Returns:
            float: 预期变动百分比
        """
        # 基于趋势线斜率和历史数据计算
        base_move = abs(price_trend.slope) * 10  # 基础变动
        
        # 根据背离类型调整
        if divergence_type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BEARISH_REGULAR]:
            type_multiplier = 1.0
        else:
            type_multiplier = 0.8
        
        expected_move = base_move * type_multiplier
        
        # 限制在合理范围内
        return min(0.1, max(0.01, expected_move))
    
    def _validate_signal_strength(self, pattern: DivergencePattern) -> bool:
        """
        验证信号强度
        
        Args:
            pattern: 背离模式
            
        Returns:
            bool: 是否通过强度验证
        """
        # 检查置信度
        if pattern.confidence < self.strength_filter:
            return False
        
        # 检查趋势线质量
        if pattern.price_trend.r_squared < self.min_r_squared:
            return False
        
        if pattern.macd_trend.r_squared < self.min_r_squared:
            return False
        
        # 检查持续时间
        duration_hours = pattern.duration.total_seconds() / 3600
        if duration_hours < self.min_duration or duration_hours > self.max_duration:
            return False
        
        return True
    
    def _record_signal_history(self, pattern: DivergencePattern, symbol: str):
        """
        记录信号历史
        
        Args:
            pattern: 背离模式
            symbol: 交易品种
        """
        signal_record = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'divergence_type': pattern.divergence_type.value,
            'confidence': pattern.confidence,
            'signal_strength': pattern.signal_strength.value,
            'expected_move': pattern.expected_move
        }
        
        self.signal_history.append(signal_record)
        
        # 限制历史记录数量
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-50:]
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        获取信号统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.signal_history:
            return {}
        
        # 计算统计信息
        total_signals = len(self.signal_history)
        avg_confidence = sum(s['confidence'] for s in self.signal_history) / total_signals
        
        # 按类型统计
        type_counts = {}
        for signal in self.signal_history:
            signal_type = signal['divergence_type']
            type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
        
        return {
            'total_signals': total_signals,
            'average_confidence': avg_confidence,
            'signal_type_distribution': type_counts,
            'recent_signals': self.signal_history[-10:],
            'detection_mode': 'enhanced_continuous'
        }


# 主要导出接口
__all__ = [
    'DivergenceType',
    'SignalStrength',
    'Peak',
    'TrendLine',
    'DivergencePattern',
    'MACDDivergenceSignal',
    'PeakDetector',
    'TrendLineAnalyzer',
    'DivergenceValidator',
    'MACDDivergenceDetector'
] 