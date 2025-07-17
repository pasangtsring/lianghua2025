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
    """MACD背离检测器 - 集成专家建议的优化算法"""
    
    def __init__(self, config: Optional[DivergenceDetectionConfig] = None):
        self.config = config or DivergenceDetectionConfig()
        self.logger = get_logger(__name__)
        
        # 初始化子组件
        self.enhanced_peak_detector = EnhancedPeakDetector(self.config)
        self.consecutive_detector = ConsecutiveDivergenceDetector(self.config)
        
        # 保持原有组件的兼容性
        self.peak_detector = PeakDetector(self.config)
        self.trend_analyzer = TrendLineAnalyzer(self.config)
        self.validator = DivergenceValidator(self.config)
        
        # 性能监控
        self.detection_count = 0
        self.last_detection_time = datetime.now()
    
    @performance_monitor
    def detect_divergence_enhanced(self, prices: List[float], macd_results: List[MACDResult], 
                                 volumes: List[float] = None, symbol: str = "", 
                                 timeframe: str = "1h") -> List[MACDDivergenceSignal]:
        """增强版背离检测 - 结合专家建议"""
        
        if len(prices) < self.config.lookback_period or len(macd_results) != len(prices):
            return []
        
        # 提取最近的数据
        recent_prices = prices[-self.config.lookback_period:]
        recent_macd_results = macd_results[-self.config.lookback_period:]
        recent_volumes = volumes[-self.config.lookback_period:] if volumes else None
        
        # 提取MACD数据
        macd_lines = [r.macd_line for r in recent_macd_results]
        macd_histograms = [r.histogram for r in recent_macd_results]
        
        # 使用增强版峰值检测
        price_peaks, price_valleys = self.enhanced_peak_detector.detect_peaks_with_scipy(
            recent_prices, recent_volumes
        )
        
        macd_peaks, macd_valleys = self.enhanced_peak_detector.detect_peaks_with_scipy(
            macd_histograms, recent_volumes
        )
        
        # 过滤显著峰值
        significant_price_peaks = self.enhanced_peak_detector.filter_significant_peaks(price_peaks)
        significant_price_valleys = self.enhanced_peak_detector.filter_significant_peaks(price_valleys)
        significant_macd_peaks = self.enhanced_peak_detector.filter_significant_peaks(macd_peaks)
        significant_macd_valleys = self.enhanced_peak_detector.filter_significant_peaks(macd_valleys)
        
        divergence_signals = []
        
        # 检测连续看跌背离
        bearish_patterns = self.consecutive_detector.detect_consecutive_divergence(
            significant_price_peaks, significant_macd_peaks, 
            recent_prices, macd_histograms, is_bearish=True
        )
        
        # 检测连续看涨背离
        bullish_patterns = self.consecutive_detector.detect_consecutive_divergence(
            significant_price_valleys, significant_macd_valleys, 
            recent_prices, macd_histograms, is_bearish=False
        )
        
        # 处理检测到的模式
        all_patterns = []
        
        # 转换连续背离为标准格式
        for pattern in bearish_patterns:
            divergence_pattern = self._convert_consecutive_to_standard(
                pattern, DivergenceType.BEARISH_REGULAR, recent_prices, macd_lines, macd_histograms
            )
            if divergence_pattern:
                all_patterns.append(divergence_pattern)
        
        for pattern in bullish_patterns:
            divergence_pattern = self._convert_consecutive_to_standard(
                pattern, DivergenceType.BULLISH_REGULAR, recent_prices, macd_lines, macd_histograms
            )
            if divergence_pattern:
                all_patterns.append(divergence_pattern)
        
        # 验证和生成信号
        for pattern in all_patterns:
            is_valid, confidence, issues = self.validator.validate_divergence(pattern)
            
            if is_valid:
                signal = self._generate_trading_signal(pattern, recent_prices, symbol, timeframe)
                if signal:
                    divergence_signals.append(signal)
                    self.logger.info(f"检测到连续{pattern.divergence_type.value}背离信号 - "
                                   f"置信度: {confidence:.3f}, 强度: {pattern.confidence:.3f}")
        
        self.detection_count += len(divergence_signals)
        return divergence_signals
    
    def _convert_consecutive_to_standard(self, consecutive_pattern: Dict, 
                                       divergence_type: DivergenceType,
                                       prices: List[float], macd_lines: List[float], 
                                       macd_histograms: List[float]) -> Optional[DivergencePattern]:
        """将连续背离模式转换为标准背离模式"""
        try:
            price_peaks = consecutive_pattern['price_sequence']
            macd_peaks = consecutive_pattern['macd_sequence']
            
            # 创建趋势线
            price_trend = self.trend_analyzer.create_trend_line(price_peaks)
            macd_trend = self.trend_analyzer.create_trend_line(macd_peaks)
            
            if not price_trend or not macd_trend:
                return None
            
            # 计算风险回报比
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                divergence_type, price_peaks, prices
            )
            
            # 计算预期移动
            expected_move = self._calculate_expected_move(
                divergence_type, price_peaks, prices, macd_histograms
            )
            
            # 创建背离模式
            pattern = DivergencePattern(
                divergence_type=divergence_type,
                price_trend=price_trend,
                macd_trend=macd_trend,
                start_time=consecutive_pattern['start_time'],
                end_time=consecutive_pattern['end_time'],
                duration=consecutive_pattern['end_time'] - consecutive_pattern['start_time'],
                signal_strength=self._calculate_signal_strength(consecutive_pattern['strength']),
                confidence=consecutive_pattern['confidence'],
                risk_reward_ratio=risk_reward_ratio,
                expected_move=expected_move,
                supporting_indicators=[],
                metadata={
                    'consecutive_count': consecutive_pattern['divergence_count'],
                    'pattern_strength': consecutive_pattern['strength'],
                    'detection_method': 'enhanced_consecutive'
                }
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"转换连续背离模式失败: {e}")
            return None
    
    def _calculate_signal_strength(self, strength_value: float) -> SignalStrength:
        """根据强度值计算信号强度"""
        if strength_value >= 0.8:
            return SignalStrength.VERY_STRONG
        elif strength_value >= 0.6:
            return SignalStrength.STRONG
        elif strength_value >= 0.4:
            return SignalStrength.MODERATE
        elif strength_value >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    @performance_monitor
    def detect_divergence(self, prices: List[float], macd_results: List[MACDResult], 
                         volumes: List[float] = None, symbol: str = "UNKNOWN", 
                         timeframe: str = "1h") -> List[MACDDivergenceSignal]:
        """
        检测MACD背离
        
        Args:
            prices: 价格数据
            macd_results: MACD结果数据
            volumes: 成交量数据（可选）
            symbol: 交易对符号
            timeframe: 时间周期
            
        Returns:
            List[MACDDivergenceSignal]: 背离信号列表
        """
        if len(prices) < self.config.lookback_period or len(macd_results) < self.config.lookback_period:
            logger.warning(f"数据不足：价格数据{len(prices)}，MACD数据{len(macd_results)}")
            return []
        
        # 提取MACD线数据
        macd_lines = [result.macd_line for result in macd_results]
        macd_histograms = [result.histogram for result in macd_results]
        
        # 只使用最近的数据
        recent_prices = prices[-self.config.lookback_period:]
        recent_macd_lines = macd_lines[-self.config.lookback_period:]
        recent_macd_histograms = macd_histograms[-self.config.lookback_period:]
        recent_volumes = volumes[-self.config.lookback_period:] if volumes else None
        
        # 检测价格峰值
        price_peaks, price_valleys = self.peak_detector.detect_peaks(
            recent_prices, recent_volumes
        )
        
        # 检测MACD峰值
        macd_peaks, macd_valleys = self.peak_detector.detect_peaks(
            recent_macd_lines, recent_volumes
        )
        
        # 过滤显著峰值
        significant_price_peaks = self.peak_detector.filter_significant_peaks(price_peaks)
        significant_price_valleys = self.peak_detector.filter_significant_peaks(price_valleys)
        significant_macd_peaks = self.peak_detector.filter_significant_peaks(macd_peaks)
        significant_macd_valleys = self.peak_detector.filter_significant_peaks(macd_valleys)
        
        # 检测背离模式
        divergence_signals = []
        
        # 检测看涨背离（价格低点 vs MACD高点）
        bullish_patterns = self._detect_bullish_divergence(
            significant_price_valleys, significant_macd_valleys,
            recent_prices, recent_macd_lines, recent_macd_histograms
        )
        
        # 检测看跌背离（价格高点 vs MACD低点）
        bearish_patterns = self._detect_bearish_divergence(
            significant_price_peaks, significant_macd_peaks,
            recent_prices, recent_macd_lines, recent_macd_histograms
        )
        
        # 生成交易信号
        all_patterns = bullish_patterns + bearish_patterns
        
        for pattern in all_patterns:
            # 验证背离模式
            is_valid, confidence, issues = self.validator.validate_divergence(pattern)
            
            if is_valid:
                # 计算交易参数
                signal = self._generate_trading_signal(
                    pattern, recent_prices, symbol, timeframe
                )
                
                if signal:
                    divergence_signals.append(signal)
                    logger.info(f"检测到{pattern.divergence_type.value}背离信号 - "
                              f"置信度: {confidence:.3f}, 风险回报比: {pattern.risk_reward_ratio:.2f}")
            else:
                logger.debug(f"背离模式验证失败: {issues}")
        
        return divergence_signals
    
    def _detect_bullish_divergence(self, price_valleys: List[Peak], macd_valleys: List[Peak],
                                  prices: List[float], macd_lines: List[float], 
                                  macd_histograms: List[float]) -> List[DivergencePattern]:
        """检测看涨背离"""
        patterns = []
        
        if len(price_valleys) < 2 or len(macd_valleys) < 2:
            return patterns
        
        # 寻找连续的价格低点和MACD低点
        for i in range(len(price_valleys) - 1):
            for j in range(i + 1, len(price_valleys)):
                price_valley1 = price_valleys[i]
                price_valley2 = price_valleys[j]
                
                # 价格创新低
                if price_valley2.value < price_valley1.value:
                    # 寻找对应的MACD低点
                    macd_valley1 = self._find_closest_peak(macd_valleys, price_valley1.index)
                    macd_valley2 = self._find_closest_peak(macd_valleys, price_valley2.index)
                    
                    if macd_valley1 and macd_valley2:
                        # MACD不创新低（背离）
                        if macd_valley2.value > macd_valley1.value:
                            pattern = self._create_divergence_pattern(
                                DivergenceType.BULLISH_REGULAR,
                                [price_valley1, price_valley2],
                                [macd_valley1, macd_valley2],
                                prices, macd_lines, macd_histograms
                            )
                            if pattern:
                                patterns.append(pattern)
        
        return patterns
    
    def _detect_bearish_divergence(self, price_peaks: List[Peak], macd_peaks: List[Peak],
                                  prices: List[float], macd_lines: List[float], 
                                  macd_histograms: List[float]) -> List[DivergencePattern]:
        """检测看跌背离"""
        patterns = []
        
        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return patterns
        
        # 寻找连续的价格高点和MACD高点
        for i in range(len(price_peaks) - 1):
            for j in range(i + 1, len(price_peaks)):
                price_peak1 = price_peaks[i]
                price_peak2 = price_peaks[j]
                
                # 价格创新高
                if price_peak2.value > price_peak1.value:
                    # 寻找对应的MACD高点
                    macd_peak1 = self._find_closest_peak(macd_peaks, price_peak1.index)
                    macd_peak2 = self._find_closest_peak(macd_peaks, price_peak2.index)
                    
                    if macd_peak1 and macd_peak2:
                        # MACD不创新高（背离）
                        if macd_peak2.value < macd_peak1.value:
                            pattern = self._create_divergence_pattern(
                                DivergenceType.BEARISH_REGULAR,
                                [price_peak1, price_peak2],
                                [macd_peak1, macd_peak2],
                                prices, macd_lines, macd_histograms
                            )
                            if pattern:
                                patterns.append(pattern)
        
        return patterns
    
    def _find_closest_peak(self, peaks: List[Peak], target_index: int, max_distance: int = 20) -> Optional[Peak]:
        """寻找最接近的峰值"""
        if not peaks:
            return None
        
        closest_peak = None
        min_distance = float('inf')
        
        for peak in peaks:
            distance = abs(peak.index - target_index)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_peak = peak
        
        return closest_peak
    
    def _create_divergence_pattern(self, divergence_type: DivergenceType, 
                                 price_peaks: List[Peak], macd_peaks: List[Peak],
                                 prices: List[float], macd_lines: List[float], 
                                 macd_histograms: List[float]) -> Optional[DivergencePattern]:
        """创建背离模式"""
        # 创建价格趋势线
        price_trend = self.trend_analyzer.create_trend_line(price_peaks)
        if not price_trend:
            return None
        
        # 创建MACD趋势线
        macd_trend = self.trend_analyzer.create_trend_line(macd_peaks)
        if not macd_trend:
            return None
        
        # 分析趋势背离
        detected_type, divergence_strength = self.trend_analyzer.analyze_trend_divergence(
            price_trend, macd_trend
        )
        
        # 检查背离类型是否匹配
        if detected_type != divergence_type:
            return None
        
        # 计算时间信息
        start_time = datetime.now() - timedelta(hours=len(prices))
        end_time = datetime.now()
        
        # 计算风险回报比
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            divergence_type, price_peaks, prices
        )
        
        # 计算预期移动幅度
        expected_move = self._calculate_expected_move(
            divergence_type, price_peaks, prices, macd_histograms
        )
        
        # 创建背离模式
        pattern = DivergencePattern(
            divergence_type=divergence_type,
            price_trend=price_trend,
            macd_trend=macd_trend,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,  # 添加duration参数
            signal_strength=self.validator.calculate_signal_strength(
                divergence_strength, risk_reward_ratio
            ),
            confidence=divergence_strength,
            risk_reward_ratio=risk_reward_ratio,
            expected_move=expected_move,
            supporting_indicators=[],
            metadata={
                'price_peaks_count': len(price_peaks),
                'macd_peaks_count': len(macd_peaks),
                'divergence_strength': divergence_strength,
                'trend_line_quality': (price_trend.r_squared + macd_trend.r_squared) / 2
            }
        )
        
        return pattern
    
    def _calculate_risk_reward_ratio(self, divergence_type: DivergenceType, 
                                   peaks: List[Peak], prices: List[float]) -> float:
        """计算风险回报比"""
        if len(peaks) < 2:
            return 1.0
        
        current_price = prices[-1]
        
        if divergence_type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BULLISH_HIDDEN]:
            # 看涨背离
            stop_loss = min(peak.value for peak in peaks) * 0.98  # 2%止损
            take_profit = current_price * 1.06  # 6%止盈
        else:
            # 看跌背离
            stop_loss = max(peak.value for peak in peaks) * 1.02  # 2%止损
            take_profit = current_price * 0.94  # 6%止盈
        
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        
        return reward / risk if risk > 0 else 1.0
    
    def _calculate_expected_move(self, divergence_type: DivergenceType, 
                               peaks: List[Peak], prices: List[float], 
                               macd_histograms: List[float]) -> float:
        """计算预期移动幅度"""
        if len(peaks) < 2:
            return 0.02  # 默认2%
        
        # 基于历史波动率
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                        for i in range(1, len(prices))]
        avg_volatility = np.mean(price_changes)
        
        # 基于MACD柱状图强度
        macd_strength = abs(macd_histograms[-1]) / max(abs(h) for h in macd_histograms[-20:])
        
        # 基于峰值间距
        peak_distance = abs(peaks[-1].index - peaks[0].index)
        distance_factor = min(peak_distance / 20, 2.0)
        
        # 综合计算
        expected_move = avg_volatility * macd_strength * distance_factor * 2.0
        
        return min(expected_move, 0.15)  # 最大15%
    
    def _generate_trading_signal(self, pattern: DivergencePattern, prices: List[float], 
                               symbol: str, timeframe: str) -> Optional[MACDDivergenceSignal]:
        """生成交易信号"""
        current_price = prices[-1]
        
        if pattern.divergence_type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BULLISH_HIDDEN]:
            # 看涨信号
            entry_price = current_price * 1.001  # 小幅上涨确认
            stop_loss = min(peak.value for peak in pattern.price_trend.points) * 0.98
            take_profit = entry_price * (1 + pattern.expected_move)
        else:
            # 看跌信号
            entry_price = current_price * 0.999  # 小幅下跌确认
            stop_loss = max(peak.value for peak in pattern.price_trend.points) * 1.02
            take_profit = entry_price * (1 - pattern.expected_move)
        
        # 计算仓位大小（基于风险管理）
        risk_per_trade = 0.02  # 2%风险
        position_size = risk_per_trade / (abs(entry_price - stop_loss) / entry_price)
        
        # 计算失效水平
        if pattern.divergence_type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BULLISH_HIDDEN]:
            invalidation_level = stop_loss * 0.95
        else:
            invalidation_level = stop_loss * 1.05
        
        # 预期持续时间
        expected_duration = pattern.duration * 2  # 预期持续时间为背离形成时间的2倍
        
        signal = MACDDivergenceSignal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            divergence_pattern=pattern,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_percentage=risk_per_trade,
            expected_duration=expected_duration,
            invalidation_level=invalidation_level,
            notes=f"MACD背离信号 - 类型: {pattern.divergence_type.value}, "
                  f"置信度: {pattern.confidence:.3f}, 强度: {pattern.signal_strength.value}"
        )
        
        return signal
    
    @performance_monitor
    async def detect_divergence_async(self, prices: List[float], macd_results: List[MACDResult], 
                                    volumes: List[float] = None, symbol: str = "UNKNOWN", 
                                    timeframe: str = "1h") -> List[MACDDivergenceSignal]:
        """异步检测MACD背离"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中运行同步检测
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = loop.run_in_executor(
                executor,
                self.detect_divergence,
                prices, macd_results, volumes, symbol, timeframe
            )
            
            return await future
    
    @performance_monitor
    def get_detection_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        return {
            'lookback_period': self.config.lookback_period,
            'min_peak_distance': self.config.min_peak_distance,
            'dynamic_threshold': self.config.dynamic_threshold,
            'cache_size': len(self._cache),
            'detector_components': {
                'peak_detector': type(self.peak_detector).__name__,
                'trend_analyzer': type(self.trend_analyzer).__name__,
                'validator': type(self.validator).__name__
            }
        }
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("MACD背离检测器缓存已清空")


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