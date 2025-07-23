#!/usr/bin/env python3
"""
完整的MACD背离检测器
集成专家建议的所有优化和改进
适用于真实市场数据的背离检测
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
from scipy.signal import find_peaks
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# 设置高精度计算
getcontext().prec = 50

# 创建logger
logger = logging.getLogger(__name__)


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
class MACDResult:
    """MACD指标结果"""
    macd_line: float
    signal_line: float
    histogram: float
    fast_ema: float
    slow_ema: float
    timestamp: datetime
    
    def __post_init__(self):
        """计算MACD基本特征"""
        self.divergence = self.macd_line - self.signal_line
        self.momentum = abs(self.histogram)
        self.trend_strength = abs(self.macd_line)


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
class DivergenceSignal:
    """背离信号数据结构"""
    divergence_type: DivergenceType
    signal_strength: SignalStrength
    confidence: float
    risk_reward_ratio: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    expected_return: float
    max_risk: float
    entry_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergenceDetectionConfig:
    """背离检测配置参数"""
    # 专家建议的核心参数
    lookback_period: int = 100               # 检测窗口（增加到100）
    min_peak_distance: int = 3              # 峰值最小间隔
    prominence_multiplier: float = 0.15     # 噪音过滤倍数（更宽松）
    min_divergence_gap: float = 0.02        # 最小背离强度阈值（更宽松）
    min_consecutive_count: int = 2          # 最小连续背离次数
    time_alignment_tolerance: int = 8       # 时间对齐容忍度（更宽松）
    max_time_window: int = 50              # 最大时间窗口
    
    # 进阶参数
    min_macd_threshold: float = 0.01        # MACD最小阈值
    min_price_change_pct: float = 0.5       # 最小价格变化百分比
    peak_confirmation_period: int = 2       # 峰值确认周期
    trend_strength_threshold: float = 0.3   # 趋势强度阈值
    
    # 风险管理参数
    risk_reward_min_ratio: float = 1.2      # 最小风险回报比
    stop_loss_pct: float = 2.0             # 止损百分比
    take_profit_pct: float = 4.0           # 止盈百分比
    position_size_pct: float = 1.0         # 仓位大小百分比


class CompleteMACDDivergenceDetector:
    """完整的MACD背离检测器"""
    
    def __init__(self, config_manager=None):
        """初始化MACD背离检测器"""
        if config_manager:
            self.config = config_manager.get_macd_divergence_config()
        else:
            # 使用默认配置
            from config.config_manager import MACDDivergenceConfig
            self.config = MACDDivergenceConfig()
        
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 添加缓存机制
        self.cache_results = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5分钟缓存
        
        logger.info(f"完整MACD背离检测器初始化完成 - 回看周期: {self.config.lookback_period}")
    
    def detect_divergence(self, prices: List[float], macd_results: List[MACDResult], 
                         volumes: List[float] = None, symbol: str = "", 
                         timeframe: str = "1h") -> List[DivergenceSignal]:
        """主要的背离检测方法"""
        
        if len(prices) < self.config.lookback_period or len(macd_results) != len(prices):
            return []
        
        # 提取最近数据
        recent_prices = prices[-self.config.lookback_period:]
        recent_macd_results = macd_results[-self.config.lookback_period:]
        recent_volumes = volumes[-self.config.lookback_period:] if volumes else None
        
        # 提取MACD数据
        macd_lines = [r.macd_line for r in recent_macd_results]
        macd_histograms = [r.histogram for r in recent_macd_results]
        
        # 峰值检测
        price_peaks, price_valleys = self._detect_peaks_enhanced(recent_prices)
        macd_peaks, macd_valleys = self._detect_peaks_enhanced(macd_histograms)
        
        # 过滤有效峰值
        valid_price_peaks = self._filter_valid_peaks(price_peaks, recent_prices)
        valid_price_valleys = self._filter_valid_peaks(price_valleys, recent_prices)
        valid_macd_peaks = self._filter_valid_peaks(macd_peaks, macd_histograms)
        valid_macd_valleys = self._filter_valid_peaks(macd_valleys, macd_histograms)
        
        signals = []
        
        # 检测看跌背离
        bearish_signals = self._detect_bearish_divergence(
            valid_price_peaks, valid_macd_peaks, 
            recent_prices, macd_histograms, symbol, timeframe
        )
        signals.extend(bearish_signals)
        
        # 检测看涨背离
        bullish_signals = self._detect_bullish_divergence(
            valid_price_valleys, valid_macd_valleys, 
            recent_prices, macd_histograms, symbol, timeframe
        )
        signals.extend(bullish_signals)
        
        # 检测隐藏背离
        hidden_signals = self._detect_hidden_divergence(
            valid_price_peaks, valid_price_valleys,
            valid_macd_peaks, valid_macd_valleys,
            recent_prices, macd_histograms, symbol, timeframe
        )
        signals.extend(hidden_signals)
        
        # 更新统计
        self.detection_count += 1
        self.signal_count += len(signals)
        
        return signals
    
    def _detect_peaks_enhanced(self, data: List[float]) -> Tuple[List[Peak], List[Peak]]:
        """增强版峰值检测"""
        if len(data) < self.config.min_peak_distance * 2:
            return [], []
        
        data_array = np.array(data)
        
        # 计算prominence阈值
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
                peak_type='high'
            )
            peaks.append(peak)
        
        valleys = []
        for idx in valley_indices:
            valley = Peak(
                index=idx,
                value=data_array[idx],
                timestamp=datetime.now() - timedelta(hours=int(len(data) - idx)),
                peak_type='low'
            )
            valleys.append(valley)
        
        return peaks, valleys
    
    def _filter_valid_peaks(self, peaks: List[Peak], data: List[float]) -> List[Peak]:
        """过滤有效峰值"""
        if len(peaks) < 2:
            return peaks
        
        # 按时间排序
        sorted_peaks = sorted(peaks, key=lambda p: p.index)
        
        # 计算动态阈值
        data_array = np.array(data)
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        
        filtered_peaks = []
        for peak in sorted_peaks:
            # 检查峰值是否显著
            if peak.peak_type == 'high':
                if peak.value > mean_val + std_val * 0.5:
                    filtered_peaks.append(peak)
            else:  # low
                if peak.value < mean_val - std_val * 0.5:
                    filtered_peaks.append(peak)
        
        return filtered_peaks
    
    def _detect_bearish_divergence(self, price_peaks: List[Peak], macd_peaks: List[Peak],
                                  prices: List[float], macd_histograms: List[float],
                                  symbol: str, timeframe: str) -> List[DivergenceSignal]:
        """检测看跌背离"""
        signals = []
        
        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return signals
        
        # 寻找连续的价格高点创新高，但MACD高点不创新高
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
                            # 验证背离强度
                            price_change = (price_peak2.value - price_peak1.value) / price_peak1.value
                            macd_change = (macd_peak2.value - macd_peak1.value) / abs(macd_peak1.value)
                            
                            divergence_strength = abs(macd_change) / price_change if price_change > 0 else 0
                            
                            if divergence_strength > self.config.min_divergence_gap:
                                signal = self._create_divergence_signal(
                                    DivergenceType.BEARISH_REGULAR,
                                    price_peak2, macd_peak2,
                                    divergence_strength, prices, symbol, timeframe
                                )
                                if signal:
                                    signals.append(signal)
        
        return signals
    
    def _detect_bullish_divergence(self, price_valleys: List[Peak], macd_valleys: List[Peak],
                                  prices: List[float], macd_histograms: List[float],
                                  symbol: str, timeframe: str) -> List[DivergenceSignal]:
        """检测看涨背离"""
        signals = []
        
        if len(price_valleys) < 2 or len(macd_valleys) < 2:
            return signals
        
        # 寻找连续的价格低点创新低，但MACD低点不创新低
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
                            # 验证背离强度
                            price_change = abs(price_valley2.value - price_valley1.value) / price_valley1.value
                            macd_change = abs(macd_valley2.value - macd_valley1.value) / abs(macd_valley1.value)
                            
                            divergence_strength = macd_change / price_change if price_change > 0 else 0
                            
                            if divergence_strength > self.config.min_divergence_gap:
                                signal = self._create_divergence_signal(
                                    DivergenceType.BULLISH_REGULAR,
                                    price_valley2, macd_valley2,
                                    divergence_strength, prices, symbol, timeframe
                                )
                                if signal:
                                    signals.append(signal)
        
        return signals
    
    def _detect_hidden_divergence(self, price_peaks: List[Peak], price_valleys: List[Peak],
                                 macd_peaks: List[Peak], macd_valleys: List[Peak],
                                 prices: List[float], macd_histograms: List[float],
                                 symbol: str, timeframe: str) -> List[DivergenceSignal]:
        """检测隐藏背离"""
        signals = []
        
        # 隐藏看跌背离：价格不创新高，但MACD创新高
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            for i in range(len(price_peaks) - 1):
                for j in range(i + 1, len(price_peaks)):
                    price_peak1 = price_peaks[i]
                    price_peak2 = price_peaks[j]
                    
                    # 价格不创新高
                    if price_peak2.value <= price_peak1.value:
                        macd_peak1 = self._find_closest_peak(macd_peaks, price_peak1.index)
                        macd_peak2 = self._find_closest_peak(macd_peaks, price_peak2.index)
                        
                        if macd_peak1 and macd_peak2:
                            # MACD创新高（隐藏背离）
                            if macd_peak2.value > macd_peak1.value:
                                price_change = abs(price_peak2.value - price_peak1.value) / price_peak1.value
                                macd_change = (macd_peak2.value - macd_peak1.value) / abs(macd_peak1.value)
                                
                                if price_change > 0.001 and macd_change > self.config.min_divergence_gap:
                                    signal = self._create_divergence_signal(
                                        DivergenceType.BEARISH_HIDDEN,
                                        price_peak2, macd_peak2,
                                        macd_change, prices, symbol, timeframe
                                    )
                                    if signal:
                                        signals.append(signal)
        
        # 隐藏看涨背离：价格不创新低，但MACD创新低
        if len(price_valleys) >= 2 and len(macd_valleys) >= 2:
            for i in range(len(price_valleys) - 1):
                for j in range(i + 1, len(price_valleys)):
                    price_valley1 = price_valleys[i]
                    price_valley2 = price_valleys[j]
                    
                    # 价格不创新低
                    if price_valley2.value >= price_valley1.value:
                        macd_valley1 = self._find_closest_peak(macd_valleys, price_valley1.index)
                        macd_valley2 = self._find_closest_peak(macd_valleys, price_valley2.index)
                        
                        if macd_valley1 and macd_valley2:
                            # MACD创新低（隐藏背离）
                            if macd_valley2.value < macd_valley1.value:
                                price_change = abs(price_valley2.value - price_valley1.value) / price_valley1.value
                                macd_change = abs(macd_valley2.value - macd_valley1.value) / abs(macd_valley1.value)
                                
                                if price_change > 0.001 and macd_change > self.config.min_divergence_gap:
                                    signal = self._create_divergence_signal(
                                        DivergenceType.BULLISH_HIDDEN,
                                        price_valley2, macd_valley2,
                                        macd_change, prices, symbol, timeframe
                                    )
                                    if signal:
                                        signals.append(signal)
        
        return signals
    
    def _find_closest_peak(self, peaks: List[Peak], target_index: int) -> Optional[Peak]:
        """寻找最接近的峰值"""
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
    
    def _create_divergence_signal(self, divergence_type: DivergenceType, 
                                 price_peak: Peak, macd_peak: Peak,
                                 divergence_strength: float, prices: List[float],
                                 symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """创建背离信号"""
        
        # 计算信号强度
        signal_strength = self._calculate_signal_strength(divergence_strength)
        
        # 计算置信度
        confidence = min(divergence_strength / 0.1, 1.0)
        
        # 计算交易参数
        entry_price = price_peak.value
        
        if divergence_type in [DivergenceType.BEARISH_REGULAR, DivergenceType.BEARISH_HIDDEN]:
            # 看跌信号
            stop_loss = entry_price * (1 + self.config.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.config.take_profit_pct / 100)
            expected_return = -self.config.take_profit_pct
        else:
            # 看涨信号
            stop_loss = entry_price * (1 - self.config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.config.take_profit_pct / 100)
            expected_return = self.config.take_profit_pct
        
        # 计算风险回报比
        risk = abs(entry_price - stop_loss) / entry_price * 100
        reward = abs(take_profit - entry_price) / entry_price * 100
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # 检查风险回报比
        if risk_reward_ratio < self.config.risk_reward_min_ratio:
            return None
        
        # 计算仓位大小
        position_size = self.config.position_size_pct
        
        # 计算最大风险
        max_risk = risk * position_size / 100
        
        # 创建信号
        signal = DivergenceSignal(
            divergence_type=divergence_type,
            signal_strength=signal_strength,
            confidence=confidence,
            risk_reward_ratio=risk_reward_ratio,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            expected_return=expected_return,
            max_risk=max_risk,
            entry_time=price_peak.timestamp,
            metadata={
                'symbol': symbol,
                'timeframe': timeframe,
                'divergence_strength': divergence_strength,
                'price_peak_index': price_peak.index,
                'macd_peak_index': macd_peak.index,
                'detection_time': datetime.now().isoformat()
            }
        )
        
        return signal
    
    def _calculate_signal_strength(self, divergence_strength: float) -> SignalStrength:
        """计算信号强度"""
        if divergence_strength >= 0.15:
            return SignalStrength.VERY_STRONG
        elif divergence_strength >= 0.10:
            return SignalStrength.STRONG
        elif divergence_strength >= 0.06:
            return SignalStrength.MODERATE
        elif divergence_strength >= 0.03:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测统计"""
        return {
            'total_detections': self.detection_count,
            'total_signals': self.signal_count,
            'signal_rate': self.signal_count / max(self.detection_count, 1),
            'last_detection_time': self.last_detection_time.isoformat(),
            'config': {
                'lookback_period': self.config.lookback_period,
                'min_peak_distance': self.config.min_peak_distance,
                'prominence_multiplier': self.config.prominence_multiplier,
                'min_divergence_gap': self.config.min_divergence_gap,
                'time_alignment_tolerance': self.config.time_alignment_tolerance
            }
        }


# 便捷函数
def detect_macd_divergence(prices: List[float], macd_results: List[MACDResult],
                          volumes: List[float] = None, symbol: str = "",
                          timeframe: str = "1h", 
                          config: Optional[DivergenceDetectionConfig] = None) -> List[DivergenceSignal]:
    """便捷的MACD背离检测函数"""
    detector = CompleteMACDDivergenceDetector(config)
    return detector.detect_divergence(prices, macd_results, volumes, symbol, timeframe)


def create_optimized_config(market_type: str = "crypto") -> DivergenceDetectionConfig:
    """创建优化的配置"""
    if market_type == "crypto":
        # 加密货币市场配置
        return DivergenceDetectionConfig(
            lookback_period=80,
            min_peak_distance=2,
            prominence_multiplier=0.12,
            min_divergence_gap=0.015,
            min_consecutive_count=2,
            time_alignment_tolerance=6,
            stop_loss_pct=1.5,
            take_profit_pct=3.0
        )
    elif market_type == "forex":
        # 外汇市场配置
        return DivergenceDetectionConfig(
            lookback_period=120,
            min_peak_distance=4,
            prominence_multiplier=0.2,
            min_divergence_gap=0.025,
            min_consecutive_count=2,
            time_alignment_tolerance=4,
            stop_loss_pct=1.0,
            take_profit_pct=2.0
        )
    else:
        # 股票市场配置
        return DivergenceDetectionConfig(
            lookback_period=100,
            min_peak_distance=3,
            prominence_multiplier=0.15,
            min_divergence_gap=0.02,
            min_consecutive_count=2,
            time_alignment_tolerance=5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        )


if __name__ == "__main__":
    # 测试示例
    import random
    
    # 生成测试数据
    prices = [100 + i * 0.1 + random.uniform(-2, 2) for i in range(200)]
    macd_results = []
    
    for i, price in enumerate(prices):
        macd_results.append(MACDResult(
            macd_line=random.uniform(-1, 1),
            signal_line=random.uniform(-0.5, 0.5),
            histogram=random.uniform(-0.3, 0.3),
            fast_ema=price * 1.01,
            slow_ema=price * 0.99,
            timestamp=datetime.now() - timedelta(hours=200-i)
        ))
    
    # 使用优化配置
    config = create_optimized_config("crypto")
    
    # 检测背离
    signals = detect_macd_divergence(prices, macd_results, symbol="TEST", config=config)
    
    print(f"检测到 {len(signals)} 个背离信号")
    for i, signal in enumerate(signals):
        print(f"信号 {i+1}: {signal.divergence_type.value}, 强度: {signal.signal_strength.value}, 置信度: {signal.confidence:.3f}") 