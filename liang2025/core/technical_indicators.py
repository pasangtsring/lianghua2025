"""
技术指标计算模块
实现MACD、MA、ATR、波动率等核心技术指标
严格遵循零简化原则，确保计算精度和性能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

from config.config_manager import ConfigManager
from utils.logger import get_logger, performance_monitor

# 设置高精度计算
getcontext().prec = 50

# 创建logger实例
logger = get_logger(__name__)


@dataclass
class IndicatorResult:
    """技术指标计算结果"""
    indicator_name: str
    symbol: str
    timeframe: str
    timestamp: datetime
    values: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    calculation_time: float = 0.0
    confidence: float = 1.0


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
class MAResult:
    """移动平均线结果"""
    ma_type: str  # SMA, EMA, WMA等
    period: int
    value: float
    timestamp: datetime
    trend_direction: str  # UP, DOWN, FLAT
    trend_strength: float
    
    def __post_init__(self):
        """计算MA特征"""
        if self.trend_strength > 0.7:
            self.signal_strength = "STRONG"
        elif self.trend_strength > 0.3:
            self.signal_strength = "MODERATE"
        else:
            self.signal_strength = "WEAK"


@dataclass
class ATRResult:
    """平均真实范围结果"""
    atr_value: float
    volatility_level: str  # LOW, MODERATE, HIGH
    price_range: float
    timestamp: datetime
    
    def __post_init__(self):
        """计算ATR特征"""
        if self.atr_value > 0.05:
            self.volatility_level = "HIGH"
        elif self.atr_value > 0.02:
            self.volatility_level = "MODERATE"
        else:
            self.volatility_level = "LOW"


@dataclass
class RSIResult:
    """RSI指标结果"""
    rsi_value: float
    overbought: bool
    oversold: bool
    timestamp: datetime
    
    def __post_init__(self):
        """计算RSI信号"""
        self.overbought = self.rsi_value > 70
        self.oversold = self.rsi_value < 30
        self.neutral = 30 <= self.rsi_value <= 70
        
        # 添加强度和趋势分析
        if self.rsi_value >= 80:
            self.strength = "Very Strong"
            self.trend = "Overbought"
        elif self.rsi_value >= 70:
            self.strength = "Strong"
            self.trend = "Overbought"
        elif self.rsi_value >= 60:
            self.strength = "Above Average"
            self.trend = "Bullish"
        elif self.rsi_value >= 40:
            self.strength = "Average"
            self.trend = "Neutral"
        elif self.rsi_value >= 30:
            self.strength = "Below Average"
            self.trend = "Bearish"
        elif self.rsi_value >= 20:
            self.strength = "Weak"
            self.trend = "Oversold"
        else:
            self.strength = "Very Weak"
            self.trend = "Oversold"


@dataclass
class BollingerResult:
    """布林带结果"""
    upper_band: float
    middle_band: float
    lower_band: float
    current_price: float
    bandwidth: float
    position: str  # ABOVE_UPPER, BETWEEN, BELOW_LOWER
    timestamp: datetime
    
    def __post_init__(self):
        """计算布林带位置"""
        if self.current_price > self.upper_band:
            self.position = "ABOVE_UPPER"
        elif self.current_price < self.lower_band:
            self.position = "BELOW_LOWER"
        else:
            self.position = "BETWEEN"


@dataclass
class KDJResult:
    """KDJ随机指标结果"""
    k_value: float
    d_value: float
    j_value: float
    timestamp: datetime
    
    def __post_init__(self):
        """计算KDJ信号"""
        self.overbought = self.k_value > 80 and self.d_value > 80
        self.oversold = self.k_value < 20 and self.d_value < 20
        self.golden_cross = self.k_value > self.d_value
        self.death_cross = self.k_value < self.d_value


class TechnicalIndicatorCalculator:
    """技术指标计算器"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.calculation_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 60  # 1分钟缓存
        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    @performance_monitor
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """计算简单移动平均线"""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
        
        return sma_values
    
    @performance_monitor
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """计算指数移动平均线"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = []
        
        # 第一个EMA值使用SMA
        first_ema = sum(prices[:period]) / period
        ema_values.append(first_ema)
        
        # 后续EMA计算
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    @performance_monitor
    def calculate_wma(self, prices: List[float], period: int) -> List[float]:
        """计算加权移动平均线"""
        if len(prices) < period:
            return []
        
        wma_values = []
        weights = list(range(1, period + 1))
        weight_sum = sum(weights)
        
        for i in range(period - 1, len(prices)):
            weighted_sum = sum(prices[i - period + 1 + j] * weights[j] for j in range(period))
            wma = weighted_sum / weight_sum
            wma_values.append(wma)
        
        return wma_values
    
    @performance_monitor
    def calculate_macd(self, prices: List[float], fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> List[MACDResult]:
        """计算MACD指标"""
        if len(prices) < slow_period + signal_period:
            return []
        
        # 计算快速和慢速EMA
        fast_ema = self.calculate_ema(prices, fast_period)
        slow_ema = self.calculate_ema(prices, slow_period)
        
        # 调整长度，确保两个EMA长度一致
        min_length = min(len(fast_ema), len(slow_ema))
        if min_length == 0:
            return []
        
        fast_ema = fast_ema[-min_length:]
        slow_ema = slow_ema[-min_length:]
        
        # 计算MACD线
        macd_line = [fast_ema[i] - slow_ema[i] for i in range(min_length)]
        
        # 计算信号线（MACD的EMA）
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # 计算柱状图
        histogram = []
        signal_start = len(macd_line) - len(signal_line)
        
        for i in range(len(signal_line)):
            hist = macd_line[signal_start + i] - signal_line[i]
            histogram.append(hist)
        
        # 构建结果
        results = []
        base_index = len(prices) - len(histogram)
        
        for i, hist in enumerate(histogram):
            idx = signal_start + i
            result = MACDResult(
                macd_line=macd_line[idx],
                signal_line=signal_line[i],
                histogram=hist,
                fast_ema=fast_ema[idx],
                slow_ema=slow_ema[idx],
                timestamp=datetime.now()
            )
            results.append(result)
        
        return results
    
    @performance_monitor
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[RSIResult]:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return []
        
        # 计算价格变化
        price_changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        
        # 分离上涨和下跌
        gains = [change if change > 0 else 0 for change in price_changes]
        losses = [-change if change < 0 else 0 for change in price_changes]
        
        # 计算平均涨幅和跌幅
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(gains)):
            # 使用指数移动平均
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            result = RSIResult(
                rsi_value=rsi,
                overbought=rsi > 70,
                oversold=rsi < 30,
                timestamp=datetime.now()
            )
            rsi_values.append(result)
        
        return rsi_values
    
    @performance_monitor
    def calculate_atr(self, highs: List[float], lows: List[float], 
                     closes: List[float], period: int = 14) -> List[ATRResult]:
        """计算平均真实范围"""
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return []
        
        true_ranges = []
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # 计算ATR
        atr_values = []
        current_atr = sum(true_ranges[:period]) / period
        
        for i in range(period, len(true_ranges)):
            current_atr = (current_atr * (period - 1) + true_ranges[i]) / period
            
            result = ATRResult(
                atr_value=current_atr,
                volatility_level="",  # 在__post_init__中计算
                price_range=current_atr / closes[i + 1] if closes[i + 1] > 0 else 0,
                timestamp=datetime.now()
            )
            atr_values.append(result)
        
        return atr_values
    
    @performance_monitor
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                 std_dev: float = 2.0) -> List[BollingerResult]:
        """计算布林带"""
        if len(prices) < period:
            return []
        
        results = []
        
        for i in range(period - 1, len(prices)):
            # 计算移动平均（中轨）
            period_prices = prices[i - period + 1:i + 1]
            middle_band = sum(period_prices) / period
            
            # 计算标准差
            variance = sum((p - middle_band) ** 2 for p in period_prices) / period
            std = variance ** 0.5
            
            # 计算上轨和下轨
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            # 计算带宽
            bandwidth = (upper_band - lower_band) / middle_band if middle_band > 0 else 0
            
            result = BollingerResult(
                upper_band=upper_band,
                middle_band=middle_band,
                lower_band=lower_band,
                current_price=prices[i],
                bandwidth=bandwidth,
                position="",  # 在__post_init__中计算
                timestamp=datetime.now()
            )
            results.append(result)
        
        return results
    
    @performance_monitor
    def calculate_kdj(self, highs: List[float], lows: List[float], 
                     closes: List[float], k_period: int = 14, 
                     d_period: int = 3, j_period: int = 3) -> List[KDJResult]:
        """计算KDJ随机指标"""
        if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
            return []
        
        rsv_values = []
        
        # 计算RSV
        for i in range(k_period - 1, len(closes)):
            period_highs = highs[i - k_period + 1:i + 1]
            period_lows = lows[i - k_period + 1:i + 1]
            
            highest = max(period_highs)
            lowest = min(period_lows)
            
            if highest == lowest:
                rsv = 50  # 避免除零
            else:
                rsv = (closes[i] - lowest) / (highest - lowest) * 100
            
            rsv_values.append(rsv)
        
        # 计算K值
        k_values = []
        k_value = 50  # 初始值
        
        for rsv in rsv_values:
            k_value = (k_value * (d_period - 1) + rsv) / d_period
            k_values.append(k_value)
        
        # 计算D值
        d_values = []
        d_value = 50  # 初始值
        
        for k in k_values:
            d_value = (d_value * (j_period - 1) + k) / j_period
            d_values.append(d_value)
        
        # 计算J值和结果
        results = []
        for i in range(len(k_values)):
            j_value = 3 * k_values[i] - 2 * d_values[i]
            
            result = KDJResult(
                k_value=k_values[i],
                d_value=d_values[i],
                j_value=j_value,
                timestamp=datetime.now()
            )
            results.append(result)
        
        return results
    
    @performance_monitor
    def calculate_williams_r(self, highs: List[float], lows: List[float], 
                           closes: List[float], period: int = 14) -> List[float]:
        """计算威廉指标"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return []
        
        wr_values = []
        
        for i in range(period - 1, len(closes)):
            period_highs = highs[i - period + 1:i + 1]
            period_lows = lows[i - period + 1:i + 1]
            
            highest = max(period_highs)
            lowest = min(period_lows)
            
            if highest == lowest:
                wr = -50  # 避免除零
            else:
                wr = (highest - closes[i]) / (highest - lowest) * -100
            
            wr_values.append(wr)
        
        return wr_values
    
    @performance_monitor
    def calculate_volume_sma(self, volumes: List[float], period: int = 20) -> List[float]:
        """计算成交量移动平均"""
        return self.calculate_sma(volumes, period)
    
    @performance_monitor
    def calculate_volume_ratio(self, volumes: List[float], period: int = 20) -> List[float]:
        """计算成交量比率"""
        volume_sma = self.calculate_volume_sma(volumes, period)
        if not volume_sma:
            return []
        
        volume_ratios = []
        for i, vol in enumerate(volumes[len(volumes) - len(volume_sma):]):
            ratio = vol / volume_sma[i] if volume_sma[i] > 0 else 0
            volume_ratios.append(ratio)
        
        return volume_ratios
    
    @performance_monitor
    def calculate_momentum(self, prices: List[float], period: int = 10) -> List[float]:
        """计算动量指标"""
        if len(prices) < period + 1:
            return []
        
        momentum_values = []
        for i in range(period, len(prices)):
            momentum = prices[i] - prices[i - period]
            momentum_values.append(momentum)
        
        return momentum_values
    
    @performance_monitor
    def calculate_roc(self, prices: List[float], period: int = 12) -> List[float]:
        """计算变动率指标"""
        if len(prices) < period + 1:
            return []
        
        roc_values = []
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                roc = (prices[i] - prices[i - period]) / prices[i - period] * 100
            else:
                roc = 0
            roc_values.append(roc)
        
        return roc_values
    
    @performance_monitor
    def calculate_volatility(self, prices: List[float], period: int = 20) -> List[float]:
        """计算波动率"""
        if len(prices) < period + 1:
            return []
        
        # 计算收益率
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] 
                  for i in range(1, len(prices)) if prices[i - 1] != 0]
        
        if len(returns) < period:
            return []
        
        volatility_values = []
        for i in range(period - 1, len(returns)):
            period_returns = returns[i - period + 1:i + 1]
            
            # 计算标准差
            mean_return = sum(period_returns) / period
            variance = sum((r - mean_return) ** 2 for r in period_returns) / period
            volatility = variance ** 0.5
            
            # 年化波动率
            annual_volatility = volatility * (252 ** 0.5)
            volatility_values.append(annual_volatility)
        
        return volatility_values
    
    @performance_monitor
    def calculate_support_resistance(self, highs: List[float], lows: List[float], 
                                   window: int = 20) -> Dict[str, List[float]]:
        """计算支撑阻力位"""
        if len(highs) < window or len(lows) < window:
            return {"support": [], "resistance": []}
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(lows) - window):
            # 寻找支撑位（局部最小值）
            is_support = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] < lows[i]:
                    is_support = False
                    break
            
            if is_support:
                support_levels.append(lows[i])
            
            # 寻找阻力位（局部最大值）
            is_resistance = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] > highs[i]:
                    is_resistance = False
                    break
            
            if is_resistance:
                resistance_levels.append(highs[i])
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    @performance_monitor
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """计算枢轴点"""
        pivot = (high + low + close) / 3
        
        # 支撑位
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # 阻力位
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        return {
            "pivot": pivot,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "r1": r1,
            "r2": r2,
            "r3": r3
        }
    
    @performance_monitor
    def calculate_fibonacci_retracement(self, high: float, low: float) -> Dict[str, float]:
        """计算斐波那契回撤位"""
        diff = high - low
        
        return {
            "0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "100%": low
        }
    
    def _get_cache_key(self, indicator_name: str, symbol: str, 
                      timeframe: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [indicator_name, symbol, timeframe]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return "_".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """获取缓存结果"""
        with self.cache_lock:
            if cache_key in self.calculation_cache:
                result, timestamp = self.calculation_cache[cache_key]
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return result
                else:
                    del self.calculation_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """缓存结果"""
        with self.cache_lock:
            self.calculation_cache[cache_key] = (result, datetime.now().timestamp())
    
    @performance_monitor
    def calculate_all_indicators(self, ohlcv_data: Dict[str, List[float]], 
                               symbol: str, timeframe: str) -> Dict[str, Any]:
        """计算所有技术指标"""
        start_time = datetime.now()
        
        # 提取OHLCV数据
        opens = ohlcv_data.get('open', [])
        highs = ohlcv_data.get('high', [])
        lows = ohlcv_data.get('low', [])
        closes = ohlcv_data.get('close', [])
        volumes = ohlcv_data.get('volume', [])
        
        if not closes:
            return {}
        
        results = {}
        
        # 移动平均线
        if len(closes) >= 20:
            results['sma_20'] = self.calculate_sma(closes, 20)
            results['ema_20'] = self.calculate_ema(closes, 20)
            results['wma_20'] = self.calculate_wma(closes, 20)
        
        # MACD
        results['macd'] = self.calculate_macd(closes, 12, 26, 9)
        
        # RSI
        results['rsi'] = self.calculate_rsi(closes, 14)
        
        # ATR
        if len(highs) >= 14 and len(lows) >= 14:
            results['atr'] = self.calculate_atr(highs, lows, closes, 14)
        
        # 布林带
        if len(closes) >= 20:
            results['bollinger'] = self.calculate_bollinger_bands(closes, 20, 2.0)
        
        # KDJ
        if len(highs) >= 14 and len(lows) >= 14:
            results['kdj'] = self.calculate_kdj(highs, lows, closes, 14, 3, 3)
        
        # 威廉指标
        if len(highs) >= 14 and len(lows) >= 14:
            results['williams_r'] = self.calculate_williams_r(highs, lows, closes, 14)
        
        # 成交量指标
        if volumes:
            results['volume_sma'] = self.calculate_volume_sma(volumes, 20)
            results['volume_ratio'] = self.calculate_volume_ratio(volumes, 20)
        
        # 动量指标
        results['momentum'] = self.calculate_momentum(closes, 10)
        results['roc'] = self.calculate_roc(closes, 12)
        
        # 波动率
        results['volatility'] = self.calculate_volatility(closes, 20)
        
        # 支撑阻力位
        if len(highs) >= 40 and len(lows) >= 40:
            results['support_resistance'] = self.calculate_support_resistance(highs, lows, 20)
        
        # 枢轴点和斐波那契回撤（基于最新数据）
        if len(closes) >= 1:
            latest_high = max(highs[-20:]) if len(highs) >= 20 else highs[-1]
            latest_low = min(lows[-20:]) if len(lows) >= 20 else lows[-1]
            latest_close = closes[-1]
            
            results['pivot_points'] = self.calculate_pivot_points(latest_high, latest_low, latest_close)
            results['fibonacci'] = self.calculate_fibonacci_retracement(latest_high, latest_low)
        
        # 计算总体指标
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        results['meta'] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data_points': len(closes),
            'calculation_time': calculation_time,
            'timestamp': datetime.now().isoformat(),
            'indicators_count': len([k for k in results.keys() if k != 'meta'])
        }
        
        logger.info(f"技术指标计算完成 {symbol}:{timeframe} - "
                   f"指标数量: {results['meta']['indicators_count']}, "
                   f"计算时间: {calculation_time:.3f}s")
        
        return results
    
    async def calculate_indicators_async(self, ohlcv_data: Dict[str, List[float]], 
                                       symbol: str, timeframe: str) -> Dict[str, Any]:
        """异步计算技术指标"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行计算
        result = await loop.run_in_executor(
            self.executor, 
            self.calculate_all_indicators, 
            ohlcv_data, 
            symbol, 
            timeframe
        )
        
        return result
    
    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.calculation_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        with self.cache_lock:
            return {
                'cache_size': len(self.calculation_cache),
                'cache_ttl': self.cache_ttl,
                'cache_keys': list(self.calculation_cache.keys())
            }


class IndicatorAnalyzer:
    """技术指标分析器"""
    
    def __init__(self, calculator: TechnicalIndicatorCalculator):
        self.calculator = calculator
    
    @performance_monitor
    def analyze_macd_signals(self, macd_results: List[MACDResult]) -> Dict[str, Any]:
        """分析MACD信号"""
        if len(macd_results) < 2:
            return {}
        
        latest = macd_results[-1]
        previous = macd_results[-2]
        
        # 检测金叉和死叉
        golden_cross = (latest.macd_line > latest.signal_line and 
                       previous.macd_line <= previous.signal_line)
        death_cross = (latest.macd_line < latest.signal_line and 
                      previous.macd_line >= previous.signal_line)
        
        # 检测背离
        macd_trend = "UP" if latest.macd_line > previous.macd_line else "DOWN"
        histogram_trend = "UP" if latest.histogram > previous.histogram else "DOWN"
        
        # 计算信号强度
        signal_strength = abs(latest.histogram) / max(abs(latest.macd_line), 0.001)
        
        return {
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            'macd_trend': macd_trend,
            'histogram_trend': histogram_trend,
            'signal_strength': signal_strength,
            'trend_consistency': macd_trend == histogram_trend,
            'current_position': "ABOVE" if latest.macd_line > latest.signal_line else "BELOW"
        }
    
    @performance_monitor
    def analyze_trend_strength(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析趋势强度"""
        trend_signals = []
        
        # MACD趋势
        if 'macd' in results and results['macd']:
            latest_macd = results['macd'][-1]
            if latest_macd.macd_line > latest_macd.signal_line:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
        
        # RSI趋势
        if 'rsi' in results and results['rsi']:
            latest_rsi = results['rsi'][-1]
            if latest_rsi.rsi_value > 50:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
        
        # 移动平均线趋势
        if 'sma_20' in results and len(results['sma_20']) >= 2:
            sma_current = results['sma_20'][-1]
            sma_previous = results['sma_20'][-2]
            if sma_current > sma_previous:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
        
        # 计算总体趋势
        if trend_signals:
            trend_score = sum(trend_signals) / len(trend_signals)
            if trend_score > 0.3:
                overall_trend = "BULLISH"
            elif trend_score < -0.3:
                overall_trend = "BEARISH"
            else:
                overall_trend = "NEUTRAL"
        else:
            overall_trend = "UNKNOWN"
            trend_score = 0
        
        return {
            'overall_trend': overall_trend,
            'trend_score': trend_score,
            'signal_count': len(trend_signals),
            'trend_strength': abs(trend_score)
        }
    
    @performance_monitor
    def detect_reversal_patterns(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测反转模式"""
        patterns = []
        
        # RSI超买超卖
        if 'rsi' in results and results['rsi']:
            latest_rsi = results['rsi'][-1]
            if latest_rsi.oversold:
                patterns.append({
                    'type': 'RSI_OVERSOLD',
                    'confidence': 0.7,
                    'direction': 'BULLISH_REVERSAL',
                    'value': latest_rsi.rsi_value
                })
            elif latest_rsi.overbought:
                patterns.append({
                    'type': 'RSI_OVERBOUGHT',
                    'confidence': 0.7,
                    'direction': 'BEARISH_REVERSAL',
                    'value': latest_rsi.rsi_value
                })
        
        # 布林带突破
        if 'bollinger' in results and results['bollinger']:
            latest_bb = results['bollinger'][-1]
            if latest_bb.position == "BELOW_LOWER":
                patterns.append({
                    'type': 'BOLLINGER_OVERSOLD',
                    'confidence': 0.6,
                    'direction': 'BULLISH_REVERSAL',
                    'value': latest_bb.current_price
                })
            elif latest_bb.position == "ABOVE_UPPER":
                patterns.append({
                    'type': 'BOLLINGER_OVERBOUGHT',
                    'confidence': 0.6,
                    'direction': 'BEARISH_REVERSAL',
                    'value': latest_bb.current_price
                })
        
        # KDJ信号
        if 'kdj' in results and results['kdj']:
            latest_kdj = results['kdj'][-1]
            if latest_kdj.oversold and latest_kdj.golden_cross:
                patterns.append({
                    'type': 'KDJ_BULLISH_SIGNAL',
                    'confidence': 0.8,
                    'direction': 'BULLISH_REVERSAL',
                    'value': latest_kdj.k_value
                })
            elif latest_kdj.overbought and latest_kdj.death_cross:
                patterns.append({
                    'type': 'KDJ_BEARISH_SIGNAL',
                    'confidence': 0.8,
                    'direction': 'BEARISH_REVERSAL',
                    'value': latest_kdj.k_value
                })
        
        return patterns
    
    @performance_monitor
    def generate_trading_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'hold_signals': [],
            'overall_signal': 'HOLD',
            'confidence': 0.0
        }
        
        # MACD信号
        if 'macd' in results and results['macd']:
            macd_analysis = self.analyze_macd_signals(results['macd'])
            if macd_analysis.get('golden_cross'):
                signals['buy_signals'].append({
                    'type': 'MACD_GOLDEN_CROSS',
                    'confidence': 0.8,
                    'strength': macd_analysis.get('signal_strength', 0)
                })
            elif macd_analysis.get('death_cross'):
                signals['sell_signals'].append({
                    'type': 'MACD_DEATH_CROSS',
                    'confidence': 0.8,
                    'strength': macd_analysis.get('signal_strength', 0)
                })
        
        # 趋势分析
        trend_analysis = self.analyze_trend_strength(results)
        if trend_analysis['overall_trend'] == 'BULLISH':
            signals['buy_signals'].append({
                'type': 'TREND_BULLISH',
                'confidence': trend_analysis['trend_strength'],
                'strength': trend_analysis['trend_score']
            })
        elif trend_analysis['overall_trend'] == 'BEARISH':
            signals['sell_signals'].append({
                'type': 'TREND_BEARISH',
                'confidence': trend_analysis['trend_strength'],
                'strength': abs(trend_analysis['trend_score'])
            })
        
        # 反转模式
        reversal_patterns = self.detect_reversal_patterns(results)
        for pattern in reversal_patterns:
            if pattern['direction'] == 'BULLISH_REVERSAL':
                signals['buy_signals'].append({
                    'type': pattern['type'],
                    'confidence': pattern['confidence'],
                    'strength': pattern['value']
                })
            elif pattern['direction'] == 'BEARISH_REVERSAL':
                signals['sell_signals'].append({
                    'type': pattern['type'],
                    'confidence': pattern['confidence'],
                    'strength': pattern['value']
                })
        
        # 计算综合信号
        buy_strength = sum(s['confidence'] * s.get('strength', 1) for s in signals['buy_signals'])
        sell_strength = sum(s['confidence'] * s.get('strength', 1) for s in signals['sell_signals'])
        
        if buy_strength > sell_strength and buy_strength > 0.5:
            signals['overall_signal'] = 'BUY'
            signals['confidence'] = min(buy_strength, 1.0)
        elif sell_strength > buy_strength and sell_strength > 0.5:
            signals['overall_signal'] = 'SELL'
            signals['confidence'] = min(sell_strength, 1.0)
        else:
            signals['overall_signal'] = 'HOLD'
            signals['confidence'] = 0.5
        
        return signals


# 主要导出接口
__all__ = [
    'IndicatorResult',
    'MACDResult',
    'MAResult',
    'ATRResult',
    'RSIResult',
    'BollingerResult',
    'KDJResult',
    'TechnicalIndicatorCalculator',
    'IndicatorAnalyzer'
] 