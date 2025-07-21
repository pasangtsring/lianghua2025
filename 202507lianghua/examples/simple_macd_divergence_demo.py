#!/usr/bin/env python3
"""
简化版MACD背离检测器演示
独立运行，无复杂依赖
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告：未安装scipy，将使用简化的峰值检测")


class DivergenceType(Enum):
    """背离类型"""
    BULLISH_REGULAR = "看涨背离"
    BEARISH_REGULAR = "看跌背离"
    BULLISH_HIDDEN = "隐藏看涨背离"
    BEARISH_HIDDEN = "隐藏看跌背离"


@dataclass
class MACDResult:
    """MACD结果"""
    macd_line: float
    signal_line: float
    histogram: float
    timestamp: datetime


@dataclass
class DivergenceSignal:
    """背离信号"""
    divergence_type: DivergenceType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime


class SimpleMACDDivergenceDetector:
    """简化版MACD背离检测器"""
    
    def __init__(self, lookback_period: int = 50, min_peak_distance: int = 3):
        self.lookback_period = lookback_period
        self.min_peak_distance = min_peak_distance
    
    def detect_peaks(self, data: List[float]) -> List[int]:
        """检测峰值"""
        if HAS_SCIPY:
            peaks, _ = find_peaks(data, distance=self.min_peak_distance)
            return peaks.tolist()
        else:
            # 简化的峰值检测
            peaks = []
            for i in range(self.min_peak_distance, len(data) - self.min_peak_distance):
                is_peak = True
                for j in range(i - self.min_peak_distance, i + self.min_peak_distance + 1):
                    if j != i and data[j] >= data[i]:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(i)
            return peaks
    
    def detect_valleys(self, data: List[float]) -> List[int]:
        """检测谷值"""
        if HAS_SCIPY:
            valleys, _ = find_peaks([-x for x in data], distance=self.min_peak_distance)
            return valleys.tolist()
        else:
            # 简化的谷值检测
            valleys = []
            for i in range(self.min_peak_distance, len(data) - self.min_peak_distance):
                is_valley = True
                for j in range(i - self.min_peak_distance, i + self.min_peak_distance + 1):
                    if j != i and data[j] <= data[i]:
                        is_valley = False
                        break
                if is_valley:
                    valleys.append(i)
            return valleys
    
    def detect_divergence(self, prices: List[float], 
                         macd_results: List[MACDResult]) -> List[DivergenceSignal]:
        """检测背离"""
        if len(prices) < self.lookback_period:
            return []
        
        # 获取最近数据
        recent_prices = prices[-self.lookback_period:]
        recent_macd = macd_results[-self.lookback_period:]
        macd_histograms = [r.histogram for r in recent_macd]
        
        # 检测价格和MACD峰值
        price_peaks = self.detect_peaks(recent_prices)
        price_valleys = self.detect_valleys(recent_prices)
        macd_peaks = self.detect_peaks(macd_histograms)
        macd_valleys = self.detect_valleys(macd_histograms)
        
        signals = []
        
        # 检测看跌背离
        signals.extend(self._detect_bearish_divergence(
            price_peaks, macd_peaks, recent_prices, macd_histograms, recent_macd
        ))
        
        # 检测看涨背离
        signals.extend(self._detect_bullish_divergence(
            price_valleys, macd_valleys, recent_prices, macd_histograms, recent_macd
        ))
        
        return signals
    
    def _detect_bearish_divergence(self, price_peaks: List[int], macd_peaks: List[int],
                                  prices: List[float], macd_histograms: List[float],
                                  macd_results: List[MACDResult]) -> List[DivergenceSignal]:
        """检测看跌背离"""
        signals = []
        
        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return signals
        
        # 寻找价格新高但MACD不创新高的情况
        for i in range(len(price_peaks) - 1):
            for j in range(i + 1, len(price_peaks)):
                price_peak1 = price_peaks[i]
                price_peak2 = price_peaks[j]
                
                # 价格创新高
                if prices[price_peak2] > prices[price_peak1]:
                    # 寻找对应的MACD峰值
                    macd_peak1 = self._find_closest_peak(macd_peaks, price_peak1)
                    macd_peak2 = self._find_closest_peak(macd_peaks, price_peak2)
                    
                    if macd_peak1 is not None and macd_peak2 is not None:
                        # MACD不创新高（背离）
                        if macd_histograms[macd_peak2] < macd_histograms[macd_peak1]:
                            confidence = self._calculate_confidence(
                                prices[price_peak1], prices[price_peak2],
                                macd_histograms[macd_peak1], macd_histograms[macd_peak2]
                            )
                            
                            signal = DivergenceSignal(
                                divergence_type=DivergenceType.BEARISH_REGULAR,
                                confidence=confidence,
                                entry_price=prices[price_peak2],
                                stop_loss=prices[price_peak2] * 1.02,
                                take_profit=prices[price_peak2] * 0.96,
                                timestamp=macd_results[price_peak2].timestamp
                            )
                            signals.append(signal)
        
        return signals
    
    def _detect_bullish_divergence(self, price_valleys: List[int], macd_valleys: List[int],
                                  prices: List[float], macd_histograms: List[float],
                                  macd_results: List[MACDResult]) -> List[DivergenceSignal]:
        """检测看涨背离"""
        signals = []
        
        if len(price_valleys) < 2 or len(macd_valleys) < 2:
            return signals
        
        # 寻找价格新低但MACD不创新低的情况
        for i in range(len(price_valleys) - 1):
            for j in range(i + 1, len(price_valleys)):
                price_valley1 = price_valleys[i]
                price_valley2 = price_valleys[j]
                
                # 价格创新低
                if prices[price_valley2] < prices[price_valley1]:
                    # 寻找对应的MACD谷值
                    macd_valley1 = self._find_closest_peak(macd_valleys, price_valley1)
                    macd_valley2 = self._find_closest_peak(macd_valleys, price_valley2)
                    
                    if macd_valley1 is not None and macd_valley2 is not None:
                        # MACD不创新低（背离）
                        if macd_histograms[macd_valley2] > macd_histograms[macd_valley1]:
                            confidence = self._calculate_confidence(
                                prices[price_valley1], prices[price_valley2],
                                macd_histograms[macd_valley1], macd_histograms[macd_valley2]
                            )
                            
                            signal = DivergenceSignal(
                                divergence_type=DivergenceType.BULLISH_REGULAR,
                                confidence=confidence,
                                entry_price=prices[price_valley2],
                                stop_loss=prices[price_valley2] * 0.98,
                                take_profit=prices[price_valley2] * 1.04,
                                timestamp=macd_results[price_valley2].timestamp
                            )
                            signals.append(signal)
        
        return signals
    
    def _find_closest_peak(self, peaks: List[int], target_index: int) -> Optional[int]:
        """寻找最接近的峰值"""
        if not peaks:
            return None
        
        min_distance = float('inf')
        closest_peak = None
        
        for peak in peaks:
            distance = abs(peak - target_index)
            if distance < min_distance and distance <= 5:  # 容忍度
                min_distance = distance
                closest_peak = peak
        
        return closest_peak
    
    def _calculate_confidence(self, price1: float, price2: float, 
                            macd1: float, macd2: float) -> float:
        """计算置信度"""
        price_change = abs(price2 - price1) / price1
        macd_change = abs(macd2 - macd1) / abs(macd1) if macd1 != 0 else 0
        
        # 价格变化大而MACD变化小时，置信度高
        if price_change > 0 and macd_change < price_change:
            return min(1.0, (price_change - macd_change) / price_change)
        else:
            return 0.3


def create_sample_data(length: int = 100) -> tuple:
    """创建示例数据"""
    np.random.seed(42)  # 固定随机种子
    
    # 生成带趋势的价格数据
    prices = []
    base_price = 100
    trend = 0
    
    for i in range(length):
        # 添加趋势变化
        if i < length // 3:
            trend = -0.05  # 下跌
        elif i < 2 * length // 3:
            trend = 0.1   # 上涨
        else:
            trend = -0.02  # 小幅回调
        
        noise = np.random.normal(0, 0.5)
        price = base_price + trend * i + noise
        prices.append(max(80, price))  # 确保价格不低于80
    
    # 生成MACD数据
    macd_results = []
    for i in range(length):
        if i < 26:
            macd_line = 0
            signal_line = 0
        else:
            # 简化的MACD计算
            fast_ema = np.mean(prices[max(0, i-12):i+1])
            slow_ema = np.mean(prices[max(0, i-26):i+1])
            macd_line = fast_ema - slow_ema
            
            if i < 35:
                signal_line = macd_line
            else:
                signal_line = np.mean([r.macd_line for r in macd_results[max(0, i-9):i]])
        
        histogram = macd_line - signal_line
        
        macd_results.append(MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            timestamp=datetime.now() - timedelta(hours=length-i)
        ))
    
    return prices, macd_results


def main():
    """主演示函数"""
    print("🚀 简化版MACD背离检测器演示")
    print("=" * 50)
    
    # 创建示例数据
    print("📊 创建示例数据...")
    prices, macd_results = create_sample_data(100)
    
    print(f"✅ 生成数据完成")
    print(f"   价格范围: ${min(prices):.2f} - ${max(prices):.2f}")
    print(f"   数据点数: {len(prices)}")
    
    # 创建检测器
    print("\n⚙️ 初始化检测器...")
    detector = SimpleMACDDivergenceDetector(lookback_period=80, min_peak_distance=3)
    
    # 执行背离检测
    print("\n🔍 执行背离检测...")
    signals = detector.detect_divergence(prices, macd_results)
    
    print(f"✅ 检测完成，发现 {len(signals)} 个背离信号")
    
    # 显示结果
    if signals:
        print("\n📈 背离信号详情:")
        for i, signal in enumerate(signals):
            print(f"\n  信号 {i+1}:")
            print(f"    类型: {signal.divergence_type.value}")
            print(f"    置信度: {signal.confidence:.3f}")
            print(f"    入场价: ${signal.entry_price:.2f}")
            print(f"    止损价: ${signal.stop_loss:.2f}")
            print(f"    止盈价: ${signal.take_profit:.2f}")
            print(f"    时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("\n📊 未检测到背离信号")
        print("  可能原因:")
        print("    - 数据量不足")
        print("    - 市场趋势太强")
        print("    - 参数设置过于严格")
    
    # 显示价格和MACD摘要
    print("\n📈 数据摘要:")
    print(f"  价格变化: {(prices[-1] - prices[0]) / prices[0] * 100:.2f}%")
    print(f"  MACD范围: {min(r.macd_line for r in macd_results):.3f} 到 {max(r.macd_line for r in macd_results):.3f}")
    
    print("\n🎉 演示完成！")


if __name__ == "__main__":
    main() 