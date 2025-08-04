"""
🆕 任务4.1: 改进版W底/双顶检测器
基于大神算法的改进版本，修复关键bug，适配15分钟高频交易

主要改进：
1. 修复中间峰/谷验证逻辑的关键bug
2. 增强稳健性，处理NaN值和边界情况  
3. 优化参数适配15分钟K线数据
4. 简化置信度计算，提高准确性
"""

import numpy as np
from scipy.signal import find_peaks
import talib as ta
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

from utils.logger import get_logger

class DoublePatternType(Enum):
    """双重形态类型枚举"""
    DOUBLE_BOTTOM_BULL = "double_bottom_bull"  # W底看涨
    DOUBLE_TOP_BEAR = "double_top_bear"        # 双顶看跌

@dataclass  
class PatternSignal:
    """形态信号数据结构"""
    type: DoublePatternType
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    formation_bars: int
    volume_ratio: float
    similarity_score: float
    rsi_confirmation: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'target_price': self.target_price, 
            'stop_loss': self.stop_loss,
            'formation_bars': self.formation_bars,
            'volume_ratio': self.volume_ratio,
            'similarity_score': self.similarity_score,
            'rsi_confirmation': self.rsi_confirmation,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class ImprovedDoublePatternDetector:
    """
    🔧 改进版W底/双顶形态检测器
    
    主要改进：
    1. ✅ 修复中间峰/谷验证的关键逻辑错误
    2. ✅ 增强数据稳健性处理（NaN、边界情况）
    3. ✅ 优化高频交易参数（15分钟K线适配）
    4. ✅ 简化置信度计算逻辑
    """
    
    def __init__(
        self,
        min_bars: int = 10,          # 最小形态周期（降低为10，提高敏感度）
        max_bars: int = 50,          # 最大形态周期（增加为50，适配更多形态）
        similarity_thresh: float = 0.1,   # 相似度阈值（放宽到10%，避免过严）
        volume_mult: float = 1.2,    # 成交量确认倍数（降低到1.2）
        rsi_bull_thresh: float = 50, # RSI看涨阈值（设为中性50）
        rsi_bear_thresh: float = 50, # RSI看跌阈值（设为中性50）
        min_confidence: float = 0.5  # 最小置信度阈值（降低到0.5）
    ):
        self.min_bars = min_bars
        self.max_bars = max_bars  
        self.similarity_thresh = similarity_thresh
        self.volume_mult = volume_mult
        self.rsi_bull_thresh = rsi_bull_thresh
        self.rsi_bear_thresh = rsi_bear_thresh
        self.min_confidence = min_confidence
        
        self.logger = get_logger(__name__)
        
        # 日志记录参数设置
        self.logger.info(f"🔧 初始化改进版双重形态检测器:")
        self.logger.info(f"   📊 形态周期: {min_bars}-{max_bars} 根K线")
        self.logger.info(f"   📏 相似度阈值: {similarity_thresh*100:.1f}%")
        self.logger.info(f"   📦 成交量倍数: {volume_mult}x")
        self.logger.info(f"   📈 RSI阈值: 看涨<{rsi_bull_thresh}, 看跌>{rsi_bear_thresh}")
        self.logger.info(f"   🎯 最小置信度: {min_confidence}")

    def _safe_rsi_calculation(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """
        🛡️ 安全的RSI计算，处理NaN值和数据不足情况
        这是对原算法的重要改进，避免因数据问题导致的计算失败
        """
        try:
            if len(closes) < period:
                # 数据不足时返回中性值50
                return np.full(len(closes), 50.0)
            
            rsi = ta.RSI(closes.astype(float), timeperiod=period)
            
            # 处理NaN值：前面的NaN用50填充，避免计算错误
            if np.isnan(rsi).any():
                # 找到第一个非NaN的位置
                first_valid = period - 1
                rsi[:first_valid] = 50.0  # NaN部分填充50（中性）
                
                # 处理可能的其他NaN值
                rsi = np.nan_to_num(rsi, nan=50.0)
            
            return rsi
            
        except Exception as e:
            self.logger.warning(f"⚠️ RSI计算异常: {e}, 返回中性值")
            return np.full(len(closes), 50.0)

    def detect_double_bottom(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        volumes: np.ndarray
    ) -> Optional[PatternSignal]:
        """
        🔍 检测W底（双底）形态
        
        关键改进：修复了原算法中间峰验证的严重逻辑错误
        """
        try:
            if len(lows) < self.min_bars:
                return None
            
            # 1. 寻找低点（谷值）
            valley_indices, _ = find_peaks(-lows, distance=max(5, self.min_bars//3))
            
            if len(valley_indices) < 2:
                return None
            
            # 2. 检查最后两个低点作为双底候选
            if len(valley_indices) >= 2:
                left_valley_idx = valley_indices[-2]  # 左底
                right_valley_idx = valley_indices[-1] # 右底
            else:
                return None
            
            # 3. 验证形态周期合理性
            formation_bars = right_valley_idx - left_valley_idx
            if formation_bars < self.min_bars or formation_bars > self.max_bars:
                return None
            
            # 4. 检查底部相似性
            left_low = lows[left_valley_idx]
            right_low = lows[right_valley_idx]
            similarity = abs(left_low - right_low) / min(left_low, right_low)
            
            if similarity > self.similarity_thresh:
                return None
            
            # 5. 🔧 关键修复：正确的中间峰验证逻辑
            # 原算法错误：与全局最高点比较 —— if middle_peak < max(highs)
            # 修复版本：找到两个底部之间的实际最高点作为颈线
            middle_section = highs[left_valley_idx:right_valley_idx+1]
            middle_peak_relative_idx = np.argmax(middle_section)
            middle_peak_idx = left_valley_idx + middle_peak_relative_idx
            middle_peak = highs[middle_peak_idx]
            
            # 中间峰应该明显高于两个底部
            if middle_peak <= max(left_low, right_low) * 1.02:  # 至少高2%
                return None
            
            # 6. 验证突破确认（当前价格突破颈线）
            current_price = closes[-1]
            neckline = middle_peak
            
            if current_price <= neckline:
                return None  # 还未突破，不产生信号
            
            # 7. 成交量确认
            recent_volume = np.mean(volumes[-5:])  # 最近5根K线平均成交量
            base_volume = np.mean(volumes[left_valley_idx:right_valley_idx+1])
            volume_ratio = recent_volume / base_volume if base_volume > 0 else 1.0
            
            # 8. RSI确认（使用安全计算）
            rsi_values = self._safe_rsi_calculation(closes)
            current_rsi = rsi_values[-1]
            
            # 9. 计算置信度（简化版本，更稳定）
            confidence_factors = {
                'similarity': max(0, 1 - similarity / self.similarity_thresh) * 0.25,  # 25%权重
                'volume': min(volume_ratio / self.volume_mult, 1.0) * 0.25,           # 25%权重  
                'rsi': (1.0 if current_rsi < self.rsi_bull_thresh else 0.5) * 0.25,  # 25%权重
                'breakout': min((current_price - neckline) / neckline / 0.02, 1.0) * 0.25  # 25%权重
            }
            
            total_confidence = sum(confidence_factors.values())
            
            # 10. 置信度检查
            if total_confidence < self.min_confidence:
                return None
            
            # 11. 计算交易参数
            entry_price = current_price
            height = neckline - max(left_low, right_low)  # 形态高度
            target_price = neckline + height              # 目标价格
            stop_loss = min(left_low, right_low) * 0.98   # 止损设在最低点下方2%
            
            # 12. 创建信号
            signal = PatternSignal(
                type=DoublePatternType.DOUBLE_BOTTOM_BULL,
                confidence=total_confidence,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                formation_bars=formation_bars,
                volume_ratio=volume_ratio,
                similarity_score=1 - similarity,
                rsi_confirmation=current_rsi,
                metadata={
                    'left_valley_idx': int(left_valley_idx),
                    'right_valley_idx': int(right_valley_idx),
                    'middle_peak_idx': int(middle_peak_idx),
                    'neckline': float(neckline),
                    'confidence_breakdown': confidence_factors
                }
            )
            
            self.logger.info(f"✅ 检测到W底形态: 置信度={total_confidence:.2f}, "
                           f"形态周期={formation_bars}, 相似度={1-similarity:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ W底检测异常: {e}")
            return None

    def detect_double_top(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        volumes: np.ndarray
    ) -> Optional[PatternSignal]:
        """
        🔍 检测双顶形态
        
        这是W底检测的镜像版本，应用相同的逻辑修复
        """
        try:
            if len(highs) < self.min_bars:
                return None
            
            # 1. 寻找高点（峰值）
            peak_indices, _ = find_peaks(highs, distance=max(5, self.min_bars//3))
            
            if len(peak_indices) < 2:
                return None
            
            # 2. 检查最后两个高点作为双顶候选
            if len(peak_indices) >= 2:
                left_peak_idx = peak_indices[-2]   # 左顶
                right_peak_idx = peak_indices[-1]  # 右顶
            else:
                return None
            
            # 3. 验证形态周期合理性
            formation_bars = right_peak_idx - left_peak_idx
            if formation_bars < self.min_bars or formation_bars > self.max_bars:
                return None
            
            # 4. 检查顶部相似性
            left_high = highs[left_peak_idx]
            right_high = highs[right_peak_idx]
            similarity = abs(left_high - right_high) / max(left_high, right_high)
            
            if similarity > self.similarity_thresh:
                return None
            
            # 5. 🔧 关键修复：正确的中间谷验证逻辑
            # 找到两个顶部之间的实际最低点作为颈线
            middle_section = lows[left_peak_idx:right_peak_idx+1]
            middle_valley_relative_idx = np.argmin(middle_section)
            middle_valley_idx = left_peak_idx + middle_valley_relative_idx
            middle_valley = lows[middle_valley_idx]
            
            # 中间谷应该明显低于两个顶部
            if middle_valley >= min(left_high, right_high) * 0.98:  # 至少低2%
                return None
            
            # 6. 验证突破确认（当前价格跌破颈线）
            current_price = closes[-1]
            neckline = middle_valley
            
            if current_price >= neckline:
                return None  # 还未跌破，不产生信号
            
            # 7. 成交量确认
            recent_volume = np.mean(volumes[-5:])
            base_volume = np.mean(volumes[left_peak_idx:right_peak_idx+1])
            volume_ratio = recent_volume / base_volume if base_volume > 0 else 1.0
            
            # 8. RSI确认（使用安全计算）
            rsi_values = self._safe_rsi_calculation(closes)
            current_rsi = rsi_values[-1]
            
            # 9. 计算置信度（镜像逻辑）
            confidence_factors = {
                'similarity': max(0, 1 - similarity / self.similarity_thresh) * 0.25,
                'volume': min(volume_ratio / self.volume_mult, 1.0) * 0.25,
                'rsi': (1.0 if current_rsi > self.rsi_bear_thresh else 0.5) * 0.25,
                'breakout': min((neckline - current_price) / neckline / 0.02, 1.0) * 0.25
            }
            
            total_confidence = sum(confidence_factors.values())
            
            # 10. 置信度检查
            if total_confidence < self.min_confidence:
                return None
            
            # 11. 计算交易参数
            entry_price = current_price
            height = min(left_high, right_high) - neckline  # 形态高度
            target_price = neckline - height                # 目标价格（做空）
            stop_loss = max(left_high, right_high) * 1.02   # 止损设在最高点上方2%
            
            # 12. 创建信号
            signal = PatternSignal(
                type=DoublePatternType.DOUBLE_TOP_BEAR,
                confidence=total_confidence,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                formation_bars=formation_bars,
                volume_ratio=volume_ratio,
                similarity_score=1 - similarity,
                rsi_confirmation=current_rsi,
                metadata={
                    'left_peak_idx': int(left_peak_idx),
                    'right_peak_idx': int(right_peak_idx),
                    'middle_valley_idx': int(middle_valley_idx),
                    'neckline': float(neckline),
                    'confidence_breakdown': confidence_factors
                }
            )
            
            self.logger.info(f"✅ 检测到双顶形态: 置信度={total_confidence:.2f}, "
                           f"形态周期={formation_bars}, 相似度={1-similarity:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ 双顶检测异常: {e}")
            return None

    def detect_patterns(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        volumes: np.ndarray
    ) -> List[PatternSignal]:
        """
        🔍 检测所有双重形态
        
        Returns:
            List[PatternSignal]: 检测到的形态信号列表
        """
        patterns = []
        
        try:
            # 检测W底形态
            double_bottom = self.detect_double_bottom(highs, lows, closes, volumes)
            if double_bottom:
                patterns.append(double_bottom)
            
            # 检测双顶形态  
            double_top = self.detect_double_top(highs, lows, closes, volumes)
            if double_top:
                patterns.append(double_top)
            
            if patterns:
                self.logger.info(f"🎯 双重形态检测完成: 发现 {len(patterns)} 个有效形态")
            else:
                self.logger.debug("🔍 双重形态检测完成: 未发现有效形态")
                
        except Exception as e:
            self.logger.error(f"❌ 双重形态检测异常: {e}")
        
        return patterns

    def get_pattern_summary(self, patterns: List[PatternSignal]) -> Dict[str, Any]:
        """
        📊 获取形态检测摘要统计
        """
        if not patterns:
            return {
                'total_patterns': 0,
                'pattern_types': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0
            }
        
        pattern_types = {}
        confidences = []
        
        for pattern in patterns:
            pattern_type = pattern.type.value
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            confidences.append(pattern.confidence)
        
        return {
            'total_patterns': len(patterns),
            'pattern_types': pattern_types,
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences)
        } 