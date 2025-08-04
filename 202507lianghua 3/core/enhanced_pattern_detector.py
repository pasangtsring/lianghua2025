"""
增强形态检测器模块
集成专家级MACD背离 + 形态识别算法
基于大佬提供的专业代码，适配现有系统架构
"""

import numpy as np
from scipy.signal import find_peaks
import talib as ta
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import pandas as pd

from config.config_manager import ConfigManager
from utils.logger import get_logger, performance_monitor
# 🆕 任务4.2: 导入改进版双重形态检测器
from core.improved_double_pattern_detector import ImprovedDoublePatternDetector, PatternSignal as DoublePatternSignal

class PatternType(Enum):
    """形态类型枚举"""
    ENGULFING_BULL = "engulfing_bull"
    ENGULFING_BEAR = "engulfing_bear"
    HEAD_SHOULDER_BEAR = "head_shoulder_bear"
    HEAD_SHOULDER_BULL = "head_shoulder_bull"
    CONVERGENCE_TRIANGLE_BULL = "convergence_triangle_bull"
    CONVERGENCE_TRIANGLE_BEAR = "convergence_triangle_bear"
    # 🆕 任务4.2: 添加W底/双顶形态类型
    DOUBLE_BOTTOM_BULL = "double_bottom_bull"  # W底看涨
    DOUBLE_TOP_BEAR = "double_top_bear"        # 双顶看跌

class DivergenceType(Enum):
    """背离类型枚举"""
    BEARISH = "bearish"
    BULLISH = "bullish"

@dataclass
class DivergenceSignal:
    """背离信号数据结构"""
    type: DivergenceType
    strength: float
    confidence: float
    indices: List[int]
    timestamp: datetime = field(default_factory=datetime.now)
    macd_values: List[float] = field(default_factory=list)
    price_values: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'indices': self.indices,
            'timestamp': self.timestamp.isoformat(),
            'macd_values': self.macd_values,
            'price_values': self.price_values
        }

@dataclass
class PatternSignal:
    """形态信号数据结构"""
    type: PatternType
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    validity_period: int = 24  # 有效期（小时）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'confidence': self.confidence,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'validity_period': self.validity_period
        }

class EnhancedPatternDetector:
    """
    增强形态检测器 - 基于专家代码优化
    集成MACD背离检测和形态识别功能
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 获取配置
        self.trading_config = self.config.get_trading_config()
        self.macd_config = self.trading_config.macd
        self.divergence_config = self.trading_config.macd_divergence
        self.pattern_config = self.config.get_signal_config()
        
        # MACD参数
        self.macd_fast = self.macd_config.fast
        self.macd_slow = self.macd_config.slow
        self.macd_signal = self.macd_config.signal
        
        # 检测参数
        self.lookback = self.divergence_config.lookback_period
        self.min_distance = self.divergence_config.min_peak_distance
        self.prominence_mult = self.divergence_config.prominence_mult
        self.min_gap = 0.1
        self.min_consecutive = self.divergence_config.consecutive_signals
        self.tolerance = 2
        self.vol_factor_mult = self.pattern_config.vol_factor_mult
        
        # 支持的形态
        pattern_list = self.config.get_param('market_conditions.pattern_recognition.patterns', 
                                           ["ENGULFING", "HEAD_SHOULDER", "CONVERGENCE_TRIANGLE", "DOUBLE_PATTERNS"])
        self.morph_patterns = pattern_list
        
        # 🆕 任务4.2: 初始化改进版双重形态检测器
        self.double_pattern_detector = ImprovedDoublePatternDetector()
        
        # 检测统计
        self.detection_stats = {
            'total_detections': 0,
            'divergence_count': 0,
            'pattern_count': 0,
            'double_pattern_count': 0,  # 🆕 添加双重形态统计
            'high_confidence_signals': 0,
            'false_positive_rate': 0.0
        }
        
        # 信号历史
        self.signal_history: List[Dict] = []
        self.max_history = 100
        
        self.logger.info("增强形态检测器初始化完成")
    
    @performance_monitor
    def compute_macd(self, closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算MACD指标 (使用ta-lib)
        
        Args:
            closes: 收盘价数组
            
        Returns:
            (macd_line, signal_line, histogram)
        """
        try:
            if len(closes) < self.macd_slow + self.macd_signal:
                raise ValueError(f"数据长度不足，需要至少{self.macd_slow + self.macd_signal}个数据点")
            
            macd, signal, hist = ta.MACD(
                closes, 
                fastperiod=self.macd_fast, 
                slowperiod=self.macd_slow, 
                signalperiod=self.macd_signal
            )
            
            return macd, signal, hist
            
        except Exception as e:
            self.logger.error(f"计算MACD失败: {e}")
            raise
    
    @performance_monitor
    def detect_divergence(self, highs: np.ndarray, lows: np.ndarray, 
                         closes: np.ndarray, vol_factor: float = 0.0) -> List[DivergenceSignal]:
        """
        检测MACD背离 - 基于专家算法优化
        
        Args:
            highs: 最高价数组
            lows: 最低价数组  
            closes: 收盘价数组
            vol_factor: 波动性因子，用于动态调整阈值
            
        Returns:
            背离信号列表
        """
        try:
            if len(closes) < self.lookback:
                self.logger.warning(f"数据长度不足: {len(closes)} < {self.lookback}")
                return []
            
            # 计算MACD
            macd, signal, hist = self.compute_macd(closes[-self.lookback:])
            
            # 过滤NaN值
            valid_mask = ~(np.isnan(macd) | np.isnan(signal) | np.isnan(hist))
            if not np.any(valid_mask):
                self.logger.warning("MACD计算结果全为NaN")
                return []
            
            macd = macd[valid_mask]
            signal = signal[valid_mask]
            hist = hist[valid_mask]
            
            # 调整对应的价格数据
            price_start_idx = len(closes) - len(macd)
            adjusted_highs = highs[price_start_idx:]
            adjusted_lows = lows[price_start_idx:]
            
            # 动态阈值
            gap_thresh = self.min_gap + vol_factor * self.vol_factor_mult
            
            # 计算prominence阈值
            price_prominence = self.prominence_mult * np.std(adjusted_highs)
            macd_prominence = self.prominence_mult * np.std(hist)
            
            # 检测价格峰谷
            price_peaks, _ = find_peaks(
                adjusted_highs, 
                distance=self.min_distance, 
                prominence=max(price_prominence, 0.001)  # 防止prominence为0
            )
            price_valleys, _ = find_peaks(
                -adjusted_lows, 
                distance=self.min_distance, 
                prominence=max(price_prominence, 0.001)
            )
            
            # 检测MACD柱状图峰谷
            macd_peaks, _ = find_peaks(
                hist, 
                distance=self.min_distance, 
                prominence=max(macd_prominence, 0.001)
            )
            macd_valleys, _ = find_peaks(
                -hist, 
                distance=self.min_distance, 
                prominence=max(macd_prominence, 0.001)
            )
            
            signals = []
            
            # 检测看跌背离
            bear_signals = self._find_consecutive_divergence(
                price_peaks, macd_peaks, adjusted_highs, hist, 
                is_bearish=True, gap_thresh=gap_thresh
            )
            signals.extend(bear_signals)
            
            # 检测看涨背离
            bull_signals = self._find_consecutive_divergence(
                price_valleys, macd_valleys, -adjusted_lows, -hist, 
                is_bearish=False, gap_thresh=gap_thresh
            )
            signals.extend(bull_signals)
            
            # 更新统计
            self.detection_stats['total_detections'] += 1
            self.detection_stats['divergence_count'] += len(signals)
            
            # 记录高置信度信号
            high_conf_signals = [s for s in signals if s.confidence > 0.7]
            self.detection_stats['high_confidence_signals'] += len(high_conf_signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"背离检测失败: {e}")
            return []
    
    def _find_consecutive_divergence(self, price_extrema: np.ndarray, macd_extrema: np.ndarray,
                                   prices: np.ndarray, macd: np.ndarray, 
                                   is_bearish: bool, gap_thresh: float) -> List[DivergenceSignal]:
        """
        查找连续背离 - 核心算法
        
        Args:
            price_extrema: 价格极值索引
            macd_extrema: MACD极值索引
            prices: 价格数组
            macd: MACD数组
            is_bearish: 是否为看跌背离
            gap_thresh: 间隔阈值
            
        Returns:
            背离信号列表
        """
        signals = []
        
        if len(price_extrema) < self.min_consecutive or len(macd_extrema) < self.min_consecutive:
            return signals
        
        price_extrema = np.sort(price_extrema)
        macd_extrema = np.sort(macd_extrema)
        
        for start in range(len(price_extrema) - self.min_consecutive + 1):
            seq_price = price_extrema[start:start + self.min_consecutive]
            seq_macd = []
            
            # 寻找对应的MACD极值
            for idx in seq_price:
                closest_macd = self._find_closest(macd_extrema, idx)
                if closest_macd is None:
                    break
                seq_macd.append(closest_macd)
            
            if len(seq_macd) != self.min_consecutive:
                continue
            
            # 检查柱转虚条件 - 专家建议的关键过滤
            turn_virtual = self._check_turn_virtual(macd, seq_macd, is_bearish)
            if not turn_virtual:
                continue
            
            # 计算背离强度
            div_count = 0
            total_strength = 0
            price_values = []
            macd_values = []
            
            for i in range(1, self.min_consecutive):
                price_diff = prices[seq_price[i]] - prices[seq_price[i-1]]
                macd_diff = macd[seq_macd[i]] - macd[seq_macd[i-1]]
                
                price_values.append(prices[seq_price[i]])
                macd_values.append(macd[seq_macd[i]])
                
                # 背离条件检查
                is_divergence = False
                if is_bearish:
                    # 看跌背离：价格新高，MACD新低
                    is_divergence = price_diff > 0 and macd_diff < 0
                else:
                    # 看涨背离：价格新低，MACD新高
                    is_divergence = price_diff < 0 and macd_diff > 0
                
                if is_divergence:
                    # 计算背离强度
                    strength = abs(macd_diff / price_diff) if abs(price_diff) > 1e-8 else 0
                    if strength > gap_thresh:
                        div_count += 1
                        total_strength += strength
            
            # 生成信号
            if div_count >= self.min_consecutive - 1:
                avg_strength = total_strength / div_count if div_count > 0 else 0
                confidence = self._calculate_divergence_confidence(
                    avg_strength, gap_thresh, div_count
                )
                
                signal_type = DivergenceType.BEARISH if is_bearish else DivergenceType.BULLISH
                
                signal = DivergenceSignal(
                    type=signal_type,
                    strength=avg_strength,
                    confidence=confidence,
                    indices=seq_price.tolist(),
                    macd_values=macd_values,
                    price_values=price_values
                )
                
                signals.append(signal)
        
        return signals
    
    def _check_turn_virtual(self, macd: np.ndarray, seq_macd: List[int], is_bearish: bool) -> bool:
        """
        检查柱转虚条件 - 专家建议的关键过滤
        
        Args:
            macd: MACD柱状图数组
            seq_macd: MACD极值序列
            is_bearish: 是否为看跌背离
            
        Returns:
            是否满足柱转虚条件
        """
        try:
            if not seq_macd:
                return False
            
            latest_macd = macd[seq_macd[-1]]
            
            if is_bearish:
                # 看跌背离：MACD柱应该从正转负（转虚）
                return latest_macd < 0
            else:
                # 看涨背离：MACD柱应该从负转正（转虚）
                return latest_macd > 0
                
        except Exception as e:
            self.logger.error(f"检查柱转虚失败: {e}")
            return False
    
    def _calculate_divergence_confidence(self, avg_strength: float, gap_thresh: float, div_count: int) -> float:
        """
        计算背离置信度
        
        Args:
            avg_strength: 平均强度
            gap_thresh: 间隔阈值
            div_count: 背离计数
            
        Returns:
            置信度 (0-1)
        """
        try:
            # 基础置信度
            base_confidence = min((avg_strength / gap_thresh) * 0.5 + 0.5, 1.0)
            
            # 连续性加成
            consecutive_bonus = min((div_count - 1) * 0.1, 0.3)
            
            # 最终置信度
            final_confidence = min(base_confidence + consecutive_bonus, 1.0)
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"计算背离置信度失败: {e}")
            return 0.5
    
    def _find_closest(self, extrema: np.ndarray, target_idx: int) -> Optional[int]:
        """
        寻找最接近的极值索引
        
        Args:
            extrema: 极值索引数组
            target_idx: 目标索引
            
        Returns:
            最接近的极值索引或None
        """
        if len(extrema) == 0:
            return None
        
        distances = np.abs(extrema - target_idx)
        min_dist_idx = np.argmin(distances)
        
        if distances[min_dist_idx] <= self.tolerance:
            return extrema[min_dist_idx]
        
        return None
    
    @performance_monitor
    def detect_pattern(self, opens: np.ndarray, highs: np.ndarray, 
                      lows: np.ndarray, closes: np.ndarray) -> List[PatternSignal]:
        """
        检测形态 - 基于专家算法优化
        支持ENGULFING/HEAD_SHOULDER/CONVERGENCE_TRIANGLE
        
        Args:
            opens: 开盘价数组
            highs: 最高价数组
            lows: 最低价数组
            closes: 收盘价数组
            
        Returns:
            形态信号列表
        """
        try:
            signals = []
            
            if len(closes) < self.lookback:
                self.logger.warning(f"数据长度不足进行形态检测: {len(closes)} < {self.lookback}")
                return signals
            
            # 使用lookback长度的数据
            opens_slice = opens[-self.lookback:]
            highs_slice = highs[-self.lookback:]
            lows_slice = lows[-self.lookback:]
            closes_slice = closes[-self.lookback:]
            
            # 🔧 为_detect_double_patterns提供虚拟volumes数据
            # 由于detect_pattern方法不接受volumes参数，我们创建虚拟数据
            volumes_slice = np.ones_like(closes_slice) * 1000000  # 虚拟成交量数据
            
            # 1. 检测吞没形态
            if "ENGULFING" in self.morph_patterns:
                engulfing_signals = self._detect_engulfing_pattern(
                    opens_slice, highs_slice, lows_slice, closes_slice
                )
                signals.extend(engulfing_signals)
            
            # 2. 检测头肩形态
            if "HEAD_SHOULDER" in self.morph_patterns:
                head_shoulder_signals = self._detect_head_shoulder_pattern(
                    highs_slice, lows_slice, closes_slice
                )
                signals.extend(head_shoulder_signals)
            
            # 3. 检测收敛三角形
            if "CONVERGENCE_TRIANGLE" in self.morph_patterns:
                triangle_signals = self._detect_convergence_triangle_pattern(
                    opens_slice, highs_slice, lows_slice, closes_slice
                )
                signals.extend(triangle_signals)
            
            # 🆕 任务4.2: 检测W底/双顶形态
            if "DOUBLE_PATTERNS" in self.morph_patterns:
                double_pattern_signals = self._detect_double_patterns(
                    highs_slice, lows_slice, closes_slice, volumes_slice
                )
                signals.extend(double_pattern_signals)
            
            # 更新统计
            self.detection_stats['pattern_count'] += len(signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"形态检测失败: {e}")
            return []
    
    def _detect_double_patterns(self, highs: np.ndarray, lows: np.ndarray, 
                               closes: np.ndarray, volumes: np.ndarray) -> List[PatternSignal]:
        """
        🆕 任务4.2: 检测W底/双顶形态
        
        Args:
            highs, lows, closes, volumes: OHLCV数据
            
        Returns:
            检测到的双重形态信号列表
        """
        patterns = []
        
        try:
            # 调用改进版双重形态检测器
            double_patterns = self.double_pattern_detector.detect_patterns(
                highs, lows, closes, volumes
            )
            
            # 转换为标准PatternSignal格式
            for dp in double_patterns:
                try:
                    # 映射形态类型
                    if dp.type.value == "double_bottom_bull":
                        pattern_type = PatternType.DOUBLE_BOTTOM_BULL
                    elif dp.type.value == "double_top_bear":
                        pattern_type = PatternType.DOUBLE_TOP_BEAR
                    else:
                        continue
                    
                    # 创建标准形态信号
                    pattern_signal = PatternSignal(
                        type=pattern_type,
                        confidence=dp.confidence,
                        start_idx=dp.metadata.get('left_valley_idx', 0) if 'left_valley_idx' in dp.metadata else dp.metadata.get('left_peak_idx', 0),
                        end_idx=dp.metadata.get('right_valley_idx', len(closes)-1) if 'right_valley_idx' in dp.metadata else dp.metadata.get('right_peak_idx', len(closes)-1),
                        strength=min(dp.confidence * 100, 100),
                        direction='BULLISH' if pattern_type == PatternType.DOUBLE_BOTTOM_BULL else 'BEARISH',
                        entry_price=dp.entry_price,
                        stop_loss=dp.stop_loss,
                        take_profit=dp.target_price,
                        formation_bars=dp.formation_bars,
                        volume_confirmation=dp.volume_ratio > 1.0,
                        metadata={
                            'similarity_score': dp.similarity_score,
                            'rsi_confirmation': dp.rsi_confirmation,
                            'neckline': dp.metadata.get('neckline', dp.entry_price),
                            'confidence_breakdown': dp.metadata.get('confidence_breakdown', {}),
                            'double_pattern_detector': True  # 标识来源
                        }
                    )
                    
                    patterns.append(pattern_signal)
                    self.detection_stats['double_pattern_count'] += 1
                    
                    self.logger.info(f"✅ 检测到{pattern_type.value}形态: 置信度={dp.confidence:.2f}, "
                                   f"周期={dp.formation_bars}, 相似度={dp.similarity_score:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"转换双重形态信号失败: {e}")
                    continue
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"双重形态检测失败: {e}")
            return []
    
    def _detect_engulfing_pattern(self, opens: np.ndarray, highs: np.ndarray, 
                                lows: np.ndarray, closes: np.ndarray) -> List[PatternSignal]:
        """
        检测吞没形态 - 使用ta-lib
        
        Args:
            opens, highs, lows, closes: OHLC数据
            
        Returns:
            吞没形态信号列表
        """
        signals = []
        
        try:
            engulfing = ta.CDLENGULFING(opens, highs, lows, closes)
            
            if engulfing[-1] != 0:
                if engulfing[-1] > 0:
                    # 看涨吞没
                    pattern_type = PatternType.ENGULFING_BULL
                    confidence = 0.8
                else:
                    # 看跌吞没
                    pattern_type = PatternType.ENGULFING_BEAR
                    confidence = 0.7
                
                # 增强置信度计算
                confidence = self._enhance_engulfing_confidence(
                    opens, highs, lows, closes, confidence
                )
                
                signal = PatternSignal(
                    type=pattern_type,
                    confidence=confidence,
                    details={
                        'candle_index': len(closes) - 1,
                        'engulfing_value': int(engulfing[-1]),
                        'prev_candle_size': abs(closes[-2] - opens[-2]),
                        'curr_candle_size': abs(closes[-1] - opens[-1])
                    }
                )
                
                signals.append(signal)
                
        except Exception as e:
            self.logger.error(f"检测吞没形态失败: {e}")
        
        return signals
    
    def _enhance_engulfing_confidence(self, opens: np.ndarray, highs: np.ndarray,
                                    lows: np.ndarray, closes: np.ndarray, 
                                    base_confidence: float) -> float:
        """
        增强吞没形态置信度计算
        
        Args:
            opens, highs, lows, closes: OHLC数据
            base_confidence: 基础置信度
            
        Returns:
            增强后的置信度
        """
        try:
            # 计算吞没强度
            prev_body = abs(closes[-2] - opens[-2])
            curr_body = abs(closes[-1] - opens[-1])
            engulfing_ratio = curr_body / prev_body if prev_body > 0 else 1
            
            # 计算成交量确认（如果有成交量数据）
            volume_confirmation = 1.0  # 简化处理
            
            # 计算趋势背景
            trend_strength = self._calculate_trend_strength(closes)
            
            # 综合调整置信度
            confidence_multiplier = min(1.2, 0.8 + engulfing_ratio * 0.1 + trend_strength * 0.1)
            enhanced_confidence = min(1.0, base_confidence * confidence_multiplier)
            
            return enhanced_confidence
            
        except Exception as e:
            self.logger.error(f"增强吞没置信度计算失败: {e}")
            return base_confidence
    
    def _detect_head_shoulder_pattern(self, highs: np.ndarray, lows: np.ndarray, 
                                    closes: np.ndarray) -> List[PatternSignal]:
        """
        检测头肩形态 - 基于专家算法
        
        Args:
            highs: 最高价数组
            lows: 最低价数组
            closes: 收盘价数组
            
        Returns:
            头肩形态信号列表
        """
        signals = []
        
        try:
            # 检测峰值
            peaks_idx, _ = find_peaks(
                highs, 
                distance=self.min_distance, 
                prominence=self.prominence_mult * np.std(highs)
            )
            
            if len(peaks_idx) >= 3:
                # 取最近的三个峰值
                left, head, right = peaks_idx[-3:]
                
                # 验证头肩形态
                if (highs[head] > highs[left] and 
                    highs[head] > highs[right]):
                    
                    # 计算肩部相似度
                    shoulder_diff = abs(highs[left] - highs[right])
                    shoulder_similarity = 1.0 - (shoulder_diff / highs[head])
                    
                    if shoulder_diff < np.std(highs) * 0.3:  # 肩部相似度检查
                        # 计算颈线
                        left_low = np.min(lows[left:head])
                        right_low = np.min(lows[head:right])
                        neckline = np.mean([left_low, right_low])
                        
                        # 检查颈线突破
                        recent_closes = closes[-3:]
                        neckline_break = np.min(recent_closes) < neckline
                        
                        if neckline_break:
                            # 计算置信度
                            confidence = max(0.5, 0.85 - shoulder_diff / highs[head])
                            confidence *= shoulder_similarity  # 肩部相似度加成
                            
                            # 检查成交量确认
                            volume_confirmation = self._check_volume_confirmation(closes)
                            confidence *= volume_confirmation
                            
                            signal = PatternSignal(
                                type=PatternType.HEAD_SHOULDER_BEAR,
                                confidence=min(1.0, confidence),
                                details={
                                    'neckline': float(neckline),
                                    'peaks': [int(left), int(head), int(right)],
                                    'head_height': float(highs[head]),
                                    'left_shoulder': float(highs[left]),
                                    'right_shoulder': float(highs[right]),
                                    'shoulder_similarity': float(shoulder_similarity),
                                    'neckline_break_confirmed': neckline_break
                                }
                            )
                            
                            signals.append(signal)
            
            # 检测倒头肩（头肩底）
            valley_signals = self._detect_inverse_head_shoulder(lows, highs, closes)
            signals.extend(valley_signals)
            
        except Exception as e:
            self.logger.error(f"检测头肩形态失败: {e}")
        
        return signals
    
    def _detect_inverse_head_shoulder(self, lows: np.ndarray, highs: np.ndarray,
                                    closes: np.ndarray) -> List[PatternSignal]:
        """
        检测倒头肩形态（头肩底）
        
        Args:
            lows: 最低价数组
            highs: 最高价数组
            closes: 收盘价数组
            
        Returns:
            倒头肩形态信号列表
        """
        signals = []
        
        try:
            # 检测谷值
            valleys_idx, _ = find_peaks(
                -lows,
                distance=self.min_distance,
                prominence=self.prominence_mult * np.std(lows)
            )
            
            if len(valleys_idx) >= 3:
                # 取最近的三个谷值
                left, head, right = valleys_idx[-3:]
                
                # 验证倒头肩形态（头应该是最低点）
                if (lows[head] < lows[left] and 
                    lows[head] < lows[right]):
                    
                    # 计算肩部相似度
                    shoulder_diff = abs(lows[left] - lows[right])
                    shoulder_similarity = 1.0 - (shoulder_diff / abs(lows[head]))
                    
                    if shoulder_diff < np.std(lows) * 0.3:
                        # 计算颈线（阻力线）
                        left_high = np.max(highs[left:head])
                        right_high = np.max(highs[head:right])
                        neckline = np.mean([left_high, right_high])
                        
                        # 检查颈线突破（向上突破）
                        recent_closes = closes[-3:]
                        neckline_break = np.max(recent_closes) > neckline
                        
                        if neckline_break:
                            confidence = max(0.5, 0.85 - shoulder_diff / abs(lows[head]))
                            confidence *= shoulder_similarity
                            
                            signal = PatternSignal(
                                type=PatternType.HEAD_SHOULDER_BULL,
                                confidence=min(1.0, confidence),
                                details={
                                    'neckline': float(neckline),
                                    'valleys': [int(left), int(head), int(right)],
                                    'head_depth': float(lows[head]),
                                    'left_shoulder': float(lows[left]),
                                    'right_shoulder': float(lows[right]),
                                    'shoulder_similarity': float(shoulder_similarity),
                                    'neckline_break_confirmed': neckline_break
                                }
                            )
                            
                            signals.append(signal)
                            
        except Exception as e:
            self.logger.error(f"检测倒头肩形态失败: {e}")
        
        return signals
    
    def _detect_convergence_triangle_pattern(self, opens: np.ndarray, highs: np.ndarray,
                                           lows: np.ndarray, closes: np.ndarray) -> List[PatternSignal]:
        """
        检测收敛三角形形态 - 使用np.polyfit
        
        Args:
            opens, highs, lows, closes: OHLC数据
            
        Returns:
            收敛三角形信号列表
        """
        signals = []
        
        try:
            x = np.arange(len(highs))
            
            # 使用线性回归拟合趋势线
            high_slope, high_inter = np.polyfit(x, highs, 1)
            low_slope, low_inter = np.polyfit(x, lows, 1)
            
            # 检查收敛条件：上降下升
            if high_slope < 0 and low_slope > 0:
                # 计算收敛点
                convergence_point = (high_inter - low_inter) / (low_slope - high_slope)
                
                # 验证收敛点的合理性
                if 0 < convergence_point < len(highs) * 2:
                    # 检查波动收窄
                    recent_vol = np.mean(highs[-3:] - lows[-3:])
                    avg_vol = np.mean(highs - lows)
                    
                    if recent_vol < avg_vol * 0.7:  # 波动收窄确认
                        # 计算置信度
                        slope_strength = abs(high_slope) + abs(low_slope)
                        confidence = min(1.0, 0.75 + slope_strength * 0.1)
                        
                        # 确定突破方向
                        last_close = closes[-1]
                        last_open = opens[-1]
                        
                        pattern_type = (PatternType.CONVERGENCE_TRIANGLE_BULL 
                                      if last_close > last_open 
                                      else PatternType.CONVERGENCE_TRIANGLE_BEAR)
                        
                        # 增强置信度计算
                        confidence = self._enhance_triangle_confidence(
                            highs, lows, closes, confidence, convergence_point
                        )
                        
                        signal = PatternSignal(
                            type=pattern_type,
                            confidence=confidence,
                            details={
                                'convergence_point': float(convergence_point),
                                'high_slope': float(high_slope),
                                'low_slope': float(low_slope),
                                'high_intercept': float(high_inter),
                                'low_intercept': float(low_inter),
                                'volume_compression': float(recent_vol / avg_vol),
                                'slope_strength': float(slope_strength)
                            }
                        )
                        
                        signals.append(signal)
                        
        except Exception as e:
            self.logger.error(f"检测收敛三角形失败: {e}")
        
        return signals
    
    def _enhance_triangle_confidence(self, highs: np.ndarray, lows: np.ndarray,
                                   closes: np.ndarray, base_confidence: float,
                                   convergence_point: float) -> float:
        """
        增强三角形形态置信度
        
        Args:
            highs, lows, closes: 价格数据
            base_confidence: 基础置信度
            convergence_point: 收敛点
            
        Returns:
            增强后的置信度
        """
        try:
            # 计算趋势线拟合度
            x = np.arange(len(highs))
            high_slope, high_inter = np.polyfit(x, highs, 1)
            low_slope, low_inter = np.polyfit(x, lows, 1)
            
            # 计算R²值
            high_trend = high_slope * x + high_inter
            low_trend = low_slope * x + low_inter
            
            high_r2 = 1 - np.sum((highs - high_trend)**2) / np.sum((highs - np.mean(highs))**2)
            low_r2 = 1 - np.sum((lows - low_trend)**2) / np.sum((lows - np.mean(lows))**2)
            
            avg_r2 = (high_r2 + low_r2) / 2
            
            # 计算距离收敛点的远近
            distance_factor = min(1.0, convergence_point / len(highs))
            
            # 计算波动性压缩程度
            early_vol = np.mean(highs[:len(highs)//2] - lows[:len(lows)//2])
            late_vol = np.mean(highs[len(highs)//2:] - lows[len(lows)//2:])
            compression_ratio = late_vol / early_vol if early_vol > 0 else 1
            
            # 综合调整置信度
            r2_bonus = max(0, avg_r2 - 0.5) * 0.2
            distance_bonus = (1 - distance_factor) * 0.1
            compression_bonus = max(0, 1 - compression_ratio) * 0.1
            
            enhanced_confidence = min(1.0, 
                base_confidence + r2_bonus + distance_bonus + compression_bonus
            )
            
            return enhanced_confidence
            
        except Exception as e:
            self.logger.error(f"增强三角形置信度失败: {e}")
            return base_confidence
    
    def _calculate_trend_strength(self, closes: np.ndarray) -> float:
        """
        计算趋势强度
        
        Args:
            closes: 收盘价数组
            
        Returns:
            趋势强度 (-1到1)
        """
        try:
            if len(closes) < 10:
                return 0.0
            
            # 短期和长期移动平均
            short_ma = np.mean(closes[-5:])
            long_ma = np.mean(closes[-10:])
            
            # 计算趋势强度
            trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
            
            # 标准化到[-1, 1]范围
            return np.clip(trend_strength * 10, -1, 1)
            
        except Exception as e:
            self.logger.error(f"计算趋势强度失败: {e}")
            return 0.0
    
    def _check_volume_confirmation(self, closes: np.ndarray) -> float:
        """
        检查成交量确认（简化实现）
        
        Args:
            closes: 收盘价数组
            
        Returns:
            成交量确认因子 (0.8-1.2)
        """
        try:
            # 这里是简化实现，实际应该使用真实成交量数据
            # 基于价格变化模拟成交量确认
            price_volatility = np.std(closes[-5:]) / np.mean(closes[-5:])
            
            # 高波动性假设成交量放大
            if price_volatility > 0.02:
                return 1.1
            elif price_volatility < 0.01:
                return 0.9
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"检查成交量确认失败: {e}")
            return 1.0
    
    def analyze_market_structure(self, highs: np.ndarray, lows: np.ndarray, 
                               closes: np.ndarray, vol_factor: float = 0.0) -> Dict[str, Any]:
        """
        综合分析市场结构
        
        Args:
            highs: 最高价数组
            lows: 最低价数组
            closes: 收盘价数组
            vol_factor: 波动性因子
            
        Returns:
            市场结构分析结果
        """
        try:
            # 检测背离信号
            divergence_signals = self.detect_divergence(highs, lows, closes, vol_factor)
            
            # 检测形态信号
            opens = closes - 0.5  # 简化开盘价
            pattern_signals = self.detect_pattern(opens, highs, lows, closes)
            
            # 分析信号质量
            signal_quality = self._analyze_signal_quality(divergence_signals, pattern_signals)
            
            # 综合评分
            overall_score = self._calculate_overall_score(divergence_signals, pattern_signals)
            
            analysis_result = {
                'divergence_signals': [signal.to_dict() for signal in divergence_signals],
                'pattern_signals': [signal.to_dict() for signal in pattern_signals],
                'signal_quality': signal_quality,
                'overall_score': overall_score,
                'market_condition': self._determine_market_condition(overall_score),
                'recommendation': self._generate_recommendation(divergence_signals, pattern_signals),
                'timestamp': datetime.now().isoformat()
            }
            
            # 记录到历史
            self._add_to_history(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"市场结构分析失败: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_signal_quality(self, divergence_signals: List[DivergenceSignal],
                              pattern_signals: List[PatternSignal]) -> Dict[str, Any]:
        """
        分析信号质量
        
        Args:
            divergence_signals: 背离信号列表
            pattern_signals: 形态信号列表
            
        Returns:
            信号质量分析结果
        """
        try:
            total_signals = len(divergence_signals) + len(pattern_signals)
            
            if total_signals == 0:
                return {
                    'total_signals': 0,
                    'high_confidence_ratio': 0.0,
                    'avg_confidence': 0.0,
                    'signal_consistency': 0.0
                }
            
            # 计算高置信度信号比例
            high_conf_count = 0
            total_confidence = 0
            
            for signal in divergence_signals:
                if signal.confidence > 0.7:
                    high_conf_count += 1
                total_confidence += signal.confidence
            
            for signal in pattern_signals:
                if signal.confidence > 0.7:
                    high_conf_count += 1
                total_confidence += signal.confidence
            
            high_confidence_ratio = high_conf_count / total_signals
            avg_confidence = total_confidence / total_signals
            
            # 计算信号一致性
            signal_consistency = self._calculate_signal_consistency(
                divergence_signals, pattern_signals
            )
            
            return {
                'total_signals': total_signals,
                'high_confidence_ratio': high_confidence_ratio,
                'avg_confidence': avg_confidence,
                'signal_consistency': signal_consistency,
                'divergence_count': len(divergence_signals),
                'pattern_count': len(pattern_signals)
            }
            
        except Exception as e:
            self.logger.error(f"分析信号质量失败: {e}")
            return {}
    
    def _calculate_signal_consistency(self, divergence_signals: List[DivergenceSignal],
                                    pattern_signals: List[PatternSignal]) -> float:
        """
        计算信号一致性
        
        Args:
            divergence_signals: 背离信号列表
            pattern_signals: 形态信号列表
            
        Returns:
            信号一致性分数 (0-1)
        """
        try:
            if not divergence_signals and not pattern_signals:
                return 0.0
            
            # 统计看涨和看跌信号
            bullish_count = 0
            bearish_count = 0
            
            for signal in divergence_signals:
                if signal.type == DivergenceType.BULLISH:
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            for signal in pattern_signals:
                if 'BULL' in signal.type.value:
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            total_count = bullish_count + bearish_count
            
            if total_count == 0:
                return 0.0
            
            # 计算一致性：主导方向的信号占比
            dominant_ratio = max(bullish_count, bearish_count) / total_count
            
            return dominant_ratio
            
        except Exception as e:
            self.logger.error(f"计算信号一致性失败: {e}")
            return 0.0
    
    def _calculate_overall_score(self, divergence_signals: List[DivergenceSignal],
                               pattern_signals: List[PatternSignal]) -> float:
        """
        计算综合评分
        
        Args:
            divergence_signals: 背离信号列表
            pattern_signals: 形态信号列表
            
        Returns:
            综合评分 (0-100)
        """
        try:
            if not divergence_signals and not pattern_signals:
                return 50.0  # 中性
            
            total_score = 0
            total_weight = 0
            
            # 背离信号评分（权重0.6）
            for signal in divergence_signals:
                signal_score = signal.confidence * 100
                if signal.type == DivergenceType.BULLISH:
                    signal_score = signal_score
                else:
                    signal_score = 100 - signal_score
                
                total_score += signal_score * 0.6
                total_weight += 0.6
            
            # 形态信号评分（权重0.4）
            for signal in pattern_signals:
                signal_score = signal.confidence * 100
                if 'BULL' in signal.type.value:
                    signal_score = signal_score
                else:
                    signal_score = 100 - signal_score
                
                total_score += signal_score * 0.4
                total_weight += 0.4
            
            if total_weight == 0:
                return 50.0
            
            overall_score = total_score / total_weight
            
            return np.clip(overall_score, 0, 100)
            
        except Exception as e:
            self.logger.error(f"计算综合评分失败: {e}")
            return 50.0
    
    def _determine_market_condition(self, overall_score: float) -> str:
        """
        确定市场状态
        
        Args:
            overall_score: 综合评分
            
        Returns:
            市场状态描述
        """
        if overall_score >= 75:
            return "强烈看涨"
        elif overall_score >= 60:
            return "看涨"
        elif overall_score >= 40:
            return "中性"
        elif overall_score >= 25:
            return "看跌"
        else:
            return "强烈看跌"
    
    def _generate_recommendation(self, divergence_signals: List[DivergenceSignal],
                               pattern_signals: List[PatternSignal]) -> str:
        """
        生成交易建议
        
        Args:
            divergence_signals: 背离信号列表
            pattern_signals: 形态信号列表
            
        Returns:
            交易建议
        """
        try:
            if not divergence_signals and not pattern_signals:
                return "无明确信号，建议观望"
            
            # 统计高质量信号
            high_quality_bullish = 0
            high_quality_bearish = 0
            
            for signal in divergence_signals:
                if signal.confidence > 0.7:
                    if signal.type == DivergenceType.BULLISH:
                        high_quality_bullish += 1
                    else:
                        high_quality_bearish += 1
            
            for signal in pattern_signals:
                if signal.confidence > 0.7:
                    if 'BULL' in signal.type.value:
                        high_quality_bullish += 1
                    else:
                        high_quality_bearish += 1
            
            if high_quality_bullish > high_quality_bearish:
                return f"建议看涨，发现{high_quality_bullish}个高质量看涨信号"
            elif high_quality_bearish > high_quality_bullish:
                return f"建议看跌，发现{high_quality_bearish}个高质量看跌信号"
            else:
                return "信号混合，建议谨慎操作"
                
        except Exception as e:
            self.logger.error(f"生成交易建议失败: {e}")
            return "分析出错，建议观望"
    
    def _add_to_history(self, analysis_result: Dict[str, Any]):
        """
        添加分析结果到历史记录
        
        Args:
            analysis_result: 分析结果
        """
        try:
            self.signal_history.append(analysis_result)
            
            # 限制历史记录大小
            if len(self.signal_history) > self.max_history:
                self.signal_history = self.signal_history[-self.max_history//2:]
                
        except Exception as e:
            self.logger.error(f"添加历史记录失败: {e}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            检测统计信息
        """
        try:
            total_detections = self.detection_stats['total_detections']
            
            if total_detections == 0:
                return {
                    'total_detections': 0,
                    'detection_rates': {},
                    'accuracy_metrics': {}
                }
            
            return {
                'total_detections': total_detections,
                'divergence_count': self.detection_stats['divergence_count'],
                'pattern_count': self.detection_stats['pattern_count'],
                'high_confidence_signals': self.detection_stats['high_confidence_signals'],
                'detection_rates': {
                    'divergence_rate': self.detection_stats['divergence_count'] / total_detections,
                    'pattern_rate': self.detection_stats['pattern_count'] / total_detections,
                    'high_confidence_rate': self.detection_stats['high_confidence_signals'] / total_detections
                },
                'accuracy_metrics': {
                    'false_positive_rate': self.detection_stats['false_positive_rate'],
                    'signal_history_size': len(self.signal_history)
                },
                'configuration': {
                    'lookback_period': self.lookback,
                    'min_distance': self.min_distance,
                    'prominence_multiplier': self.prominence_mult,
                    'supported_patterns': self.morph_patterns
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取检测统计失败: {e}")
            return {}
    
    def reset_statistics(self):
        """
        重置检测统计
        """
        try:
            self.detection_stats = {
                'total_detections': 0,
                'divergence_count': 0,
                'pattern_count': 0,
                'high_confidence_signals': 0,
                'false_positive_rate': 0.0
            }
            
            self.signal_history.clear()
            
            self.logger.info("检测统计已重置")
            
        except Exception as e:
            self.logger.error(f"重置检测统计失败: {e}")
    
    def update_configuration(self, **kwargs):
        """
        更新检测配置
        
        Args:
            **kwargs: 配置参数
        """
        try:
            if 'lookback' in kwargs:
                self.lookback = kwargs['lookback']
            
            if 'min_distance' in kwargs:
                self.min_distance = kwargs['min_distance']
            
            if 'prominence_mult' in kwargs:
                self.prominence_mult = kwargs['prominence_mult']
            
            if 'min_consecutive' in kwargs:
                self.min_consecutive = kwargs['min_consecutive']
            
            if 'morph_patterns' in kwargs:
                self.morph_patterns = kwargs['morph_patterns']
            
            self.logger.info(f"检测配置已更新: {kwargs}")
            
        except Exception as e:
            self.logger.error(f"更新检测配置失败: {e}")

    def detect_patterns(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> List[Dict]:
        """批量检测形态的简化接口"""
        try:
            if len(df) < 20:
                return []
                
            # 转换数据
            opens = df['open'].astype(float).values
            highs = df['high'].astype(float).values  
            lows = df['low'].astype(float).values
            closes = df['close'].astype(float).values
            
            # 检测单个形态
            pattern = self.detect_pattern(opens, highs, lows, closes)
            
            # 包装成列表格式
            results = []
            if pattern and pattern.get('pattern_type') != 'UNKNOWN':
                results.append({
                    'symbol': symbol,
                    'pattern_type': pattern.get('pattern_type', 'UNKNOWN'),
                    'confidence': pattern.get('confidence', 0.0),
                    'timestamp': df.index[-1] if hasattr(df, 'index') else len(df)-1
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量形态检测失败: {e}")
            return []


# 测试函数
def test_enhanced_pattern_detector():
    """
    测试增强形态检测器
    """
    try:
        # 创建模拟配置
        from config.config_manager import ConfigManager
        config = ConfigManager()
        
        # 创建检测器
        detector = EnhancedPatternDetector(config)
        
        # 生成测试数据
        np.random.seed(42)
        length = 100
        
        base_price = 100
        trend = np.linspace(0, 10, length)
        noise = np.random.normal(0, 1, length)
        
        closes = base_price + trend + noise
        highs = closes + np.abs(np.random.normal(0, 0.5, length))
        lows = closes - np.abs(np.random.normal(0, 0.5, length))
        opens = closes - np.random.normal(0, 0.2, length)
        
        # 测试市场结构分析
        analysis = detector.analyze_market_structure(highs, lows, closes)
        
        print("=== 增强形态检测器测试结果 ===")
        print(f"检测到背离信号: {len(analysis.get('divergence_signals', []))}")
        print(f"检测到形态信号: {len(analysis.get('pattern_signals', []))}")
        print(f"综合评分: {analysis.get('overall_score', 0):.2f}")
        print(f"市场状态: {analysis.get('market_condition', 'Unknown')}")
        print(f"交易建议: {analysis.get('recommendation', 'Unknown')}")
        
        # 获取统计信息
        stats = detector.get_detection_statistics()
        print(f"\n检测统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    test_enhanced_pattern_detector() 