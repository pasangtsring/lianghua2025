"""
支撑阻力分析器 - 针对持仓优化的技术分析
基于大佬建议的W底/双顶识别算法，专门为渐进式止损优化设计
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime

class SupportResistanceAnalyzer:
    """支撑阻力分析器 - 专门为止损优化设计"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 加密货币市场优化参数 - 针对止损应用专门调整
        self.crypto_params = {
            'diff_thresh': 0.05,      # 价格相似性阈值 (5% - 更宽松)
            'vol_mult': 1.2,          # 突破成交量倍数 (降低要求)
            'rsi_thresh': 45,         # RSI确认阈值 (更宽松)
            'prominence_mult': 0.4,   # 峰谷识别敏感度 (更敏感)
            'confidence_thresh': 0.3, # 最小置信度阈值 (大幅降低)
            'min_distance': 5,        # 最小峰谷距离 (更短)
            'lookback_period': 30,    # 回看周期 (更短，适合快速市场)
            'enable_lightweight_mode': True  # 启用轻量级模式
        }
        
        self.logger.info("支撑阻力分析器初始化完成 - 加密货币市场优化版本")
    
    def analyze(self, kline_data: List) -> Dict:
        """
        分析K线数据，识别关键支撑阻力位
        
        Args:
            kline_data: K线数据列表 [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            分析结果字典 {
                'support': float | None,
                'resistance': float | None,
                'confidence': float,
                'pattern_type': str,
                'fibonacci_levels': dict,
                'analysis_timestamp': str
            }
        """
        try:
            if not kline_data or len(kline_data) < self.crypto_params['lookback_period']:
                return self._empty_result(f"数据不足，需要至少{self.crypto_params['lookback_period']}条K线")
            
            # 提取OHLCV数据
            highs = np.array([float(k[2]) for k in kline_data])
            lows = np.array([float(k[3]) for k in kline_data])  
            closes = np.array([float(k[4]) for k in kline_data])
            volumes = np.array([float(k[5]) for k in kline_data])
            
            self.logger.debug(f"分析K线数据: {len(kline_data)}条，价格范围 {np.min(lows):.6f} - {np.max(highs):.6f}")
            
            # 计算技术指标
            rsi = self._calculate_rsi(closes)
            atr = self._calculate_atr(highs, lows, closes)
            
            # 识别W底支撑
            w_bottom_result = self._identify_w_bottom(
                lows, highs, closes, volumes, rsi, atr
            )
            
            # 识别双顶阻力
            double_top_result = self._identify_double_top(
                highs, lows, closes, volumes, rsi, atr
            )
            
            # 计算Fibonacci回撤位
            fib_levels = self._calculate_fibonacci_levels(highs, lows)
            
            # 计算关键均线水平
            ma_levels = self._calculate_ma_levels(closes)
            
            # 轻量级模式：如果W底/双顶都未识别，使用简单支撑阻力
            if (self.crypto_params.get('enable_lightweight_mode', False) and 
                w_bottom_result.get('confidence', 0) < 0.3 and 
                double_top_result.get('confidence', 0) < 0.3):
                simple_levels = self._identify_simple_support_resistance(highs, lows, closes)
                if simple_levels['confidence'] > 0.3:
                    self.logger.info(f"使用简单支撑阻力识别: 支撑={simple_levels.get('support')}, 阻力={simple_levels.get('resistance')}")
                    return simple_levels
            
            # 综合分析结果
            result = self._combine_analysis_results(
                w_bottom_result, double_top_result, fib_levels, ma_levels
            )
            
            self.logger.info(f"技术分析完成: 支撑={result.get('support')}, 阻力={result.get('resistance')}, 置信度={result.get('confidence', 0):.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"支撑阻力分析失败: {e}")
            return self._empty_result(f"分析异常: {e}")
    
    def _identify_w_bottom(self, lows, highs, closes, volumes, rsi, atr) -> Dict:
        """识别W底支撑形态 - 基于大佬算法优化"""
        try:
            if len(lows) < 20:
                return {'support': None, 'confidence': 0.0, 'pattern': None}
            
            # 使用scipy.signal.find_peaks识别谷值
            prominence = self.crypto_params['prominence_mult'] * np.std(lows[-50:])
            valleys_idx = find_peaks(
                -lows, 
                distance=self.crypto_params['min_distance'], 
                prominence=prominence
            )[0]
            
            if len(valleys_idx) < 2:
                return {'support': None, 'confidence': 0.0, 'pattern': None}
            
            # 取最近两个低点
            low1_idx, low2_idx = valleys_idx[-2:]
            low1_price, low2_price = lows[low1_idx], lows[low2_idx]
            
            # 价格相似性检查
            price_diff = abs(low2_price - low1_price) / low1_price
            if price_diff > self.crypto_params['diff_thresh']:
                return {'support': None, 'confidence': 0.0, 'pattern': 'price_diff_too_large'}
            
            # 中间峰值检查 - 确保是W形态
            if low2_idx <= low1_idx:
                return {'support': None, 'confidence': 0.0, 'pattern': 'invalid_sequence'}
                
            middle_section = highs[low1_idx:low2_idx]
            if len(middle_section) == 0:
                return {'support': None, 'confidence': 0.0, 'pattern': 'no_middle_section'}
                
            middle_peak = np.max(middle_section)
            recent_high = np.max(highs[-20:])
            
            # 中间峰值不能太高（避免假W底）
            if middle_peak >= recent_high * 0.95:
                return {'support': None, 'confidence': 0.0, 'pattern': 'middle_peak_too_high'}
            
            # 轻量级模式：不要求完整突破确认，只验证形态完整性
            current_price = closes[-1]
            neckline = middle_peak
            
            # 如果启用轻量级模式，则跳过突破确认（专为止损应用）
            if not self.crypto_params.get('enable_lightweight_mode', False):
                if current_price <= neckline:
                    return {'support': None, 'confidence': 0.0, 'pattern': 'no_breakout'}
            
            # 轻量级模式：记录但不强制突破确认
            breakout_status = "confirmed" if current_price > neckline else "pending"
            
            # 成交量确认（轻量级模式更宽松）
            recent_volume = volumes[-1]
            avg_volume = np.mean(volumes[-20:-1]) if len(volumes) > 20 else np.mean(volumes[:-1])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 轻量级模式：降低成交量要求
            if self.crypto_params.get('enable_lightweight_mode', False):
                # 轻量级模式只要求基本成交量活跃度
                if volume_ratio < 0.5:  # 成交量不能过于萎缩
                    return {'support': None, 'confidence': 0.0, 'pattern': 'volume_too_low'}
            else:
                # 传统模式：严格成交量要求
                if volume_ratio < self.crypto_params['vol_mult']:
                    return {'support': None, 'confidence': 0.0, 'pattern': 'volume_insufficient'}
            
            # RSI确认（轻量级模式更宽松）
            current_rsi = rsi[-1] if len(rsi) > 0 else 50
            if self.crypto_params.get('enable_lightweight_mode', False):
                # 轻量级模式：RSI范围更宽松 (30-70)
                if not (30 <= current_rsi <= 70):
                    rsi_factor = 0.5  # 降低但不完全排除
                else:
                    rsi_factor = 1.0
            else:
                # 传统模式：严格RSI要求
                if current_rsi < self.crypto_params['rsi_thresh']:
                    return {'support': None, 'confidence': 0.0, 'pattern': 'rsi_weak'}
                rsi_factor = 1.0
            
            # ATR确认（轻量级模式更宽松）
            current_atr = atr[-1] if len(atr) > 0 else 0
            avg_atr = np.mean(atr[-10:-1]) if len(atr) > 10 else current_atr
            if self.crypto_params.get('enable_lightweight_mode', False):
                # 轻量级模式：允许更高波动率
                atr_factor = max(0.3, 1 - min(current_atr / avg_atr if avg_atr > 0 else 0, 3.0) / 3.0)
            else:
                # 传统模式：严格ATR限制
                if current_atr > avg_atr * 1.5:
                    return {'support': None, 'confidence': 0.0, 'pattern': 'volatility_too_high'}
                atr_factor = 1 - min(current_atr / avg_atr if avg_atr > 0 else 0, 2.0) / 2.0
            
            # 计算综合置信度（轻量级模式优化）
            price_factor = 1 - price_diff  # 价格越相似越好
            
            if self.crypto_params.get('enable_lightweight_mode', False):
                # 轻量级模式：更注重形态，降低指标要求
                volume_factor = min(volume_ratio / 0.8, 1.5) / 1.5 if volume_ratio > 0.5 else 0.3
                breakout_factor = 1.0 if breakout_status == "confirmed" else 0.7  # 未突破也给予较高评分
                
                confidence = (price_factor * 0.4 +      # 价格相似性最重要
                             volume_factor * 0.2 +       # 成交量权重降低  
                             rsi_factor * 0.2 +          # RSI权重保持
                             atr_factor * 0.1 +          # ATR权重降低
                             breakout_factor * 0.1)      # 突破状态加分
            else:
                # 传统模式：严格指标要求
                volume_factor = min(volume_ratio / self.crypto_params['vol_mult'], 2.0) / 2.0
                rsi_factor = min(current_rsi / 100.0, 1.0)
                
                confidence = (price_factor * 0.3 + volume_factor * 0.3 + 
                             rsi_factor * 0.2 + atr_factor * 0.2)
            
            if confidence > self.crypto_params['confidence_thresh']:
                support_price = min(low1_price, low2_price)
                self.logger.info(f"W底支撑识别成功: {support_price:.6f}, 置信度: {confidence:.3f}")
                return {
                    'support': support_price,
                    'confidence': confidence,
                    'pattern': 'w_bottom',
                    'details': {
                        'low1': low1_price,
                        'low2': low2_price,
                        'neckline': neckline,
                        'volume_ratio': volume_ratio,
                        'rsi': current_rsi
                    }
                }
            
            return {'support': None, 'confidence': confidence, 'pattern': 'confidence_too_low'}
            
        except Exception as e:
            self.logger.error(f"W底识别失败: {e}")
            return {'support': None, 'confidence': 0.0, 'pattern': f'error: {e}'}
    
    def _identify_double_top(self, highs, lows, closes, volumes, rsi, atr) -> Dict:
        """识别双顶阻力形态 - 基于大佬算法优化"""
        try:
            if len(highs) < 20:
                return {'resistance': None, 'confidence': 0.0, 'pattern': None}
            
            # 使用scipy.signal.find_peaks识别峰值
            prominence = self.crypto_params['prominence_mult'] * np.std(highs[-50:])
            peaks_idx = find_peaks(
                highs, 
                distance=self.crypto_params['min_distance'],
                prominence=prominence
            )[0]
            
            if len(peaks_idx) < 2:
                return {'resistance': None, 'confidence': 0.0, 'pattern': None}
                
            # 取最近两个高点
            peak1_idx, peak2_idx = peaks_idx[-2:]
            peak1_price, peak2_price = highs[peak1_idx], highs[peak2_idx]
            
            # 价格相似性检查
            price_diff = abs(peak2_price - peak1_price) / peak1_price
            if price_diff > self.crypto_params['diff_thresh']:
                return {'resistance': None, 'confidence': 0.0, 'pattern': 'price_diff_too_large'}
            
            # 中间谷值检查 - 确保是双顶形态
            if peak2_idx <= peak1_idx:
                return {'resistance': None, 'confidence': 0.0, 'pattern': 'invalid_sequence'}
                
            middle_section = lows[peak1_idx:peak2_idx]
            if len(middle_section) == 0:
                return {'resistance': None, 'confidence': 0.0, 'pattern': 'no_middle_section'}
                
            middle_valley = np.min(middle_section)
            recent_low = np.min(lows[-20:])
            
            # 中间谷值不能太低（避免假双顶）
            if middle_valley <= recent_low * 1.05:
                return {'resistance': None, 'confidence': 0.0, 'pattern': 'middle_valley_too_low'}
            
            # 轻量级模式：不要求完整跌破确认，只验证形态完整性
            current_price = closes[-1]
            neckline = middle_valley
            
            # 如果启用轻量级模式，则跳过跌破确认（专为止损应用）
            if not self.crypto_params.get('enable_lightweight_mode', False):
                if current_price >= neckline:
                    return {'resistance': None, 'confidence': 0.0, 'pattern': 'no_breakdown'}
            
            # 轻量级模式：记录但不强制跌破确认
            breakdown_status = "confirmed" if current_price < neckline else "pending"
            
            # 成交量确认（轻量级模式更宽松）
            recent_volume = volumes[-1]
            avg_volume = np.mean(volumes[-20:-1]) if len(volumes) > 20 else np.mean(volumes[:-1])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 轻量级模式：降低成交量要求
            if self.crypto_params.get('enable_lightweight_mode', False):
                if volume_ratio < 0.5:
                    return {'resistance': None, 'confidence': 0.0, 'pattern': 'volume_too_low'}
            else:
                if volume_ratio < self.crypto_params['vol_mult']:
                    return {'resistance': None, 'confidence': 0.0, 'pattern': 'volume_insufficient'}
            
            # RSI确认（轻量级模式更宽松）
            current_rsi = rsi[-1] if len(rsi) > 0 else 50
            if self.crypto_params.get('enable_lightweight_mode', False):
                # 轻量级模式：RSI范围更宽松 (30-70)
                if not (30 <= current_rsi <= 70):
                    rsi_factor = 0.5
                else:
                    rsi_factor = 1.0
            else:
                # 传统模式：严格RSI要求
                if current_rsi > (100 - self.crypto_params['rsi_thresh']):
                    return {'resistance': None, 'confidence': 0.0, 'pattern': 'rsi_too_strong'}
                rsi_factor = 1 - min(current_rsi / 100.0, 1.0)
            
            # 计算综合置信度（轻量级模式优化）
            price_factor = 1 - price_diff
            
            if self.crypto_params.get('enable_lightweight_mode', False):
                # 轻量级模式：更注重形态，降低指标要求
                volume_factor = min(volume_ratio / 0.8, 1.5) / 1.5 if volume_ratio > 0.5 else 0.3
                breakdown_factor = 1.0 if breakdown_status == "confirmed" else 0.7
                
                confidence = (price_factor * 0.4 +      # 价格相似性最重要 
                             volume_factor * 0.2 +       # 成交量权重降低
                             rsi_factor * 0.2 +          # RSI权重保持
                             breakdown_factor * 0.2)     # 跌破状态权重
            else:
                # 传统模式：严格指标要求
                volume_factor = min(volume_ratio / self.crypto_params['vol_mult'], 2.0) / 2.0
                
                confidence = (price_factor * 0.4 + volume_factor * 0.3 + rsi_factor * 0.3)
            
            if confidence > self.crypto_params['confidence_thresh']:
                resistance_price = max(peak1_price, peak2_price)
                self.logger.info(f"双顶阻力识别成功: {resistance_price:.6f}, 置信度: {confidence:.3f}")
                return {
                    'resistance': resistance_price,
                    'confidence': confidence,
                    'pattern': 'double_top',
                    'details': {
                        'peak1': peak1_price,
                        'peak2': peak2_price,
                        'neckline': neckline,
                        'volume_ratio': volume_ratio,
                        'rsi': current_rsi
                    }
                }
            
            return {'resistance': None, 'confidence': confidence, 'pattern': 'confidence_too_low'}
            
        except Exception as e:
            self.logger.error(f"双顶识别失败: {e}")
            return {'resistance': None, 'confidence': 0.0, 'pattern': f'error: {e}'}
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """计算RSI指标"""
        try:
            if len(closes) < period + 1:
                return np.full(len(closes), 50.0)
            
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.zeros(len(closes))
            avg_losses = np.zeros(len(closes))
            
            # 初始平均值
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])
            
            # 指数移动平均
            for i in range(period + 1, len(closes)):
                avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
                avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
            
            # 计算RSI
            rs = avg_gains / (avg_losses + 1e-10)  # 避免除零
            rsi = 100 - (100 / (1 + rs))
            
            # 填充前面的值
            rsi[:period] = 50.0
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"RSI计算失败: {e}")
            return np.full(len(closes), 50.0)
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """计算ATR指标"""
        try:
            if len(closes) < 2:
                return np.full(len(closes), 0.01)
            
            # 计算真实波幅
            tr1 = highs - lows
            tr2 = np.abs(highs[1:] - closes[:-1])
            tr3 = np.abs(lows[1:] - closes[:-1])
            
            tr = np.maximum(tr1[1:], np.maximum(tr2, tr3))
            tr = np.concatenate([[tr1[0]], tr])  # 第一个值使用high-low
            
            # 计算ATR (简单移动平均)
            atr = np.zeros(len(tr))
            for i in range(len(tr)):
                start_idx = max(0, i - period + 1)
                atr[i] = np.mean(tr[start_idx:i+1])
            
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR计算失败: {e}")
            return np.full(len(closes), np.std(closes[-20:]) * 0.1)
    
    def _calculate_fibonacci_levels(self, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """计算Fibonacci回撤位"""
        try:
            if len(highs) < 20 or len(lows) < 20:
                return {}
            
            recent_high = np.max(highs[-50:])
            recent_low = np.min(lows[-50:])
            
            if recent_high <= recent_low:
                return {}
            
            fib_levels = [0.236, 0.382, 0.618, 0.786]
            fib_prices = {}
            
            for level in fib_levels:
                fib_price = recent_low + (recent_high - recent_low) * level
                fib_prices[f'fib_{int(level*1000)}'] = fib_price
            
            fib_prices['fib_high'] = recent_high
            fib_prices['fib_low'] = recent_low
            
            return fib_prices
            
        except Exception as e:
            self.logger.error(f"Fibonacci计算失败: {e}")
            return {}
    
    def _calculate_ma_levels(self, closes: np.ndarray) -> Dict:
        """计算关键均线水平"""
        try:
            ma_levels = {}
            periods = [20, 50]
            
            for period in periods:
                if len(closes) >= period:
                    ma = np.mean(closes[-period:])
                    ma_levels[f'ma{period}'] = ma
            
            return ma_levels
            
        except Exception as e:
            self.logger.error(f"均线计算失败: {e}")
            return {}
    
    def _combine_analysis_results(self, w_bottom_result: Dict, double_top_result: Dict, 
                                 fib_levels: Dict, ma_levels: Dict) -> Dict:
        """综合分析结果，调整支撑阻力位"""
        try:
            support = w_bottom_result.get('support')
            resistance = double_top_result.get('resistance')
            
            # 综合置信度取最高值
            confidence = max(
                w_bottom_result.get('confidence', 0.0),
                double_top_result.get('confidence', 0.0)
            )
            
            # 确定主要形态类型
            if w_bottom_result.get('confidence', 0.0) > double_top_result.get('confidence', 0.0):
                pattern_type = w_bottom_result.get('pattern', 'unknown')
            else:
                pattern_type = double_top_result.get('pattern', 'unknown')
            
            # Fibonacci和均线调整
            if support and fib_levels:
                support = self._adjust_level_with_fibonacci_ma(support, fib_levels, ma_levels, 'support')
            
            if resistance and fib_levels:
                resistance = self._adjust_level_with_fibonacci_ma(resistance, fib_levels, ma_levels, 'resistance')
            
            return {
                'support': support,
                'resistance': resistance,
                'confidence': confidence,
                'pattern_type': pattern_type,
                'fibonacci_levels': fib_levels,
                'ma_levels': ma_levels,
                'w_bottom_details': w_bottom_result.get('details'),
                'double_top_details': double_top_result.get('details'),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"结果综合失败: {e}")
            return self._empty_result(f"结果综合异常: {e}")
    
    def _adjust_level_with_fibonacci_ma(self, level: float, fib_levels: Dict, ma_levels: Dict, level_type: str) -> float:
        """使用Fibonacci和均线调整支撑阻力位"""
        try:
            if not fib_levels and not ma_levels:
                return level
            
            # 收集所有可能的调整位
            adjustment_levels = []
            adjustment_levels.extend(fib_levels.values())
            adjustment_levels.extend(ma_levels.values())
            
            if not adjustment_levels:
                return level
            
            # 找到最接近原始水平的调整位
            closest_level = min(adjustment_levels, key=lambda x: abs(x - level))
            
            # 如果调整位与原始位相近（在合理范围内），则使用调整位
            tolerance = abs(level) * 0.02  # 2%容差
            if abs(closest_level - level) <= tolerance:
                self.logger.debug(f"{level_type}位调整: {level:.6f} → {closest_level:.6f}")
                return closest_level
            
            return level
            
        except Exception as e:
            self.logger.error(f"水平调整失败: {e}")
            return level
    
    def _identify_simple_support_resistance(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """
        简单支撑阻力识别 - 轻量级模式备选方案
        基于近期高低点和价格聚集区域
        """
        try:
            if len(highs) < 10:
                return self._empty_result("数据不足进行简单分析")
            
            current_price = closes[-1]
            
            # 识别近期支撑（最近20根K线的低点聚集区）
            recent_lows = lows[-20:]
            recent_highs = highs[-20:]
            
            # 找到价格聚集度较高的区域
            support_candidates = []
            resistance_candidates = []
            
            # 支撑识别：找到多次测试的低点区域
            for i in range(len(recent_lows)):
                low_price = recent_lows[i]
                if low_price < current_price:  # 只考虑当前价格下方的低点
                    # 计算在此价格附近±2%范围内的测试次数
                    similar_count = sum(1 for l in recent_lows 
                                      if abs(l - low_price) / low_price <= 0.02)
                    if similar_count >= 2:  # 至少被测试2次
                        support_candidates.append((low_price, similar_count))
            
            # 阻力识别：找到多次测试的高点区域  
            for i in range(len(recent_highs)):
                high_price = recent_highs[i]
                if high_price > current_price:  # 只考虑当前价格上方的高点
                    similar_count = sum(1 for h in recent_highs 
                                      if abs(h - high_price) / high_price <= 0.02)
                    if similar_count >= 2:
                        resistance_candidates.append((high_price, similar_count))
            
            support = None
            resistance = None
            confidence = 0.0
            
            # 选择最强的支撑（最近且测试次数最多）
            if support_candidates:
                support_candidates.sort(key=lambda x: (-x[1], -x[0]))  # 按测试次数和价格排序
                support = support_candidates[0][0]
                confidence += 0.3 + min(support_candidates[0][1] * 0.1, 0.3)
            
            # 选择最强的阻力
            if resistance_candidates:
                resistance_candidates.sort(key=lambda x: (-x[1], x[0]))  # 按测试次数排序，价格从低到高
                resistance = resistance_candidates[0][0]
                confidence += 0.3 + min(resistance_candidates[0][1] * 0.1, 0.3)
            
            # 额外的置信度调整
            if support and resistance:
                price_range_factor = (resistance - support) / current_price
                if 0.05 <= price_range_factor <= 0.3:  # 合理的价格区间
                    confidence += 0.2
            
            confidence = min(confidence, 0.8)  # 简单方法置信度上限
            
            if support or resistance:
                support_str = f"{support:.6f}" if support else "None"
                resistance_str = f"{resistance:.6f}" if resistance else "None"
                self.logger.info(f"简单支撑阻力识别: 支撑={support_str}, "
                               f"阻力={resistance_str}, 置信度={confidence:.3f}")
                
                return {
                    'support': support,
                    'resistance': resistance,
                    'confidence': confidence,
                    'pattern_type': 'simple_levels',
                    'fibonacci_levels': {},
                    'ma_levels': {},
                    'analysis_timestamp': datetime.now().isoformat(),
                    'method': 'simple_support_resistance'
                }
            
            return self._empty_result("未找到有效的简单支撑阻力位")
            
        except Exception as e:
            self.logger.error(f"简单支撑阻力识别失败: {e}")
            return self._empty_result(f"简单分析异常: {e}")
    
    def _empty_result(self, reason: str) -> Dict:
        """返回空结果"""
        return {
            'support': None,
            'resistance': None,
            'confidence': 0.0,
            'pattern_type': 'none',
            'fibonacci_levels': {},
            'ma_levels': {},
            'reason': reason,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_technical_stop_loss(self, technical_result: Dict, position_side: str, current_price: float = None) -> Optional[float]:
        """
        基于技术分析结果计算合理的止损位
        
        Args:
            technical_result: analyze()方法返回的技术分析结果
            position_side: 持仓方向 ('LONG' 或 'SHORT')
            current_price: 当前价格（可选，用于验证）
            
        Returns:
            技术止损价格，如果无法确定则返回None
        """
        try:
            if technical_result['confidence'] < self.crypto_params['confidence_thresh']:
                self.logger.debug(f"技术分析置信度({technical_result['confidence']:.3f})过低")
                return None
            
            if position_side == 'LONG':
                # 多头：使用支撑位作为止损
                if technical_result['support']:
                    # 在支撑位下方设置止损，预留0.5%缓冲
                    technical_stop = technical_result['support'] * 0.995
                    self.logger.info(f"技术止损(LONG): 支撑位{technical_result['support']:.6f} → 止损{technical_stop:.6f}")
                    return technical_stop
                    
            else:  # SHORT
                # 空头：使用阻力位作为止损
                if technical_result['resistance']:
                    # 在阻力位上方设置止损，预留0.5%缓冲
                    technical_stop = technical_result['resistance'] * 1.005
                    self.logger.info(f"技术止损(SHORT): 阻力位{technical_result['resistance']:.6f} → 止损{technical_stop:.6f}")
                    return technical_stop
            
            self.logger.debug(f"无有效技术位可用于{position_side}止损计算")
            return None
            
        except Exception as e:
            self.logger.error(f"技术止损计算失败: {e}")
            return None
    
    def validate_stop_loss_with_technical_analysis(self, proposed_stop: float, technical_result: Dict, 
                                                 position_side: str) -> Dict:
        """
        用技术分析验证提议的止损位是否合理
        
        Args:
            proposed_stop: 提议的止损价格
            technical_result: 技术分析结果
            position_side: 持仓方向
            
        Returns:
            验证结果字典 {
                'safe': bool,           # 是否安全
                'reason': str,          # 原因说明
                'suggested_stop_loss': float,  # 建议的止损价格（如果不安全）
                'confidence': float     # 验证置信度
            }
        """
        try:
            if technical_result['confidence'] < 0.5:
                # 技术分析置信度太低，无法验证
                return {
                    'safe': True,  # 默认认为安全
                    'reason': f"技术分析置信度({technical_result['confidence']:.2f})过低，无法验证",
                    'suggested_stop_loss': proposed_stop,
                    'confidence': 0.1
                }
            
            if position_side == 'LONG':
                resistance = technical_result.get('resistance')
                if resistance and proposed_stop > resistance * 0.98:  # 止损太接近阻力位
                    suggested_stop = resistance * 0.995  # 在阻力位下方
                    return {
                        'safe': False,
                        'reason': f"LONG止损({proposed_stop:.6f})过于接近阻力位({resistance:.6f})",
                        'suggested_stop_loss': suggested_stop,
                        'confidence': technical_result['confidence']
                    }
                    
            else:  # SHORT
                support = technical_result.get('support')
                if support and proposed_stop < support * 1.02:  # 止损太接近支撑位
                    suggested_stop = support * 1.005  # 在支撑位上方
                    return {
                        'safe': False,
                        'reason': f"SHORT止损({proposed_stop:.6f})过于接近支撑位({support:.6f})",
                        'suggested_stop_loss': suggested_stop,
                        'confidence': technical_result['confidence']
                    }
            
            # 验证通过
            return {
                'safe': True,
                'reason': "技术验证通过",
                'suggested_stop_loss': proposed_stop,
                'confidence': technical_result['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"技术验证失败: {e}")
            return {
                'safe': True,  # 默认认为安全
                'reason': f"验证异常: {e}",
                'suggested_stop_loss': proposed_stop,
                'confidence': 0.1
            } 