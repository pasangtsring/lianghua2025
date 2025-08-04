"""
增强反转检测器 - 基于大佬算法的多指标反转识别
融合MACD、成交量、RSI、ATR、形态识别的科学反转检测
"""

import numpy as np
from scipy.signal import find_peaks
import talib as ta
from typing import Tuple, Optional, Dict, List
import logging
from datetime import datetime

class EnhancedReversalDetector:
    """增强反转检测器 - 专门为急拉急跌防护设计"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 基于大佬建议的加密货币优化参数
        self.crypto_params = {
            'min_length': 15,           # 15min K最小数据量
            'prominence_mult': 0.5,     # 峰值识别敏感度（0.3-0.6）
            'vol_mult': 1.5,            # 成交量放大确认倍数
            'rsi_thresh': 50,           # RSI中位阈值
            'atr_mult': 1.0,            # ATR稳定性倍数  
            'confidence_thresh': 0.5    # 置信度触发阈值
        }
        
        self.logger.info("增强反转检测器初始化完成 - 基于大佬算法优化")
    
    async def detect_reversal(self, kline_data: List) -> Tuple[Optional[str], float, Dict]:
        """
        15min K行情反转检测 - 基于大佬核心算法
        
        检测逻辑：
        - bull reversal: MACD金叉 + W底/ENGULFING_BULL + vol>1.5*avg + RSI>50 + ATR<avg
        - bear reversal: MACD死叉 + 双顶/ENGULFING_BEAR + vol>1.5*avg + RSI<50 + ATR<avg
        - 置信度: (cross_strength + morph_conf + vol_ratio + rsi_factor + atr_factor) / 5
        
        Args:
            kline_data: K线数据 [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            Tuple[signal_type, confidence, details]
            - signal_type: 'BULL_REVERSAL' | 'BEAR_REVERSAL' | None
            - confidence: 0.0-1.0
            - details: 详细分析结果
        """
        try:
            if not kline_data or len(kline_data) < self.crypto_params['min_length']:
                return None, 0.0, {'reason': '数据不足'}
            
            # 提取OHLCV数据
            opens = np.array([float(k[1]) for k in kline_data])
            highs = np.array([float(k[2]) for k in kline_data])
            lows = np.array([float(k[3]) for k in kline_data])
            closes = np.array([float(k[4]) for k in kline_data])
            volumes = np.array([float(k[5]) for k in kline_data])
            
            # 计算技术指标 - 处理nan值
            macd_line, macd_signal, _ = ta.MACD(closes, 12, 26, 9)
            rsi = ta.RSI(closes, 14)
            atr = ta.ATR(highs, lows, closes, 14)
            
            # 处理nan值
            if macd_line is not None and len(macd_line) > 0:
                macd_line = np.nan_to_num(macd_line, nan=0.0)
                macd_signal = np.nan_to_num(macd_signal, nan=0.0)
            
            if rsi is not None and len(rsi) > 0:
                rsi = np.nan_to_num(rsi, nan=50.0)
                
            if atr is not None and len(atr) > 0:
                atr = np.nan_to_num(atr, nan=0.01)
            
            # 1. MACD交叉强度计算
            cross_strength = self._calculate_macd_cross_strength(macd_line, macd_signal)
            
            # 2. 形态置信度计算 
            morph_conf = self._calculate_morph_confidence(opens, highs, lows, closes, volumes, rsi)
            
            # 3. 成交量确认
            vol_ratio = self._calculate_volume_ratio(volumes)
            
            # 4. RSI因子
            rsi_factor = self._calculate_rsi_factor(rsi, self.crypto_params['rsi_thresh'])
            
            # 5. ATR稳定性因子
            atr_factor = self._calculate_atr_factor(atr, self.crypto_params['atr_mult'])
            
            # 综合置信度计算（大佬公式）
            confidence = (cross_strength + morph_conf + vol_ratio + rsi_factor + atr_factor) / 5.0
            confidence = min(max(confidence, 0.0), 1.0)  # 限制在0-1范围
            
            details = {
                'cross_strength': cross_strength,
                'morph_conf': morph_conf, 
                'vol_ratio': vol_ratio,
                'rsi_factor': rsi_factor,
                'atr_factor': atr_factor,
                'macd_diff': macd_line[-1] - macd_signal[-1] if len(macd_line) > 0 else 0,
                'rsi_current': rsi[-1] if len(rsi) > 0 else 50,
                'atr_current': atr[-1] if len(atr) > 0 else 0,
                'vol_current': volumes[-1],
                'vol_avg': np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
            }
            
            # 判断反转类型和触发条件（优化逻辑：不要求所有条件都满足）
            if confidence > self.crypto_params['confidence_thresh']:
                macd_diff = macd_line[-1] - macd_signal[-1] if len(macd_line) > 0 else 0
                prev_diff = macd_line[-2] - macd_signal[-2] if len(macd_line) > 1 else 0
                
                # 优化判断逻辑：至少满足3个关键条件
                key_conditions = 0
                signal_direction = None
                
                # 条件1：成交量放大确认
                if vol_ratio > self.crypto_params['vol_mult']:
                    key_conditions += 1
                
                # 条件2：形态确认或RSI确认
                if morph_conf > 0 or rsi_factor > 0.5:
                    key_conditions += 1
                
                # 条件3：MACD交叉或方向确认（降低阈值）
                if cross_strength > 0:
                    key_conditions += 1
                    if macd_diff > 0 and prev_diff <= 0:  # 金叉
                        signal_direction = 'BULL_REVERSAL'
                    elif macd_diff < 0 and prev_diff >= 0:  # 死叉  
                        signal_direction = 'BEAR_REVERSAL'
                elif abs(macd_diff) > 0.001:  # MACD方向明确（降低阈值）
                    key_conditions += 0.5  # 半分
                    if macd_diff > 0:
                        signal_direction = 'BULL_REVERSAL'
                    else:
                        signal_direction = 'BEAR_REVERSAL'
                
                # 条件4：ATR稳定性（降低阈值）
                if atr_factor > 0.1:  # 从0.3降低到0.1
                    key_conditions += 0.5
                
                # 价格趋势确认（新增条件）
                if len(closes) >= 3:
                    recent_trend = (closes[-1] - closes[-3]) / closes[-3]
                    if abs(recent_trend) > 0.01:  # 1%的价格变化
                        key_conditions += 0.5
                        if signal_direction is None:
                            signal_direction = 'BULL_REVERSAL' if recent_trend > 0 else 'BEAR_REVERSAL'
                
                # 至少满足2个条件才触发（降低门槛）
                if key_conditions >= 2.0 and signal_direction:
                    self.logger.info(f"反转检测: {signal_direction}, 置信度: {confidence:.3f}, 满足条件: {key_conditions}")
                    return signal_direction, confidence, details
            
            return None, confidence, details
            
        except Exception as e:
            self.logger.error(f"反转检测失败: {e}")
            return None, 0.0, {'reason': f'检测异常: {e}'}
    
    def _calculate_macd_cross_strength(self, macd_line: np.ndarray, macd_signal: np.ndarray) -> float:
        """计算MACD交叉强度"""
        if len(macd_line) < 2:
            return 0.0
            
        macd_diff = macd_line[-1] - macd_signal[-1]
        prev_diff = macd_line[-2] - macd_signal[-2]
        
        # 检测交叉
        if (macd_diff > 0 and prev_diff <= 0) or (macd_diff < 0 and prev_diff >= 0):
            return abs(macd_diff / macd_line[-1]) if macd_line[-1] != 0 else 0
        
        return 0.0
    
    def _calculate_morph_confidence(self, opens: np.ndarray, highs: np.ndarray, 
                                  lows: np.ndarray, closes: np.ndarray, 
                                  volumes: np.ndarray, rsi: np.ndarray) -> float:
        """计算形态置信度（W底/双顶/吞没）"""
        try:
            # 简化形态识别（可以后续扩展集成现有的support_resistance_analyzer）
            if len(closes) < 10:
                return 0.0
            
            # 检测吞没形态
            engulfing = ta.CDLENGULFING(opens, highs, lows, closes)
            if len(engulfing) > 0 and engulfing[-1] != 0:
                return 0.7  # 吞没形态置信度
            
            # 简化的W底/双顶检测（基于价格模式）
            recent_lows = lows[-10:]
            recent_highs = highs[-10:]
            
            # 检测双底形态迹象
            if len(recent_lows) >= 5:
                min_low = np.min(recent_lows)
                low_indices = np.where(recent_lows <= min_low * 1.02)[0]  # 2%相似度
                if len(low_indices) >= 2:
                    return 0.6  # 双底迹象
            
            # 检测双顶形态迹象  
            if len(recent_highs) >= 5:
                max_high = np.max(recent_highs)
                high_indices = np.where(recent_highs >= max_high * 0.98)[0]  # 2%相似度
                if len(high_indices) >= 2:
                    return 0.6  # 双顶迹象
                    
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"形态识别失败: {e}")
            return 0.0
    
    def _calculate_volume_ratio(self, volumes: np.ndarray) -> float:
        """计算成交量比率"""
        if len(volumes) < 2:
            return 0.0
            
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[:-1])
        
        return current_vol / avg_vol if avg_vol > 0 else 0.0
    
    def _calculate_rsi_factor(self, rsi: np.ndarray, rsi_thresh: float) -> float:
        """计算RSI因子"""
        if len(rsi) == 0:
            return 0.5
            
        current_rsi = rsi[-1]
        
        # 牛市反转：RSI>50，熊市反转：RSI<50
        if current_rsi > rsi_thresh:
            return current_rsi / 100.0  # 牛市因子
        else:
            return (100 - current_rsi) / 100.0  # 熊市因子
    
    def _calculate_atr_factor(self, atr: np.ndarray, atr_mult: float) -> float:
        """计算ATR稳定性因子"""
        if len(atr) < 2:
            return 0.5
            
        current_atr = atr[-1]
        avg_atr = np.mean(atr[:-1])
        
        if avg_atr == 0:
            return 0.5
            
        # ATR越低越稳定，反转可靠性越高
        if current_atr < avg_atr * atr_mult:
            return 1.0 - (current_atr / avg_atr)
        else:
            return 0.0 