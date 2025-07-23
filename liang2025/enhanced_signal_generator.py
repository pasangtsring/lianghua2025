#!/usr/bin/env python3
"""
增强信号生成器 - 融合大佬合理建议和技术实现
采纳：做空功能、置信度过滤、双向交易
修复：代码错误、导入问题、数据结构不匹配
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import logging
from dataclasses import dataclass
from enum import Enum

class MarketCycle(Enum):
    """市场周期"""
    SPRING = "spring"  # 积累期
    SUMMER = "summer"  # 上升期  
    AUTUMN = "autumn"  # 分配期
    WINTER = "winter"  # 下降期

class SignalType(Enum):
    """信号类型"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"

@dataclass
class TradingSignal:
    """交易信号"""
    signal_type: SignalType
    confidence: float
    entry_price: float
    leverage: int
    stop_loss: float
    take_profit: float
    reason: str

class EnhancedSignalGenerator:
    """增强信号生成器（融合大佬合理建议）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 从配置读取参数（修复大佬的配置错误）
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.max_leverage_bull = config.get('max_leverage', 10)
        self.max_leverage_bear = min(config.get('max_leverage', 10), 5)  # 做空杠杆更保守
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.funding_threshold = config.get('funding_threshold', 0.0001)
        
        self.logger.info("增强信号生成器初始化完成")
    
    def detect_market_cycle(self, daily_data: pd.DataFrame) -> MarketCycle:
        """
        检测市场周期（修复大佬的detect_cycle函数）
        采纳大佬的双向交易理念，但用安全的实现
        """
        try:
            if len(daily_data) < 50:
                return MarketCycle.SPRING
            
            # 计算趋势指标
            closes = daily_data['close'].values
            ma_20 = pd.Series(closes).rolling(20).mean().iloc[-1]
            ma_50 = pd.Series(closes).rolling(50).mean().iloc[-1]
            
            current_price = closes[-1]
            
            # 价格趋势
            price_trend = (current_price - closes[-20]) / closes[-20]
            
            # 成交量趋势  
            volumes = daily_data['volume'].values
            volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:])
            
            # 周期判断逻辑（采纳大佬的分类思路）
            if price_trend > 0.1 and current_price > ma_20 > ma_50 and volume_trend > 1.2:
                return MarketCycle.SUMMER  # 强势上升
            elif price_trend > 0 and current_price > ma_50:
                return MarketCycle.SPRING  # 积累阶段
            elif price_trend < -0.1 and current_price < ma_20 < ma_50:
                return MarketCycle.WINTER  # 下降阶段
            else:
                return MarketCycle.AUTUMN  # 分配/调整阶段
                
        except Exception as e:
            self.logger.error(f"市场周期检测失败: {e}")
            return MarketCycle.SPRING
    
    def calculate_macd_confidence(self, data: pd.DataFrame) -> float:
        """
        计算MACD信号置信度（采纳大佬的置信度过滤想法）
        """
        try:
            if len(data) < 34:
                return 0.0
            
            closes = data['close']
            
            # 计算MACD
            ema_12 = closes.ewm(span=12).mean()
            ema_26 = closes.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            # 计算置信度（基于MACD的强度和一致性）
            recent_histogram = histogram.tail(5)
            histogram_strength = abs(recent_histogram.mean())
            histogram_consistency = 1.0 - (recent_histogram.std() / (abs(recent_histogram.mean()) + 0.001))
            
            # 标准化到0-1
            confidence = min(histogram_strength * histogram_consistency * 10, 1.0)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"MACD置信度计算失败: {e}")
            return 0.0
    
    def detect_pattern_confidence(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        检测技术形态置信度（简化大佬的复杂HEAD_SHOULDER检测）
        """
        try:
            if len(data) < 20:
                return "NONE", 0.0
            
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # 简化的形态检测（避免大佬复杂实现的问题）
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            
            # 突破形态检测
            current_high = highs[-1]
            prev_resistance = np.max(recent_highs[:-1])
            
            current_low = lows[-1]
            prev_support = np.min(recent_lows[:-1])
            
            # 突破确认
            if current_high > prev_resistance * 1.02:  # 向上突破
                confidence = min((current_high - prev_resistance) / prev_resistance * 20, 1.0)
                return "BULLISH_BREAKOUT", confidence
                
            elif current_low < prev_support * 0.98:  # 向下突破
                confidence = min((prev_support - current_low) / prev_support * 20, 1.0)
                return "BEARISH_BREAKOUT", confidence
            
            return "NONE", 0.0
            
        except Exception as e:
            self.logger.error(f"形态检测失败: {e}")
            return "NONE", 0.0
    
    def generate_trading_signal(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame], 
                              market_state: Dict) -> Optional[TradingSignal]:
        """
        生成交易信号（融合大佬的双向交易和我的安全实现）
        """
        try:
            # 获取不同时间框架数据
            daily_data = timeframe_data.get('1d', pd.DataFrame())
            hourly_data = timeframe_data.get('1h', pd.DataFrame())
            
            if daily_data.empty or hourly_data.empty:
                return None
            
            # 检测市场周期（采纳大佬理念）
            cycle = self.detect_market_cycle(daily_data)
            
            # 计算置信度（采纳大佬的置信度过滤）
            macd_confidence = self.calculate_macd_confidence(hourly_data)
            pattern_type, pattern_confidence = self.detect_pattern_confidence(hourly_data)
            
            # 综合置信度
            total_confidence = (macd_confidence + pattern_confidence) / 2
            
            # 置信度过滤（采纳大佬建议）
            if total_confidence < self.confidence_threshold:
                self.logger.debug(f"{symbol} 信号置信度不足: {total_confidence:.2f} < {self.confidence_threshold}")
                return None
            
            # 当前价格和风险指标
            current_price = hourly_data['close'].iloc[-1]
            atr = self._calculate_atr(hourly_data, 14)
            funding_rate = market_state.get('funding_rate', 0)
            
            # 信号生成逻辑（融合大佬的双向交易理念）
            signal_type = SignalType.NONE
            leverage = 1
            reason = ""
            
            # 做多条件（牛市周期）
            if cycle in [MarketCycle.SPRING, MarketCycle.SUMMER]:
                if (pattern_type == "BULLISH_BREAKOUT" and 
                    macd_confidence > 0.4 and 
                    funding_rate < self.funding_threshold):
                    
                    signal_type = SignalType.LONG
                    leverage = min(int(total_confidence * self.max_leverage_bull), self.max_leverage_bull)
                    reason = f"牛市{cycle.value}阶段,{pattern_type},MACD置信度{macd_confidence:.2f}"
            
            # 做空条件（熊市周期 - 采纳大佬的做空理念）
            elif cycle in [MarketCycle.AUTUMN, MarketCycle.WINTER]:
                if (pattern_type == "BEARISH_BREAKOUT" and 
                    macd_confidence > 0.4 and 
                    funding_rate > -self.funding_threshold):
                    
                    signal_type = SignalType.SHORT
                    leverage = min(int(total_confidence * self.max_leverage_bear), self.max_leverage_bear)
                    reason = f"熊市{cycle.value}阶段,{pattern_type},MACD置信度{macd_confidence:.2f}"
            
            # 如果没有信号
            if signal_type == SignalType.NONE:
                return None
            
            # 计算止损止盈
            if signal_type == SignalType.LONG:
                stop_loss = current_price - (atr * self.atr_multiplier)
                take_profit = current_price + (atr * self.atr_multiplier * 2)
                entry_price = current_price * 1.001  # 稍微高于市价
            else:  # SHORT
                stop_loss = current_price + (atr * self.atr_multiplier)
                take_profit = current_price - (atr * self.atr_multiplier * 2)
                entry_price = current_price * 0.999  # 稍微低于市价
            
            return TradingSignal(
                signal_type=signal_type,
                confidence=total_confidence,
                entry_price=entry_price,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )
            
        except Exception as e:
            self.logger.error(f"生成交易信号失败: {e}")
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """计算ATR（平均真实波幅）"""
        try:
            if len(data) < period:
                return data['close'].std()
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr if not np.isnan(atr) else data['close'].std()
            
        except Exception as e:
            self.logger.error(f"ATR计算失败: {e}")
            return data['close'].std()

# 使用示例
def create_enhanced_signal_generator(config: Dict) -> EnhancedSignalGenerator:
    """创建增强信号生成器"""
    return EnhancedSignalGenerator(config)

if __name__ == "__main__":
    # 测试配置
    config = {
        'confidence_threshold': 0.5,
        'max_leverage': 10,
        'atr_multiplier': 2.0,
        'funding_threshold': 0.0001
    }
    
    generator = create_enhanced_signal_generator(config)
    print("增强信号生成器创建成功")
    print("✅ 支持双向交易（采纳大佬建议）")
    print("✅ 置信度过滤（采纳大佬建议）")
    print("✅ 安全的代码实现（修复大佬问题）")
    print("✅ 完整的错误处理机制") 