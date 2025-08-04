"""
信号生成器模块
整合所有分析模块，生成交易信号、风险评估、置信度计算
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd

from utils.logger import get_logger
from config.config_manager import ConfigManager
from core.complete_macd_divergence_detector import CompleteMACDDivergenceDetector
from core.technical_indicators import TechnicalIndicatorCalculator
from core.pattern_recognizer import PatternRecognizer
from core.cycle_analyzer import CycleAnalyzer, CyclePhase
from core.enhanced_pattern_detector import EnhancedPatternDetector, PatternType, DivergenceType

class SignalType(Enum):
    """信号类型枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalStrength(Enum):
    """信号强度枚举"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TradingSignal:
    """交易信号数据类"""
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    timestamp: datetime
    reasons: List[str]
    technical_score: float
    market_condition: str
    technical_indicators: Dict = None  # 添加技术指标字段
    
class SignalGenerator:
    """信号生成器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 初始化各种分析器
        self.macd_detector = CompleteMACDDivergenceDetector()  # 使用默认配置
        self.technical_indicators = TechnicalIndicatorCalculator(config_manager)
        self.pattern_recognizer = PatternRecognizer(config_manager)
        self.cycle_analyzer = CycleAnalyzer(config_manager)
        
        # 添加增强形态检测器
        self.enhanced_pattern_detector = EnhancedPatternDetector(config_manager)
        
        # 获取配置
        self.trading_config = config_manager.get_trading_config()
        self.signal_config = config_manager.get_signal_config()
        
        # 信号配置
        self.min_confidence = 0.45  # 降低至45%，增加信号生成机会
        self.risk_reward_min = 0.8  # 大幅降低风险回报比要求，确保信号生成
        
        # 权重配置
        self.weights = {
            'macd_divergence': 0.4,
            'technical_indicators': 0.3,
            'pattern_recognition': 0.2,
            'cycle_analysis': 0.1
        }
        
        # 统计信息
        self.signal_stats = {
            'total_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'enhanced_pattern_usage': 0
        }
        
        self.logger.info("信号生成器初始化完成（集成增强形态检测器）")
    
    def generate_signal(self, kline_data: List[Dict]) -> Optional[TradingSignal]:
        """
        生成交易信号 - 集成增强形态检测
        
        Args:
            kline_data: K线数据
            
        Returns:
            交易信号或None
        """
        try:
            if len(kline_data) < 100:
                self.logger.warning("K线数据不足，无法生成信号")
                return None
            
            # 获取当前价格
            current_price = float(kline_data[-1]['close'])
            
            # 1. 增强MACD背离分析
            macd_result = self.analyze_macd_divergence(kline_data)
            
            # 2. 技术指标分析
            technical_result = self.analyze_technical_indicators(kline_data)
            
            # 3. 增强形态识别分析
            pattern_result = self.analyze_patterns(kline_data)
            
            # 4. 周期分析
            cycle_result = self.analyze_cycle(kline_data)
            
            # 5. 综合市场结构分析
            market_structure = self.analyze_market_structure_comprehensive(kline_data)
            
            # 6. 综合评分（增强版）
            composite_score = self.calculate_composite_score_enhanced(
                macd_result, technical_result, pattern_result, cycle_result, market_structure
            )
            
            # 7. 生成信号
            signal = self.create_signal_enhanced(
                composite_score, current_price, 
                macd_result, technical_result, pattern_result, cycle_result, market_structure
            )
            
            if signal:
                # 更新统计
                self.signal_stats['total_generated'] += 1
                if signal.signal_type == SignalType.BUY:
                    self.signal_stats['buy_signals'] += 1
                elif signal.signal_type == SignalType.SELL:
                    self.signal_stats['sell_signals'] += 1
                else:
                    self.signal_stats['hold_signals'] += 1
                
                # 更新平均置信度
                total_signals = self.signal_stats['total_generated']
                self.signal_stats['avg_confidence'] = (
                    (self.signal_stats['avg_confidence'] * (total_signals - 1) + signal.confidence) / total_signals
                )
                
                self.logger.info(f"生成增强交易信号: {signal.signal_type.value}, 置信度: {signal.confidence:.2f}")
                self.logger.info(f"✅ 成功生成{signal.signal_type.value}信号: "
                               f"价格{signal.entry_price}, 止损{signal.stop_loss_price:.2f}, "
                               f"止盈{signal.take_profit_price:.2f}, RR比{signal.risk_reward_ratio:.2f}")
            else:
                self.logger.info("信号生成失败，跳过过滤")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成增强信号失败: {e}")
            return None
    
    def generate_signals(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> List[Dict]:
        """批量生成信号的简化接口"""
        try:
            # 转换DataFrame为kline_data格式
            kline_data = []
            for i, row in df.iterrows():
                kline = {
                    'open_time': i if isinstance(i, int) else int(row.get('open_time', 0)),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                kline_data.append(kline)
            
            # 生成信号
            signal = self.generate_signal(kline_data)
            
            # 包装为列表格式
            signals = []
            if signal:
                signals.append({
                    'symbol': symbol,
                    'side': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'strength': signal.signal_strength.value,
                    'price': signal.entry_price,
                    'timestamp': signal.timestamp.isoformat(),
                    'reasons': signal.reasons
                })
                
                self.logger.info(f"✅ 成功生成{signal.signal_type.value}信号: "
                               f"价格{signal.entry_price}, 止损{signal.stop_loss_price:.2f}, "
                               f"止盈{signal.take_profit_price:.2f}, RR比{signal.risk_reward_ratio:.2f}")
            else:
                self.logger.info("信号生成失败，跳过过滤")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"批量生成信号失败: {e}")
            return []
    
    def analyze_macd_divergence(self, kline_data: List[Dict]) -> Dict:
        """
        分析MACD背离 - 使用增强背离检测
        """
        try:
            # 准备价格数据
            highs = np.array([float(k['high']) for k in kline_data])
            lows = np.array([float(k['low']) for k in kline_data])
            closes = np.array([float(k['close']) for k in kline_data])
            
            # 计算波动性因子
            vol_factor = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0
            
            # 使用增强背离检测器
            divergence_signals = self.enhanced_pattern_detector.detect_divergence(
                highs, lows, closes, vol_factor
            )
            
            if divergence_signals:
                # 取最高置信度的背离信号
                best_divergence = max(divergence_signals, key=lambda d: d.confidence)
                
                # 确定信号类型
                signal_type = 'bullish' if best_divergence.type == DivergenceType.BULLISH else 'bearish'
                
                return {
                    'has_divergence': True,
                    'divergence_type': signal_type,
                    'confidence': best_divergence.confidence,
                    'strength': best_divergence.strength,
                    'score': best_divergence.confidence * 100,
                    'indices': best_divergence.indices,
                    'macd_values': best_divergence.macd_values,
                    'price_values': best_divergence.price_values,
                    'enhanced_detection': True
                }
            else:
                # 回退到原有的MACD检测器
                prices = [float(k['close']) for k in kline_data]
                macd_data = self.technical_indicators.calculate_macd(prices)
                
                if macd_data:
                    # 简化的背离检测逻辑
                    # 没有背离时，根据MACD趋势给分
                    macd_line = macd_data[-1].macd_line if macd_data else 0
                    signal_line = macd_data[-1].signal_line if macd_data else 0
                    
                    if macd_line < signal_line and macd_line < 0:
                        macd_trend_score = 30  # 看跌趋势
                    elif macd_line > signal_line and macd_line > 0:
                        macd_trend_score = 70  # 看涨趋势
                    else:
                        macd_trend_score = 50  # 中性
                    
                    return {
                        'has_divergence': False,
                        'divergence_type': 'none',
                        'confidence': 0.0,
                        'strength': 0.0,
                        'score': macd_trend_score,
                        'enhanced_detection': False
                    }
                else:
                    return {
                        'has_divergence': False,
                        'divergence_type': 'none', 
                        'confidence': 0.0,
                        'strength': 0.0,
                        'score': 50,
                        'enhanced_detection': False
                    }
                
        except Exception as e:
            self.logger.error(f"增强MACD背离分析失败: {e}")
            return {
                'has_divergence': False,
                'divergence_type': 'none',
                'confidence': 0.0,
                'strength': 0.0,
                'score': 50,
                'enhanced_detection': False,
                'error': str(e)
            }
    
    def analyze_technical_indicators(self, kline_data: List[Dict]) -> Dict:
        """分析技术指标"""
        try:
            # 提取价格和成交量数据
            prices = [float(k['close']) for k in kline_data]
            volumes = [float(k['volume']) for k in kline_data]
            
            # 计算技术指标
            macd_data = self.technical_indicators.calculate_macd(prices)
            rsi_data = self.technical_indicators.calculate_rsi(prices)
            sma_data = self.technical_indicators.calculate_sma(prices, 20)
            
            # 获取最新值
            current_price = prices[-1]
            current_rsi = rsi_data[-1].rsi_value if rsi_data else 50
            current_sma = sma_data[-1] if sma_data else current_price
            
            # 技术指标评分
            score = 0.0
            signals = []
            
            # RSI评分 - 考虑趋势方向
            # 判断趋势：价格与均线关系
            is_downtrend = current_price < current_sma
            
            if current_rsi < 30:
                if is_downtrend:
                    # 下跌趋势中RSI超卖是做空确认信号
                    score -= 30  # 做空信号
                    signals.append("RSI超卖(下跌趋势-做空信号)")
                else:
                    # 上涨趋势中RSI超卖是反弹机会
                    score += 30  # 做多信号
                    signals.append("RSI超卖(上涨趋势-反弹机会)")
            elif current_rsi > 70:
                if is_downtrend:
                    # 下跌趋势中RSI超买是反弹结束信号
                    score -= 30  # 做空信号
                    signals.append("RSI超买(下跌趋势-反弹结束)")
                else:
                    # 上涨趋势中RSI超买是做多警告
                    score -= 30  # 减分
                    signals.append("RSI超买(上涨趋势-警告)")
            
            # 价格相对均线
            if current_price > current_sma:
                score += 20  # 价格在均线上方
                signals.append("价格在均线上方")
            else:
                score -= 20  # 价格在均线下方
                signals.append("价格在均线下方")
            
            # MACD趋势
            if macd_data:
                macd_line = macd_data[-1].macd_line
                signal_line = macd_data[-1].signal_line
                if macd_line > signal_line:
                    score += 15
                    signals.append("MACD上穿信号线")
                else:
                    score -= 15
                    signals.append("MACD下穿信号线")
            
            
            # 趋势强度评分
            price_vs_sma = (current_price - current_sma) / current_sma * 100
            if abs(price_vs_sma) > 2:  # 价格偏离均线超过2%
                if price_vs_sma < 0:  # 强烈下跌趋势
                    score -= 20
                    signals.append(f"强烈下跌趋势({price_vs_sma:.1f}%)")
                else:  # 强烈上涨趋势
                    score += 20
                    signals.append(f"强烈上涨趋势({price_vs_sma:.1f}%)")
            
            
            # 成交量确认
            if len(volumes) >= 20:
                current_volume = volumes[-1]
                avg_volume = sum(volumes[-20:]) / 20
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 1.5 and price_vs_sma < -1:  # 放量下跌
                    score -= 15
                    signals.append(f"放量下跌(成交量{volume_ratio:.1f}倍)")
                elif volume_ratio > 1.5 and price_vs_sma > 1:  # 放量上涨
                    score += 15
                    signals.append(f"放量上涨(成交量{volume_ratio:.1f}倍)")
            
            # 归一化评分
            normalized_score = max(0, min(100, score + 50))
            
            return {
                'score': normalized_score,
                'signals': signals,
                'rsi': current_rsi,
                'sma': current_sma,
                'trend': 'bullish' if score > 0 else 'bearish'
            }
            
        except Exception as e:
            self.logger.error(f"技术指标分析失败: {e}")
            return {
                'score': 50,
                'signals': [],
                'rsi': 50,
                'sma': 0,
                'trend': 'neutral'
            }
    
    def analyze_patterns(self, kline_data: List[Dict]) -> Dict:
        """
        分析形态 - 使用增强形态检测器
        """
        try:
            # 准备OHLC数据
            opens = np.array([float(k['open']) for k in kline_data])
            highs = np.array([float(k['high']) for k in kline_data])
            lows = np.array([float(k['low']) for k in kline_data])
            closes = np.array([float(k['close']) for k in kline_data])
            
            # 使用增强形态检测器
            pattern_signals = self.enhanced_pattern_detector.detect_pattern(opens, highs, lows, closes)
            
            if pattern_signals:
                # 取最高置信度的形态
                best_pattern = max(pattern_signals, key=lambda p: p.confidence)
                
                # 统计增强形态检测器的使用
                self.signal_stats['enhanced_pattern_usage'] += 1
                
                # 确定形态类型
                pattern_type = 'bullish' if 'BULL' in best_pattern.type.value else 'bearish'
                
                return {
                    'has_pattern': True,
                    'pattern_name': best_pattern.type.value,
                    'confidence': best_pattern.confidence,
                    'pattern_type': pattern_type,
                    'score': best_pattern.confidence * 100,
                    'details': best_pattern.details,
                    'enhanced_detection': True
                }
            else:
                # 回退到原有的形态识别器
                patterns = self.pattern_recognizer.recognize_patterns(kline_data)
                
                if patterns:
                    best_pattern = max(patterns, key=lambda p: p.confidence)
                    
                    return {
                        'has_pattern': True,
                        'pattern_name': best_pattern.name,
                        'confidence': best_pattern.confidence,
                        'pattern_type': best_pattern.pattern_type,
                        'score': best_pattern.confidence * 100,
                        'enhanced_detection': False
                    }
                else:
                    return {
                        'has_pattern': False,
                        'pattern_name': 'none',
                        'confidence': 0.0,
                        'pattern_type': 'neutral',
                        'score': 50,
                        'enhanced_detection': False
                    }
                
        except Exception as e:
            self.logger.error(f"增强形态分析失败: {e}")
            return {
                'has_pattern': False,
                'pattern_name': 'none',
                'confidence': 0.0,
                'pattern_type': 'neutral',
                'score': 50,
                'enhanced_detection': False,
                'error': str(e)
            }
    
    def analyze_cycle(self, kline_data: List[Dict]) -> Dict:
        """分析周期"""
        try:
            cycle_analysis = self.cycle_analyzer.analyze_cycle(kline_data)
            
            # 根据周期阶段评分
            phase_scores = {
                CyclePhase.SPRING: 70,  # 积累期 - 适合做多
                CyclePhase.SUMMER: 85,  # 上升期 - 适合做多
                CyclePhase.AUTUMN: 70,  # 分配期 - 适合做空（评分提升）
                CyclePhase.WINTER: 85   # 衰退期 - 适合做空（评分提升）
            }
            
            base_score = phase_scores.get(cycle_analysis.current_phase, 50)
            confidence_adjusted_score = base_score * cycle_analysis.phase_confidence
            
            return {
                'phase': cycle_analysis.current_phase.value,
                'phase_confidence': cycle_analysis.phase_confidence,
                'trend_strength': cycle_analysis.trend_strength,
                'score': confidence_adjusted_score,
                'bias': 'bullish' if confidence_adjusted_score > 50 else 'bearish'
            }
            
        except Exception as e:
            self.logger.error(f"周期分析失败: {e}")
            return {
                'phase': 'spring',
                'phase_confidence': 0.0,
                'trend_strength': 0.0,
                'score': 50,
                'bias': 'neutral'
            }
    
    def calculate_composite_score(self, macd_result: Dict, technical_result: Dict, 
                                pattern_result: Dict, cycle_result: Dict) -> Dict:
        """计算综合评分"""
        try:
            # 加权计算总分
            total_score = (
                macd_result['score'] * self.weights['macd_divergence'] +
                technical_result['score'] * self.weights['technical_indicators'] +
                pattern_result['score'] * self.weights['pattern_recognition'] +
                cycle_result['score'] * self.weights['cycle_analysis']
            )
            
            # 计算置信度
            confidence = total_score / 100.0
            
            # 多空平衡阈值设计
            if total_score > 50:      # 做多阈值降低至50分，增加机会
                signal_type = SignalType.BUY
                signal_strength = SignalStrength.STRONG if total_score > 80 else SignalStrength.MEDIUM
            elif total_score < 45:    # 做空阈值提高至45分，更对称（与做多对称）
                signal_type = SignalType.SELL
                signal_strength = SignalStrength.STRONG if total_score < 20 else SignalStrength.MEDIUM
            else:
                signal_type = SignalType.HOLD
                signal_strength = SignalStrength.WEAK
            
            # 收集原因
            reasons = []
            if macd_result['has_signal']:
                reasons.append(f"MACD{macd_result['signal_type']}信号")
            
            reasons.extend(technical_result['signals'])
            
            if pattern_result['has_pattern']:
                reasons.append(f"{pattern_result['pattern_name']}形态")
            
            reasons.append(f"市场周期: {cycle_result['phase']}")
            
            return {
                'total_score': total_score,
                'confidence': confidence,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'reasons': reasons,
                'components': {
                    'macd': macd_result['score'],
                    'technical': technical_result['score'],
                    'pattern': pattern_result['score'],
                    'cycle': cycle_result['score']
                }
            }
            
        except Exception as e:
            self.logger.error(f"计算综合评分失败: {e}")
            return {
                'total_score': 50,
                'confidence': 0.5,
                'signal_type': SignalType.HOLD,
                'signal_strength': SignalStrength.WEAK,
                'reasons': ['计算失败'],
                'components': {}
            }
    
    def create_signal(self, composite_score: Dict, current_price: float,
                     macd_result: Dict, technical_result: Dict, 
                     pattern_result: Dict, cycle_result: Dict) -> Optional[TradingSignal]:
        """创建交易信号"""
        try:
            # 检查最小置信度
            if composite_score['confidence'] < self.min_confidence:
                return None
            
            # 检查是否为持有信号
            if composite_score['signal_type'] == SignalType.HOLD:
                return None
            
            # 计算止损止盈
            stop_loss_price, take_profit_price = self.calculate_stop_loss_take_profit(
                composite_score['signal_type'], current_price, composite_score['confidence']
            )
            
            # 计算风险回报比
            risk_reward_ratio = self.calculate_risk_reward_ratio(
                composite_score['signal_type'], current_price, stop_loss_price, take_profit_price
            )
            
            # 检查风险回报比
            self.logger.info(f"📊 风险回报比计算: {risk_reward_ratio:.2f}, 要求: {self.risk_reward_min}")
            if risk_reward_ratio < self.risk_reward_min:
                self.logger.warning(f"⚠️ 风险回报比不足: {risk_reward_ratio:.2f} < {self.risk_reward_min}")
                # 对于强信号，放宽风险回报比要求
                if composite_score['confidence'] >= 0.7 and risk_reward_ratio >= 0.5:
                    self.logger.info(f"✅ 强信号(置信度{composite_score['confidence']:.2f})，放宽风险回报比要求")
                else:
                    return None
            
            # 确定市场状态
            market_condition = self.determine_market_condition(cycle_result, technical_result)
            
            # 创建信号
            signal = TradingSignal(
                signal_type=composite_score['signal_type'],
                signal_strength=composite_score['signal_strength'],
                confidence=composite_score['confidence'],
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_reward_ratio=risk_reward_ratio,
                timestamp=datetime.now(),
                reasons=composite_score['reasons'],
                technical_score=composite_score['total_score'],
                market_condition=market_condition,
                technical_indicators=technical_result  # 添加技术指标
            )
            
            self.logger.info(f"✅ 成功生成{signal.signal_type.value}信号: "
                           f"价格{signal.entry_price}, 止损{signal.stop_loss_price:.2f}, "
                           f"止盈{signal.take_profit_price:.2f}, RR比{signal.risk_reward_ratio:.2f}")
            return signal
            
        except Exception as e:
            self.logger.error(f"创建信号失败: {e}")
            return None
    
    def calculate_stop_loss_take_profit(self, signal_type: SignalType, 
                                      current_price: float, confidence: float) -> Tuple[float, float]:
        """计算止损止盈"""
        try:
            # 基础止损止盈比例
            base_stop_loss = self.trading_config.get('stop_loss_pct', 0.02)
            base_take_profit = self.trading_config.get('take_profit_pct', 0.04)
            
            # 根据置信度调整
            confidence_multiplier = 0.5 + (confidence * 0.5)
            adjusted_stop_loss = base_stop_loss * confidence_multiplier
            adjusted_take_profit = base_take_profit * confidence_multiplier
            
            if signal_type == SignalType.BUY:
                stop_loss_price = current_price * (1 - adjusted_stop_loss)
                take_profit_price = current_price * (1 + adjusted_take_profit)
            else:  # SELL
                stop_loss_price = current_price * (1 + adjusted_stop_loss)
                take_profit_price = current_price * (1 - adjusted_take_profit)
            
            return stop_loss_price, take_profit_price
            
        except Exception as e:
            self.logger.error(f"计算止损止盈失败: {e}")
            return current_price, current_price
    
    def calculate_risk_reward_ratio(self, signal_type: SignalType, entry_price: float,
                                  stop_loss_price: float, take_profit_price: float) -> float:
        """计算风险回报比"""
        try:
            self.logger.debug(f"📊 RR计算: 类型={signal_type.value}, 入场={entry_price}, "
                            f"止损={stop_loss_price}, 止盈={take_profit_price}")
            if signal_type == SignalType.BUY:
                risk = entry_price - stop_loss_price
                reward = take_profit_price - entry_price
            else:  # SELL
                risk = stop_loss_price - entry_price
                reward = entry_price - take_profit_price
            
            if risk > 0:
                return reward / risk
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"计算风险回报比失败: {e}")
            return 0.0
    
    def determine_market_condition(self, cycle_result: Dict, technical_result: Dict) -> str:
        """确定市场状态"""
        try:
            cycle_phase = cycle_result['phase']
            technical_trend = technical_result['trend']
            
            if cycle_phase in ['spring', 'summer'] and technical_trend == 'bullish':
                return 'strong_bull'
            elif cycle_phase in ['autumn', 'winter'] and technical_trend == 'bearish':
                return 'strong_bear'
            elif cycle_phase in ['spring', 'summer']:
                return 'bull_market'
            elif cycle_phase in ['autumn', 'winter']:
                return 'bear_market'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"确定市场状态失败: {e}")
            return 'unknown'
    
    def get_signal_summary(self, signal: TradingSignal) -> Dict:
        """获取信号摘要"""
        try:
            return {
                'signal_type': signal.signal_type.value,
                'signal_strength': signal.signal_strength.value,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss_price': signal.stop_loss_price,
                'take_profit_price': signal.take_profit_price,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'technical_score': signal.technical_score,
                'market_condition': signal.market_condition,
                'reasons': signal.reasons,
                'timestamp': signal.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取信号摘要失败: {e}")
            return {} 

# 在SignalGenerator类末尾添加增强的置信度过滤机制

import numpy as np
from typing import Set

class EnhancedSignalFilter:
    """
    增强信号过滤器 - 实现专家建议的置信度过滤和信号质量提升
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 获取置信度配置
        self.risk_config = self.config.get_risk_config()
        self.signal_config = self.config.get_signal_config()
        
        # 置信度阈值
        self.min_confidence = 0.4  # 适度降低，保持质量
        self.high_confidence = 0.6
        self.very_high_confidence = 0.7
        
        # 多重过滤条件 - 智能调整
        self.enable_strength_filter = True
        self.enable_consistency_filter = True  # 重新启用，但优化逻辑
        self.enable_market_condition_filter = True
        self.enable_volatility_filter = True  # 重新启用，但优化逻辑
        
        # 信号历史（用于一致性检查）
        self.signal_history: List[Dict] = []
        self.max_history_size = 20
        
        # 过滤统计
        self.filter_stats = {
            'total_signals': 0,
            'passed_confidence': 0,
            'passed_strength': 0,
            'passed_consistency': 0,
            'passed_market_condition': 0,
            'passed_volatility': 0,
            'final_passed': 0,
            'filter_rates': {}
        }
        
        self.logger.info("增强信号过滤器初始化完成")
    
    def filter_signal(self, signal: TradingSignal, market_data: Dict) -> Optional[TradingSignal]:
        """
        过滤信号 - 多重过滤条件
        
        Args:
            signal: 原始信号
            market_data: 市场数据
            
        Returns:
            过滤后的信号或None
        """
        try:
            self.filter_stats['total_signals'] += 1
            
            # 详细记录原始信号信息
            self.logger.info(f"🔍 开始过滤信号: 类型={signal.signal_type.value}, "
                           f"置信度={signal.confidence:.3f}, 强度={signal.signal_strength.value}, "
                           f"风险回报比={signal.risk_reward_ratio:.2f}")
            
            # 1. 基础置信度过滤
            if not self.pass_confidence_filter(signal):
                self.logger.debug(f"信号被置信度过滤: {signal.confidence:.3f} < {self.min_confidence}")
                return None
            self.filter_stats['passed_confidence'] += 1
            
            # 2. 信号强度过滤
            if self.enable_strength_filter and not self.pass_strength_filter(signal):
                self.logger.debug(f"信号被强度过滤: {signal.signal_strength.value}")
                return None
            self.filter_stats['passed_strength'] += 1
            
            # 3. 一致性过滤
            if self.enable_consistency_filter and not self.pass_consistency_filter(signal):
                self.logger.debug(f"信号被一致性过滤")
                return None
            self.filter_stats['passed_consistency'] += 1
            
            # 4. 市场条件过滤
            if self.enable_market_condition_filter:
                market_passed = self.pass_market_condition_filter(signal, market_data)
                self.logger.info(f"🌍 市场条件过滤检查: {'通过' if market_passed else '未通过'}")
                if not market_passed:
                    self.logger.info(f"❌ 信号被市场条件过滤")
                    return None
                self.logger.info(f"✅ 通过市场条件过滤")
            self.filter_stats['passed_market_condition'] += 1
            
            # 5. 波动性过滤
            if self.enable_volatility_filter:
                volatility_passed = self.pass_volatility_filter(signal, market_data)
                self.logger.info(f"📊 波动性过滤检查: {'通过' if volatility_passed else '未通过'}")
                if not volatility_passed:
                    self.logger.info(f"❌ 信号被波动性过滤")
                    return None
                self.logger.info(f"✅ 通过波动性过滤")
            self.filter_stats['passed_volatility'] += 1
            
            # 6. 最终质量检查
            self.logger.info(f"🔧 开始信号质量增强...")
            enhanced_signal = self.enhance_signal_quality(signal, market_data)
            
            if enhanced_signal:
                self.logger.info(f"✅ 信号质量增强成功")
                self.filter_stats['final_passed'] += 1
                self.add_to_signal_history(enhanced_signal)
                self.logger.info(f"✅ 信号通过所有过滤器: 类型={enhanced_signal.signal_type.value}, 置信度={enhanced_signal.confidence:.3f}")
                return enhanced_signal
            else:
                self.logger.info(f"❌ 信号在最终质量增强步骤被过滤")
            
            return None
            
        except Exception as e:
            self.logger.error(f"信号过滤失败: {e}")
            return None
    
    def pass_confidence_filter(self, signal: TradingSignal) -> bool:
        """
        置信度过滤 - 核心过滤条件
        
        Args:
            signal: 交易信号
            
        Returns:
            是否通过置信度过滤
        """
        try:
            # 基础置信度检查
            if signal.confidence < self.min_confidence:
                return False
            
            # 多空平衡的置信度要求
            if signal.signal_type == SignalType.BUY:
                # 看涨信号置信度要求
                required_confidence = self.min_confidence + 0.03
            elif signal.signal_type == SignalType.SELL:
                # 看跌信号置信度要求（与看涨对称）
                required_confidence = self.min_confidence + 0.03
            else:
                required_confidence = self.min_confidence
            
            self.logger.info(f"✅ 成功生成{signal.signal_type.value}信号: "
                           f"价格{signal.entry_price}, 止损{signal.stop_loss_price:.2f}, "
                           f"止盈{signal.take_profit_price:.2f}, RR比{signal.risk_reward_ratio:.2f}")
            return signal.confidence >= required_confidence
            
        except Exception as e:
            self.logger.error(f"置信度过滤检查失败: {e}")
            return False
    
    def pass_strength_filter(self, signal: TradingSignal) -> bool:
        """
        信号强度过滤 - 更灵活
        
        Args:
            signal: 交易信号
            
        Returns:
            是否通过强度过滤
        """
        try:
            # 接受中等及以上强度的信号
            acceptable_strengths = {
                SignalStrength.MEDIUM,
                SignalStrength.STRONG,
                SignalStrength.VERY_STRONG,
                SignalStrength.MODERATE
            }
            
            # 更灵活的弱信号处理
            if signal.signal_strength == SignalStrength.WEAK:
                # 置信度超过0.5就接受
                if signal.confidence >= 0.5:
                    self.logger.info(f"置信度({signal.confidence:.2f})弱信号，允许通过")
                    return True
                # 或者风险回报比很好（>2.0）也接受
                if hasattr(signal, 'risk_reward_ratio') and signal.risk_reward_ratio >= 2.0:
                    self.logger.info(f"高风险回报比({signal.risk_reward_ratio:.2f})弱信号，允许通过")
                    return True
            
            return signal.signal_strength in acceptable_strengths
            
        except Exception as e:
            self.logger.error(f"强度过滤检查失败: {e}")
            return False
    
    def pass_consistency_filter(self, signal: TradingSignal) -> bool:
        """
        检查信号与技术指标的一致性
        
        修复版本：正确处理SELL信号的评分逻辑
        """
        if not self.enable_consistency_filter:
            return True
        
        # 不需要在这里增加统计，filter_signal函数会处理
        
        # 获取技术指标
        indicators = getattr(signal, 'technical_indicators', {})
        if not indicators:
            self.logger.warning("信号缺少技术指标，跳过一致性检查")
            return True
        
        consistency_score = 0
        total_checks = 0
        
        try:
            # 1. 价格与SMA的关系 - 修复SELL信号的评分逻辑
            if 'sma20' in indicators and 'current_price' in indicators:
                total_checks += 1
                price = indicators['current_price']
                sma20 = indicators['sma20']
                
                if signal.signal_type == SignalType.BUY:
                    # BUY信号：价格在均线上方得高分
                    if price > sma20:
                        consistency_score += 1.0
                    else:
                        consistency_score += 0.3
                else:  # SELL信号
                    # SELL信号：价格在均线下方得高分
                    if price < sma20:
                        consistency_score += 1.0
                    else:
                        consistency_score += 0.3
            
            # 2. RSI指标 - 修复SELL信号的评分逻辑
            if 'rsi' in indicators:
                total_checks += 1
                rsi = indicators['rsi']
                
                if signal.signal_type == SignalType.BUY:
                    # BUY信号：RSI超卖得高分
                    if rsi < 30:
                        consistency_score += 1.0
                    elif rsi < 40:
                        consistency_score += 0.7
                    else:
                        consistency_score += 0.3
                else:  # SELL信号
                    # SELL信号：RSI超买得高分，超卖也可接受（趋势延续）
                    if rsi > 70:
                        consistency_score += 1.0
                    elif rsi < 30:
                        # 超卖时做空也是合理的（趋势延续）
                        consistency_score += 0.8
                    elif rsi > 60:
                        consistency_score += 0.7
                    else:
                        consistency_score += 0.5
            
            # 3. MACD指标
            if 'macd' in indicators and 'macd_signal' in indicators:
                total_checks += 1
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                
                if signal.signal_type == SignalType.BUY:
                    # BUY信号：MACD在信号线上方得高分
                    if macd > macd_signal:
                        consistency_score += 1.0
                    else:
                        consistency_score += 0.3
                else:  # SELL信号
                    # SELL信号：MACD在信号线下方得高分
                    if macd < macd_signal:
                        consistency_score += 1.0
                    else:
                        consistency_score += 0.3
            
            # 计算平均一致性分数
            if total_checks > 0:
                avg_consistency = consistency_score / total_checks
            else:
                avg_consistency = 0.5  # 没有指标时给中性分数
            
            # 根据信号置信度动态调整要求
            required_consistency = 0.5  # 基础要求
            
            if signal.confidence >= 0.8:
                required_consistency = 0.4  # 高置信度信号降低一致性要求
            elif signal.confidence >= 0.6:
                required_consistency = 0.45
            elif signal.confidence >= 0.5:
                required_consistency = 0.5
            else:
                required_consistency = 0.55
            
            # 对于风险回报比高的信号，降低一致性要求
            if signal.risk_reward_ratio >= 3.0:
                required_consistency -= 0.1
            elif signal.risk_reward_ratio >= 2.0:
                required_consistency -= 0.05
            
            passed = avg_consistency >= required_consistency
            
            if passed:
                # 统计在filter_signal函数中处理
                self.logger.info(f"✅ 一致性检查通过: {avg_consistency:.2f} >= {required_consistency:.2f}")
            else:
                self.logger.info(f"❌ 一致性检查未通过: {avg_consistency:.2f} < {required_consistency:.2f}")
            
            return passed
            
        except Exception as e:
            self.logger.error(f"一致性过滤检查失败: {str(e)}")
            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"波动性过滤检查失败: {e}")
            return True  # 出错时默认通过

    def pass_market_condition_filter(self, signal: TradingSignal, market_data: Dict) -> bool:
        """
        市场条件过滤器 - 优化版本，更加智能和宽松
        
        Args:
            signal: 交易信号
            market_data: 市场数据
            
        Returns:
            是否通过市场条件过滤
        """
        try:
            # 详细调试信息
            self.logger.info(f"🔍 市场条件过滤器调试:")
            self.logger.info(f"   📊 market_data类型: {type(market_data)}")
            self.logger.info(f"   📊 market_data键: {list(market_data.keys()) if market_data else 'None'}")
            self.logger.info(f"   🎯 信号类型: {signal.signal_type}")
            self.logger.info(f"   🎯 信号置信度: {signal.confidence:.3f}")
            
            # 基础检查：如果没有市场数据，默认通过
            if not market_data:
                self.logger.info(f"   ✅ 无市场数据，默认通过")
                return True
            
            # 检查市场趋势与信号方向的一致性
            if 'trend' in market_data:
                market_trend = market_data['trend']
                self.logger.info(f"   📈 市场趋势: {market_trend}")
                
                # 熊市中放宽做空信号
                if market_trend == 'BEARISH' and signal.signal_type == SignalType.SELL:
                    self.logger.info(f"   ✅ 熊市做空信号，通过")
                    return True
                
                # 牛市中放宽做多信号
                if market_trend == 'BULLISH' and signal.signal_type == SignalType.BUY:
                    self.logger.info(f"   ✅ 牛市做多信号，通过")
                    return True
            else:
                self.logger.info(f"   ⚠️ 无趋势数据，继续其他检查")
            
            # 优化波动性检查 - 更加宽松
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                self.logger.info(f"   📊 波动性: {volatility:.4f}")
                
                # 只有极高波动才需要更高置信度
                if volatility > 0.08:  # 从0.05提高到0.08
                    required_confidence = 0.55  # 从0.6降低到0.55
                    passed = signal.confidence >= required_confidence
                    self.logger.info(f"   {'✅' if passed else '❌'} 极高波动({volatility:.4f} > 0.08): 置信度{signal.confidence:.3f} {'≥' if passed else '<'} {required_confidence}")
                    return passed
                
                # 极低波动要求降低
                if volatility < 0.005:  # 从0.01降低到0.005
                    required_confidence = 0.55  # 从0.7降低到0.55
                    passed = signal.confidence >= required_confidence
                    self.logger.info(f"   {'✅' if passed else '❌'} 极低波动({volatility:.4f} < 0.005): 置信度{signal.confidence:.3f} {'≥' if passed else '<'} {required_confidence}")
                    return passed
                else:
                    self.logger.info(f"   ✅ 正常波动({volatility:.4f})，通过")
                    return True
            else:
                self.logger.info(f"   ⚠️ 无波动性数据，继续其他检查")
            
            # 优化成交量检查 - 更加宽松
            if 'volume_ratio' in market_data:
                volume_ratio = market_data['volume_ratio']
                self.logger.info(f"   📊 成交量比率: {volume_ratio:.2f}")
                
                # 只有极端成交量异常才需要更高置信度
                if volume_ratio < 0.1 or volume_ratio > 5.0:  # 从0.5-2.0放宽到0.1-5.0
                    required_confidence = 0.55  # 从0.6降低到0.55
                    passed = signal.confidence >= required_confidence
                    self.logger.info(f"   {'✅' if passed else '❌'} 极端成交量异常({volume_ratio:.2f}): 置信度{signal.confidence:.3f} {'≥' if passed else '<'} {required_confidence}")
                    return passed
                else:
                    self.logger.info(f"   ✅ 成交量正常({volume_ratio:.2f})，通过")
                    return True
            else:
                self.logger.info(f"   ⚠️ 无成交量数据，继续其他检查")
            
            # 默认通过 - 如果没有明确的拒绝理由，就允许通过
            self.logger.info(f"   ✅ 所有条件检查完毕，默认通过")
            return True
            
        except Exception as e:
            self.logger.error(f"市场条件过滤检查失败: {e}")
            return True  # 出错时默认通过

    def pass_volatility_filter(self, signal: TradingSignal, market_data: Dict) -> bool:
        """
        波动性过滤器 - 严格模式
        
        Args:
            signal: 交易信号
            market_data: 市场数据
            
        Returns:
            是否通过波动性过滤
        """
        try:
            # 严格检查：如果没有市场数据，要求更高置信度
            if not market_data:
                if signal.confidence < 0.6:
                    self.logger.info(f"   ❌ 无市场数据且置信度不足: {signal.confidence} < 0.6")
                    return False
                return True
            
            # 严格检查价格波动性
            if 'price_volatility' in market_data:
                price_vol = market_data['price_volatility']
                
                # 根据风险回报比严格调整波动性要求
                if signal.risk_reward_ratio < 2.0:
                    # 风险回报比较低时，严格要求波动性适中
                    if price_vol > 0.06:  # 6%波动率（更严格）
                        self.logger.info(f"   ❌ 低风险回报比({signal.risk_reward_ratio})且高波动({price_vol:.4f} > 0.06)")
                        return False
                
                # 极端波动性严格过滤
                if price_vol > 0.12:  # 12%波动率（更严格）
                    if signal.confidence < 0.75:  # 更高置信度要求
                        self.logger.info(f"   ❌ 极端波动({price_vol:.4f} > 0.12)且置信度不足: {signal.confidence} < 0.75")
                        return False
            
            # 严格检查MACD波动性
            if 'macd_volatility' in market_data:
                macd_vol = market_data['macd_volatility']
                
                # MACD波动过大时严格要求
                if macd_vol > 0.3:  # 更严格的阈值
                    if signal.confidence < 0.65:  # 更高置信度要求
                        self.logger.info(f"   ❌ MACD高波动({macd_vol:.4f} > 0.3)且置信度不足: {signal.confidence} < 0.65")
                        return False
            
            # 通过所有检查
            return True
            
        except Exception as e:
            self.logger.error(f"波动性过滤检查失败: {e}")
            # 严格模式：出错时拒绝通过
            return False

    def enhance_signal_quality(self, signal: TradingSignal, market_data: Dict) -> Optional[TradingSignal]:
        """
        增强信号质量
        
        Args:
            signal: 原始信号
            market_data: 市场数据
            
        Returns:
            增强后的信号或None
        """
        try:
            # 创建增强信号副本
            enhanced_signal = TradingSignal(
                signal_type=signal.signal_type,
                signal_strength=signal.signal_strength,
                confidence=signal.confidence,
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
                risk_reward_ratio=signal.risk_reward_ratio,
                timestamp=signal.timestamp,
                reasons=signal.reasons.copy(),
                technical_score=signal.technical_score,
                market_condition=signal.market_condition
            )
            
            # 根据过滤结果调整置信度
            confidence_bonus = 0.0
            self.logger.info(f"🔍 信号增强开始: 初始置信度={signal.confidence:.3f}")
            
            # 高强度信号奖励
            if signal.signal_strength == SignalStrength.VERY_STRONG:
                confidence_bonus += 0.05
            elif signal.signal_strength == SignalStrength.STRONG:
                confidence_bonus += 0.03
            
            # 一致性奖励
            consistency_result = self.check_signal_consistency(signal)
            self.logger.info(f"📊 一致性检查: {consistency_result}, 历史信号数: {len(self.signal_history)}")
            if consistency_result:
                confidence_bonus += 0.02
                enhanced_signal.reasons.append("信号一致性强")
            
            # 市场条件奖励
            market_result = self.check_favorable_market_conditions(market_data)
            self.logger.info(f"📈 市场条件检查: {market_result}, 数据字段: {list(market_data.keys())}")
            if market_result:
                confidence_bonus += 0.03
                enhanced_signal.reasons.append("市场条件良好")
            
            # 应用置信度调整
            enhanced_signal.confidence = min(1.0, signal.confidence + confidence_bonus)
            self.logger.info(f"💡 置信度调整: 奖励={confidence_bonus:.3f}, 最终置信度={enhanced_signal.confidence:.3f}")
            
            # 最终置信度检查
            if enhanced_signal.confidence < self.min_confidence:
                self.logger.info(f"❌ 信号在最终质量检查被过滤: 置信度{enhanced_signal.confidence:.3f} < {self.min_confidence}")
                return None
            
            # 调整信号强度
            if enhanced_signal.confidence >= self.very_high_confidence:
                if enhanced_signal.signal_strength != SignalStrength.VERY_STRONG:
                    enhanced_signal.signal_strength = SignalStrength.VERY_STRONG
            elif enhanced_signal.confidence >= self.high_confidence:
                if enhanced_signal.signal_strength == SignalStrength.MODERATE:
                    enhanced_signal.signal_strength = SignalStrength.STRONG
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"增强信号质量失败: {e}")
            self.logger.info(f"✅ 成功生成{signal.signal_type.value}信号: "
                           f"价格{signal.entry_price}, 止损{signal.stop_loss_price:.2f}, "
                           f"止盈{signal.take_profit_price:.2f}, RR比{signal.risk_reward_ratio:.2f}")
            return signal
    
    def check_signal_consistency(self, signal: TradingSignal) -> bool:
        """
        检查信号一致性
        
        Args:
            signal: 交易信号
            
        Returns:
            是否具有一致性
        """
        try:
            if len(self.signal_history) < 2:
                return False
            
            recent_signals = self.signal_history[-2:]
            
            # 检查信号方向一致性
            same_direction = all(s['signal_type'] == signal.signal_type.value for s in recent_signals)
            
            # 检查置信度趋势
            confidences = [s['confidence'] for s in recent_signals]
            avg_confidence = sum(confidences) / len(confidences)
            confidence_improving = signal.confidence >= avg_confidence
            
            return same_direction and confidence_improving
            
        except Exception as e:
            self.logger.error(f"检查信号一致性失败: {e}")
            return False
    
    def check_favorable_market_conditions(self, market_data: Dict) -> bool:
        """
        检查有利的市场条件
        
        Args:
            market_data: 市场数据
            
        Returns:
            是否为有利市场条件
        """
        try:
            favorable_count = 0
            total_checks = 0
            
            # 检查波动性
            if 'volatility' in market_data:
                total_checks += 1
                volatility = market_data['volatility']
                if 0.02 <= volatility <= 0.06:  # 适中波动性
                    favorable_count += 1
            
            # 检查成交量
            if 'volume_ratio' in market_data:
                total_checks += 1
                volume_ratio = market_data['volume_ratio']
                if 0.8 <= volume_ratio <= 1.5:  # 正常成交量
                    favorable_count += 1
            
            # 检查趋势强度
            if 'trend_strength' in market_data:
                total_checks += 1
                trend_strength = market_data['trend_strength']
                if abs(trend_strength) >= 0.5:  # 明确趋势
                    favorable_count += 1
            
            # 需要至少2/3的条件有利
            return total_checks > 0 and (favorable_count / total_checks) >= 0.67
            
        except Exception as e:
            self.logger.error(f"检查市场条件失败: {e}")
            return False
    
    def add_to_signal_history(self, signal: TradingSignal):
        """
        添加信号到历史记录
        
        Args:
            signal: 交易信号
        """
        try:
            signal_record = {
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp.isoformat(),
                'signal_strength': signal.signal_strength.value,
                'risk_reward_ratio': signal.risk_reward_ratio
            }
            
            self.signal_history.append(signal_record)
            
            # 限制历史记录大小
            if len(self.signal_history) > self.max_history_size:
                self.signal_history = self.signal_history[-self.max_history_size//2:]
                
        except Exception as e:
            self.logger.error(f"添加信号历史失败: {e}")
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        获取过滤统计信息
        
        Returns:
            过滤统计信息
        """
        try:
            total = self.filter_stats['total_signals']
            
            if total == 0:
                return {
                    'total_signals': 0,
                    'final_pass_rate': 0.0,
                    'filter_rates': {}
                }
            
            filter_rates = {
                'confidence_pass_rate': self.filter_stats['passed_confidence'] / total,
                'strength_pass_rate': self.filter_stats['passed_strength'] / total,
                'consistency_pass_rate': self.filter_stats['passed_consistency'] / total,
                'market_condition_pass_rate': self.filter_stats['passed_market_condition'] / total,
                'volatility_pass_rate': self.filter_stats['passed_volatility'] / total,
                'final_pass_rate': self.filter_stats['final_passed'] / total
            }
            
            return {
                'total_signals': total,
                'passed_signals': self.filter_stats['final_passed'],
                'final_pass_rate': filter_rates['final_pass_rate'],
                'filter_rates': filter_rates,
                'signal_history_size': len(self.signal_history),
                'min_confidence_threshold': self.min_confidence,
                'high_confidence_threshold': self.high_confidence,
                'very_high_confidence_threshold': self.very_high_confidence
            }
            
        except Exception as e:
            self.logger.error(f"获取过滤统计失败: {e}")
            return {}
    
    def reset_statistics(self):
        """
        重置过滤统计
        """
        try:
            self.filter_stats = {
                'total_signals': 0,
                'passed_confidence': 0,
                'passed_strength': 0,
                'passed_consistency': 0,
                'passed_market_condition': 0,
                'passed_volatility': 0,
                'final_passed': 0,
                'filter_rates': {}
            }
            
            self.logger.info("过滤统计已重置")
            
        except Exception as e:
            self.logger.error(f"重置过滤统计失败: {e}")
    
    def update_confidence_thresholds(self, min_confidence: float = None, 
                                   high_confidence: float = None,
                                   very_high_confidence: float = None):
        """
        更新置信度阈值
        
        Args:
            min_confidence: 最小置信度阈值
            high_confidence: 高置信度阈值
            very_high_confidence: 极高置信度阈值
        """
        try:
            if min_confidence is not None:
                self.min_confidence = min_confidence
            
            if high_confidence is not None:
                self.high_confidence = high_confidence
                
            if very_high_confidence is not None:
                self.very_high_confidence = very_high_confidence
            
            self.logger.info(f"置信度阈值已更新: min={self.min_confidence}, high={self.high_confidence}, very_high={self.very_high_confidence}")
            
        except Exception as e:
            self.logger.error(f"更新置信度阈值失败: {e}")
    
    def enable_filter(self, filter_name: str, enabled: bool = True):
        """
        启用/禁用特定过滤器
        
        Args:
            filter_name: 过滤器名称
            enabled: 是否启用
        """
        try:
            filter_map = {
                'strength': 'enable_strength_filter',
                'consistency': 'enable_consistency_filter',
                'market_condition': 'enable_market_condition_filter',
                'volatility': 'enable_volatility_filter'
            }
            
            if filter_name in filter_map:
                setattr(self, filter_map[filter_name], enabled)
                self.logger.info(f"过滤器 {filter_name} 已{'启用' if enabled else '禁用'}")
            else:
                self.logger.warning(f"未知的过滤器名称: {filter_name}")
                
        except Exception as e:
            self.logger.error(f"设置过滤器状态失败: {e}")

# 扩展SignalGenerator类以集成增强过滤器
class SignalGeneratorWithEnhancedFilter(SignalGenerator):
    """
    带增强过滤器的信号生成器
    """
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.enhanced_filter = EnhancedSignalFilter(config_manager)
        self.logger.info("增强过滤信号生成器初始化完成")
    
    def generate_signal(self, kline_data: List[Dict]) -> Optional[TradingSignal]:
        """
        生成带增强过滤的交易信号
        
        Args:
            kline_data: K线数据
            
        Returns:
            经过增强过滤的交易信号或None
        """
        try:
            # 使用原始逻辑生成信号
            raw_signal = super().generate_signal(kline_data)
            
            if not raw_signal:
                return None
            
            # 准备市场数据
            market_data = self.prepare_market_data(kline_data)
            
            # 应用增强过滤
            filtered_signal = self.enhanced_filter.filter_signal(raw_signal, market_data)
            
            if filtered_signal:
                self.logger.info(f"信号通过增强过滤: 置信度 {filtered_signal.confidence:.3f}")
            else:
                self.logger.debug(f"信号被增强过滤器拦截")
            
            return filtered_signal
            
        except Exception as e:
            self.logger.error(f"增强过滤信号生成失败: {e}")
            return None
    
    def prepare_market_data(self, kline_data: List[Dict]) -> Dict[str, Any]:
        """
        准备市场数据用于过滤
        
        Args:
            kline_data: K线数据
            
        Returns:
            市场数据字典
        """
        try:
            # 计算价格波动性
            prices = [float(k['close']) for k in kline_data[-20:]]
            price_volatility = np.std(prices) / np.mean(prices) if len(prices) > 1 else 0
            
            # 计算成交量比率
            volumes = [float(k['volume']) for k in kline_data[-20:]]
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else 1
            current_volume = volumes[-1] if volumes else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # 计算趋势强度
            if len(prices) >= 10:
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-10:])
                trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
            else:
                trend_strength = 0
            
            # 基于trend_strength判断趋势方向
            if trend_strength > 0.02:  # 上升趋势阈值2%
                trend = 'BULLISH'
            elif trend_strength < -0.02:  # 下降趋势阈值-2%
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'  # 横盘趋势
            
            return {
                'volatility': price_volatility,
                'price_volatility': price_volatility,
                'volume_ratio': volume_ratio,
                'trend_strength': trend_strength,
                'trend': trend,  # 添加缺失的trend字段
                'macd_volatility': 0.1  # 简化MACD波动性
            }
            
        except Exception as e:
            self.logger.error(f"准备市场数据失败: {e}")
            return {}
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        获取增强过滤器统计信息
        
        Returns:
            过滤器统计信息
        """
        return self.enhanced_filter.get_filter_statistics()
    
    def reset_filter_statistics(self):
        """
        重置过滤器统计信息
        """
        self.enhanced_filter.reset_statistics()
    
    def update_filter_settings(self, **kwargs):
        """
        更新过滤器设置
        """
        if 'min_confidence' in kwargs:
            self.enhanced_filter.update_confidence_thresholds(
                min_confidence=kwargs['min_confidence']
            )
        
        for filter_name in ['strength', 'consistency', 'market_condition', 'volatility']:
            if filter_name in kwargs:
                self.enhanced_filter.enable_filter(filter_name, kwargs[filter_name])

    def analyze_market_structure_comprehensive(self, kline_data: List[Dict]) -> Dict:
        """
        综合市场结构分析 - 使用增强形态检测器
        
        Args:
            kline_data: K线数据
            
        Returns:
            综合市场结构分析结果
        """
        try:
            # 准备OHLC数据
            highs = np.array([float(k['high']) for k in kline_data])
            lows = np.array([float(k['low']) for k in kline_data])
            closes = np.array([float(k['close']) for k in kline_data])
            
            # 计算波动性因子
            vol_factor = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0
            
            # 使用增强形态检测器进行综合分析
            market_analysis = self.enhanced_pattern_detector.analyze_market_structure(
                highs, lows, closes, vol_factor
            )
            
            return {
                'market_analysis': market_analysis,
                'signal_quality': market_analysis.get('signal_quality', {}),
                'overall_score': market_analysis.get('overall_score', 50),
                'market_condition': market_analysis.get('market_condition', 'neutral'),
                'recommendation': market_analysis.get('recommendation', '无明确建议'),
                'enhanced_analysis': True
            }
            
        except Exception as e:
            self.logger.error(f"综合市场结构分析失败: {e}")
            return {
                'market_analysis': {},
                'signal_quality': {},
                'overall_score': 50,
                'market_condition': 'neutral',
                'recommendation': '分析出错',
                'enhanced_analysis': False,
                'error': str(e)
            }

    def calculate_composite_score_enhanced(self, macd_result: Dict, technical_result: Dict, 
                                         pattern_result: Dict, cycle_result: Dict,
                                         market_structure: Dict) -> Dict:
        """
        计算综合评分 - 增强版
        
        Args:
            macd_result: MACD分析结果
            technical_result: 技术指标分析结果
            pattern_result: 形态分析结果
            cycle_result: 周期分析结果
            market_structure: 市场结构分析结果
            
        Returns:
            综合评分结果
        """
        try:
            # 基础评分
            macd_score = macd_result['score']
            technical_score = technical_result['score']
            pattern_score = pattern_result['score']
            cycle_score = cycle_result['score']
            
            # 市场结构评分
            structure_score = market_structure.get('overall_score', 50)
            
            # 增强权重分配
            enhanced_weights = {
                'macd': 0.3,
                'technical': 0.25,
                'pattern': 0.25,
                'cycle': 0.1,
                'structure': 0.1
            }
            
            # 计算加权评分
            total_score = (
                macd_score * enhanced_weights['macd'] +
                technical_score * enhanced_weights['technical'] +
                pattern_score * enhanced_weights['pattern'] +
                cycle_score * enhanced_weights['cycle'] +
                structure_score * enhanced_weights['structure']
            )
            
            # 调试日志
            self.logger.info(f"📊 评分详情: MACD={macd_score:.1f}, 技术={technical_score:.1f}, "
                           f"形态={pattern_score:.1f}, 周期={cycle_score:.1f}, 结构={structure_score:.1f}")
            self.logger.info(f"📊 加权总分: {total_score:.1f}")
            
            # 增强置信度计算
            confidence_factors = []
            
            # MACD置信度
            if macd_result.get('enhanced_detection', False):
                confidence_factors.append(macd_result['confidence'])
            else:
                confidence_factors.append(0.5)
            
            # 形态置信度
            if pattern_result.get('enhanced_detection', False):
                confidence_factors.append(pattern_result['confidence'])
            else:
                confidence_factors.append(0.5)
            
            # 市场结构置信度
            signal_quality = market_structure.get('signal_quality', {})
            structure_confidence = signal_quality.get('avg_confidence', 0.5)
            confidence_factors.append(structure_confidence)
            
            # 综合置信度 - 调整计算方式，提高基础置信度
            # 如果有强烈的技术信号，提高置信度
            if abs(technical_score - 50) > 30:  # 技术分数偏离中性超过30分
                confidence_boost = 0.2
            else:
                confidence_boost = 0
            
            overall_confidence = np.mean(confidence_factors) + confidence_boost
            overall_confidence = min(overall_confidence, 0.95)  # 限制最高置信度
            
            # 信号强度增强计算
            if overall_confidence >= 0.8:
                signal_strength = SignalStrength.VERY_STRONG
            elif overall_confidence >= 0.7:
                signal_strength = SignalStrength.STRONG
            elif overall_confidence >= 0.6:
                signal_strength = SignalStrength.MODERATE
            elif overall_confidence >= 0.4:
                signal_strength = SignalStrength.WEAK
            else:
                signal_strength = SignalStrength.VERY_WEAK
            
            # 🔥 修复：多空平衡的信号类型判断 - 大幅平衡阈值
            market_condition = market_structure.get('market_condition', 'neutral')
            
            # 根据市场环境动态调整阈值
            if market_condition == 'bullish':
                # 牛市：偏向做多
                buy_threshold = 48
                sell_threshold = 45
            elif market_condition == 'bearish':
                # 熊市：适度平衡
                buy_threshold = 55
                sell_threshold = 50
            elif market_condition == 'sideways':
                # 震荡市：完全平衡
                buy_threshold = 50
                sell_threshold = 50
            else:
                # 中性市场：平衡阈值
                buy_threshold = 52
                sell_threshold = 48
            
            if total_score >= buy_threshold:
                signal_type = SignalType.BUY
            elif total_score <= sell_threshold:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # 生成原因列表
            reasons = []
            
            if macd_result.get('enhanced_detection', False):
                if macd_result['has_divergence']:
                    reasons.append(f"增强MACD{macd_result['divergence_type']}背离")
            
            if pattern_result.get('enhanced_detection', False):
                if pattern_result['has_pattern']:
                    reasons.append(f"增强{pattern_result['pattern_name']}形态")
            
            if technical_result['rsi'] < 30:
                reasons.append("RSI超卖")
            elif technical_result['rsi'] > 70:
                reasons.append("RSI超买")
            
            market_condition = market_structure.get('market_condition', 'neutral')
            reasons.append(f"市场状态: {market_condition}")
            
            return {
                'total_score': total_score,
                'confidence': overall_confidence,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'reasons': reasons,
                'components': {
                    'macd': macd_score,
                    'technical': technical_score,
                    'pattern': pattern_score,
                    'cycle': cycle_score,
                    'structure': structure_score
                },
                'enhanced_features': {
                    'macd_enhanced': macd_result.get('enhanced_detection', False),
                    'pattern_enhanced': pattern_result.get('enhanced_detection', False),
                    'structure_analysis': True,
                    'confidence_factors': confidence_factors
                }
            }
            
        except Exception as e:
            self.logger.error(f"计算增强综合评分失败: {e}")
            return {
                'total_score': 50,
                'confidence': 0.5,
                'signal_type': SignalType.HOLD,
                'signal_strength': SignalStrength.WEAK,
                'reasons': ['计算失败'],
                'components': {},
                'enhanced_features': {}
            }

# 添加增强版信号创建方法
    def create_signal_enhanced(self, composite_score: Dict, current_price: float,
                             macd_result: Dict, technical_result: Dict, 
                             pattern_result: Dict, cycle_result: Dict,
                             market_structure: Dict) -> Optional[TradingSignal]:
        """
        创建交易信号 - 增强版
        
        Args:
            composite_score: 综合评分
            current_price: 当前价格
            macd_result: MACD分析结果
            technical_result: 技术指标分析结果
            pattern_result: 形态分析结果
            cycle_result: 周期分析结果
            market_structure: 市场结构分析结果
            
        Returns:
            增强交易信号或None
        """
        try:
            # 检查最小置信度
            if composite_score['confidence'] < self.min_confidence:
                return None
            
            # 检查是否为持有信号
            if composite_score['signal_type'] == SignalType.HOLD:
                return None
            
            # 增强版止损止盈计算
            stop_loss_price, take_profit_price = self.calculate_stop_loss_take_profit_enhanced(
                composite_score['signal_type'], current_price, composite_score['confidence'],
                market_structure
            )
            
            # 计算风险回报比
            risk_reward_ratio = self.calculate_risk_reward_ratio(
                composite_score['signal_type'], current_price, stop_loss_price, take_profit_price
            )
            
            # 检查风险回报比
            self.logger.info(f"📊 风险回报比计算: {risk_reward_ratio:.2f}, 要求: {self.risk_reward_min}")
            if risk_reward_ratio < self.risk_reward_min:
                self.logger.warning(f"⚠️ 风险回报比不足: {risk_reward_ratio:.2f} < {self.risk_reward_min}")
                # 对于强信号，放宽风险回报比要求
                if composite_score['confidence'] >= 0.7 and risk_reward_ratio >= 0.5:
                    self.logger.info(f"✅ 强信号(置信度{composite_score['confidence']:.2f})，放宽风险回报比要求")
                else:
                    return None
            
            # 确定市场状态
            market_condition = market_structure.get('market_condition', 'neutral')
            
            # 创建增强信号
            signal = TradingSignal(
                signal_type=composite_score['signal_type'],
                signal_strength=composite_score['signal_strength'],
                confidence=composite_score['confidence'],
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_reward_ratio=risk_reward_ratio,
                timestamp=datetime.now(),
                reasons=composite_score['reasons'],
                technical_score=composite_score['total_score'],
                market_condition=market_condition,
                technical_indicators=technical_result  # 添加技术指标
            )
            
            # 添加增强功能标记
            signal.enhanced_features = composite_score.get('enhanced_features', {})
            signal.market_structure = market_structure
            
            self.logger.info(f"✅ 成功生成{signal.signal_type.value}信号: "
                           f"价格{signal.entry_price}, 止损{signal.stop_loss_price:.2f}, "
                           f"止盈{signal.take_profit_price:.2f}, RR比{signal.risk_reward_ratio:.2f}")
            return signal
            
        except Exception as e:
            self.logger.error(f"创建增强信号失败: {e}")
            return None

# 添加增强版止损止盈计算
    def calculate_stop_loss_take_profit_enhanced(self, signal_type: SignalType, 
                                               current_price: float, confidence: float,
                                               market_structure: Dict) -> Tuple[float, float]:
        """
        计算止损止盈 - 增强版，考虑市场结构
        
        Args:
            signal_type: 信号类型
            current_price: 当前价格
            confidence: 置信度
            market_structure: 市场结构分析
            
        Returns:
            (止损价格, 止盈价格)
        """
        try:
            # 获取配置
            risk_config = self.config.get_risk_config()
            base_stop_loss = risk_config.stop_loss_pct
            base_take_profit = base_stop_loss * risk_config.take_profit_ratio
            
            # 根据置信度调整
            confidence_multiplier = 0.5 + (confidence * 0.5)
            
            # 根据市场结构调整
            structure_score = market_structure.get('overall_score', 50)
            signal_quality = market_structure.get('signal_quality', {})
            avg_confidence = signal_quality.get('avg_confidence', 0.5)
            
            # 市场强度调整因子
            if structure_score >= 75 or structure_score <= 25:
                # 强烈趋势市场，扩大止盈，收紧止损
                structure_multiplier = 1.2
                stop_loss_tightening = 0.8
            else:
                # 震荡市场，保守设置
                structure_multiplier = 0.9
                stop_loss_tightening = 1.1
            
            # 综合调整
            adjusted_stop_loss = base_stop_loss * confidence_multiplier * stop_loss_tightening
            adjusted_take_profit = base_take_profit * confidence_multiplier * structure_multiplier
            
            if signal_type == SignalType.BUY:
                stop_loss_price = current_price * (1 - adjusted_stop_loss)
                take_profit_price = current_price * (1 + adjusted_take_profit)
            else:  # SELL
                stop_loss_price = current_price * (1 + adjusted_stop_loss)
                take_profit_price = current_price * (1 - adjusted_take_profit)
            
            return stop_loss_price, take_profit_price
            
        except Exception as e:
            self.logger.error(f"增强止损止盈计算失败: {e}")
            # 回退到简单计算
            return self.calculate_stop_loss_take_profit(signal_type, current_price, confidence)

# 添加获取增强统计信息的方法
    def get_enhanced_signal_statistics(self) -> Dict[str, Any]:
        """
        获取增强信号统计信息
        
        Returns:
            增强信号统计信息
        """
        try:
            # 基础统计
            base_stats = self.get_signal_summary()
            
            # 增强统计
            enhanced_stats = {
                'enhanced_pattern_usage': self.signal_stats['enhanced_pattern_usage'],
                'enhanced_usage_rate': (
                    self.signal_stats['enhanced_pattern_usage'] / self.signal_stats['total_generated']
                    if self.signal_stats['total_generated'] > 0 else 0
                ),
                'pattern_detector_stats': self.enhanced_pattern_detector.get_detection_statistics()
            }
            
            # 合并统计信息
            combined_stats = {**base_stats, **enhanced_stats}
            
            return combined_stats
            
        except Exception as e:
            self.logger.error(f"获取增强统计失败: {e}")
            return {}

# 添加重置增强统计的方法
    def reset_enhanced_statistics(self):
        """
        重置增强统计信息
        """
        try:
            # 重置基础统计
            self.signal_stats = {
                'total_generated': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_confidence': 0.0,
                'enhanced_pattern_usage': 0
            }
            
            # 重置增强形态检测器统计
            self.enhanced_pattern_detector.reset_statistics()
            
            self.logger.info("增强统计信息已重置")
            
        except Exception as e:
            self.logger.error(f"重置增强统计失败: {e}") 