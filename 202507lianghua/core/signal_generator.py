"""
信号生成器模块
整合所有分析模块，生成交易信号、风险评估、置信度计算
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from utils.logger import get_logger
from config.config_manager import ConfigManager
from core.complete_macd_divergence_detector import CompleteMACDDivergenceDetector
from core.technical_indicators import TechnicalIndicatorCalculator
from core.pattern_recognizer import PatternRecognizer
from core.cycle_analyzer import CycleAnalyzer, CyclePhase

class SignalType(Enum):
    """信号类型枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

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
    
class SignalGenerator:
    """信号生成器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 初始化分析模块
        self.macd_detector = CompleteMACDDivergenceDetector()  # 使用默认配置
        self.technical_indicators = TechnicalIndicatorCalculator(config_manager)
        self.pattern_recognizer = PatternRecognizer(config_manager)
        self.cycle_analyzer = CycleAnalyzer(config_manager)
        
        # 信号配置
        self.trading_config = config_manager.config.trading
        self.min_confidence = 0.65
        self.risk_reward_min = 1.5
        
        # 权重配置
        self.weights = {
            'macd_divergence': 0.4,
            'technical_indicators': 0.3,
            'pattern_recognition': 0.2,
            'cycle_analysis': 0.1
        }
        
        self.logger.info("信号生成器初始化完成")
    
    def generate_signal(self, kline_data: List[Dict]) -> Optional[TradingSignal]:
        """
        生成交易信号
        
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
            
            # 1. MACD背离分析
            macd_result = self.analyze_macd_divergence(kline_data)
            
            # 2. 技术指标分析
            technical_result = self.analyze_technical_indicators(kline_data)
            
            # 3. 形态识别分析
            pattern_result = self.analyze_patterns(kline_data)
            
            # 4. 周期分析
            cycle_result = self.analyze_cycle(kline_data)
            
            # 5. 综合评分
            composite_score = self.calculate_composite_score(
                macd_result, technical_result, pattern_result, cycle_result
            )
            
            # 6. 生成信号
            signal = self.create_signal(
                composite_score, current_price, 
                macd_result, technical_result, pattern_result, cycle_result
            )
            
            if signal:
                self.logger.info(f"生成交易信号: {signal.signal_type.value}, 置信度: {signal.confidence:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {e}")
            return None
    
    def analyze_macd_divergence(self, kline_data: List[Dict]) -> Dict:
        """分析MACD背离"""
        try:
            # 提取价格数据
            prices = [float(k['close']) for k in kline_data]
            
            # 计算MACD
            macd_data = self.technical_indicators.calculate_macd(prices)
            
            # 转换为MACDResult格式
            from core.complete_macd_divergence_detector import MACDResult
            macd_results = []
            for i, data in enumerate(macd_data):
                macd_result = MACDResult(
                    macd_line=data.macd_line,
                    signal_line=data.signal_line,
                    histogram=data.histogram,
                    fast_ema=data.fast_ema,
                    slow_ema=data.slow_ema,
                    timestamp=datetime.fromtimestamp(kline_data[i].get('timestamp', 0) / 1000)
                )
                macd_results.append(macd_result)
            
            # 检测背离
            divergence_signals = self.macd_detector.detect_divergence(prices, macd_results)
            
            if divergence_signals:
                # 取最强的信号
                best_signal = max(divergence_signals, key=lambda x: x.strength)
                return {
                    'has_signal': True,
                    'signal_type': best_signal.signal_type,
                    'confidence': best_signal.confidence,
                    'strength': best_signal.strength,
                    'score': best_signal.confidence * 100
                }
            else:
                return {
                    'has_signal': False,
                    'signal_type': 'none',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'score': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"MACD背离分析失败: {e}")
            return {
                'has_signal': False,
                'signal_type': 'none',
                'confidence': 0.0,
                'strength': 0.0,
                'score': 0.0
            }
    
    def analyze_technical_indicators(self, kline_data: List[Dict]) -> Dict:
        """分析技术指标"""
        try:
            # 提取价格数据
            prices = [float(k['close']) for k in kline_data]
            
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
            
            # RSI评分
            if current_rsi < 30:
                score += 30  # 超卖
                signals.append("RSI超卖")
            elif current_rsi > 70:
                score -= 30  # 超买
                signals.append("RSI超买")
            
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
        """分析形态"""
        try:
            patterns = self.pattern_recognizer.recognize_patterns(kline_data)
            
            if patterns:
                # 取最高置信度的形态
                best_pattern = max(patterns, key=lambda p: p.confidence)
                
                return {
                    'has_pattern': True,
                    'pattern_name': best_pattern.name,
                    'confidence': best_pattern.confidence,
                    'pattern_type': best_pattern.pattern_type,
                    'score': best_pattern.confidence * 100
                }
            else:
                return {
                    'has_pattern': False,
                    'pattern_name': 'none',
                    'confidence': 0.0,
                    'pattern_type': 'neutral',
                    'score': 50
                }
                
        except Exception as e:
            self.logger.error(f"形态分析失败: {e}")
            return {
                'has_pattern': False,
                'pattern_name': 'none',
                'confidence': 0.0,
                'pattern_type': 'neutral',
                'score': 50
            }
    
    def analyze_cycle(self, kline_data: List[Dict]) -> Dict:
        """分析周期"""
        try:
            cycle_analysis = self.cycle_analyzer.analyze_cycle(kline_data)
            
            # 根据周期阶段评分
            phase_scores = {
                CyclePhase.SPRING: 70,  # 积累期 - 偏多
                CyclePhase.SUMMER: 85,  # 上升期 - 强多
                CyclePhase.AUTUMN: 30,  # 分配期 - 偏空
                CyclePhase.WINTER: 15   # 衰退期 - 强空
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
            
            # 确定信号类型
            if total_score > 65:
                signal_type = SignalType.BUY
                signal_strength = SignalStrength.STRONG if total_score > 80 else SignalStrength.MEDIUM
            elif total_score < 35:
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
            if risk_reward_ratio < self.risk_reward_min:
                self.logger.info(f"风险回报比不足: {risk_reward_ratio:.2f} < {self.risk_reward_min}")
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
                market_condition=market_condition
            )
            
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