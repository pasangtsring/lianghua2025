"""
🚀 任务5.1: 终极多时间周期信号生成器
集成多时间周期分析 + 增强形态识别(含W底/双顶) + 传统技术指标

核心功能：
1. 多时间周期决策融合（4h趋势 + 1h信号 + 15m入场）
2. 增强形态识别集成（W底/双顶作为辅助确认）
3. 传统技术指标补充
4. 智能信号融合和权重分配
5. 冲突信号处理和置信度评估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

from utils.logger import get_logger, performance_monitor
from config.config_manager import ConfigManager
from core.signal_generator import TradingSignal, SignalType, SignalStrength
from core.technical_indicators import MultiTimeframeIndicators, MultiTimeframeIndicatorResult
from core.enhanced_pattern_detector import EnhancedPatternDetector, PatternType
from core.technical_indicators import TechnicalIndicatorCalculator

class TimeframeRole(Enum):
    """时间周期角色枚举"""
    TREND = "trend"          # 趋势判断（4h）
    SIGNAL = "signal"        # 信号确认（1h）
    ENTRY = "entry"          # 入场时机（15m）
    CONFIRMATION = "confirm"  # 突破确认（5m）

class SignalFusionResult(Enum):
    """信号融合结果枚举"""
    STRONG_AGREEMENT = "strong_agreement"      # 强一致
    PARTIAL_AGREEMENT = "partial_agreement"    # 部分一致
    WEAK_AGREEMENT = "weak_agreement"         # 弱一致
    CONFLICT = "conflict"                     # 冲突
    NO_SIGNAL = "no_signal"                   # 无信号

@dataclass
class EnhancedTradingSignal:
    """增强版交易信号数据结构"""
    # 基础信号信息
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    timestamp: datetime
    
    # 多时间周期分析结果
    timeframe_analysis: Dict[str, Any] = field(default_factory=dict)
    trend_alignment: str = "NEUTRAL"
    signal_consistency: SignalFusionResult = SignalFusionResult.NO_SIGNAL
    
    # 形态确认信息
    pattern_confirmation: Optional[Dict[str, Any]] = None
    double_pattern_signal: Optional[Dict[str, Any]] = None
    
    # 技术指标支持
    technical_support: Dict[str, float] = field(default_factory=dict)
    
    # 权重和评分
    component_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    
    # 附加信息
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_standard_signal(self) -> TradingSignal:
        """转换为标准交易信号格式"""
        return TradingSignal(
            signal_type=self.signal_type,
            signal_strength=self.signal_strength,
            confidence=self.confidence,
            entry_price=self.entry_price,
            stop_loss_price=self.stop_loss_price,
            take_profit_price=self.take_profit_price,
            risk_reward_ratio=self.risk_reward_ratio,
            timestamp=self.timestamp,
            reasons=self.reasons,
            technical_score=self.final_score,
            market_condition=self.trend_alignment,
            technical_indicators=self.timeframe_analysis
        )

class UltimateMultiTimeframeSignalGenerator:
    """
    🎯 终极多时间周期信号生成器
    
    架构设计：
    1. 多时间周期分析权重：70%
       - 4h趋势判断：主导方向
       - 1h信号确认：强度验证
       - 15m入场时机：精确定位
    
    2. 增强形态识别权重：18%
       - W底/双顶检测：辅助确认
       - 传统形态：补充支持
    
    3. 传统技术指标权重：12%
       - RSI、MACD：辅助验证
       - 成交量分析：确认强度
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_logger(__name__)
        
        # 初始化组件
        self.multi_tf_indicators = MultiTimeframeIndicators(config)
        self.enhanced_pattern_detector = EnhancedPatternDetector(config)
        self.tech_calculator = TechnicalIndicatorCalculator(config)
        
        # 权重配置
        self.component_weights = {
            'multi_timeframe': 0.70,      # 多时间周期主导
            'enhanced_patterns': 0.18,    # 增强形态识别
            'traditional_tech': 0.12      # 传统技术指标
        }
        
        # 时间周期权重
        self.timeframe_weights = {
            'trend': 0.40,      # 4h趋势权重40%
            'signal': 0.35,     # 1h信号权重35%
            'entry': 0.25       # 15m入场权重25%
        }
        
        # 信号强度阈值
        self.strength_thresholds = {
            'very_strong': 0.85,
            'strong': 0.70,
            'medium': 0.55,
            'weak': 0.40,
            'very_weak': 0.20
        }
        
        # 统计信息
        self.generation_stats = {
            'total_signals': 0,
            'strong_signals': 0,
            'pattern_enhanced': 0,
            'timeframe_conflicts': 0
        }
        
        self.logger.info("🚀 终极多时间周期信号生成器初始化完成")
        self.logger.info(f"   📊 组件权重: {self.component_weights}")
        self.logger.info(f"   ⏰时间周期权重: {self.timeframe_weights}")
    
    @performance_monitor
    async def generate_ultimate_signal(self, symbol: str, multi_data: Dict[str, pd.DataFrame]) -> Optional[EnhancedTradingSignal]:
        """
        🎯 生成终极交易信号
        
        Args:
            symbol: 交易对符号
            multi_data: 多时间周期数据 {'trend': 4h_data, 'signal': 1h_data, 'entry': 15m_data}
        
        Returns:
            增强版交易信号或None
        """
        try:
            self.logger.debug(f"🔍 开始为 {symbol} 生成终极信号...")
            
            # 1. 多时间周期技术指标分析
            multi_tf_result = await self._analyze_multi_timeframe(symbol, multi_data)
            if not multi_tf_result:
                self.logger.debug(f"多时间周期分析无结果: {symbol}")
                return None
            
            # 2. 增强形态识别分析
            pattern_result = await self._analyze_enhanced_patterns(multi_data.get('entry'))
            
            # 3. 传统技术指标分析
            tech_result = await self._analyze_traditional_indicators(multi_data.get('entry'))
            
            # 4. 信号融合和决策
            ultimate_signal = await self._fuse_signals(
                symbol, multi_tf_result, pattern_result, tech_result, multi_data
            )
            
            if ultimate_signal:
                self.generation_stats['total_signals'] += 1
                if ultimate_signal.signal_strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                    self.generation_stats['strong_signals'] += 1
                if ultimate_signal.pattern_confirmation:
                    self.generation_stats['pattern_enhanced'] += 1
                
                self.logger.info(f"✅ 生成终极信号: {symbol} {ultimate_signal.signal_type.value.upper()} "
                               f"强度={ultimate_signal.signal_strength.value} 置信度={ultimate_signal.confidence:.2f}")
            
            return ultimate_signal
            
        except Exception as e:
            self.logger.error(f"❌ 终极信号生成失败 {symbol}: {e}")
            return None
    
    async def _analyze_multi_timeframe(self, symbol: str, multi_data: Dict[str, pd.DataFrame]) -> Optional[MultiTimeframeIndicatorResult]:
        """
        📊 多时间周期技术指标分析
        """
        try:
            # 确保数据完整性
            required_timeframes = ['trend', 'signal', 'entry']
            for tf in required_timeframes:
                if tf not in multi_data or multi_data[tf].empty:
                    self.logger.warning(f"缺少{tf}时间周期数据: {symbol}")
                    return None
            
            # 调用多时间周期指标计算器
            result = await self.multi_tf_indicators.calculate_multi_timeframe_indicators(multi_data, symbol)
            
            if result and result.trend_alignment != 'NEUTRAL':
                self.logger.debug(f"多时间周期分析: {symbol} 趋势={result.trend_alignment} "
                                f"信号强度={result.signal_strength} 置信度={result.confidence:.2f}")
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"多时间周期分析失败 {symbol}: {e}")
            return None
    
    async def _analyze_enhanced_patterns(self, entry_data: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        🔍 增强形态识别分析（包含W底/双顶）
        """
        try:
            if entry_data is None or entry_data.empty or len(entry_data) < 20:
                return None
            
            # 提取OHLCV数据
            highs = entry_data['high'].values
            lows = entry_data['low'].values
            closes = entry_data['close'].values
            volumes = entry_data['volume'].values
            
            # 调用增强形态检测器
            patterns = self.enhanced_pattern_detector.detect_pattern(
                opens=entry_data['open'].values,
                highs=highs,
                lows=lows,
                closes=closes
            )
            
            if not patterns:
                return None
            
            # 处理形态结果
            pattern_result = {
                'total_patterns': len(patterns),
                'pattern_types': {},
                'max_confidence': 0.0,
                'double_pattern_found': False,
                'strongest_pattern': None
            }
            
            for pattern in patterns:
                pattern_type = pattern.type.value
                pattern_result['pattern_types'][pattern_type] = pattern_result['pattern_types'].get(pattern_type, 0) + 1
                
                if pattern.confidence > pattern_result['max_confidence']:
                    pattern_result['max_confidence'] = pattern.confidence
                    pattern_result['strongest_pattern'] = pattern
                
                # 检查是否包含W底/双顶
                if pattern_type in ['double_bottom_bull', 'double_top_bear']:
                    pattern_result['double_pattern_found'] = True
            
            self.logger.debug(f"形态识别结果: 发现{len(patterns)}个形态，最高置信度={pattern_result['max_confidence']:.2f}")
            return pattern_result
            
        except Exception as e:
            self.logger.error(f"增强形态分析失败: {e}")
            return None
    
    async def _analyze_traditional_indicators(self, entry_data: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
        """
        📈 传统技术指标分析
        """
        try:
            if entry_data is None or entry_data.empty or len(entry_data) < 20:
                return None
            
            # 计算传统技术指标
            closes = entry_data['close'].values
            highs = entry_data['high'].values
            lows = entry_data['low'].values
            volumes = entry_data['volume'].values
            
            # RSI分析（修复RSIResult对象访问）
            rsi_results = self.tech_calculator.calculate_rsi(closes, period=14)
            current_rsi = rsi_results[-1].rsi_value if len(rsi_results) > 0 else 50
            
            # MACD分析（修复返回值解包问题）
            macd_results = self.tech_calculator.calculate_macd(closes)
            current_macd = macd_results[-1].macd_line if len(macd_results) > 0 else 0
            
            # 成交量分析
            volume_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            current_volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
            
            result = {
                'rsi': current_rsi,
                'macd': current_macd,
                'volume_ratio': current_volume_ratio,
                'rsi_signal': self._evaluate_rsi_signal(current_rsi),
                'macd_signal': self._evaluate_macd_signal(current_macd),
                'volume_signal': self._evaluate_volume_signal(current_volume_ratio)
            }
            
            self.logger.debug(f"传统指标分析: RSI={current_rsi:.1f} MACD={current_macd:.4f} Vol比率={current_volume_ratio:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"传统技术指标分析失败: {e}")
            return None
    
    def _evaluate_rsi_signal(self, rsi: float) -> float:
        """评估RSI信号强度 (-1到1)"""
        if rsi > 70:
            return -0.8  # 超买，看跌
        elif rsi > 60:
            return -0.4  # 偏超买
        elif rsi < 30:
            return 0.8   # 超卖，看涨
        elif rsi < 40:
            return 0.4   # 偏超卖
        else:
            return 0.0   # 中性
    
    def _evaluate_macd_signal(self, macd: float) -> float:
        """评估MACD信号强度 (-1到1)"""
        if macd > 0.001:
            return 0.6   # 看涨
        elif macd < -0.001:
            return -0.6  # 看跌
        else:
            return 0.0   # 中性
    
    def _evaluate_volume_signal(self, volume_ratio: float) -> float:
        """评估成交量信号强度 (0到1)"""
        if volume_ratio > 2.0:
            return 1.0   # 成交量爆发
        elif volume_ratio > 1.5:
            return 0.7   # 成交量活跃
        elif volume_ratio > 1.2:
            return 0.4   # 成交量温和放大
        else:
            return 0.0   # 成交量正常
    
    async def _fuse_signals(self, symbol: str, multi_tf_result: MultiTimeframeIndicatorResult,
                          pattern_result: Optional[Dict], tech_result: Optional[Dict],
                          multi_data: Dict[str, pd.DataFrame]) -> Optional[EnhancedTradingSignal]:
        """
        🔀 信号融合和最终决策
        """
        try:
            # 1. 检查多时间周期信号强度
            if multi_tf_result.signal_strength == 'WEAK':
                self.logger.debug(f"多时间周期信号过弱，跳过: {symbol}")
                return None
            
            # 2. 计算各组件得分
            component_scores = {}
            
            # 多时间周期得分
            tf_score = self._calculate_timeframe_score(multi_tf_result)
            component_scores['multi_timeframe'] = tf_score
            
            # 形态识别得分
            pattern_score = self._calculate_pattern_score(pattern_result)
            component_scores['enhanced_patterns'] = pattern_score
            
            # 传统技术指标得分
            tech_score = self._calculate_technical_score(tech_result, multi_tf_result.trend_alignment)
            component_scores['traditional_tech'] = tech_score
            
            # 3. 计算加权总分
            final_score = (
                component_scores['multi_timeframe'] * self.component_weights['multi_timeframe'] +
                component_scores['enhanced_patterns'] * self.component_weights['enhanced_patterns'] +
                component_scores['traditional_tech'] * self.component_weights['traditional_tech']
            )
            
            # 4. 评估信号强度
            signal_strength = self._determine_signal_strength(final_score)
            if signal_strength == SignalStrength.VERY_WEAK:
                self.logger.debug(f"最终信号过弱，跳过: {symbol} 得分={final_score:.2f}")
                return None
            
            # 5. 确定信号方向
            signal_type = SignalType.BUY if multi_tf_result.trend_alignment == 'ALIGNED_BULLISH' else SignalType.SELL
            
            # 6. 计算交易参数
            entry_data = multi_data.get('entry')
            current_price = entry_data['close'].iloc[-1] if entry_data is not None and not entry_data.empty else 0
            
            stop_loss, take_profit = self._calculate_trade_levels(
                current_price, signal_type, multi_tf_result, pattern_result
            )
            
            # 7. 创建增强信号
            enhanced_signal = EnhancedTradingSignal(
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence=min(final_score / 100.0, 1.0),
                entry_price=current_price,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                risk_reward_ratio=abs(take_profit - current_price) / abs(current_price - stop_loss) if abs(current_price - stop_loss) > 0 else 2.0,
                timestamp=datetime.now(),
                timeframe_analysis=multi_tf_result.timeframe_results,
                trend_alignment=multi_tf_result.trend_alignment,
                signal_consistency=self._evaluate_signal_consistency(multi_tf_result, pattern_result),
                pattern_confirmation=pattern_result,
                technical_support=tech_result or {},
                component_scores=component_scores,
                final_score=final_score,
                reasons=self._generate_signal_reasons(multi_tf_result, pattern_result, tech_result),
                metadata={
                    'symbol': symbol,
                    'generation_time': datetime.now().isoformat(),
                    'multi_tf_confidence': multi_tf_result.confidence,
                    'pattern_enhanced': pattern_result is not None,
                    'double_pattern_found': pattern_result.get('double_pattern_found', False) if pattern_result else False
                }
            )
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"信号融合失败 {symbol}: {e}")
            return None
    
    def _calculate_timeframe_score(self, multi_tf_result: MultiTimeframeIndicatorResult) -> float:
        """计算多时间周期得分 (0-100)"""
        base_score = multi_tf_result.confidence * 100
        
        # 趋势一致性加分
        if multi_tf_result.trend_alignment in ['ALIGNED_BULLISH', 'ALIGNED_BEARISH']:
            base_score *= 1.1
        
        # 信号强度加分
        strength_multiplier = {
            'VERY_STRONG': 1.2,
            'STRONG': 1.1,
            'MEDIUM': 1.0,
            'WEAK': 0.8
        }.get(multi_tf_result.signal_strength, 1.0)
        
        return min(base_score * strength_multiplier, 100)
    
    def _calculate_pattern_score(self, pattern_result: Optional[Dict]) -> float:
        """计算形态识别得分 (0-100)"""
        if not pattern_result:
            return 0
        
        base_score = pattern_result.get('max_confidence', 0) * 100
        
        # W底/双顶额外加分
        if pattern_result.get('double_pattern_found', False):
            base_score *= 1.15  # 15%加分
        
        # 形态数量加分
        pattern_count = pattern_result.get('total_patterns', 0)
        if pattern_count > 1:
            base_score *= (1 + 0.05 * (pattern_count - 1))  # 每个额外形态5%加分
        
        return min(base_score, 100)
    
    def _calculate_technical_score(self, tech_result: Optional[Dict], trend_alignment: str) -> float:
        """计算传统技术指标得分 (0-100)"""
        if not tech_result:
            return 0
        
        # 基础指标得分
        rsi_score = abs(tech_result.get('rsi_signal', 0)) * 30
        macd_score = abs(tech_result.get('macd_signal', 0)) * 30
        volume_score = tech_result.get('volume_signal', 0) * 40
        
        total_score = rsi_score + macd_score + volume_score
        
        # 与趋势一致性检查
        is_bullish_trend = trend_alignment == 'ALIGNED_BULLISH'
        rsi_consistent = (tech_result.get('rsi_signal', 0) > 0) == is_bullish_trend
        macd_consistent = (tech_result.get('macd_signal', 0) > 0) == is_bullish_trend
        
        # 一致性加分
        if rsi_consistent:
            total_score *= 1.1
        if macd_consistent:
            total_score *= 1.1
        
        return min(total_score, 100)
    
    def _determine_signal_strength(self, final_score: float) -> SignalStrength:
        """根据最终得分确定信号强度"""
        if final_score >= self.strength_thresholds['very_strong']:
            return SignalStrength.VERY_STRONG
        elif final_score >= self.strength_thresholds['strong']:
            return SignalStrength.STRONG
        elif final_score >= self.strength_thresholds['medium']:
            return SignalStrength.MEDIUM
        elif final_score >= self.strength_thresholds['weak']:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _evaluate_signal_consistency(self, multi_tf_result: MultiTimeframeIndicatorResult,
                                   pattern_result: Optional[Dict]) -> SignalFusionResult:
        """评估信号一致性"""
        if multi_tf_result.trend_alignment in ['ALIGNED_BULLISH', 'ALIGNED_BEARISH']:
            if pattern_result and pattern_result.get('max_confidence', 0) > 0.7:
                return SignalFusionResult.STRONG_AGREEMENT
            elif pattern_result:
                return SignalFusionResult.PARTIAL_AGREEMENT
            else:
                return SignalFusionResult.WEAK_AGREEMENT
        else:
            return SignalFusionResult.CONFLICT
    
    def _calculate_trade_levels(self, current_price: float, signal_type: SignalType,
                              multi_tf_result: MultiTimeframeIndicatorResult,
                              pattern_result: Optional[Dict]) -> Tuple[float, float]:
        """计算止损和止盈位"""
        # 基础ATR倍数
        atr_multiplier = 1.5 if multi_tf_result.signal_strength == 'VERY_STRONG' else 2.0
        
        # 从形态结果获取更精确的止损位
        if pattern_result and pattern_result.get('strongest_pattern'):
            pattern = pattern_result['strongest_pattern']
            if hasattr(pattern, 'stop_loss') and hasattr(pattern, 'take_profit'):
                return pattern.stop_loss, pattern.take_profit
        
        # 默认计算
        if signal_type == SignalType.BUY:
            stop_loss = current_price * (1 - 0.02 * atr_multiplier)  # 2-3%止损
            take_profit = current_price * (1 + 0.04 * atr_multiplier)  # 4-6%止盈
        else:
            stop_loss = current_price * (1 + 0.02 * atr_multiplier)
            take_profit = current_price * (1 - 0.04 * atr_multiplier)
        
        return stop_loss, take_profit
    
    def _generate_signal_reasons(self, multi_tf_result: MultiTimeframeIndicatorResult,
                               pattern_result: Optional[Dict], tech_result: Optional[Dict]) -> List[str]:
        """生成信号原因说明"""
        reasons = []
        
        # 多时间周期原因
        reasons.append(f"多时间周期{multi_tf_result.trend_alignment}，信号强度{multi_tf_result.signal_strength}")
        
        # 形态原因
        if pattern_result:
            if pattern_result.get('double_pattern_found'):
                reasons.append("检测到W底/双顶形态确认")
            reasons.append(f"发现{pattern_result['total_patterns']}个形态，最高置信度{pattern_result['max_confidence']:.2f}")
        
        # 技术指标原因
        if tech_result:
            if abs(tech_result.get('rsi_signal', 0)) > 0.5:
                reasons.append(f"RSI指标{'超卖' if tech_result['rsi_signal'] > 0 else '超买'}确认")
            if tech_result.get('volume_signal', 0) > 0.5:
                reasons.append("成交量放大确认")
        
        return reasons
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        return {
            **self.generation_stats,
            'strong_signal_rate': self.generation_stats['strong_signals'] / max(self.generation_stats['total_signals'], 1),
            'pattern_enhanced_rate': self.generation_stats['pattern_enhanced'] / max(self.generation_stats['total_signals'], 1)
        } 