"""
🧪 任务5.2: A/B测试框架 - 信号生成器性能对比
对比原版信号生成器 vs 增强版终极信号生成器

核心功能：
1. 并行运行两套信号生成系统
2. 实时收集性能指标
3. 统计分析和对比报告
4. 自动化决策支持系统
5. 风险控制和回退机制
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import statistics

from utils.logger import get_logger, performance_monitor
from config.config_manager import ConfigManager
from core.signal_generator import SignalGeneratorWithEnhancedFilter, TradingSignal
from core.ultimate_multi_timeframe_signal_generator import UltimateMultiTimeframeSignalGenerator, EnhancedTradingSignal

class TestGroup(Enum):
    """测试组别枚举"""
    GROUP_A = "group_a"  # 原版信号生成器
    GROUP_B = "group_b"  # 增强版终极信号生成器

class TestPhase(Enum):
    """测试阶段枚举"""
    INITIALIZATION = "initialization"    # 初始化
    RUNNING = "running"                 # 运行中
    ANALYSIS = "analysis"               # 分析阶段
    DECISION = "decision"               # 决策阶段
    COMPLETED = "completed"             # 已完成

@dataclass
class SignalPerformanceMetrics:
    """信号性能指标数据结构"""
    # 基础统计
    total_signals: int = 0
    valid_signals: int = 0
    
    # 信号质量
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 信号类型分布
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    
    # 信号强度分布
    strength_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 时间统计
    avg_generation_time: float = 0.0
    total_generation_time: float = 0.0
    
    # 形态增强统计（仅B组）
    pattern_enhanced_signals: int = 0
    double_pattern_found: int = 0
    
    def calculate_derived_metrics(self):
        """计算衍生指标"""
        self.signal_success_rate = self.valid_signals / max(self.total_signals, 1)
        self.pattern_enhancement_rate = self.pattern_enhanced_signals / max(self.valid_signals, 1)
        self.double_pattern_rate = self.double_pattern_found / max(self.valid_signals, 1)

@dataclass
class ABTestConfig:
    """A/B测试配置"""
    # 测试持续时间
    test_duration_days: int = 30
    test_duration_hours: int = 24 * 30  # 30天 = 720小时 
    
    # 流量分配
    traffic_split: float = 0.5  # 50%流量给B组
    
    # 成功指标
    success_metrics: List[str] = field(default_factory=lambda: [
        'signal_success_rate', 'avg_confidence', 'pattern_enhancement_rate'
    ])
    
    # 统计显著性
    min_sample_size: int = 100      # 最小样本量
    significance_level: float = 0.05 # 显著性水平
    
    # 风险控制
    max_performance_degradation: float = 0.1  # 最大性能下降10%
    emergency_stop_threshold: float = 0.2     # 紧急停止阈值20%
    
    # 数据收集
    metrics_collection_interval: int = 3600   # 每小时收集一次指标
    detailed_logging: bool = True             # 详细日志
    
    # 决策标准
    improvement_threshold: float = 0.05       # 改进阈值5%
    confidence_threshold: float = 0.95        # 置信度阈值95%

class SignalGeneratorABTest:
    """
    🧪 信号生成器A/B测试管理器
    
    测试设计：
    - A组：原版SignalGeneratorWithEnhancedFilter
    - B组：增强版UltimateMultiTimeframeSignalGenerator
    - 评估指标：信号质量、生成速度、增强效果
    """
    
    def __init__(self, config: ConfigManager, test_config: Optional[ABTestConfig] = None):
        self.config = config
        self.test_config = test_config or ABTestConfig()
        self.logger = get_logger(__name__)
        
        # 初始化信号生成器
        self.group_a_generator = SignalGeneratorWithEnhancedFilter(config)
        self.group_b_generator = UltimateMultiTimeframeSignalGenerator(config)
        
        # 测试状态
        self.current_phase = TestPhase.INITIALIZATION
        self.test_start_time = None
        self.test_end_time = None
        
        # 性能指标收集
        self.group_a_metrics = SignalPerformanceMetrics()
        self.group_b_metrics = SignalPerformanceMetrics()
        
        # 详细数据记录
        self.detailed_results = {
            'group_a': [],
            'group_b': []
        }
        
        # 实时统计
        self.hourly_metrics = {}
        self.comparison_results = {}
        
        self.logger.info("🧪 A/B测试框架初始化完成")
        self.logger.info(f"   📊 测试配置: {self.test_config.test_duration_days}天, 流量分配={self.test_config.traffic_split}")
        self.logger.info(f"   🎯 成功指标: {self.test_config.success_metrics}")
    
    async def start_ab_test(self) -> Dict[str, Any]:
        """
        🚀 启动A/B测试
        
        Returns:
            测试启动结果
        """
        try:
            self.logger.info("🚀 启动A/B测试...")
            
            # 1. 初始化测试环境
            await self._initialize_test_environment()
            
            # 2. 开始测试
            self.current_phase = TestPhase.RUNNING
            self.test_start_time = datetime.now()
            self.test_end_time = self.test_start_time + timedelta(days=self.test_config.test_duration_days)
            
            self.logger.info(f"✅ A/B测试已启动")
            self.logger.info(f"   ⏰ 开始时间: {self.test_start_time}")
            self.logger.info(f"   ⏰ 结束时间: {self.test_end_time}")
            
            return {
                'status': 'started',
                'start_time': self.test_start_time.isoformat(),
                'end_time': self.test_end_time.isoformat(),
                'phase': self.current_phase.value
            }
            
        except Exception as e:
            self.logger.error(f"❌ A/B测试启动失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _initialize_test_environment(self):
        """初始化测试环境"""
        # 验证两个生成器都正常工作
        test_data = pd.DataFrame({
            'open': [100, 101, 99, 102],
            'high': [102, 103, 101, 104],
            'low': [99, 100, 98, 101],
            'close': [101, 99, 102, 103],
            'volume': [1000, 1200, 800, 1500]
        })
        
        # A组测试
        try:
            a_signal = self.group_a_generator.generate_signal(test_data.to_dict('records'))
            self.logger.info("✅ A组信号生成器测试通过")
        except Exception as e:
            self.logger.error(f"❌ A组信号生成器测试失败: {e}")
            raise
        
        # B组测试 (需要多时间周期数据)
        try:
            multi_data = {
                'trend': test_data,
                'signal': test_data, 
                'entry': test_data
            }
            b_signal = await self.group_b_generator.generate_ultimate_signal('BTCUSDT', multi_data)
            self.logger.info("✅ B组信号生成器测试通过")
        except Exception as e:
            self.logger.error(f"❌ B组信号生成器测试失败: {e}")
            raise
    
    async def run_parallel_signal_generation(self, symbol: str, single_tf_data: List[Dict], multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        🔄 并行运行两套信号生成系统
        
        Args:
            symbol: 交易对符号
            single_tf_data: 单时间周期数据（A组使用）
            multi_tf_data: 多时间周期数据（B组使用）
            
        Returns:
            并行测试结果
        """
        try:
            # 根据流量分配决定是否运行B组
            run_group_b = np.random.random() < self.test_config.traffic_split
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'group_a_result': None,
                'group_b_result': None,
                'comparison': None
            }
            
            # A组信号生成
            start_time_a = datetime.now()
            try:
                a_signal = self.group_a_generator.generate_signal(single_tf_data)
                generation_time_a = (datetime.now() - start_time_a).total_seconds()
                
                results['group_a_result'] = {
                    'signal': a_signal.to_dict() if a_signal else None,
                    'generation_time': generation_time_a,
                    'success': a_signal is not None
                }
                
                # 更新A组指标
                await self._update_group_metrics(TestGroup.GROUP_A, a_signal, generation_time_a)
                
            except Exception as e:
                self.logger.error(f"A组信号生成失败 {symbol}: {e}")
                results['group_a_result'] = {'error': str(e), 'success': False}
            
            # B组信号生成（按流量分配）
            if run_group_b:
                start_time_b = datetime.now()
                try:
                    b_signal = await self.group_b_generator.generate_ultimate_signal(symbol, multi_tf_data)
                    generation_time_b = (datetime.now() - start_time_b).total_seconds()
                    
                    results['group_b_result'] = {
                        'signal': b_signal.__dict__ if b_signal else None,
                        'generation_time': generation_time_b,
                        'success': b_signal is not None
                    }
                    
                    # 更新B组指标
                    await self._update_group_metrics(TestGroup.GROUP_B, b_signal, generation_time_b)
                    
                except Exception as e:
                    self.logger.error(f"B组信号生成失败 {symbol}: {e}")
                    results['group_b_result'] = {'error': str(e), 'success': False}
            
            # 记录详细结果
            if results['group_a_result']:
                self.detailed_results['group_a'].append(results['group_a_result'])
            if results['group_b_result']:
                self.detailed_results['group_b'].append(results['group_b_result'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"并行信号生成失败 {symbol}: {e}")
            return {'error': str(e)}
    
    async def _update_group_metrics(self, group: TestGroup, signal: Any, generation_time: float):
        """更新组别性能指标"""
        metrics = self.group_a_metrics if group == TestGroup.GROUP_A else self.group_b_metrics
        
        # 基础统计
        metrics.total_signals += 1
        metrics.total_generation_time += generation_time
        metrics.avg_generation_time = metrics.total_generation_time / metrics.total_signals
        
        if signal:
            metrics.valid_signals += 1
            
            # 信号置信度
            confidence = getattr(signal, 'confidence', 0.0)
            metrics.avg_confidence = (metrics.avg_confidence * (metrics.valid_signals - 1) + confidence) / metrics.valid_signals
            metrics.max_confidence = max(metrics.max_confidence, confidence)
            
            # 信号类型统计
            signal_type = getattr(signal, 'signal_type', None)
            if signal_type:
                if signal_type.name == 'BUY':
                    metrics.buy_signals += 1
                elif signal_type.name == 'SELL':
                    metrics.sell_signals += 1
                else:
                    metrics.hold_signals += 1
            
            # 信号强度统计
            signal_strength = getattr(signal, 'signal_strength', None)
            if signal_strength:
                strength_key = signal_strength.name if hasattr(signal_strength, 'name') else str(signal_strength)
                metrics.strength_distribution[strength_key] = metrics.strength_distribution.get(strength_key, 0) + 1
            
            # B组特有的增强统计
            if group == TestGroup.GROUP_B and hasattr(signal, 'pattern_confirmation'):
                if signal.pattern_confirmation:
                    metrics.pattern_enhanced_signals += 1
                    # 修复NoneType错误：添加metadata空值检查
                    if hasattr(signal, 'metadata') and signal.metadata and signal.metadata.get('double_pattern_found', False):
                        metrics.double_pattern_found += 1
        
        # 计算衍生指标
        metrics.calculate_derived_metrics()
    
    def get_current_comparison(self) -> Dict[str, Any]:
        """
        📊 获取当前对比结果
        
        Returns:
            详细的对比分析结果  
        """
        try:
            # 基础指标对比
            comparison = {
                'test_info': {
                    'phase': self.current_phase.value,
                    'start_time': self.test_start_time.isoformat() if self.test_start_time else None,
                    'runtime_hours': (datetime.now() - self.test_start_time).total_seconds() / 3600 if self.test_start_time else 0
                },
                'group_a_metrics': self._metrics_to_dict(self.group_a_metrics),
                'group_b_metrics': self._metrics_to_dict(self.group_b_metrics),
                'comparison_analysis': {}
            }
            
            # 对比分析
            a_metrics = self.group_a_metrics
            b_metrics = self.group_b_metrics
            
            if a_metrics.total_signals > 0 and b_metrics.total_signals > 0:
                comparison['comparison_analysis'] = {
                    'signal_generation': {
                        'total_signals_ratio': b_metrics.total_signals / a_metrics.total_signals,
                        'valid_signals_ratio': b_metrics.valid_signals / max(a_metrics.valid_signals, 1),
                        'success_rate_diff': b_metrics.signal_success_rate - a_metrics.signal_success_rate
                    },
                    'quality_metrics': {
                        'confidence_improvement': b_metrics.avg_confidence - a_metrics.avg_confidence,
                        'max_confidence_diff': b_metrics.max_confidence - a_metrics.max_confidence
                    },
                    'performance_metrics': {
                        'speed_ratio': a_metrics.avg_generation_time / max(b_metrics.avg_generation_time, 0.001),
                        'pattern_enhancement_rate': b_metrics.pattern_enhancement_rate,
                        'double_pattern_rate': b_metrics.double_pattern_rate
                    }
                }
                
                # 统计显著性分析（简化版）
                comparison['statistical_analysis'] = self._perform_statistical_analysis()
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"获取对比结果失败: {e}")
            return {'error': str(e)}
    
    def _metrics_to_dict(self, metrics: SignalPerformanceMetrics) -> Dict[str, Any]:
        """将指标对象转换为字典"""
        return {
            'total_signals': metrics.total_signals,
            'valid_signals': metrics.valid_signals,
            'signal_success_rate': metrics.signal_success_rate,
            'avg_confidence': metrics.avg_confidence,
            'max_confidence': metrics.max_confidence,
            'avg_generation_time': metrics.avg_generation_time,
            'buy_signals': metrics.buy_signals,
            'sell_signals': metrics.sell_signals,
            'strength_distribution': metrics.strength_distribution,
            'pattern_enhanced_signals': metrics.pattern_enhanced_signals,
            'double_pattern_found': metrics.double_pattern_found,
            'pattern_enhancement_rate': metrics.pattern_enhancement_rate,
            'double_pattern_rate': metrics.double_pattern_rate
        }
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """执行统计显著性分析（简化版）"""
        a_success_rate = self.group_a_metrics.signal_success_rate
        b_success_rate = self.group_b_metrics.signal_success_rate
        
        # 样本量检查
        min_sample_met = (self.group_a_metrics.total_signals >= self.test_config.min_sample_size and 
                         self.group_b_metrics.total_signals >= self.test_config.min_sample_size)
        
        # 改进程度
        improvement = b_success_rate - a_success_rate
        relative_improvement = improvement / max(a_success_rate, 0.001)
        
        # 是否显著改进
        significant_improvement = (abs(improvement) > self.test_config.improvement_threshold and 
                                 min_sample_met)
        
        return {
            'min_sample_size_met': min_sample_met,
            'absolute_improvement': improvement,
            'relative_improvement': relative_improvement,
            'significant_improvement': significant_improvement,
            'recommendation': self._generate_recommendation(improvement, significant_improvement, min_sample_met)
        }
    
    def _generate_recommendation(self, improvement: float, significant: bool, sufficient_sample: bool) -> str:
        """生成测试建议"""
        if not sufficient_sample:
            return "样本量不足，需要继续测试"
        
        if significant and improvement > 0:
            return "B组显著优于A组，建议采用增强版信号生成器"
        elif significant and improvement < 0:
            return "A组显著优于B组，建议保持原版信号生成器"
        else:
            return "两组差异不显著，建议延长测试时间或保持现状"
    
    async def check_emergency_stop_conditions(self) -> bool:
        """检查紧急停止条件"""
        if self.group_b_metrics.total_signals < 10:  # 样本太少
            return False
        
        # 检查B组性能是否严重下降
        a_success_rate = self.group_a_metrics.signal_success_rate
        b_success_rate = self.group_b_metrics.signal_success_rate
        
        if a_success_rate > 0:
            performance_degradation = (a_success_rate - b_success_rate) / a_success_rate
            if performance_degradation > self.test_config.emergency_stop_threshold:
                self.logger.warning(f"⚠️ 检测到严重性能下降: {performance_degradation:.2%}")
                return True
        
        return False
    
    def export_test_results(self, filepath: str) -> bool:
        """
        📤 导出测试结果
        
        Args:
            filepath: 导出文件路径
            
        Returns:
            导出是否成功
        """
        try:
            results = {
                'test_config': {
                    'duration_days': self.test_config.test_duration_days,
                    'traffic_split': self.test_config.traffic_split,
                    'success_metrics': self.test_config.success_metrics
                },
                'test_summary': {
                    'start_time': self.test_start_time.isoformat() if self.test_start_time else None,
                    'end_time': self.test_end_time.isoformat() if self.test_end_time else None,
                    'current_phase': self.current_phase.value
                },
                'results': self.get_current_comparison(),
                'detailed_data': {
                    'group_a_signals': len(self.detailed_results['group_a']),
                    'group_b_signals': len(self.detailed_results['group_b'])
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 测试结果已导出: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 导出测试结果失败: {e}")
            return False
    
    def get_test_status(self) -> Dict[str, Any]:
        """获取测试状态摘要"""
        return {
            'phase': self.current_phase.value,
            'runtime_info': {
                'start_time': self.test_start_time.isoformat() if self.test_start_time else None,
                'runtime_hours': (datetime.now() - self.test_start_time).total_seconds() / 3600 if self.test_start_time else 0,
                'remaining_hours': (self.test_end_time - datetime.now()).total_seconds() / 3600 if self.test_end_time else 0
            },
            'sample_sizes': {
                'group_a': self.group_a_metrics.total_signals,
                'group_b': self.group_b_metrics.total_signals
            },
            'quick_comparison': {
                'group_a_success_rate': self.group_a_metrics.signal_success_rate,
                'group_b_success_rate': self.group_b_metrics.signal_success_rate,
                'group_b_pattern_rate': self.group_b_metrics.pattern_enhancement_rate
            }
        } 