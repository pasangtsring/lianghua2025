#!/usr/bin/env python3
"""
MACD背离检测器参数优化脚本
根据专家建议调整和测试最优参数
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.macd_divergence_detector import (
    MACDDivergenceDetector, 
    DivergenceDetectionConfig,
    EnhancedPeakDetector
)
from core.technical_indicators import MACDResult
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_realistic_divergence_data(length: int = 80) -> Tuple[List[float], List[MACDResult]]:
    """生成更真实的背离数据"""
    prices = []
    macd_values = []
    
    # 创建明显的背离模式
    for i in range(length):
        # 价格：波动上升趋势，包含明显的三个峰值
        if i < 20:
            # 第一阶段：缓慢上升
            price = 100 + i * 0.8 + np.sin(i * 0.3) * 2 + np.random.normal(0, 0.3)
        elif i < 40:
            # 第二阶段：第一个峰值
            peak_height = 8 + 4 * np.sin((i - 20) * 0.2)
            price = 116 + peak_height + np.random.normal(0, 0.4)
        elif i < 60:
            # 第三阶段：第二个峰值（更高）
            peak_height = 10 + 6 * np.sin((i - 40) * 0.15)
            price = 125 + peak_height + np.random.normal(0, 0.5)
        else:
            # 第四阶段：第三个峰值（最高）
            peak_height = 12 + 8 * np.sin((i - 60) * 0.1)
            price = 135 + peak_height + np.random.normal(0, 0.6)
        
        prices.append(price)
        
        # MACD：递减的峰值（背离）
        if i < 20:
            macd = 0.5 + i * 0.02 + np.sin(i * 0.2) * 0.1 + np.random.normal(0, 0.01)
        elif i < 40:
            # 第一个MACD峰值
            macd = 0.9 + (i - 20) * 0.01 + np.sin((i - 20) * 0.3) * 0.05 + np.random.normal(0, 0.015)
        elif i < 60:
            # 第二个MACD峰值（更低）
            macd = 1.0 - (i - 40) * 0.015 + np.sin((i - 40) * 0.25) * 0.03 + np.random.normal(0, 0.012)
        else:
            # 第三个MACD峰值（最低）
            macd = 0.8 - (i - 60) * 0.02 + np.sin((i - 60) * 0.2) * 0.02 + np.random.normal(0, 0.01)
        
        macd_values.append(macd)
    
    # 生成MACD结果
    macd_results = []
    for i in range(length):
        signal_line = macd_values[i] * 0.85
        histogram = macd_values[i] - signal_line
        macd_results.append(MACDResult(
            macd_line=macd_values[i],
            signal_line=signal_line,
            histogram=histogram,
            fast_ema=macd_values[i] * 1.15,
            slow_ema=macd_values[i] * 0.85,
            timestamp=datetime.now() - timedelta(hours=length - i)
        ))
    
    return prices, macd_results


def test_parameter_combination(config: DivergenceDetectionConfig, 
                             prices: List[float], 
                             macd_results: List[MACDResult]) -> Dict:
    """测试参数组合的效果"""
    detector = MACDDivergenceDetector(config)
    
    # 测试峰值检测
    enhanced_detector = EnhancedPeakDetector(config)
    price_peaks, price_valleys = enhanced_detector.detect_peaks_with_scipy(prices)
    
    # 测试背离检测
    signals = detector.detect_divergence_enhanced(prices, macd_results, symbol="TEST")
    
    # 计算结果指标
    result = {
        'config': config,
        'price_peaks': len(price_peaks),
        'price_valleys': len(price_valleys),
        'signals': len(signals),
        'total_peaks': len(price_peaks) + len(price_valleys),
        'signal_details': [
            {
                'type': s.divergence_type.value,
                'confidence': s.confidence,
                'risk_reward': s.risk_reward_ratio,
                'strength': s.signal_strength.value
            } for s in signals
        ]
    }
    
    return result


def optimize_parameters():
    """优化参数组合"""
    print("=== MACD背离检测器参数优化 ===")
    print("基于专家建议进行参数调优...\n")
    
    # 生成测试数据
    prices, macd_results = generate_realistic_divergence_data(80)
    
    # 定义参数测试范围
    parameter_combinations = [
        # 专家建议的原始参数
        {
            'name': '专家建议（原始）',
            'config': DivergenceDetectionConfig(
                lookback_period=50,
                min_peak_distance=5,
                prominence_multiplier=0.5,
                min_divergence_gap=0.1,
                min_consecutive_count=2,
                time_alignment_tolerance=2
            )
        },
        # 调整1：降低prominence阈值
        {
            'name': '降低噪音过滤',
            'config': DivergenceDetectionConfig(
                lookback_period=50,
                min_peak_distance=5,
                prominence_multiplier=0.3,  # 降低
                min_divergence_gap=0.1,
                min_consecutive_count=2,
                time_alignment_tolerance=2
            )
        },
        # 调整2：更宽松的背离阈值
        {
            'name': '宽松背离阈值',
            'config': DivergenceDetectionConfig(
                lookback_period=50,
                min_peak_distance=5,
                prominence_multiplier=0.3,
                min_divergence_gap=0.05,  # 降低
                min_consecutive_count=2,
                time_alignment_tolerance=2
            )
        },
        # 调整3：增加时间容忍度
        {
            'name': '增加时间容忍度',
            'config': DivergenceDetectionConfig(
                lookback_period=50,
                min_peak_distance=5,
                prominence_multiplier=0.3,
                min_divergence_gap=0.05,
                min_consecutive_count=2,
                time_alignment_tolerance=5  # 增加
            )
        },
        # 调整4：减少峰值间距
        {
            'name': '减少峰值间距',
            'config': DivergenceDetectionConfig(
                lookback_period=50,
                min_peak_distance=3,  # 减少
                prominence_multiplier=0.3,
                min_divergence_gap=0.05,
                min_consecutive_count=2,
                time_alignment_tolerance=5
            )
        },
        # 调整5：增加检测窗口
        {
            'name': '增加检测窗口',
            'config': DivergenceDetectionConfig(
                lookback_period=70,  # 增加
                min_peak_distance=3,
                prominence_multiplier=0.3,
                min_divergence_gap=0.05,
                min_consecutive_count=2,
                time_alignment_tolerance=5
            )
        },
        # 调整6：最优化组合
        {
            'name': '最优化组合',
            'config': DivergenceDetectionConfig(
                lookback_period=70,
                min_peak_distance=3,
                prominence_multiplier=0.2,  # 进一步降低
                min_divergence_gap=0.03,   # 进一步降低
                min_consecutive_count=2,
                time_alignment_tolerance=5
            )
        }
    ]
    
    # 测试所有参数组合
    results = []
    for combo in parameter_combinations:
        print(f"测试配置: {combo['name']}")
        result = test_parameter_combination(combo['config'], prices, macd_results)
        result['name'] = combo['name']
        results.append(result)
        
        print(f"  峰值检测: {result['price_peaks']} 个价格峰值, {result['price_valleys']} 个价格谷值")
        print(f"  背离信号: {result['signals']} 个")
        if result['signal_details']:
            for i, signal in enumerate(result['signal_details']):
                print(f"    信号{i+1}: {signal['type']}, 置信度={signal['confidence']:.3f}, 风险回报={signal['risk_reward']:.2f}")
        print()
    
    # 分析结果
    print("=== 参数优化结果分析 ===")
    print(f"{'配置名称':<15} {'峰值总数':<8} {'背离信号':<8} {'有效性':<8}")
    print("-" * 50)
    
    for result in results:
        effectiveness = "优秀" if result['signals'] > 0 and result['total_peaks'] > 2 else \
                       "良好" if result['total_peaks'] > 0 else "待改进"
        print(f"{result['name']:<15} {result['total_peaks']:<8} {result['signals']:<8} {effectiveness:<8}")
    
    # 推荐最优配置
    best_result = max(results, key=lambda x: x['signals'] + x['total_peaks'] / 5)
    print(f"\n推荐最优配置: {best_result['name']}")
    print("推荐参数:")
    config = best_result['config']
    print(f"  lookback_period: {config.lookback_period}")
    print(f"  min_peak_distance: {config.min_peak_distance}")
    print(f"  prominence_multiplier: {config.prominence_multiplier}")
    print(f"  min_divergence_gap: {config.min_divergence_gap}")
    print(f"  min_consecutive_count: {config.min_consecutive_count}")
    print(f"  time_alignment_tolerance: {config.time_alignment_tolerance}")
    
    return best_result


def demonstrate_expert_improvements():
    """演示专家建议的改进效果"""
    print("\n=== 专家建议改进效果演示 ===")
    
    # 生成测试数据
    prices, macd_results = generate_realistic_divergence_data(80)
    
    # 原始配置（模拟专家批评的版本）
    original_config = DivergenceDetectionConfig(
        lookback_period=30,
        min_peak_distance=8,
        prominence_multiplier=0.8,  # 过严
        min_divergence_gap=0.2,     # 过严
        min_consecutive_count=3,     # 过严
        time_alignment_tolerance=1   # 过严
    )
    
    # 专家建议的优化配置
    optimized_config = DivergenceDetectionConfig(
        lookback_period=70,
        min_peak_distance=3,
        prominence_multiplier=0.2,
        min_divergence_gap=0.03,
        min_consecutive_count=2,
        time_alignment_tolerance=5
    )
    
    # 测试对比
    print("原始配置（专家批评的版本）:")
    original_result = test_parameter_combination(original_config, prices, macd_results)
    print(f"  峰值检测: {original_result['total_peaks']} 个")
    print(f"  背离信号: {original_result['signals']} 个")
    
    print("\n专家建议优化后:")
    optimized_result = test_parameter_combination(optimized_config, prices, macd_results)
    print(f"  峰值检测: {optimized_result['total_peaks']} 个")
    print(f"  背离信号: {optimized_result['signals']} 个")
    
    # 计算改进效果
    peak_improvement = optimized_result['total_peaks'] - original_result['total_peaks']
    signal_improvement = optimized_result['signals'] - original_result['signals']
    
    print(f"\n改进效果:")
    print(f"  峰值检测提升: +{peak_improvement} 个")
    print(f"  背离信号提升: +{signal_improvement} 个")
    print(f"  总体评价: {'显著改进' if signal_improvement > 0 else '需要进一步调优'}")


if __name__ == "__main__":
    try:
        # 运行参数优化
        best_config = optimize_parameters()
        
        # 演示专家改进效果
        demonstrate_expert_improvements()
        
        print("\n=== 专家建议优化完成 ===")
        print("建议将最优参数应用到配置文件中")
        
    except Exception as e:
        logger.error(f"参数优化失败: {e}")
        import traceback
        traceback.print_exc() 