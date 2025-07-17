#!/usr/bin/env python3
"""
增强版MACD背离检测器测试
验证专家建议的优化功能
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.macd_divergence_detector import (
    MACDDivergenceDetector, 
    DivergenceDetectionConfig,
    EnhancedPeakDetector,
    ConsecutiveDivergenceDetector,
    Peak,
    DivergenceType,
    SignalStrength
)
from core.technical_indicators import MACDResult, TechnicalIndicatorCalculator
from utils.logger import get_logger


class TestEnhancedMACDDivergence(unittest.TestCase):
    """增强版MACD背离检测器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = DivergenceDetectionConfig(
            lookback_period=50,
            min_peak_distance=5,
            prominence_multiplier=0.3,  # 降低以便测试
            min_divergence_gap=0.05,   # 降低以便测试
            min_consecutive_count=2,
            time_alignment_tolerance=3,
            max_time_window=20
        )
        
        self.detector = MACDDivergenceDetector(self.config)
        self.enhanced_peak_detector = EnhancedPeakDetector(self.config)
        self.consecutive_detector = ConsecutiveDivergenceDetector(self.config)
        
        self.logger = get_logger(__name__)
    
    def generate_test_data_with_divergence(self, length: int = 60) -> tuple:
        """生成包含背离的测试数据"""
        # 生成基础趋势
        x = np.linspace(0, 10, length)
        
        # 创建看跌背离场景：价格上涨，MACD下降
        prices = []
        macd_values = []
        
        for i in range(length):
            # 价格整体上涨趋势，但在特定位置创建峰值
            if i < 20:
                price = 100 + i * 0.5 + np.random.normal(0, 0.1)
            elif i < 40:
                # 第一个价格峰值
                price = 110 + (i - 20) * 0.3 + np.random.normal(0, 0.1)
            else:
                # 第二个价格峰值（更高）
                price = 116 + (i - 40) * 0.2 + np.random.normal(0, 0.1)
            
            prices.append(price)
            
            # MACD值：第一个峰值高，第二个峰值低（背离）
            if i < 20:
                macd = 0.1 + i * 0.01 + np.random.normal(0, 0.005)
            elif i < 40:
                # 第一个MACD峰值
                macd = 0.3 + (i - 20) * 0.02 + np.random.normal(0, 0.005)
            else:
                # 第二个MACD峰值（更低，形成背离）
                macd = 0.35 - (i - 40) * 0.01 + np.random.normal(0, 0.005)
            
            macd_values.append(macd)
        
        # 生成MACD结果
        macd_results = []
        for i in range(length):
            signal_line = macd_values[i] * 0.8
            histogram = macd_values[i] - signal_line
            macd_results.append(MACDResult(
                macd_line=macd_values[i],
                signal_line=signal_line,
                histogram=histogram,
                fast_ema=macd_values[i] * 1.1,
                slow_ema=macd_values[i] * 0.9,
                timestamp=datetime.now() - timedelta(hours=length - i)
            ))
        
        return prices, macd_results
    
    def test_enhanced_peak_detection(self):
        """测试增强版峰值检测"""
        print("\n=== 测试增强版峰值检测 ===")
        
        # 生成测试数据
        prices, macd_results = self.generate_test_data_with_divergence(60)
        
        # 使用增强版峰值检测
        peaks, valleys = self.enhanced_peak_detector.detect_peaks_with_scipy(prices)
        
        print(f"检测到 {len(peaks)} 个峰值, {len(valleys)} 个谷值")
        
        # 验证结果
        self.assertGreater(len(peaks), 0, "应该检测到至少一个峰值")
        self.assertGreater(len(valleys), 0, "应该检测到至少一个谷值")
        
        # 验证峰值排序
        peak_indices = [p.index for p in peaks]
        self.assertEqual(peak_indices, sorted(peak_indices), "峰值应该按时间排序")
        
        # 打印峰值信息
        for i, peak in enumerate(peaks):
            print(f"峰值 {i+1}: 索引={peak.index}, 值={peak.value:.3f}, 类型={peak.peak_type}")
    
    def test_consecutive_divergence_detection(self):
        """测试连续背离检测"""
        print("\n=== 测试连续背离检测 ===")
        
        # 生成测试数据
        prices, macd_results = self.generate_test_data_with_divergence(60)
        
        # 提取MACD直方图
        macd_histograms = [r.histogram for r in macd_results]
        
        # 检测峰值
        price_peaks, _ = self.enhanced_peak_detector.detect_peaks_with_scipy(prices)
        macd_peaks, _ = self.enhanced_peak_detector.detect_peaks_with_scipy(macd_histograms)
        
        print(f"价格峰值: {len(price_peaks)}, MACD峰值: {len(macd_peaks)}")
        
        # 检测连续看跌背离
        bearish_patterns = self.consecutive_detector.detect_consecutive_divergence(
            price_peaks, macd_peaks, prices, macd_histograms, is_bearish=True
        )
        
        print(f"检测到 {len(bearish_patterns)} 个连续看跌背离模式")
        
        # 验证结果
        if len(bearish_patterns) > 0:
            for i, pattern in enumerate(bearish_patterns):
                print(f"背离模式 {i+1}:")
                print(f"  类型: {pattern['type']}")
                print(f"  强度: {pattern['strength']:.3f}")
                print(f"  置信度: {pattern['confidence']:.3f}")
                print(f"  背离次数: {pattern['divergence_count']}")
                print(f"  价格序列数: {len(pattern['price_sequence'])}")
        
        # 基本验证
        self.assertIsInstance(bearish_patterns, list, "应该返回列表")
    
    def test_enhanced_divergence_detection(self):
        """测试增强版背离检测"""
        print("\n=== 测试增强版背离检测 ===")
        
        # 生成测试数据
        prices, macd_results = self.generate_test_data_with_divergence(60)
        
        # 使用增强版检测
        signals = self.detector.detect_divergence_enhanced(
            prices, macd_results, symbol="BTC/USDT", timeframe="1h"
        )
        
        print(f"检测到 {len(signals)} 个背离信号")
        
        # 验证结果
        self.assertIsInstance(signals, list, "应该返回信号列表")
        
        # 打印信号详情
        for i, signal in enumerate(signals):
            print(f"信号 {i+1}:")
            print(f"  类型: {signal.divergence_type.value}")
            print(f"  强度: {signal.signal_strength.value}")
            print(f"  置信度: {signal.confidence:.3f}")
            print(f"  风险回报比: {signal.risk_reward_ratio:.2f}")
            print(f"  入场价格: {signal.entry_price:.3f}")
            print(f"  止损价格: {signal.stop_loss:.3f}")
            print(f"  止盈价格: {signal.take_profit:.3f}")
    
    def test_config_parameters(self):
        """测试配置参数"""
        print("\n=== 测试配置参数 ===")
        
        # 测试不同的配置参数
        strict_config = DivergenceDetectionConfig(
            lookback_period=30,
            min_peak_distance=8,
            prominence_multiplier=0.8,
            min_divergence_gap=0.15,
            min_consecutive_count=3,
            time_alignment_tolerance=1
        )
        
        lenient_config = DivergenceDetectionConfig(
            lookback_period=80,
            min_peak_distance=3,
            prominence_multiplier=0.2,
            min_divergence_gap=0.05,
            min_consecutive_count=2,
            time_alignment_tolerance=5
        )
        
        # 生成测试数据
        prices, macd_results = self.generate_test_data_with_divergence(60)
        
        # 使用严格配置
        strict_detector = MACDDivergenceDetector(strict_config)
        strict_signals = strict_detector.detect_divergence_enhanced(
            prices, macd_results, symbol="TEST"
        )
        
        # 使用宽松配置
        lenient_detector = MACDDivergenceDetector(lenient_config)
        lenient_signals = lenient_detector.detect_divergence_enhanced(
            prices, macd_results, symbol="TEST"
        )
        
        print(f"严格配置检测到 {len(strict_signals)} 个信号")
        print(f"宽松配置检测到 {len(lenient_signals)} 个信号")
        
        # 验证：宽松配置应该检测到更多或相等数量的信号
        self.assertGreaterEqual(len(lenient_signals), len(strict_signals), 
                               "宽松配置应该检测到更多信号")
    
    def test_performance_benchmark(self):
        """测试性能基准"""
        print("\n=== 测试性能基准 ===")
        
        # 生成大量测试数据
        large_prices, large_macd_results = self.generate_test_data_with_divergence(200)
        
        # 多次运行测试
        import time
        start_time = time.time()
        
        iterations = 50
        total_signals = 0
        
        for i in range(iterations):
            signals = self.detector.detect_divergence_enhanced(
                large_prices, large_macd_results, symbol="PERF_TEST"
            )
            total_signals += len(signals)
        
        end_time = time.time()
        
        # 计算性能指标
        total_time = end_time - start_time
        avg_time_per_run = total_time / iterations
        data_points_per_second = (200 * iterations) / total_time
        
        print(f"性能测试结果:")
        print(f"  总运行次数: {iterations}")
        print(f"  总耗时: {total_time:.3f}秒")
        print(f"  平均每次: {avg_time_per_run:.3f}秒")
        print(f"  处理速度: {data_points_per_second:.0f} 数据点/秒")
        print(f"  总信号数: {total_signals}")
        
        # 验证性能要求
        self.assertLess(avg_time_per_run, 0.1, "平均检测时间应该小于0.1秒")
        self.assertGreater(data_points_per_second, 1000, "处理速度应该大于1000点/秒")
    
    def test_edge_cases(self):
        """测试边界条件"""
        print("\n=== 测试边界条件 ===")
        
        # 测试空数据
        empty_signals = self.detector.detect_divergence_enhanced([], [], symbol="EMPTY")
        self.assertEqual(len(empty_signals), 0, "空数据应该返回空信号列表")
        
        # 测试数据不足
        small_prices = [100, 101, 102]
        small_macd = [MACDResult(
            macd_line=0.1, 
            signal_line=0.08, 
            histogram=0.02,
            fast_ema=0.11,
            slow_ema=0.09,
            timestamp=datetime.now()
        ) for _ in range(3)]
        small_signals = self.detector.detect_divergence_enhanced(
            small_prices, small_macd, symbol="SMALL"
        )
        self.assertEqual(len(small_signals), 0, "数据不足应该返回空信号列表")
        
        # 测试单调数据
        monotonic_prices = list(range(100, 160))
        monotonic_macd = [MACDResult(
            macd_line=i*0.01, 
            signal_line=i*0.008, 
            histogram=i*0.002,
            fast_ema=i*0.011,
            slow_ema=i*0.009,
            timestamp=datetime.now()
        ) for i in range(60)]
        monotonic_signals = self.detector.detect_divergence_enhanced(
            monotonic_prices, monotonic_macd, symbol="MONOTONIC"
        )
        
        print(f"单调数据检测到 {len(monotonic_signals)} 个信号")
        
        # 测试高噪音数据
        noisy_prices = [100 + np.random.normal(0, 10) for _ in range(60)]
        noisy_macd = [MACDResult(
            macd_line=np.random.normal(0, 0.1), 
            signal_line=np.random.normal(0, 0.08), 
            histogram=np.random.normal(0, 0.02),
            fast_ema=np.random.normal(0, 0.11),
            slow_ema=np.random.normal(0, 0.09),
            timestamp=datetime.now()
        ) for _ in range(60)]
        noisy_signals = self.detector.detect_divergence_enhanced(
            noisy_prices, noisy_macd, symbol="NOISY"
        )
        
        print(f"高噪音数据检测到 {len(noisy_signals)} 个信号")
        
        print("边界条件测试完成")


def run_expert_validation_test():
    """运行专家验证测试"""
    print("\n" + "="*60)
    print("专家建议验证测试")
    print("="*60)
    
    # 创建检测器
    config = DivergenceDetectionConfig(
        lookback_period=50,
        min_peak_distance=5,
        prominence_multiplier=0.5,
        min_divergence_gap=0.1,
        min_consecutive_count=2,
        time_alignment_tolerance=2
    )
    
    detector = MACDDivergenceDetector(config)
    
    # 生成符合专家建议的测试数据
    def generate_expert_test_data():
        """生成符合专家建议的测试数据"""
        length = 60
        prices = []
        macd_values = []
        
        # 创建明显的连续背离模式
        for i in range(length):
            # 价格：三个递增的峰值
            if i < 15:
                price = 100 + i * 0.2 + np.random.normal(0, 0.05)
            elif i < 30:
                price = 103 + (i - 15) * 0.4 + np.random.normal(0, 0.05)  # 第一个峰值
            elif i < 45:
                price = 109 + (i - 30) * 0.3 + np.random.normal(0, 0.05)  # 第二个峰值
            else:
                price = 114 + (i - 45) * 0.2 + np.random.normal(0, 0.05)  # 第三个峰值
            
            prices.append(price)
            
            # MACD：三个递减的峰值（背离）
            if i < 15:
                macd = 0.05 + i * 0.005 + np.random.normal(0, 0.002)
            elif i < 30:
                macd = 0.125 + (i - 15) * 0.01 + np.random.normal(0, 0.002)  # 第一个峰值
            elif i < 45:
                macd = 0.12 - (i - 30) * 0.002 + np.random.normal(0, 0.002)  # 第二个峰值（更低）
            else:
                macd = 0.11 - (i - 45) * 0.003 + np.random.normal(0, 0.002)  # 第三个峰值（最低）
            
            macd_values.append(macd)
        
        # 生成MACD结果
        macd_results = []
        for i in range(length):
            signal_line = macd_values[i] * 0.8
            histogram = macd_values[i] - signal_line
            macd_results.append(MACDResult(
                macd_line=macd_values[i],
                signal_line=signal_line,
                histogram=histogram,
                fast_ema=macd_values[i] * 1.1,
                slow_ema=macd_values[i] * 0.9,
                timestamp=datetime.now() - timedelta(hours=length - i)
            ))
        
        return prices, macd_results
    
    # 生成测试数据
    prices, macd_results = generate_expert_test_data()
    
    # 运行检测
    signals = detector.detect_divergence_enhanced(
        prices, macd_results, symbol="EXPERT_TEST", timeframe="15m"
    )
    
    print(f"专家验证测试结果:")
    print(f"  检测到信号数量: {len(signals)}")
    
    if len(signals) > 0:
        for i, signal in enumerate(signals):
            print(f"  信号 {i+1}:")
            print(f"    类型: {signal.divergence_type.value}")
            print(f"    强度: {signal.signal_strength.value}")
            print(f"    置信度: {signal.confidence:.3f}")
            print(f"    风险回报比: {signal.risk_reward_ratio:.2f}")
            print(f"    元数据: {signal.metadata}")
    else:
        print("  ⚠️  未检测到背离信号")
    
    print("\n专家验证测试完成")


if __name__ == "__main__":
    # 运行专家验证测试
    run_expert_validation_test()
    
    # 运行单元测试
    print("\n" + "="*60)
    print("单元测试")
    print("="*60)
    
    unittest.main(verbosity=2) 