#!/usr/bin/env python3
"""
MACD背离检测器使用示例
简化版本，方便用户理解和使用
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complete_macd_divergence_detector import (
    CompleteMACDDivergenceDetector,
    DivergenceDetectionConfig,
    MACDResult,
    create_optimized_config,
    detect_macd_divergence
)
from datetime import datetime, timedelta
import numpy as np


def create_sample_data():
    """创建示例数据"""
    # 模拟价格数据：先下跌后上涨的趋势
    prices = []
    base_price = 100
    
    # 第一段：下跌趋势
    for i in range(50):
        noise = np.random.normal(0, 0.5)
        trend = -0.2 * i  # 下跌趋势
        price = base_price + trend + noise
        prices.append(max(80, price))  # 确保价格不低于80
    
    # 第二段：上涨趋势
    for i in range(50, 100):
        noise = np.random.normal(0, 0.5)
        trend = 0.3 * (i - 50)  # 上涨趋势
        price = prices[49] + trend + noise
        prices.append(price)
    
    return prices


def create_macd_data(prices):
    """创建MACD数据"""
    macd_results = []
    
    for i, price in enumerate(prices):
        # 简化的MACD计算
        if i < 26:
            macd_line = 0
            signal_line = 0
            histogram = 0
            fast_ema = price
            slow_ema = price
        else:
            # 模拟MACD计算
            fast_ema = np.mean(prices[i-12:i+1])
            slow_ema = np.mean(prices[i-26:i+1])
            macd_line = fast_ema - slow_ema
            
            if i < 35:
                signal_line = macd_line
            else:
                signal_line = np.mean([r.macd_line for r in macd_results[i-9:i]])
            
            histogram = macd_line - signal_line
        
        macd_results.append(MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            timestamp=datetime.now() - timedelta(hours=len(prices)-i)
        ))
    
    return macd_results


def main():
    """主函数"""
    print("🚀 MACD背离检测器使用示例")
    print("=" * 50)
    
    # 1. 创建示例数据
    print("📊 创建示例数据...")
    prices = create_sample_data()
    macd_results = create_macd_data(prices)
    
    print(f"✅ 生成 {len(prices)} 个价格数据点")
    print(f"   价格范围: ${min(prices):.2f} - ${max(prices):.2f}")
    
    # 2. 配置检测器
    print("\n⚙️ 配置检测器...")
    config = create_optimized_config("crypto")
    detector = CompleteMACDDivergenceDetector(config)
    
    print(f"✅ 使用优化配置:")
    print(f"   检测窗口: {config.lookback_period}")
    print(f"   峰值间距: {config.min_peak_distance}")
    print(f"   噪音过滤: {config.prominence_multiplier}")
    print(f"   背离阈值: {config.min_divergence_gap}")
    
    # 3. 执行背离检测
    print("\n🔍 执行背离检测...")
    signals = detector.detect_divergence(
        prices=prices,
        macd_results=macd_results,
        symbol="DEMO",
        timeframe="1h"
    )
    
    print(f"✅ 检测完成，发现 {len(signals)} 个背离信号")
    
    # 4. 分析结果
    print("\n📈 信号分析:")
    if signals:
        for i, signal in enumerate(signals):
            print(f"\n  信号 {i+1}:")
            print(f"    类型: {signal.divergence_type.value}")
            print(f"    强度: {signal.signal_strength.value}")
            print(f"    置信度: {signal.confidence:.3f}")
            print(f"    风险回报比: {signal.risk_reward_ratio:.2f}")
            print(f"    入场价: ${signal.entry_price:.2f}")
            print(f"    止损价: ${signal.stop_loss:.2f}")
            print(f"    止盈价: ${signal.take_profit:.2f}")
            print(f"    预期收益: {signal.expected_return:.2f}%")
    else:
        print("  未检测到背离信号")
    
    # 5. 统计信息
    print("\n📊 检测统计:")
    stats = detector.get_statistics()
    print(f"  总检测次数: {stats['total_detections']}")
    print(f"  总信号数: {stats['total_signals']}")
    print(f"  信号率: {stats['signal_rate']:.2%}")
    
    # 6. 便捷函数示例
    print("\n🛠️ 便捷函数示例:")
    quick_signals = detect_macd_divergence(
        prices=prices,
        macd_results=macd_results,
        symbol="DEMO",
        timeframe="1h",
        config=config
    )
    print(f"  便捷函数检测到: {len(quick_signals)} 个信号")
    
    print("\n🎉 示例完成！")


if __name__ == "__main__":
    main() 