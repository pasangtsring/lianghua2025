#!/usr/bin/env python3
"""
信号生成器测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from config.config_manager import ConfigManager
from core.signal_generator import SignalGenerator, SignalType, SignalStrength
from data.advanced_data_fetcher import AdvancedDataFetcher

def test_signal_generator():
    """测试信号生成器"""
    print("🧪 测试信号生成器模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化信号生成器
        signal_generator = SignalGenerator(config)
        print("✅ 信号生成器初始化成功")
        
        # 3. 获取测试数据
        from data.api_client import BinanceAPIClient
        api_client = BinanceAPIClient(config)
        data_fetcher = AdvancedDataFetcher(config, api_client)
        print("✅ 数据获取器初始化成功")
        
        # 获取K线数据
        kline_data = data_fetcher.get_klines_sync('BTCUSDT', '1h', limit=200)
        if not kline_data:
            print("❌ 获取K线数据失败")
            return False
        
        print(f"✅ 获取K线数据成功: {len(kline_data)} 条")
        
        # 4. 测试信号生成
        print("\n🔍 测试信号生成功能:")
        signal = signal_generator.generate_signal(kline_data)
        
        if signal:
            print("✅ 信号生成成功")
            print(f"   信号类型: {signal.signal_type.value}")
            print(f"   信号强度: {signal.signal_strength.value}")
            print(f"   置信度: {signal.confidence:.2f}")
            print(f"   入场价格: {signal.entry_price:.2f}")
            print(f"   止损价格: {signal.stop_loss_price:.2f}")
            print(f"   止盈价格: {signal.take_profit_price:.2f}")
            print(f"   风险回报比: {signal.risk_reward_ratio:.2f}")
            print(f"   技术评分: {signal.technical_score:.2f}")
            print(f"   市场状态: {signal.market_condition}")
            print(f"   信号原因: {', '.join(signal.reasons)}")
        else:
            print("⚠️  当前无交易信号")
        
        # 5. 测试信号摘要
        if signal:
            print("\n📊 测试信号摘要:")
            summary = signal_generator.get_signal_summary(signal)
            print("✅ 信号摘要生成成功")
            
            for key, value in summary.items():
                print(f"   {key}: {value}")
        
        # 6. 测试组件功能
        print("\n🔧 测试组件分析功能:")
        
        # 测试MACD分析
        macd_result = signal_generator.analyze_macd_divergence(kline_data)
        print(f"   MACD分析: {macd_result['has_signal']}, 评分: {macd_result['score']:.2f}")
        
        # 测试技术指标分析
        technical_result = signal_generator.analyze_technical_indicators(kline_data)
        print(f"   技术指标: 评分: {technical_result['score']:.2f}, 趋势: {technical_result['trend']}")
        
        # 测试形态分析
        pattern_result = signal_generator.analyze_patterns(kline_data)
        print(f"   形态分析: {pattern_result['has_pattern']}, 评分: {pattern_result['score']:.2f}")
        
        # 测试周期分析
        cycle_result = signal_generator.analyze_cycle(kline_data)
        print(f"   周期分析: {cycle_result['phase']}, 评分: {cycle_result['score']:.2f}")
        
        print("\n✅ 信号生成器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_generator()
    sys.exit(0 if success else 1) 