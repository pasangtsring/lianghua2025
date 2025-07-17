#!/usr/bin/env python3
"""
信号生成器简化测试
不依赖外部API，使用模拟数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from config.config_manager import ConfigManager
from core.signal_generator import SignalGenerator, SignalType, SignalStrength

def create_mock_kline_data():
    """创建模拟K线数据"""
    import random
    import time
    
    klines = []
    base_price = 50000
    
    for i in range(200):
        timestamp = int(time.time() * 1000) - (200 - i) * 3600000  # 1小时间隔
        
        # 模拟价格走势
        price_change = random.uniform(-0.02, 0.02)  # -2%到2%的变化
        base_price = base_price * (1 + price_change)
        
        open_price = base_price
        high_price = base_price * (1 + random.uniform(0, 0.015))
        low_price = base_price * (1 - random.uniform(0, 0.015))
        close_price = base_price * (1 + random.uniform(-0.01, 0.01))
        volume = random.uniform(1000, 5000)
        
        kline = {
            'timestamp': timestamp,
            'open': str(open_price),
            'high': str(high_price),
            'low': str(low_price),
            'close': str(close_price),
            'volume': str(volume)
        }
        
        klines.append(kline)
    
    return klines

def test_signal_generator_simple():
    """测试信号生成器（简化版）"""
    print("🧪 测试信号生成器模块（简化版）")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化信号生成器
        signal_generator = SignalGenerator(config)
        print("✅ 信号生成器初始化成功")
        
        # 3. 创建模拟数据
        kline_data = create_mock_kline_data()
        print(f"✅ 创建模拟K线数据成功: {len(kline_data)} 条")
        
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
        
        # 7. 测试多次信号生成
        print("\n🔄 测试多次信号生成:")
        for i in range(3):
            # 修改最后几个K线数据以模拟不同市场状态
            test_data = kline_data.copy()
            for j in range(5):
                idx = -(j+1)
                base_price = float(test_data[idx]['close'])
                if i == 0:  # 上涨趋势
                    new_price = base_price * (1 + 0.01 * (j+1))
                elif i == 1:  # 下跌趋势
                    new_price = base_price * (1 - 0.01 * (j+1))
                else:  # 横盘
                    new_price = base_price * (1 + 0.002 * (j+1) * (-1)**(j+1))
                
                test_data[idx]['close'] = str(new_price)
            
            signal = signal_generator.generate_signal(test_data)
            if signal:
                print(f"   测试 {i+1}: {signal.signal_type.value}, 置信度: {signal.confidence:.2f}")
            else:
                print(f"   测试 {i+1}: 无信号")
        
        print("\n✅ 信号生成器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_generator_simple()
    sys.exit(0 if success else 1) 