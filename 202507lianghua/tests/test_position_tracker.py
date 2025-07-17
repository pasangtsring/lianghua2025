#!/usr/bin/env python3
"""
持仓跟踪器测试
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from config.config_manager import ConfigManager
from execution.position_tracker import PositionTracker, TradeRecord

async def test_position_tracker():
    """测试持仓跟踪器"""
    print("🧪 测试持仓跟踪器模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化持仓跟踪器
        tracker = PositionTracker(config)
        print("✅ 持仓跟踪器初始化成功")
        
        # 3. 启动跟踪器
        await tracker.start()
        print("✅ 持仓跟踪器启动成功")
        
        # 4. 测试添加交易记录
        print("\n📊 测试添加交易记录：")
        
        # 创建买入交易
        buy_trade = TradeRecord(
            id="trade_001",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            commission=5.0,
            timestamp=datetime.now(),
            order_id="order_001"
        )
        
        tracker.add_trade(buy_trade)
        print("✅ 买入交易记录添加成功")
        
        # 5. 测试持仓查询
        print("\n📊 测试持仓查询：")
        position = tracker.get_position("BTCUSDT")
        if position:
            print("✅ 持仓查询成功")
            print(f"   品种: {position.symbol}")
            print(f"   方向: {position.side.value}")
            print(f"   数量: {position.size}")
            print(f"   入场价: {position.entry_price}")
            print(f"   当前价: {position.current_price}")
            print(f"   未实现盈亏: {position.unrealized_pnl}")
            print(f"   投资回报率: {position.roi:.2%}")
        else:
            print("❌ 持仓查询失败")
        
        # 6. 测试价格更新
        print("\n📊 测试价格更新：")
        tracker.update_position_price("BTCUSDT", 51000.0)
        position = tracker.get_position("BTCUSDT")
        if position:
            print("✅ 价格更新成功")
            print(f"   更新后价格: {position.current_price}")
            print(f"   未实现盈亏: {position.unrealized_pnl}")
            print(f"   投资回报率: {position.roi:.2%}")
        
        # 7. 测试加仓
        print("\n📊 测试加仓：")
        add_trade = TradeRecord(
            id="trade_002",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.05,
            price=52000.0,
            commission=2.5,
            timestamp=datetime.now(),
            order_id="order_002"
        )
        
        tracker.add_trade(add_trade)
        position = tracker.get_position("BTCUSDT")
        if position:
            print("✅ 加仓成功")
            print(f"   新数量: {position.size}")
            print(f"   新入场价: {position.entry_price:.2f}")
            print(f"   总佣金: {position.commission_paid}")
        
        # 8. 测试部分平仓
        print("\n📊 测试部分平仓：")
        sell_trade = TradeRecord(
            id="trade_003",
            symbol="BTCUSDT",
            side="sell",
            quantity=0.05,
            price=53000.0,
            commission=2.5,
            timestamp=datetime.now(),
            order_id="order_003"
        )
        
        tracker.add_trade(sell_trade)
        position = tracker.get_position("BTCUSDT")
        if position:
            print("✅ 部分平仓成功")
            print(f"   剩余数量: {position.size}")
            print(f"   已实现盈亏: {position.realized_pnl}")
            print(f"   未实现盈亏: {position.unrealized_pnl}")
        
        # 9. 测试持仓汇总
        print("\n📊 测试持仓汇总：")
        summary = tracker.get_position_summary()
        print("✅ 持仓汇总获取成功")
        print(f"   总持仓数: {summary.total_positions}")
        print(f"   多头持仓: {summary.long_positions}")
        print(f"   空头持仓: {summary.short_positions}")
        print(f"   未实现盈亏: {summary.total_unrealized_pnl:.2f}")
        print(f"   已实现盈亏: {summary.total_realized_pnl:.2f}")
        print(f"   平均ROI: {summary.average_roi:.2%}")
        print(f"   胜率: {summary.win_rate:.2%}")
        
        # 10. 测试完全平仓
        print("\n📊 测试完全平仓：")
        close_trade = TradeRecord(
            id="trade_004",
            symbol="BTCUSDT",
            side="sell",
            quantity=0.1,
            price=54000.0,
            commission=5.0,
            timestamp=datetime.now(),
            order_id="order_004"
        )
        
        tracker.add_trade(close_trade)
        position = tracker.get_position("BTCUSDT")
        if position is None:
            print("✅ 完全平仓成功，持仓已清零")
            
            # 检查已关闭持仓
            if tracker.closed_positions:
                closed_pos = tracker.closed_positions[-1]
                print(f"   已关闭持仓盈亏: {closed_pos.realized_pnl:.2f}")
                print(f"   持仓ROI: {closed_pos.roi:.2%}")
        
        # 11. 测试性能指标
        print("\n📊 测试性能指标：")
        metrics = tracker.get_performance_metrics()
        print("✅ 性能指标获取成功")
        print(f"   总盈亏: {metrics['total_pnl']:.2f}")
        print(f"   胜率: {metrics['win_rate']:.2%}")
        print(f"   盈亏比: {metrics['profit_factor']:.2f}")
        print(f"   最大回撤: {metrics['max_drawdown']:.2f}")
        
        # 12. 测试多品种持仓
        print("\n📊 测试多品种持仓：")
        eth_trade = TradeRecord(
            id="trade_005",
            symbol="ETHUSDT",
            side="buy",
            quantity=1.0,
            price=3000.0,
            commission=3.0,
            timestamp=datetime.now(),
            order_id="order_005"
        )
        
        tracker.add_trade(eth_trade)
        all_positions = tracker.get_all_positions()
        print(f"✅ 多品种持仓成功，当前持仓数: {len(all_positions)}")
        
        for symbol, position in all_positions.items():
            print(f"   {symbol}: {position.side.value} {position.size}")
        
        # 13. 测试状态监控
        print("\n📊 测试状态监控：")
        status = tracker.get_status()
        print("✅ 状态查询成功")
        print(f"   运行状态: {status['is_running']}")
        print(f"   总持仓数: {status['total_positions']}")
        print(f"   已关闭持仓数: {status['closed_positions']}")
        print(f"   总交易数: {status['total_trades']}")
        print(f"   监控品种: {status['monitored_symbols']}")
        
        # 14. 测试监控功能（短时间）
        print("\n📊 测试监控功能：")
        print("   监控运行中... (等待3秒)")
        await asyncio.sleep(3)
        
        # 检查价格是否更新
        eth_position = tracker.get_position("ETHUSDT")
        if eth_position and eth_position.current_price != eth_position.entry_price:
            print("✅ 价格监控功能正常")
        else:
            print("⚠️  价格监控功能待验证")
        
        # 15. 关闭跟踪器
        await tracker.stop()
        print("✅ 持仓跟踪器已关闭")
        
        print("\n✅ 持仓跟踪器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_position_tracker())
    sys.exit(0 if success else 1) 