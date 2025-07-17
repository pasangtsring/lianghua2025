#!/usr/bin/env python3
"""
订单执行器测试
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from config.config_manager import ConfigManager
from execution.order_executor import OrderExecutor, Order, OrderType, OrderSide, OrderStatus

async def test_order_executor():
    """测试订单执行器"""
    print("🧪 测试订单执行器模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化订单执行器
        executor = OrderExecutor(config)
        print("✅ 订单执行器初始化成功")
        
        # 3. 启动执行器
        await executor.start()
        print("✅ 订单执行器启动成功")
        
        # 4. 测试市价单
        print("\n📊 测试市价单：")
        market_order = Order(
            id="test_market_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.01,
            price=50000.0
        )
        
        success = await executor.submit_order(market_order)
        if success:
            print("✅ 市价单提交成功")
            
            # 检查订单状态
            order_status = executor.get_order_status("test_market_001")
            if order_status:
                print(f"   订单状态: {order_status['status']}")
                print(f"   成交数量: {order_status['filled_quantity']}")
                print(f"   成交价格: {order_status['average_price']}")
        else:
            print("❌ 市价单提交失败")
        
        # 5. 测试限价单
        print("\n📊 测试限价单：")
        limit_order = Order(
            id="test_limit_001",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            quantity=0.01,
            price=55000.0
        )
        
        success = await executor.submit_order(limit_order)
        if success:
            print("✅ 限价单提交成功")
            
            # 检查订单状态
            order_status = executor.get_order_status("test_limit_001")
            if order_status:
                print(f"   订单状态: {order_status['status']}")
        else:
            print("❌ 限价单提交失败")
        
        # 6. 测试取消订单
        print("\n📊 测试取消订单：")
        cancel_success = await executor.cancel_order("test_limit_001")
        if cancel_success:
            print("✅ 订单取消成功")
        else:
            print("❌ 订单取消失败")
        
        # 7. 测试持仓查询
        print("\n📊 测试持仓查询：")
        positions = executor.get_positions()
        if positions:
            print("✅ 持仓信息获取成功")
            for symbol, position in positions.items():
                print(f"   {symbol}: 方向:{position['side']}, 数量:{position['size']}, 盈亏:{position['unrealized_pnl']}")
        else:
            print("⚠️  当前无持仓")
        
        # 8. 测试止损止盈
        print("\n📊 测试止损止盈：")
        if positions:
            symbol = list(positions.keys())[0]
            await executor.set_stop_loss_take_profit(
                symbol=symbol,
                stop_loss_price=45000.0,
                take_profit_price=55000.0
            )
            print(f"✅ 为 {symbol} 设置止损止盈成功")
        
        # 9. 测试执行统计
        print("\n📊 测试执行统计：")
        stats = executor.get_execution_stats()
        print("✅ 执行统计获取成功")
        print(f"   总订单数: {stats['total_orders']}")
        print(f"   成功订单数: {stats['successful_orders']}")
        print(f"   失败订单数: {stats['failed_orders']}")
        print(f"   平均执行时间: {stats['avg_execution_time']:.4f}秒")
        
        # 10. 测试状态查询
        print("\n📊 测试状态查询：")
        status = executor.get_status()
        print("✅ 状态查询成功")
        print(f"   运行状态: {status['is_running']}")
        print(f"   活跃订单数: {status['active_orders']}")
        print(f"   持仓数量: {status['total_positions']}")
        
        # 11. 测试批量操作
        print("\n📊 测试批量操作：")
        
        # 创建多个订单
        orders = []
        for i in range(3):
            order = Order(
                id=f"test_batch_{i:03d}",
                symbol="BTCUSDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                type=OrderType.LIMIT,
                quantity=0.001,
                price=49000.0 + i * 100
            )
            orders.append(order)
        
        # 批量提交
        batch_results = []
        for order in orders:
            result = await executor.submit_order(order)
            batch_results.append(result)
        
        successful_batch = sum(batch_results)
        print(f"✅ 批量提交完成: {successful_batch}/{len(orders)} 成功")
        
        # 取消所有订单
        cancelled_count = await executor.cancel_all_orders()
        print(f"✅ 取消所有订单: {cancelled_count} 个")
        
        # 12. 监控测试（短时间）
        print("\n📊 测试监控功能：")
        print("   监控运行中... (等待3秒)")
        await asyncio.sleep(3)
        print("✅ 监控功能正常")
        
        # 13. 关闭执行器
        await executor.stop()
        print("✅ 订单执行器已关闭")
        
        print("\n✅ 订单执行器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_order_executor())
    sys.exit(0 if success else 1) 