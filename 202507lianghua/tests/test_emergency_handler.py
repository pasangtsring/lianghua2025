#!/usr/bin/env python3
"""
应急处理器测试
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from config.config_manager import ConfigManager
from execution.emergency_handler import EmergencyHandler, EmergencyType, EmergencyLevel, EmergencyAction

async def test_emergency_handler():
    """测试应急处理器"""
    print("🧪 测试应急处理器模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化应急处理器
        handler = EmergencyHandler(config)
        print("✅ 应急处理器初始化成功")
        
        # 3. 启动处理器
        await handler.start()
        print("✅ 应急处理器启动成功")
        
        # 4. 测试应急事件触发
        print("\n📊 测试应急事件触发：")
        
        # 触发低级别警报
        event_id1 = await handler.trigger_emergency(
            EmergencyType.NETWORK_ERROR,
            EmergencyLevel.LOW,
            "网络连接延迟增加",
            {"latency": 500, "threshold": 200}
        )
        print(f"✅ 低级别事件触发成功: {event_id1}")
        
        # 触发中级别警报
        event_id2 = await handler.trigger_emergency(
            EmergencyType.PRICE_SPIKE,
            EmergencyLevel.MEDIUM,
            "BTCUSDT价格异常波动",
            {"symbol": "BTCUSDT", "price_change": 8.5, "current_price": 54250.0}
        )
        print(f"✅ 中级别事件触发成功: {event_id2}")
        
        # 触发高级别警报
        event_id3 = await handler.trigger_emergency(
            EmergencyType.POSITION_RISK,
            EmergencyLevel.HIGH,
            "单仓位亏损超过阈值",
            {"symbol": "BTCUSDT", "roi": -0.08, "unrealized_pnl": -400.0}
        )
        print(f"✅ 高级别事件触发成功: {event_id3}")
        
        # 触发关键级别警报
        event_id4 = await handler.trigger_emergency(
            EmergencyType.ACCOUNT_RISK,
            EmergencyLevel.CRITICAL,
            "总体账户亏损超过阈值",
            {"total_pnl_pct": -0.12, "total_pnl": -1200.0}
        )
        print(f"✅ 关键级别事件触发成功: {event_id4}")
        
        # 5. 测试活跃事件查询
        print("\n📊 测试活跃事件查询：")
        active_events = handler.get_active_events()
        print(f"✅ 活跃事件查询成功，共 {len(active_events)} 个活跃事件")
        
        for event_id, event in active_events.items():
            print(f"   事件 {event_id}: {event.type.value} - {event.level.value}")
            print(f"      描述: {event.description}")
            print(f"      已执行动作: {[action.value for action in event.actions_taken]}")
        
        # 6. 测试应急动作确定
        print("\n📊 测试应急动作确定：")
        
        # 测试不同级别的动作确定
        for event_id, event in active_events.items():
            actions = handler.determine_emergency_actions(event)
            print(f"   {event.level.value}级别事件应急动作: {[action.value for action in actions]}")
        
        # 7. 测试事件解决
        print("\n📊 测试事件解决：")
        
        # 解决低级别事件
        await handler.resolve_event(event_id1, "网络延迟已恢复正常")
        print(f"✅ 事件 {event_id1} 已解决")
        
        # 检查活跃事件数量
        active_events = handler.get_active_events()
        print(f"   剩余活跃事件: {len(active_events)} 个")
        
        # 8. 测试事件历史查询
        print("\n📊 测试事件历史查询：")
        event_history = handler.get_event_history(limit=10)
        print(f"✅ 事件历史查询成功，共 {len(event_history)} 条历史记录")
        
        for event in event_history:
            duration = (event.resolution_time - event.timestamp).total_seconds() if event.resolution_time else 0
            print(f"   历史事件: {event.type.value} - 持续时间: {duration:.1f}秒")
        
        # 9. 测试统计信息
        print("\n📊 测试统计信息：")
        stats = handler.get_stats()
        print("✅ 统计信息获取成功")
        print(f"   总事件数: {stats['total_events']}")
        print(f"   活跃事件数: {stats['active_events']}")
        print(f"   应急模式: {stats['emergency_mode']}")
        print(f"   监控状态: {stats['is_monitoring']}")
        
        if stats['events_by_type']:
            print("   按类型统计:")
            for event_type, count in stats['events_by_type'].items():
                print(f"      {event_type}: {count}")
        
        if stats['events_by_level']:
            print("   按级别统计:")
            for level, count in stats['events_by_level'].items():
                print(f"      {level}: {count}")
        
        # 10. 测试风险阈值配置
        print("\n📊 测试风险阈值配置：")
        print("✅ 风险阈值配置")
        print(f"   最大单仓位亏损: {handler.risk_thresholds.max_position_loss_pct}%")
        print(f"   最大总体亏损: {handler.risk_thresholds.max_total_loss_pct}%")
        print(f"   最大回撤: {handler.risk_thresholds.max_drawdown_pct}%")
        print(f"   价格变化阈值: {handler.risk_thresholds.price_change_threshold}%")
        print(f"   成交量异常阈值: {handler.risk_thresholds.volume_spike_threshold}倍")
        
        # 11. 测试状态查询
        print("\n📊 测试状态查询：")
        status = handler.get_status()
        print("✅ 状态查询成功")
        print(f"   监控状态: {status['is_monitoring']}")
        print(f"   应急模式: {status['emergency_mode']}")
        print(f"   活跃事件数: {status['active_events_count']}")
        print(f"   总处理事件数: {status['total_events_processed']}")
        
        # 12. 测试监控功能（短时间）
        print("\n📊 测试监控功能：")
        print("   监控运行中... (等待5秒)")
        
        # 等待监控运行
        await asyncio.sleep(5)
        
        # 检查是否有新的监控事件
        current_stats = handler.get_stats()
        print(f"✅ 监控功能运行正常，总事件数: {current_stats['total_events']}")
        
        # 13. 测试批量事件解决
        print("\n📊 测试批量事件解决：")
        
        # 解决剩余的活跃事件
        remaining_events = list(handler.get_active_events().keys())
        for event_id in remaining_events:
            await handler.resolve_event(event_id, "测试完成，批量解决")
        
        final_active_count = len(handler.get_active_events())
        print(f"✅ 批量解决完成，剩余活跃事件: {final_active_count} 个")
        
        # 14. 测试错误处理
        print("\n📊 测试错误处理：")
        
        # 测试无效事件ID解决
        try:
            await handler.resolve_event("invalid_event_id", "测试无效ID")
            print("✅ 无效事件ID处理正常")
        except Exception as e:
            print(f"❌ 无效事件ID处理失败: {e}")
        
        # 15. 关闭处理器
        await handler.stop()
        print("✅ 应急处理器已关闭")
        
        # 16. 最终统计
        final_stats = handler.get_stats()
        print(f"\n📈 测试完成统计:")
        print(f"   总处理事件数: {final_stats['total_events']}")
        print(f"   最终活跃事件数: {final_stats['active_events']}")
        print(f"   监控状态: {final_stats['is_monitoring']}")
        
        print("\n✅ 应急处理器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_emergency_handler())
    sys.exit(0 if success else 1) 