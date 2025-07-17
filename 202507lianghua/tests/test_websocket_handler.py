#!/usr/bin/env python3
"""
WebSocket处理器测试
"""

import sys
import os
import asyncio
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from config.config_manager import ConfigManager
from data.websocket_handler import WebSocketHandler, MessageType, WebSocketMessage

# 全局变量存储接收到的消息
received_messages = []

async def test_message_handler(message: WebSocketMessage):
    """测试消息处理器"""
    received_messages.append(message)
    print(f"收到消息: {message.type.value} - {message.symbol}")

async def test_websocket_handler():
    """测试WebSocket处理器"""
    print("🧪 测试WebSocket处理器模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化WebSocket处理器
        handler = WebSocketHandler(config)
        print("✅ WebSocket处理器初始化成功")
        
        # 3. 注册消息处理器
        print("\n📊 测试消息处理器注册：")
        handler.register_message_handler(MessageType.TICK, test_message_handler)
        handler.register_message_handler(MessageType.KLINE, test_message_handler)
        handler.register_message_handler(MessageType.DEPTH, test_message_handler)
        print("✅ 消息处理器注册成功")
        
        # 4. 测试消息类型识别
        print("\n📊 测试消息类型识别：")
        
        # 测试ticker消息
        ticker_data = {
            "stream": "btcusdt@ticker",
            "data": {
                "s": "BTCUSDT",
                "c": "50000.00",
                "P": "1.23"
            }
        }
        
        message_type = handler.determine_message_type(ticker_data)
        print(f"   Ticker消息类型: {message_type.value}")
        
        # 测试K线消息
        kline_data = {
            "stream": "btcusdt@kline_1m",
            "data": {
                "s": "BTCUSDT",
                "k": {
                    "o": "49000.00",
                    "c": "50000.00",
                    "h": "51000.00",
                    "l": "48000.00"
                }
            }
        }
        
        message_type = handler.determine_message_type(kline_data)
        print(f"   K线消息类型: {message_type.value}")
        
        # 测试深度消息
        depth_data = {
            "stream": "btcusdt@depth",
            "data": {
                "s": "BTCUSDT",
                "bids": [["49000.00", "1.50"]],
                "asks": [["50000.00", "1.20"]]
            }
        }
        
        message_type = handler.determine_message_type(depth_data)
        print(f"   深度消息类型: {message_type.value}")
        
        # 5. 测试交易品种提取
        print("\n📊 测试交易品种提取：")
        
        symbol1 = handler.extract_symbol(ticker_data)
        symbol2 = handler.extract_symbol(kline_data)
        symbol3 = handler.extract_symbol(depth_data)
        
        print(f"   Ticker品种: {symbol1}")
        print(f"   K线品种: {symbol2}")
        print(f"   深度品种: {symbol3}")
        
        # 6. 测试消息处理
        print("\n📊 测试消息处理：")
        
        # 模拟处理ticker消息
        ticker_message = json.dumps(ticker_data)
        await handler.process_message(ticker_message)
        
        # 模拟处理K线消息
        kline_message = json.dumps(kline_data)
        await handler.process_message(kline_message)
        
        print(f"✅ 消息处理完成，共处理 {len(received_messages)} 条消息")
        
        # 7. 测试消息缓冲区
        print("\n📊 测试消息缓冲区：")
        
        buffer_messages = handler.get_message_buffer(limit=10)
        print(f"✅ 消息缓冲区获取成功，共 {len(buffer_messages)} 条消息")
        
        for i, msg in enumerate(buffer_messages):
            print(f"   消息 {i+1}: {msg.type.value} - {msg.symbol}")
        
        # 8. 测试统计信息
        print("\n📊 测试统计信息：")
        
        stats = handler.get_stats()
        print("✅ 统计信息获取成功")
        print(f"   状态: {stats['status']}")
        print(f"   总消息数: {stats['total_messages']}")
        print(f"   每秒消息数: {stats['messages_per_second']}")
        print(f"   错误数: {stats['error_count']}")
        print(f"   平均延迟: {stats['avg_latency']:.2f}ms")
        
        # 9. 测试订阅管理
        print("\n📊 测试订阅管理：")
        
        # 注意：这里不实际连接WebSocket，只是测试订阅逻辑
        print("   模拟订阅管理（无实际连接）")
        
        # 模拟添加订阅
        await handler.subscribe("BTCUSDT", "ticker", test_message_handler)
        await handler.subscribe("ETHUSDT", "kline_1m", test_message_handler)
        
        subscription_info = handler.get_subscription_info()
        print(f"✅ 订阅管理测试完成，当前订阅数: {len(subscription_info)}")
        
        for symbol, info in subscription_info.items():
            print(f"   {symbol}: {info['stream']} (活跃: {info['is_active']})")
        
        # 10. 测试状态查询
        print("\n📊 测试状态查询：")
        
        status = handler.get_status()
        print("✅ 状态查询成功")
        print(f"   运行状态: {status['is_running']}")
        print(f"   连接状态: {status['connection_status']}")
        print(f"   活跃订阅数: {status['active_subscriptions']}")
        print(f"   消息缓冲区大小: {status['message_buffer_size']}")
        print(f"   总消息数: {status['total_messages']}")
        
        # 11. 测试错误处理
        print("\n📊 测试错误处理：")
        
        # 测试无效JSON
        try:
            await handler.process_message("invalid json")
            print("✅ 无效JSON处理正常")
        except Exception as e:
            print(f"❌ 无效JSON处理失败: {e}")
        
        # 测试空消息
        try:
            await handler.process_message("")
            print("✅ 空消息处理正常")
        except Exception as e:
            print(f"❌ 空消息处理失败: {e}")
        
        # 12. 测试取消订阅
        print("\n📊 测试取消订阅：")
        
        await handler.unsubscribe("BTCUSDT")
        subscription_info = handler.get_subscription_info()
        print(f"✅ 取消订阅完成，剩余订阅数: {len(subscription_info)}")
        
        # 13. 测试消息处理器注销
        print("\n📊 测试消息处理器注销：")
        
        handler.unregister_message_handler(MessageType.TICK)
        print("✅ 消息处理器注销成功")
        
        # 14. 测试性能统计
        print("\n📊 测试性能统计：")
        
        # 模拟更多消息来测试性能
        for i in range(10):
            test_data = {
                "stream": f"test{i}usdt@ticker",
                "data": {
                    "s": f"TEST{i}USDT",
                    "c": f"{50000 + i}.00"
                }
            }
            await handler.process_message(json.dumps(test_data))
        
        final_stats = handler.get_stats()
        print(f"✅ 性能测试完成，总处理消息数: {final_stats['total_messages']}")
        
        print("\n✅ WebSocket处理器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_websocket_handler())
    sys.exit(0 if success else 1) 