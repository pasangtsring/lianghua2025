#!/usr/bin/env python3
"""
性能监控器测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
from datetime import datetime
from config.config_manager import ConfigManager
from utils.performance_monitor import PerformanceMonitor, MetricType, AlertLevel, PerformanceAlert

def test_performance_monitor():
    """测试性能监控器"""
    print("🧪 测试性能监控器模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化性能监控器
        monitor = PerformanceMonitor(config)
        print("✅ 性能监控器初始化成功")
        
        # 3. 测试基本指标记录
        print("\n📊 测试基本指标记录：")
        
        # 记录不同类型的指标
        monitor.record_metric("test_gauge", 42.5, MetricType.GAUGE, unit="MB", description="测试仪表指标")
        print("✅ 记录仪表指标")
        
        monitor.record_metric("test_counter", 100, MetricType.COUNTER, unit="count", description="测试计数器指标")
        print("✅ 记录计数器指标")
        
        monitor.record_metric("test_timer", 150.5, MetricType.TIMER, unit="ms", description="测试计时器指标")
        print("✅ 记录计时器指标")
        
        # 测试递增计数器
        monitor.increment_counter("api_requests", 1.0)
        monitor.increment_counter("api_requests", 5.0)
        monitor.increment_counter("api_requests", 2.0)
        print("✅ 递增计数器: api_requests")
        
        # 测试设置仪表值
        monitor.set_gauge("cpu_usage", 75.2)
        monitor.set_gauge("memory_usage", 68.9)
        print("✅ 设置仪表值")
        
        # 4. 测试计时器装饰器
        print("\n📊 测试计时器装饰器：")
        
        @monitor.timer("test_function")
        def slow_function():
            """模拟耗时函数"""
            time.sleep(0.1)
            return "完成"
        
        # 执行多次测试函数
        for i in range(5):
            result = slow_function()
            print(f"✅ 执行测试函数 {i+1}: {result}")
        
        # 测试手动计时器记录
        monitor.record_timer("manual_timer", 200.5)
        monitor.record_timer("manual_timer", 180.3)
        monitor.record_timer("manual_timer", 220.8)
        print("✅ 手动记录计时器")
        
        # 5. 测试系统指标收集
        print("\n📊 测试系统指标收集：")
        
        # 启动监控一小段时间
        monitor.start_monitoring()
        print("✅ 启动性能监控")
        
        # 等待收集一些数据
        time.sleep(6)  # 等待超过一个监控间隔
        
        # 获取系统摘要
        system_summary = monitor.get_system_summary()
        if system_summary:
            print("✅ 系统摘要获取成功:")
            print(f"   CPU使用率: {system_summary.get('current_cpu_percent', 0):.1f}%")
            print(f"   内存使用率: {system_summary.get('current_memory_percent', 0):.1f}%")
            print(f"   磁盘使用率: {system_summary.get('current_disk_percent', 0):.1f}%")
            print(f"   进程数量: {system_summary.get('process_count', 0)}")
        else:
            print("⚠️ 系统摘要暂未收集到数据")
        
        # 获取进程摘要
        process_summary = monitor.get_process_summary()
        if process_summary:
            print("✅ 进程摘要获取成功:")
            print(f"   进程ID: {process_summary.get('pid', 0)}")
            print(f"   进程名: {process_summary.get('name', 'unknown')}")
            print(f"   CPU使用率: {process_summary.get('cpu_percent', 0):.1f}%")
            print(f"   内存使用率: {process_summary.get('memory_percent', 0):.1f}%")
            print(f"   线程数: {process_summary.get('threads_count', 0)}")
            print(f"   运行时间: {process_summary.get('uptime_seconds', 0)}秒")
        else:
            print("⚠️ 进程摘要暂未收集到数据")
        
        # 6. 测试计时器统计
        print("\n📊 测试计时器统计：")
        
        timer_summary = monitor.get_timer_summary()
        print("✅ 计时器摘要:")
        for name, stats in timer_summary.items():
            print(f"   {name}:")
            print(f"      执行次数: {stats['count']}")
            print(f"      平均时间: {stats['avg_ms']:.2f}ms")
            print(f"      最小时间: {stats['min_ms']:.2f}ms")
            print(f"      最大时间: {stats['max_ms']:.2f}ms")
            print(f"      最近时间: {stats['latest_ms']:.2f}ms")
        
        # 7. 测试自定义指标摘要
        print("\n📊 测试自定义指标摘要：")
        
        custom_metrics = monitor.get_custom_metrics_summary()
        print("✅ 自定义指标摘要:")
        for name, metric in custom_metrics.items():
            print(f"   {name}: {metric['value']} {metric['unit']} ({metric['type']})")
            if metric['description']:
                print(f"      描述: {metric['description']}")
        
        # 8. 测试告警系统
        print("\n📊 测试告警系统：")
        
        # 添加告警回调
        def alert_callback(alert: PerformanceAlert):
            print(f"🚨 告警回调: {alert.level.value.upper()} - {alert.message}")
        
        monitor.add_alert_callback(alert_callback)
        print("✅ 添加告警回调")
        
        # 手动触发告警（通过设置高值）
        monitor.record_metric("high_cpu", 95.0, MetricType.GAUGE, unit="%", description="模拟高CPU使用率")
        
        # 创建手动告警
        test_alert = PerformanceAlert(
            level=AlertLevel.WARNING,
            metric_name="test_metric",
            current_value=85.0,
            threshold=80.0,
            message="测试告警消息",
            timestamp=datetime.now()
        )
        
        monitor.alerts.append(test_alert)
        print("✅ 创建测试告警")
        
        # 获取告警摘要
        alerts_summary = monitor.get_alerts_summary()
        print("✅ 告警摘要:")
        print(f"   总告警数: {alerts_summary['total_alerts']}")
        print(f"   活跃告警数: {alerts_summary['active_alerts']}")
        print(f"   已解决告警数: {alerts_summary['resolved_alerts']}")
        print(f"   级别统计: {alerts_summary['level_counts']}")
        
        if alerts_summary['latest_alerts']:
            print("   最近告警:")
            for alert in alerts_summary['latest_alerts'][-3:]:  # 显示最近3个
                print(f"      [{alert['level'].upper()}] {alert['message']}")
        
        # 解决告警
        if monitor.alerts:
            resolved = monitor.resolve_alert(0)
            print(f"✅ 解决告警: {resolved}")
        
        # 9. 测试性能压力
        print("\n📊 测试性能压力：")
        
        # 模拟高频指标记录
        start_time = time.time()
        for i in range(1000):
            monitor.record_metric(f"stress_test", i * 0.1, MetricType.GAUGE)
            if i % 10 == 0:
                monitor.increment_counter("stress_counter", 1.0)
        
        stress_time = time.time() - start_time
        print(f"✅ 1000次指标记录耗时: {stress_time:.4f}秒")
        print(f"   记录速度: {1000/stress_time:.1f} 指标/秒")
        
        # 10. 测试并发安全
        print("\n📊 测试并发安全：")
        
        def worker_thread(worker_id, monitor_instance, results):
            try:
                for i in range(100):
                    monitor_instance.record_metric(f"worker_{worker_id}_metric", i, MetricType.COUNTER)
                    monitor_instance.increment_counter(f"worker_{worker_id}_counter", 1.0)
                    
                results[worker_id] = True
            except Exception as e:
                print(f"工作线程{worker_id}错误: {e}")
                results[worker_id] = False
        
        # 启动多个并发线程
        results = {}
        threads = []
        
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i, monitor, results))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        successful_workers = sum(1 for success in results.values() if success)
        print(f"✅ 并发测试: {successful_workers}/{len(results)} 个线程成功")
        
        # 11. 测试数据导出
        print("\n📊 测试数据导出：")
        
        # 导出JSON格式
        exported_json = monitor.export_metrics("json")
        print(f"✅ JSON导出: {len(exported_json)} 字符")
        
        # 验证导出数据是否有效
        import json
        try:
            exported_data = json.loads(exported_json)
            print("✅ 导出数据JSON格式有效")
            print(f"   包含系统摘要: {'system_summary' in exported_data}")
            print(f"   包含进程摘要: {'process_summary' in exported_data}")
            print(f"   包含计时器摘要: {'timer_summary' in exported_data}")
            print(f"   包含自定义指标: {'custom_metrics' in exported_data}")
        except json.JSONDecodeError:
            print("❌ 导出数据JSON格式无效")
        
        # 12. 测试监控器控制
        print("\n📊 测试监控器控制：")
        
        # 检查监控状态
        status_before = monitor.get_status()
        print("✅ 获取监控状态:")
        print(f"   监控活跃: {status_before['monitoring_active']}")
        print(f"   监控间隔: {status_before['monitor_interval']}秒")
        print(f"   历史大小: {status_before['history_size']}")
        print(f"   系统指标数: {status_before['current_system_metrics_count']}")
        print(f"   进程指标数: {status_before['current_process_metrics_count']}")
        print(f"   自定义指标数: {status_before['custom_metrics_count']}")
        print(f"   活跃计时器数: {status_before['active_timers_count']}")
        print(f"   运行时间: {status_before['uptime_seconds']}秒")
        
        # 停止监控
        monitor.stop_monitoring()
        print("✅ 停止性能监控")
        
        # 再次检查状态
        status_after = monitor.get_status()
        print(f"✅ 停止后监控状态: {status_after['monitoring_active']}")
        
        # 13. 测试数据重置
        print("\n📊 测试数据重置：")
        
        # 记录一些数据
        monitor.record_metric("before_reset", 123.45, MetricType.GAUGE)
        monitor.increment_counter("counter_before_reset", 10.0)
        
        metrics_before = len(monitor.custom_metrics)
        print(f"✅ 重置前自定义指标数: {metrics_before}")
        
        # 重置所有指标
        monitor.reset_metrics()
        print("✅ 重置所有指标")
        
        metrics_after = len(monitor.custom_metrics)
        print(f"✅ 重置后自定义指标数: {metrics_after}")
        
        # 14. 测试异常处理
        print("\n📊 测试异常处理：")
        
        # 测试无效指标名称
        monitor.record_metric("", 100.0)  # 空名称
        print("✅ 空指标名称处理")
        
        # 测试无效值
        try:
            monitor.record_metric("test_invalid", float('inf'))
            print("✅ 无穷大值处理")
        except:
            print("⚠️ 无穷大值处理异常")
        
        # 测试解决不存在的告警
        resolve_result = monitor.resolve_alert(999)
        print(f"✅ 解决不存在告警: {resolve_result}")
        
        # 15. 测试资源清理
        print("\n📊 测试资源清理：")
        
        # 重新启动短时间监控
        monitor.start_monitoring()
        time.sleep(2)
        
        initial_metrics = len(monitor.custom_metrics)
        print(f"✅ 清理前指标数: {initial_metrics}")
        
        # 手动触发清理
        monitor._cleanup_expired_data()
        print("✅ 执行过期数据清理")
        
        final_metrics = len(monitor.custom_metrics)
        print(f"✅ 清理后指标数: {final_metrics}")
        
        # 最终停止监控
        monitor.stop_monitoring()
        print("✅ 最终停止监控")
        
        # 16. 测试监控器性能
        print("\n📊 测试监控器性能：")
        
        # 测试大量快速操作
        start_time = time.time()
        
        for i in range(5000):
            monitor.record_metric(f"perf_test", i % 100, MetricType.GAUGE)
            if i % 100 == 0:
                monitor.increment_counter("perf_counter", 1.0)
                monitor.set_gauge("perf_gauge", i % 1000)
        
        perf_time = time.time() - start_time
        ops_per_second = 5000 / perf_time if perf_time > 0 else 0
        
        print(f"✅ 5000次混合操作耗时: {perf_time:.4f}秒")
        print(f"   操作速度: {ops_per_second:.1f} 操作/秒")
        
        # 最终状态检查
        final_status = monitor.get_status()
        print(f"\n✅ 最终状态检查:")
        print(f"   运行状态: {final_status['is_operational']}")
        print(f"   自定义指标数: {final_status['custom_metrics_count']}")
        print(f"   总告警数: {final_status['total_alerts_count']}")
        
        print("\n✅ 性能监控器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_performance_monitor()
    sys.exit(0 if success else 1) 