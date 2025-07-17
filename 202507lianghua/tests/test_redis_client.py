#!/usr/bin/env python3
"""
Redis客户端测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import asyncio
import threading
from datetime import datetime
from config.config_manager import ConfigManager
from utils.redis_client import RedisClient, CacheLevel, LockType

def test_redis_client():
    """测试Redis客户端"""
    print("🧪 测试Redis客户端模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化Redis客户端
        redis_client = RedisClient(config)
        print("✅ Redis客户端初始化成功")
        
        # 3. 测试基本缓存操作
        print("\n📊 测试基本缓存操作：")
        
        # 设置缓存
        success = redis_client.set("test_key", "test_value", ttl=60)
        print(f"✅ 设置缓存: {success}")
        
        # 获取缓存
        value = redis_client.get("test_key")
        print(f"✅ 获取缓存: {value}")
        
        # 检查存在
        exists = redis_client.exists("test_key")
        print(f"✅ 检查存在: {exists}")
        
        # 获取TTL
        ttl = redis_client.ttl("test_key")
        print(f"✅ 获取TTL: {ttl}秒")
        
        # 删除缓存
        deleted = redis_client.delete("test_key")
        print(f"✅ 删除缓存: {deleted}")
        
        # 再次获取（应该为None）
        value_after_delete = redis_client.get("test_key")
        print(f"✅ 删除后获取: {value_after_delete}")
        
        # 4. 测试不同缓存级别
        print("\n📊 测试不同缓存级别：")
        
        # 内存缓存
        redis_client.set("memory_key", {"type": "memory", "value": 123}, 
                        ttl=30, level=CacheLevel.L1_MEMORY)
        memory_value = redis_client.get("memory_key", level=CacheLevel.L1_MEMORY)
        print(f"✅ 内存缓存: {memory_value}")
        
        # Redis缓存
        redis_client.set("redis_key", {"type": "redis", "value": 456}, 
                        ttl=30, level=CacheLevel.L2_REDIS)
        redis_value = redis_client.get("redis_key", level=CacheLevel.L2_REDIS)
        print(f"✅ Redis缓存: {redis_value}")
        
        # 磁盘缓存（占位实现）
        redis_client.set("disk_key", {"type": "disk", "value": 789}, 
                        ttl=30, level=CacheLevel.L3_DISK)
        disk_value = redis_client.get("disk_key", level=CacheLevel.L3_DISK)
        print(f"✅ 磁盘缓存: {disk_value}")
        
        # 5. 测试哈希操作
        print("\n📊 测试哈希操作：")
        
        # 设置哈希字段
        hash_set = redis_client.set_hash("user:1001", "name", "张三", ttl=60)
        print(f"✅ 设置哈希字段name: {hash_set}")
        
        hash_set = redis_client.set_hash("user:1001", "age", 25, ttl=60)
        print(f"✅ 设置哈希字段age: {hash_set}")
        
        hash_set = redis_client.set_hash("user:1001", "email", "zhangsan@example.com", ttl=60)
        print(f"✅ 设置哈希字段email: {hash_set}")
        
        # 获取哈希字段
        name = redis_client.get_hash("user:1001", "name")
        age = redis_client.get_hash("user:1001", "age")
        email = redis_client.get_hash("user:1001", "email")
        
        print(f"✅ 获取哈希字段:")
        print(f"   name: {name}")
        print(f"   age: {age}")
        print(f"   email: {email}")
        
        # 删除哈希字段
        deleted_field = redis_client.delete_hash("user:1001", "email")
        print(f"✅ 删除哈希字段email: {deleted_field}")
        
        # 再次获取被删除的字段
        email_after_delete = redis_client.get_hash("user:1001", "email")
        print(f"✅ 删除后获取email: {email_after_delete}")
        
        # 6. 测试集合操作
        print("\n📊 测试集合操作：")
        
        # 添加到集合
        added_count = redis_client.add_to_set("tags", "python", "redis", "cache", ttl=60)
        print(f"✅ 添加到集合: {added_count} 个元素")
        
        # 再次添加相同元素
        added_count = redis_client.add_to_set("tags", "python", "database")
        print(f"✅ 再次添加到集合: {added_count} 个新元素")
        
        # 检查成员
        is_member = redis_client.is_member_of_set("tags", "python")
        print(f"✅ 检查python是否在集合中: {is_member}")
        
        is_member = redis_client.is_member_of_set("tags", "java")
        print(f"✅ 检查java是否在集合中: {is_member}")
        
        # 获取所有成员
        members = redis_client.get_set_members("tags")
        print(f"✅ 获取集合所有成员: {sorted(list(members))}")
        
        # 7. 测试计数器操作
        print("\n📊 测试计数器操作：")
        
        # 递增计数器
        counter_value = redis_client.increment("page_views", 1, ttl=60)
        print(f"✅ 递增计数器: {counter_value}")
        
        counter_value = redis_client.increment("page_views", 5)
        print(f"✅ 再次递增计数器: {counter_value}")
        
        # 获取计数器值
        current_count = redis_client.get_counter("page_views")
        print(f"✅ 获取计数器值: {current_count}")
        
        # 递增不存在的计数器
        new_counter = redis_client.increment("new_counter", 10, ttl=30)
        print(f"✅ 新计数器值: {new_counter}")
        
        # 8. 测试分布式锁
        print("\n📊 测试分布式锁：")
        
        # 获取排他锁
        lock_acquired = redis_client.acquire_lock("test_lock", timeout=30, lock_type=LockType.EXCLUSIVE)
        print(f"✅ 获取排他锁: {lock_acquired}")
        
        # 尝试再次获取相同锁（应该失败）
        lock_acquired_again = redis_client.acquire_lock("test_lock", timeout=30, lock_type=LockType.EXCLUSIVE)
        print(f"✅ 再次获取相同锁: {lock_acquired_again}")
        
        # 释放锁
        lock_released = redis_client.release_lock("test_lock")
        print(f"✅ 释放锁: {lock_released}")
        
        # 获取共享锁
        shared_lock1 = redis_client.acquire_lock("shared_lock", timeout=30, lock_type=LockType.SHARED)
        print(f"✅ 获取共享锁1: {shared_lock1}")
        
        shared_lock2 = redis_client.acquire_lock("shared_lock", timeout=30, lock_type=LockType.SHARED)
        print(f"✅ 获取共享锁2: {shared_lock2}")
        
        # 释放共享锁
        redis_client.release_lock("shared_lock")
        print("✅ 释放共享锁")
        
        # 9. 测试锁上下文管理器
        print("\n📊 测试锁上下文管理器：")
        
        async def test_lock_context():
            try:
                async with redis_client.lock_context("context_lock", timeout=10):
                    print("✅ 在锁上下文中执行操作")
                    await asyncio.sleep(0.1)
                    print("✅ 操作完成")
                print("✅ 锁自动释放")
                return True
            except Exception as e:
                print(f"❌ 锁上下文测试失败: {e}")
                return False
        
        # 运行异步测试
        context_result = asyncio.run(test_lock_context())
        
        # 10. 测试并发锁操作
        print("\n📊 测试并发锁操作：")
        
        def worker_thread(worker_id, lock_name, results):
            try:
                acquired = redis_client.acquire_lock(f"{lock_name}_{worker_id}", timeout=5)
                if acquired:
                    time.sleep(0.1)  # 模拟工作
                    redis_client.release_lock(f"{lock_name}_{worker_id}")
                    results[worker_id] = True
                else:
                    results[worker_id] = False
            except Exception as e:
                print(f"工作线程{worker_id}错误: {e}")
                results[worker_id] = False
        
        # 启动多个线程
        results = {}
        threads = []
        
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i, "concurrent_lock", results))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        successful_locks = sum(1 for success in results.values() if success)
        print(f"✅ 并发锁测试: {successful_locks}/{len(results)} 个锁成功获取")
        
        # 11. 测试缓存过期
        print("\n📊 测试缓存过期：")
        
        # 设置短过期时间的缓存
        redis_client.set("expire_test", "will_expire", ttl=1)
        print("✅ 设置1秒过期的缓存")
        
        # 立即获取
        value_before = redis_client.get("expire_test")
        print(f"✅ 过期前获取: {value_before}")
        
        # 等待过期
        time.sleep(1.5)
        
        # 过期后获取
        value_after = redis_client.get("expire_test")
        print(f"✅ 过期后获取: {value_after}")
        
        # 12. 测试批量操作和性能
        print("\n📊 测试批量操作和性能：")
        
        # 批量设置
        start_time = time.time()
        for i in range(100):
            redis_client.set(f"batch_key_{i}", f"batch_value_{i}", ttl=30)
        
        set_time = time.time() - start_time
        print(f"✅ 批量设置100个键: {set_time:.4f}秒")
        
        # 批量获取
        start_time = time.time()
        retrieved_count = 0
        for i in range(100):
            value = redis_client.get(f"batch_key_{i}")
            if value is not None:
                retrieved_count += 1
        
        get_time = time.time() - start_time
        print(f"✅ 批量获取100个键: {get_time:.4f}秒, 成功获取: {retrieved_count}")
        
        # 计算操作速度
        total_ops = 200  # 100次设置 + 100次获取
        total_time = set_time + get_time
        ops_per_second = total_ops / total_time if total_time > 0 else 0
        
        print(f"✅ 操作速度: {ops_per_second:.1f} 操作/秒")
        
        # 13. 测试统计信息
        print("\n📊 测试统计信息：")
        
        stats = redis_client.get_cache_stats()
        print("✅ 缓存统计信息:")
        print(f"   缓存命中: {stats['cache_hits']}")
        print(f"   缓存未命中: {stats['cache_misses']}")
        print(f"   命中率: {stats['hit_rate']:.2%}")
        print(f"   缓存设置: {stats['cache_sets']}")
        print(f"   缓存删除: {stats['cache_deletes']}")
        print(f"   内存使用: {stats['memory_usage']}/{stats['memory_max_size']}")
        print(f"   Redis操作: {stats['redis_operations']}")
        print(f"   总键数: {stats['total_keys']}")
        print(f"   锁获取: {stats['lock_acquisitions']}")
        print(f"   锁失败: {stats['lock_failures']}")
        print(f"   锁成功率: {stats['lock_success_rate']:.2%}")
        print(f"   活跃锁: {stats['active_locks']}")
        
        # 14. 测试连接信息
        print("\n📊 测试连接信息：")
        
        conn_info = redis_client.get_connection_info()
        print("✅ 连接信息:")
        print(f"   连接状态: {conn_info['connected']}")
        print(f"   连接池大小: {conn_info['connection_pool_size']}")
        print(f"   Redis版本: {conn_info['redis_version']}")
        print(f"   内存使用: {conn_info['memory_usage_mb']:.2f} MB")
        print(f"   运行时间: {conn_info['uptime_seconds']} 秒")
        print(f"   总命令数: {conn_info['total_commands_processed']}")
        
        # 15. 测试Ping
        print("\n📊 测试Ping：")
        
        ping_result = redis_client.ping()
        print(f"✅ Ping结果: {ping_result}")
        
        # 16. 测试状态查询
        print("\n📊 测试状态查询：")
        
        status = redis_client.get_status()
        print("✅ 客户端状态:")
        print(f"   连接状态: {status['connected']}")
        print(f"   总操作数: {status['total_operations']}")
        print(f"   内存缓存大小: {status['memory_cache_size']}")
        print(f"   Redis存储大小: {status['redis_storage_size']}")
        print(f"   活跃锁数: {status['active_locks']}")
        print(f"   缓存命中率: {status['cache_hit_rate']:.2%}")
        print(f"   运行状态: {status['is_operational']}")
        
        # 17. 测试错误处理
        print("\n📊 测试错误处理：")
        
        # 测试获取不存在的键
        non_existent = redis_client.get("non_existent_key")
        print(f"✅ 获取不存在的键: {non_existent}")
        
        # 测试删除不存在的键
        delete_non_existent = redis_client.delete("non_existent_key")
        print(f"✅ 删除不存在的键: {delete_non_existent}")
        
        # 测试释放不存在的锁
        release_non_existent = redis_client.release_lock("non_existent_lock")
        print(f"✅ 释放不存在的锁: {release_non_existent}")
        
        # 18. 测试清空操作
        print("\n📊 测试清空操作：")
        
        # 添加一些测试数据
        redis_client.set("clear_test_1", "value1")
        redis_client.set("clear_test_2", "value2")
        redis_client.set_hash("clear_hash", "field", "value")
        
        # 获取清空前的状态
        before_stats = redis_client.get_cache_stats()
        print(f"✅ 清空前总键数: {before_stats['total_keys']}")
        
        # 清空所有缓存
        flush_result = redis_client.flush_all()
        print(f"✅ 清空所有缓存: {flush_result}")
        
        # 获取清空后的状态
        after_stats = redis_client.get_cache_stats()
        print(f"✅ 清空后总键数: {after_stats['total_keys']}")
        
        # 验证清空效果
        test_value = redis_client.get("clear_test_1")
        print(f"✅ 清空后获取测试值: {test_value}")
        
        print("\n✅ Redis客户端测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_redis_client()
    sys.exit(0 if success else 1) 