"""
Redis客户端模块
负责缓存管理、会话存储、分布式锁等功能
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import asyncio
from contextlib import asynccontextmanager

from utils.logger import get_logger
from config.config_manager import ConfigManager

class CacheLevel(Enum):
    """缓存级别"""
    L1_MEMORY = "l1_memory"      # 内存缓存
    L2_REDIS = "l2_redis"        # Redis缓存
    L3_DISK = "l3_disk"          # 磁盘缓存

class LockType(Enum):
    """锁类型"""
    EXCLUSIVE = "exclusive"       # 排他锁
    SHARED = "shared"            # 共享锁
    TIMEOUT = "timeout"          # 超时锁

@dataclass
class CacheConfig:
    """缓存配置"""
    default_ttl: int = 3600      # 默认过期时间（秒）
    max_memory_size: int = 100   # 最大内存条目数
    compression: bool = False     # 是否压缩
    serialize_json: bool = True   # 是否JSON序列化
    key_prefix: str = "qts"      # 键前缀

@dataclass
class LockConfig:
    """锁配置"""
    default_timeout: int = 30     # 默认锁超时时间（秒）
    retry_interval: float = 0.1   # 重试间隔（秒）
    max_retries: int = 100       # 最大重试次数

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    ttl: int
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """是否过期"""
        if self.ttl <= 0:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    @property
    def remaining_ttl(self) -> int:
        """剩余过期时间"""
        if self.ttl <= 0:
            return -1
        remaining = self.created_at + timedelta(seconds=self.ttl) - datetime.now()
        return max(0, int(remaining.total_seconds()))

class RedisClient:
    """Redis客户端（模拟实现）"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 缓存配置
        self.cache_config = CacheConfig()
        self.lock_config = LockConfig()
        
        # 模拟Redis存储
        self._redis_storage: Dict[str, Dict] = {}
        self._redis_locks: Dict[str, Dict] = {}
        
        # 内存缓存
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_sets': 0,
            'cache_deletes': 0,
            'memory_usage': 0,
            'redis_operations': 0,
            'lock_acquisitions': 0,
            'lock_failures': 0,
            'total_keys': 0
        }
        
        # 连接状态
        self.connected = True
        self.connection_pool_size = 10
        
        # 启动清理任务
        self._start_cleanup_task()
        
        self.logger.info("Redis客户端初始化完成")
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        try:
            def cleanup_worker():
                while True:
                    try:
                        self._cleanup_expired_entries()
                        time.sleep(30)  # 每30秒清理一次
                    except Exception as e:
                        self.logger.error(f"缓存清理失败: {e}")
                        time.sleep(60)
            
            cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
            cleanup_thread.start()
            
            self.logger.info("缓存清理任务启动成功")
            
        except Exception as e:
            self.logger.error(f"启动缓存清理任务失败: {e}")
    
    def _cleanup_expired_entries(self):
        """清理过期条目"""
        try:
            with self._cache_lock:
                expired_keys = [key for key, entry in self._memory_cache.items() if entry.is_expired]
                
                for key in expired_keys:
                    del self._memory_cache[key]
                    self.stats['cache_deletes'] += 1
                
                if expired_keys:
                    self.logger.debug(f"清理过期缓存条目: {len(expired_keys)} 个")
            
            # 清理Redis过期条目
            current_time = time.time()
            expired_redis_keys = []
            
            for key, data in self._redis_storage.items():
                if 'expires_at' in data and data['expires_at'] > 0:
                    if current_time > data['expires_at']:
                        expired_redis_keys.append(key)
            
            for key in expired_redis_keys:
                del self._redis_storage[key]
            
            if expired_redis_keys:
                self.logger.debug(f"清理Redis过期条目: {len(expired_redis_keys)} 个")
                
        except Exception as e:
            self.logger.error(f"清理过期条目失败: {e}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            level: CacheLevel = CacheLevel.L2_REDIS) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            level: 缓存级别
            
        Returns:
            是否设置成功
        """
        try:
            if ttl is None:
                ttl = self.cache_config.default_ttl
            
            full_key = f"{self.cache_config.key_prefix}:{key}"
            
            # 序列化值
            serialized_value = self._serialize_value(value)
            
            if level == CacheLevel.L1_MEMORY:
                return self._set_memory_cache(full_key, value, ttl)
            elif level == CacheLevel.L2_REDIS:
                return self._set_redis_cache(full_key, serialized_value, ttl)
            elif level == CacheLevel.L3_DISK:
                return self._set_disk_cache(full_key, serialized_value, ttl)
            
            return False
            
        except Exception as e:
            self.logger.error(f"设置缓存失败: {key} - {e}")
            return False
    
    def get(self, key: str, level: CacheLevel = CacheLevel.L2_REDIS) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            level: 缓存级别
            
        Returns:
            缓存值或None
        """
        try:
            full_key = f"{self.cache_config.key_prefix}:{key}"
            
            if level == CacheLevel.L1_MEMORY:
                return self._get_memory_cache(full_key)
            elif level == CacheLevel.L2_REDIS:
                return self._get_redis_cache(full_key)
            elif level == CacheLevel.L3_DISK:
                return self._get_disk_cache(full_key)
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取缓存失败: {key} - {e}")
            return None
    
    def delete(self, key: str, level: CacheLevel = CacheLevel.L2_REDIS) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            level: 缓存级别
            
        Returns:
            是否删除成功
        """
        try:
            full_key = f"{self.cache_config.key_prefix}:{key}"
            
            if level == CacheLevel.L1_MEMORY:
                return self._delete_memory_cache(full_key)
            elif level == CacheLevel.L2_REDIS:
                return self._delete_redis_cache(full_key)
            elif level == CacheLevel.L3_DISK:
                return self._delete_disk_cache(full_key)
            
            return False
            
        except Exception as e:
            self.logger.error(f"删除缓存失败: {key} - {e}")
            return False
    
    def exists(self, key: str, level: CacheLevel = CacheLevel.L2_REDIS) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            level: 缓存级别
            
        Returns:
            是否存在
        """
        try:
            full_key = f"{self.cache_config.key_prefix}:{key}"
            
            if level == CacheLevel.L1_MEMORY:
                return self._exists_memory_cache(full_key)
            elif level == CacheLevel.L2_REDIS:
                return self._exists_redis_cache(full_key)
            elif level == CacheLevel.L3_DISK:
                return self._exists_disk_cache(full_key)
            
            return False
            
        except Exception as e:
            self.logger.error(f"检查缓存存在失败: {key} - {e}")
            return False
    
    def ttl(self, key: str, level: CacheLevel = CacheLevel.L2_REDIS) -> int:
        """
        获取缓存剩余过期时间
        
        Args:
            key: 缓存键
            level: 缓存级别
            
        Returns:
            剩余过期时间（秒），-1表示永不过期，-2表示不存在
        """
        try:
            full_key = f"{self.cache_config.key_prefix}:{key}"
            
            if level == CacheLevel.L1_MEMORY:
                return self._ttl_memory_cache(full_key)
            elif level == CacheLevel.L2_REDIS:
                return self._ttl_redis_cache(full_key)
            elif level == CacheLevel.L3_DISK:
                return self._ttl_disk_cache(full_key)
            
            return -2
            
        except Exception as e:
            self.logger.error(f"获取缓存TTL失败: {key} - {e}")
            return -2
    
    def acquire_lock(self, lock_name: str, timeout: Optional[int] = None, 
                    lock_type: LockType = LockType.EXCLUSIVE) -> bool:
        """
        获取分布式锁
        
        Args:
            lock_name: 锁名称
            timeout: 超时时间（秒）
            lock_type: 锁类型
            
        Returns:
            是否获取成功
        """
        try:
            if timeout is None:
                timeout = self.lock_config.default_timeout
            
            lock_key = f"lock:{lock_name}"
            current_time = time.time()
            expires_at = current_time + timeout
            
            lock_data = {
                'owner': threading.current_thread().ident,
                'lock_type': lock_type.value,
                'acquired_at': current_time,
                'expires_at': expires_at,
                'timeout': timeout
            }
            
            # 检查锁是否已存在
            if lock_key in self._redis_locks:
                existing_lock = self._redis_locks[lock_key]
                
                # 检查锁是否过期
                if current_time > existing_lock['expires_at']:
                    # 锁已过期，删除
                    del self._redis_locks[lock_key]
                else:
                    # 锁仍有效
                    if lock_type == LockType.SHARED and existing_lock['lock_type'] == 'shared':
                        # 共享锁可以共存
                        pass
                    else:
                        # 排他锁或冲突的锁类型
                        self.stats['lock_failures'] += 1
                        return False
            
            # 设置锁
            self._redis_locks[lock_key] = lock_data
            self.stats['lock_acquisitions'] += 1
            
            self.logger.debug(f"获取锁成功: {lock_name} ({lock_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"获取锁失败: {lock_name} - {e}")
            self.stats['lock_failures'] += 1
            return False
    
    def release_lock(self, lock_name: str) -> bool:
        """
        释放分布式锁
        
        Args:
            lock_name: 锁名称
            
        Returns:
            是否释放成功
        """
        try:
            lock_key = f"lock:{lock_name}"
            current_thread_id = threading.current_thread().ident
            
            if lock_key not in self._redis_locks:
                self.logger.warning(f"尝试释放不存在的锁: {lock_name}")
                return False
            
            lock_data = self._redis_locks[lock_key]
            
            # 检查锁的所有者
            if lock_data['owner'] != current_thread_id:
                self.logger.warning(f"尝试释放不属于当前线程的锁: {lock_name}")
                return False
            
            # 释放锁
            del self._redis_locks[lock_key]
            
            self.logger.debug(f"释放锁成功: {lock_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"释放锁失败: {lock_name} - {e}")
            return False
    
    @asynccontextmanager
    async def lock_context(self, lock_name: str, timeout: Optional[int] = None,
                          lock_type: LockType = LockType.EXCLUSIVE):
        """
        锁上下文管理器
        
        Args:
            lock_name: 锁名称
            timeout: 超时时间
            lock_type: 锁类型
        """
        acquired = False
        try:
            # 尝试获取锁
            max_retries = self.lock_config.max_retries
            retry_interval = self.lock_config.retry_interval
            
            for attempt in range(max_retries):
                acquired = self.acquire_lock(lock_name, timeout, lock_type)
                if acquired:
                    break
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_interval)
            
            if not acquired:
                raise TimeoutError(f"无法在指定时间内获取锁: {lock_name}")
            
            yield
            
        finally:
            if acquired:
                self.release_lock(lock_name)
    
    def set_hash(self, key: str, field: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置哈希字段"""
        try:
            full_key = f"{self.cache_config.key_prefix}:hash:{key}"
            
            if full_key not in self._redis_storage:
                self._redis_storage[full_key] = {
                    'type': 'hash',
                    'data': {},
                    'created_at': time.time(),
                    'expires_at': 0
                }
            
            # 设置过期时间
            if ttl is not None:
                self._redis_storage[full_key]['expires_at'] = time.time() + ttl
            
            # 设置字段值
            self._redis_storage[full_key]['data'][field] = self._serialize_value(value)
            self.stats['redis_operations'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置哈希字段失败: {key}.{field} - {e}")
            return False
    
    def get_hash(self, key: str, field: str) -> Optional[Any]:
        """获取哈希字段"""
        try:
            full_key = f"{self.cache_config.key_prefix}:hash:{key}"
            
            if full_key not in self._redis_storage:
                return None
            
            hash_data = self._redis_storage[full_key]
            
            # 检查过期
            if hash_data['expires_at'] > 0 and time.time() > hash_data['expires_at']:
                del self._redis_storage[full_key]
                return None
            
            if field not in hash_data['data']:
                return None
            
            serialized_value = hash_data['data'][field]
            value = self._deserialize_value(serialized_value)
            
            self.stats['redis_operations'] += 1
            return value
            
        except Exception as e:
            self.logger.error(f"获取哈希字段失败: {key}.{field} - {e}")
            return None
    
    def delete_hash(self, key: str, field: Optional[str] = None) -> bool:
        """删除哈希字段或整个哈希"""
        try:
            full_key = f"{self.cache_config.key_prefix}:hash:{key}"
            
            if full_key not in self._redis_storage:
                return False
            
            if field is None:
                # 删除整个哈希
                del self._redis_storage[full_key]
            else:
                # 删除特定字段
                if field in self._redis_storage[full_key]['data']:
                    del self._redis_storage[full_key]['data'][field]
            
            self.stats['redis_operations'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"删除哈希字段失败: {key}.{field} - {e}")
            return False
    
    def add_to_set(self, key: str, *values: Any, ttl: Optional[int] = None) -> int:
        """添加到集合"""
        try:
            full_key = f"{self.cache_config.key_prefix}:set:{key}"
            
            if full_key not in self._redis_storage:
                self._redis_storage[full_key] = {
                    'type': 'set',
                    'data': set(),
                    'created_at': time.time(),
                    'expires_at': 0
                }
            
            # 设置过期时间
            if ttl is not None:
                self._redis_storage[full_key]['expires_at'] = time.time() + ttl
            
            # 添加值
            added_count = 0
            for value in values:
                serialized_value = self._serialize_value(value)
                if serialized_value not in self._redis_storage[full_key]['data']:
                    self._redis_storage[full_key]['data'].add(serialized_value)
                    added_count += 1
            
            self.stats['redis_operations'] += 1
            return added_count
            
        except Exception as e:
            self.logger.error(f"添加到集合失败: {key} - {e}")
            return 0
    
    def is_member_of_set(self, key: str, value: Any) -> bool:
        """检查是否是集合成员"""
        try:
            full_key = f"{self.cache_config.key_prefix}:set:{key}"
            
            if full_key not in self._redis_storage:
                return False
            
            set_data = self._redis_storage[full_key]
            
            # 检查过期
            if set_data['expires_at'] > 0 and time.time() > set_data['expires_at']:
                del self._redis_storage[full_key]
                return False
            
            serialized_value = self._serialize_value(value)
            self.stats['redis_operations'] += 1
            
            return serialized_value in set_data['data']
            
        except Exception as e:
            self.logger.error(f"检查集合成员失败: {key} - {e}")
            return False
    
    def get_set_members(self, key: str) -> Set[Any]:
        """获取集合所有成员"""
        try:
            full_key = f"{self.cache_config.key_prefix}:set:{key}"
            
            if full_key not in self._redis_storage:
                return set()
            
            set_data = self._redis_storage[full_key]
            
            # 检查过期
            if set_data['expires_at'] > 0 and time.time() > set_data['expires_at']:
                del self._redis_storage[full_key]
                return set()
            
            members = set()
            for serialized_value in set_data['data']:
                value = self._deserialize_value(serialized_value)
                members.add(value)
            
            self.stats['redis_operations'] += 1
            return members
            
        except Exception as e:
            self.logger.error(f"获取集合成员失败: {key} - {e}")
            return set()
    
    def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """递增计数器"""
        try:
            full_key = f"{self.cache_config.key_prefix}:counter:{key}"
            
            if full_key not in self._redis_storage:
                self._redis_storage[full_key] = {
                    'type': 'counter',
                    'value': 0,
                    'created_at': time.time(),
                    'expires_at': 0
                }
            
            # 设置过期时间
            if ttl is not None:
                self._redis_storage[full_key]['expires_at'] = time.time() + ttl
            
            # 递增值
            self._redis_storage[full_key]['value'] += amount
            self.stats['redis_operations'] += 1
            
            return self._redis_storage[full_key]['value']
            
        except Exception as e:
            self.logger.error(f"递增计数器失败: {key} - {e}")
            return 0
    
    def get_counter(self, key: str) -> int:
        """获取计数器值"""
        try:
            full_key = f"{self.cache_config.key_prefix}:counter:{key}"
            
            if full_key not in self._redis_storage:
                return 0
            
            counter_data = self._redis_storage[full_key]
            
            # 检查过期
            if counter_data['expires_at'] > 0 and time.time() > counter_data['expires_at']:
                del self._redis_storage[full_key]
                return 0
            
            self.stats['redis_operations'] += 1
            return counter_data['value']
            
        except Exception as e:
            self.logger.error(f"获取计数器值失败: {key} - {e}")
            return 0
    
    def flush_all(self) -> bool:
        """清空所有缓存"""
        try:
            with self._cache_lock:
                self._memory_cache.clear()
            
            self._redis_storage.clear()
            self._redis_locks.clear()
            
            # 重置统计
            for key in self.stats:
                if key not in ['cache_hits', 'cache_misses']:
                    self.stats[key] = 0
            
            self.logger.info("清空所有缓存完成")
            return True
            
        except Exception as e:
            self.logger.error(f"清空所有缓存失败: {e}")
            return False
    
    def _set_memory_cache(self, key: str, value: Any, ttl: int) -> bool:
        """设置内存缓存"""
        try:
            with self._cache_lock:
                # 检查缓存大小限制
                if len(self._memory_cache) >= self.cache_config.max_memory_size:
                    # 清理最少使用的条目
                    self._evict_lru_entries()
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl,
                    created_at=datetime.now()
                )
                
                self._memory_cache[key] = entry
                self.stats['cache_sets'] += 1
                self.stats['memory_usage'] = len(self._memory_cache)
                
                return True
                
        except Exception as e:
            self.logger.error(f"设置内存缓存失败: {key} - {e}")
            return False
    
    def _get_memory_cache(self, key: str) -> Optional[Any]:
        """获取内存缓存"""
        try:
            with self._cache_lock:
                if key not in self._memory_cache:
                    self.stats['cache_misses'] += 1
                    return None
                
                entry = self._memory_cache[key]
                
                # 检查过期
                if entry.is_expired:
                    del self._memory_cache[key]
                    self.stats['cache_misses'] += 1
                    return None
                
                # 更新访问信息
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                self.stats['cache_hits'] += 1
                return entry.value
                
        except Exception as e:
            self.logger.error(f"获取内存缓存失败: {key} - {e}")
            self.stats['cache_misses'] += 1
            return None
    
    def _delete_memory_cache(self, key: str) -> bool:
        """删除内存缓存"""
        try:
            with self._cache_lock:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    self.stats['cache_deletes'] += 1
                    self.stats['memory_usage'] = len(self._memory_cache)
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"删除内存缓存失败: {key} - {e}")
            return False
    
    def _exists_memory_cache(self, key: str) -> bool:
        """检查内存缓存是否存在"""
        try:
            with self._cache_lock:
                if key not in self._memory_cache:
                    return False
                
                entry = self._memory_cache[key]
                if entry.is_expired:
                    del self._memory_cache[key]
                    return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"检查内存缓存存在失败: {key} - {e}")
            return False
    
    def _ttl_memory_cache(self, key: str) -> int:
        """获取内存缓存TTL"""
        try:
            with self._cache_lock:
                if key not in self._memory_cache:
                    return -2
                
                entry = self._memory_cache[key]
                if entry.is_expired:
                    del self._memory_cache[key]
                    return -2
                
                return entry.remaining_ttl
                
        except Exception as e:
            self.logger.error(f"获取内存缓存TTL失败: {key} - {e}")
            return -2
    
    def _set_redis_cache(self, key: str, value: str, ttl: int) -> bool:
        """设置Redis缓存"""
        try:
            expires_at = time.time() + ttl if ttl > 0 else 0
            
            self._redis_storage[key] = {
                'type': 'string',
                'value': value,
                'created_at': time.time(),
                'expires_at': expires_at
            }
            
            self.stats['cache_sets'] += 1
            self.stats['redis_operations'] += 1
            self.stats['total_keys'] = len(self._redis_storage)
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置Redis缓存失败: {key} - {e}")
            return False
    
    def _get_redis_cache(self, key: str) -> Optional[Any]:
        """获取Redis缓存"""
        try:
            if key not in self._redis_storage:
                self.stats['cache_misses'] += 1
                return None
            
            data = self._redis_storage[key]
            
            # 检查过期
            if data['expires_at'] > 0 and time.time() > data['expires_at']:
                del self._redis_storage[key]
                self.stats['cache_misses'] += 1
                return None
            
            value = self._deserialize_value(data['value'])
            
            self.stats['cache_hits'] += 1
            self.stats['redis_operations'] += 1
            
            return value
            
        except Exception as e:
            self.logger.error(f"获取Redis缓存失败: {key} - {e}")
            self.stats['cache_misses'] += 1
            return None
    
    def _delete_redis_cache(self, key: str) -> bool:
        """删除Redis缓存"""
        try:
            if key in self._redis_storage:
                del self._redis_storage[key]
                self.stats['cache_deletes'] += 1
                self.stats['redis_operations'] += 1
                self.stats['total_keys'] = len(self._redis_storage)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"删除Redis缓存失败: {key} - {e}")
            return False
    
    def _exists_redis_cache(self, key: str) -> bool:
        """检查Redis缓存是否存在"""
        try:
            if key not in self._redis_storage:
                return False
            
            data = self._redis_storage[key]
            
            # 检查过期
            if data['expires_at'] > 0 and time.time() > data['expires_at']:
                del self._redis_storage[key]
                return False
            
            self.stats['redis_operations'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"检查Redis缓存存在失败: {key} - {e}")
            return False
    
    def _ttl_redis_cache(self, key: str) -> int:
        """获取Redis缓存TTL"""
        try:
            if key not in self._redis_storage:
                return -2
            
            data = self._redis_storage[key]
            
            if data['expires_at'] <= 0:
                return -1  # 永不过期
            
            remaining = data['expires_at'] - time.time()
            if remaining <= 0:
                del self._redis_storage[key]
                return -2  # 已过期
            
            self.stats['redis_operations'] += 1
            return int(remaining)
            
        except Exception as e:
            self.logger.error(f"获取Redis缓存TTL失败: {key} - {e}")
            return -2
    
    def _set_disk_cache(self, key: str, value: str, ttl: int) -> bool:
        """设置磁盘缓存（占位实现）"""
        # 这里可以实现磁盘缓存逻辑
        self.logger.debug(f"磁盘缓存设置: {key}")
        return True
    
    def _get_disk_cache(self, key: str) -> Optional[Any]:
        """获取磁盘缓存（占位实现）"""
        # 这里可以实现磁盘缓存逻辑
        self.logger.debug(f"磁盘缓存获取: {key}")
        return None
    
    def _delete_disk_cache(self, key: str) -> bool:
        """删除磁盘缓存（占位实现）"""
        # 这里可以实现磁盘缓存逻辑
        self.logger.debug(f"磁盘缓存删除: {key}")
        return True
    
    def _exists_disk_cache(self, key: str) -> bool:
        """检查磁盘缓存是否存在（占位实现）"""
        # 这里可以实现磁盘缓存逻辑
        return False
    
    def _ttl_disk_cache(self, key: str) -> int:
        """获取磁盘缓存TTL（占位实现）"""
        # 这里可以实现磁盘缓存逻辑
        return -2
    
    def _serialize_value(self, value: Any) -> str:
        """序列化值"""
        try:
            if self.cache_config.serialize_json:
                return json.dumps(value, ensure_ascii=False)
            else:
                return str(value)
                
        except Exception as e:
            self.logger.error(f"序列化值失败: {e}")
            return str(value)
    
    def _deserialize_value(self, value: str) -> Any:
        """反序列化值"""
        try:
            if self.cache_config.serialize_json:
                return json.loads(value)
            else:
                return value
                
        except Exception as e:
            self.logger.error(f"反序列化值失败: {e}")
            return value
    
    def _evict_lru_entries(self):
        """驱逐最少使用的条目"""
        try:
            if not self._memory_cache:
                return
            
            # 按访问次数和最后访问时间排序
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_accessed)
            )
            
            # 删除最少使用的条目（删除10%的条目）
            evict_count = max(1, len(sorted_entries) // 10)
            
            for i in range(evict_count):
                key = sorted_entries[i][0]
                del self._memory_cache[key]
                self.stats['cache_deletes'] += 1
            
            self.logger.debug(f"驱逐LRU条目: {evict_count} 个")
            
        except Exception as e:
            self.logger.error(f"驱逐LRU条目失败: {e}")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        hit_rate = (self.stats['cache_hits'] / 
                   (self.stats['cache_hits'] + self.stats['cache_misses']) 
                   if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0)
        
        lock_success_rate = (self.stats['lock_acquisitions'] / 
                           (self.stats['lock_acquisitions'] + self.stats['lock_failures'])
                           if (self.stats['lock_acquisitions'] + self.stats['lock_failures']) > 0 else 0)
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_sets': self.stats['cache_sets'],
            'cache_deletes': self.stats['cache_deletes'],
            'hit_rate': hit_rate,
            'memory_usage': self.stats['memory_usage'],
            'memory_max_size': self.cache_config.max_memory_size,
            'redis_operations': self.stats['redis_operations'],
            'total_keys': self.stats['total_keys'],
            'lock_acquisitions': self.stats['lock_acquisitions'],
            'lock_failures': self.stats['lock_failures'],
            'lock_success_rate': lock_success_rate,
            'active_locks': len(self._redis_locks)
        }
    
    def get_connection_info(self) -> Dict:
        """获取连接信息"""
        return {
            'connected': self.connected,
            'connection_pool_size': self.connection_pool_size,
            'redis_version': "模拟版本 7.0.0",
            'memory_usage_mb': len(str(self._redis_storage)) / 1024 / 1024,
            'uptime_seconds': int(time.time() - self.stats.get('start_time', time.time())),
            'total_commands_processed': self.stats['redis_operations']
        }
    
    def ping(self) -> bool:
        """检查连接状态"""
        try:
            return self.connected
        except Exception as e:
            self.logger.error(f"Ping失败: {e}")
            return False
    
    def get_status(self) -> Dict:
        """获取Redis客户端状态"""
        return {
            'connected': self.connected,
            'total_operations': self.stats['redis_operations'],
            'memory_cache_size': len(self._memory_cache),
            'redis_storage_size': len(self._redis_storage),
            'active_locks': len(self._redis_locks),
            'cache_hit_rate': (self.stats['cache_hits'] / 
                             (self.stats['cache_hits'] + self.stats['cache_misses'])
                             if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0),
            'is_operational': True
        } 