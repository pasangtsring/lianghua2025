"""
API客户端模块
功能：Binance API连接、数据缓存、错误处理、冗余备份机制
作者：Trading System Team
创建时间：2025-01-28
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import redis
import threading
from pathlib import Path
import ccxt
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceOrderException

# 导入自定义模块
import sys
sys.path.append('..')
from utils.logger import get_logger, log_system_performance, performance_monitor
from config.config_manager import ConfigManager

class APIProvider(Enum):
    """API提供商枚举"""
    BINANCE = "binance"
    COINGECKO = "coingecko"
    CCXT = "ccxt"

class DataType(Enum):
    """数据类型枚举"""
    KLINES = "klines"
    TICKER = "ticker"
    FUNDING_RATE = "funding_rate"
    ACCOUNT_INFO = "account_info"
    POSITIONS = "positions"
    ORDERS = "orders"
    TRADE_HISTORY = "trade_history"
    MARKET_DATA = "market_data"

@dataclass
class APIResponse:
    """API响应数据结构"""
    success: bool
    data: Any
    error_message: str = ""
    status_code: int = 200
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "status_code": self.status_code,
            "timestamp": self.timestamp,
            "source": self.source,
            "cached": self.cached
        }

class RateLimiter:
    """API速率限制器"""
    
    def __init__(self, max_requests: int = 1200, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def can_make_request(self) -> bool:
        """检查是否可以发起请求"""
        with self.lock:
            now = time.time()
            # 清除过期请求
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            return len(self.requests) < self.max_requests
    
    def add_request(self) -> None:
        """添加请求记录"""
        with self.lock:
            self.requests.append(time.time())
    
    async def wait_if_needed(self) -> None:
        """如果需要则等待"""
        if not self.can_make_request():
            # 计算等待时间
            if self.requests:
                oldest_request = min(self.requests)
                wait_time = self.time_window - (time.time() - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

class DataCache:
    """数据缓存系统"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 memory_cache_size: int = 1000):
        self.redis_client = redis_client
        self.memory_cache = {}
        self.memory_cache_size = memory_cache_size
        self.cache_times = {}
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def _generate_cache_key(self, data_type: DataType, symbol: str, 
                          interval: str = "", **kwargs) -> str:
        """生成缓存键"""
        key_parts = [data_type.value, symbol, interval]
        if kwargs:
            key_parts.append(json.dumps(kwargs, sort_keys=True))
        return ":".join(key_parts)
    
    def get(self, data_type: DataType, symbol: str, interval: str = "", 
            max_age: int = 300, **kwargs) -> Optional[Any]:
        """获取缓存数据"""
        cache_key = self._generate_cache_key(data_type, symbol, interval, **kwargs)
        
        try:
            # 先尝试内存缓存
            with self.lock:
                if cache_key in self.memory_cache:
                    cached_time = self.cache_times.get(cache_key, 0)
                    if time.time() - cached_time < max_age:
                        return self.memory_cache[cache_key]
                    else:
                        # 过期数据，删除
                        del self.memory_cache[cache_key]
                        del self.cache_times[cache_key]
            
            # 尝试Redis缓存
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    try:
                        data = json.loads(cached_data)
                        # 添加到内存缓存
                        self._add_to_memory_cache(cache_key, data)
                        return data
                    except json.JSONDecodeError:
                        self.logger.warning(f"Redis缓存数据解析失败: {cache_key}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取缓存数据失败: {e}")
            return None
    
    def set(self, data_type: DataType, symbol: str, data: Any, 
            interval: str = "", ttl: int = 300, **kwargs) -> bool:
        """设置缓存数据"""
        cache_key = self._generate_cache_key(data_type, symbol, interval, **kwargs)
        
        try:
            # 添加到内存缓存
            self._add_to_memory_cache(cache_key, data)
            
            # 添加到Redis缓存
            if self.redis_client:
                self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(data, default=str)
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置缓存数据失败: {e}")
            return False
    
    def _add_to_memory_cache(self, cache_key: str, data: Any) -> None:
        """添加到内存缓存"""
        with self.lock:
            # 如果缓存已满，删除最旧的数据
            if len(self.memory_cache) >= self.memory_cache_size:
                oldest_key = min(self.cache_times.keys(), 
                               key=lambda k: self.cache_times[k])
                del self.memory_cache[oldest_key]
                del self.cache_times[oldest_key]
            
            self.memory_cache[cache_key] = data
            self.cache_times[cache_key] = time.time()
    
    def invalidate(self, data_type: DataType, symbol: str, 
                  interval: str = "", **kwargs) -> bool:
        """删除缓存数据"""
        cache_key = self._generate_cache_key(data_type, symbol, interval, **kwargs)
        
        try:
            # 从内存缓存删除
            with self.lock:
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    del self.cache_times[cache_key]
            
            # 从Redis缓存删除
            if self.redis_client:
                self.redis_client.delete(cache_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"删除缓存数据失败: {e}")
            return False

class BaseAPIClient(ABC):
    """API客户端基类"""
    
    def __init__(self, provider: APIProvider, config: Dict[str, Any]):
        self.provider = provider
        self.config = config
        self.logger = get_logger(f"{__name__}.{provider.value}")
        self.rate_limiter = RateLimiter(
            max_requests=config.get('rate_limit', 1200),
            time_window=60
        )
        self.session = None
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms最小间隔
        
        # 初始化缓存
        self.cache = DataCache()
        
        # 连接统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_requests': 0,
            'last_error': None,
            'last_error_time': 0
        }
    
    @abstractmethod
    async def _make_request(self, method: str, endpoint: str, 
                           params: Dict[str, Any] = None, 
                           headers: Dict[str, str] = None) -> APIResponse:
        """发起API请求"""
        pass
    
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 1000, start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> APIResponse:
        """获取K线数据"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> APIResponse:
        """获取ticker数据"""
        pass
    
    async def _rate_limit_check(self) -> None:
        """速率限制检查"""
        # 检查最小请求间隔
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        
        # 检查速率限制
        await self.rate_limiter.wait_if_needed()
        
        # 更新请求时间
        self.last_request_time = time.time()
        self.rate_limiter.add_request()
    
    def _update_stats(self, success: bool, cached: bool = False) -> None:
        """更新统计信息"""
        self.stats['total_requests'] += 1
        
        if cached:
            self.stats['cached_requests'] += 1
        elif success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
            
    def _log_response_summary(self, data):
        """智能地记录API响应摘要，避免冗长的日志输出"""
        try:
            if isinstance(data, dict):
                # 检查是否是账户信息响应（包含大量持仓数据）
                if 'positions' in data and isinstance(data['positions'], list):
                    non_zero_positions = [p for p in data['positions'] if float(p.get('positionAmt', 0)) != 0]
                    self.logger.info(f"   📄 响应摘要: 账户信息 - 总持仓数:{len(data['positions'])}, 活跃持仓:{len(non_zero_positions)}")
                    if non_zero_positions:
                        for pos in non_zero_positions[:5]:  # 只显示前5个活跃持仓
                            symbol = pos.get('symbol', 'N/A')
                            side = pos.get('positionSide', 'N/A')
                            amt = pos.get('positionAmt', '0')
                            self.logger.info(f"   📊 活跃持仓: {symbol} {side} {amt}")
                # 检查是否是普通账户信息
                elif any(key in data for key in ['totalWalletBalance', 'availableBalance']):
                    balance = data.get('totalWalletBalance', 'N/A')
                    available = data.get('availableBalance', 'N/A')
                    self.logger.info(f"   📄 响应摘要: 账户余额 - 总资金:{balance} USDT, 可用:{available} USDT")
                # 检查是否是订单响应
                elif any(key in data for key in ['orderId', 'symbol', 'status']):
                    order_id = data.get('orderId', 'N/A')
                    symbol = data.get('symbol', 'N/A')
                    status = data.get('status', 'N/A')
                    self.logger.info(f"   📄 响应摘要: 订单 - ID:{order_id}, 品种:{symbol}, 状态:{status}")
                # 其他小型响应直接显示
                elif len(str(data)) < 500:  # 小于500字符的响应直接显示
                    self.logger.info(f"   📄 响应数据: {data}")
                else:
                    # 大型响应只显示键名和数据类型
                    keys_summary = list(data.keys())[:10]  # 只显示前10个键
                    self.logger.info(f"   📄 响应摘要: Dict包含 {len(data)} 个字段 - {keys_summary}...")
            elif isinstance(data, list):
                if len(data) > 10:  # 大型数组只显示摘要
                    self.logger.info(f"   📄 响应摘要: List包含 {len(data)} 个元素")
                    if data and isinstance(data[0], dict):
                        sample_keys = list(data[0].keys())[:5]
                        self.logger.info(f"   📊 数组元素示例字段: {sample_keys}")
                else:
                    self.logger.info(f"   📄 响应数据: {data}")
            else:
                # 简单数据类型直接显示
                self.logger.info(f"   📄 响应数据: {data}")
        except Exception as e:
            self.logger.info(f"   📄 响应数据: <无法解析响应格式: {e}>")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

class BinanceAPIClient(BaseAPIClient):
    """Binance API客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(APIProvider.BINANCE, config)
        
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.base_url = config.get('base_url', 'https://fapi.binance.com')
        
        # 初始化同步客户端（用于某些操作）
        # 临时禁用同步客户端以避免网络连接问题
        # if self.api_key and self.api_secret:
        #     self.sync_client = BinanceClient(
        #         api_key=self.api_key,
        #         api_secret=self.api_secret,
        #         testnet=config.get('testnet', False)
        #     )
        # else:
        #     self.sync_client = None
        self.sync_client = None  # 临时禁用，避免网络连接超时
    
    async def _make_request(self, method: str, endpoint: str, 
                           params: Dict[str, Any] = None, 
                           headers: Dict[str, str] = None) -> APIResponse:
        """发起Binance API请求"""
        await self._rate_limit_check()
        
        if not self.session:
            # VPN代理配置  
            connector = aiohttp.TCPConnector(verify_ssl=False)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        try:
            # 对于需要签名的请求，添加签名
            if headers and 'X-MBX-APIKEY' in headers:
                if 'timestamp' not in params:
                    params['timestamp'] = int(time.time() * 1000)
                    
                # 创建签名用的参数副本（排除已有的signature）
                sign_params = {k: v for k, v in params.items() if k != 'signature'}
                
                # 生成查询字符串用于签名（按币安官方要求）
                from urllib.parse import urlencode
                query_string = urlencode(sign_params)
                
                # 调试日志（简化）
                # self.logger.info(f"🔧 签名调试 - 查询字符串: {query_string}")
                
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                # ✅ 关键修复：签名作为URL参数，而不是POST body参数
                if method.upper() == 'POST':
                    # POST请求：签名添加到URL，参数作为body
                    signed_url = f"{url}?{query_string}&signature={signature}"
                    post_params = None  # 不发送body参数
                else:
                    # GET请求：所有参数包括签名都在URL中
                    signed_url = f"{url}?{query_string}&signature={signature}"
                    post_params = None
                
                # self.logger.info(f"🔧 签名调试 - 生成的签名: {signature[:10]}...")
            else:
                signed_url = url
                post_params = params
            
            # 发起请求
            proxy = 'http://127.0.0.1:7897'  # VPN代理
            
            # 详细的请求调试信息
            self.logger.info(f"🔍 API请求调试信息:")
            self.logger.info(f"   📍 完整URL: {signed_url}")
            self.logger.info(f"   🔧 请求方法: {method}")
            self.logger.info(f"   📋 请求头: {headers}")
            if method.upper() == 'POST':
                self.logger.info(f"   📦 POST数据: {post_params}")
            else:
                self.logger.info(f"   🔗 URL参数: {params}")
            
            # POST请求参数处理
            if method.upper() == 'POST':
                # 对于POST请求，使用签名后的URL，body为空（或指定参数）
                async with self.session.request(
                    method, 
                    signed_url,  # 已包含签名的URL
                    data=post_params,  # 对于期货下单，通常是空body
                    headers=headers,
                    proxy=proxy,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    data = await response.json()
                    self.logger.info(f"   📤 实际发送的请求: {method} {signed_url}")
                    self.logger.info(f"   📊 响应状态: {response.status}")
                    # 对于大型响应，只显示摘要信息而非完整数据
                    self._log_response_summary(data)
            else:
                # 对于GET请求，使用签名后的URL
                final_url = signed_url if headers and 'X-MBX-APIKEY' in headers else url
                final_params = None if headers and 'X-MBX-APIKEY' in headers else params
                
                async with self.session.request(
                    method, 
                    final_url,
                    params=final_params,
                    headers=headers,
                    proxy=proxy,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    data = await response.json()
                    self.logger.info(f"   📤 实际发送的请求: {method} {response.url}")
                    self.logger.info(f"   📊 响应状态: {response.status}")
                    # 对于大型响应，只显示摘要信息而非完整数据
                    self._log_response_summary(data)
                
            if response.status == 200:
                self._update_stats(True)
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status,
                    source=self.provider.value
                )
            else:
                error_message = data.get('msg', f'HTTP {response.status}')
                self.logger.error(f"Binance API错误: {error_message}")
                self._update_stats(False)
                return APIResponse(
                    success=False,
                    data=data,
                    error_message=error_message,
                    status_code=response.status,
                    source=self.provider.value
                )
        except Exception as e:
            self.logger.error(f"API请求异常: {e}")
            self._update_stats(False)
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                status_code=500,
                source=self.provider.value
            )
    
    @performance_monitor
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 1000, start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> APIResponse:
        """获取K线数据"""
        # 检查缓存
        cache_key_params = {
            'limit': limit,
            'start_time': start_time,
            'end_time': end_time
        }
        
        cached_data = self.cache.get(
            DataType.KLINES, 
            symbol, 
            interval, 
            max_age=60,  # 1分钟缓存
            **cache_key_params
        )
        
        if cached_data:
            self._update_stats(True, cached=True)
            return APIResponse(
                success=True,
                data=cached_data,
                source=self.provider.value,
                cached=True
            )
        
        # 构建请求参数
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        # 发起请求
        response = await self._make_request('GET', '/fapi/v1/klines', params)
        
        # 缓存成功的响应
        if response.success:
            self.cache.set(
                DataType.KLINES, 
                symbol, 
                response.data, 
                interval,
                ttl=60,
                **cache_key_params
            )
        
        return response
    
    @performance_monitor
    async def get_ticker(self, symbol: str) -> APIResponse:
        """获取ticker数据"""
        # 检查缓存
        cached_data = self.cache.get(
            DataType.TICKER, 
            symbol, 
            max_age=10  # 10秒缓存
        )
        
        if cached_data:
            self._update_stats(True, cached=True)
            return APIResponse(
                success=True,
                data=cached_data,
                source=self.provider.value,
                cached=True
            )
        
        # 发起请求
        params = {'symbol': symbol}
        response = await self._make_request('GET', '/fapi/v1/ticker/24hr', params)
        
        # 缓存成功的响应
        if response.success:
            self.cache.set(
                DataType.TICKER, 
                symbol, 
                response.data, 
                ttl=10
            )
        
        return response
    
    @performance_monitor
    async def get_funding_rate(self, symbol: str) -> APIResponse:
        """获取资金费率"""
        # 检查缓存
        cached_data = self.cache.get(
            DataType.FUNDING_RATE, 
            symbol, 
            max_age=300  # 5分钟缓存
        )
        
        if cached_data:
            self._update_stats(True, cached=True)
            return APIResponse(
                success=True,
                data=cached_data,
                source=self.provider.value,
                cached=True
            )
        
        # 发起请求
        params = {'symbol': symbol}
        response = await self._make_request('GET', '/fapi/v1/premiumIndex', params)
        
        # 缓存成功的响应
        if response.success:
            self.cache.set(
                DataType.FUNDING_RATE, 
                symbol, 
                response.data, 
                ttl=300
            )
        
        return response
    
    @performance_monitor
    async def get_account_info(self) -> APIResponse:
        """获取账户信息"""
        if not self.api_key or not self.api_secret:
            return APIResponse(
                success=False,
                data=None,
                error_message="缺少API密钥",
                source=self.provider.value
            )
        
        # 准备参数
        timestamp = int(time.time() * 1000)
        params = {
            'timestamp': timestamp
        }
        
        # 生成签名
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        # 设置请求头
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        # 发起请求
        response = await self._make_request('GET', '/fapi/v2/account', params=params, headers=headers)
        return response
    
    @performance_monitor
    async def get_positions(self) -> APIResponse:
        """获取持仓信息"""
        if not self.api_key or not self.api_secret:
            return APIResponse(
                success=False,
                data=None,
                error_message="缺少API密钥",
                source=self.provider.value
            )
        
        # 准备参数并签名
        timestamp = int(time.time() * 1000)
        params = {
            'timestamp': timestamp
        }
        
        # 生成签名
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        # 设置请求头
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        # 发起请求
        response = await self._make_request('GET', '/fapi/v2/positionRisk', params=params, headers=headers)
        return response
    
    @performance_monitor
    async def place_order(self, order_params: dict) -> APIResponse:
        """下单 - 期货市价单"""
        if not self.api_key or not self.api_secret:
            return APIResponse(
                success=False,
                data=None,
                error_message="缺少API密钥",
                source=self.provider.value
            )
        
        try:
            # 准备参数并签名
            timestamp = int(time.time() * 1000)
            order_params['timestamp'] = timestamp
            
            # 生成查询字符串
            query_string = '&'.join([f"{k}={v}" for k, v in order_params.items()])
            
            # 生成签名
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            order_params['signature'] = signature
            
            # 设置请求头
            headers = {
                'X-MBX-APIKEY': self.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # 发起POST请求下单
            self.logger.info(f"正在提交订单: {order_params['symbol']} {order_params['side']} {order_params.get('quantity', 'N/A')}")
            response = await self._make_request('POST', '/fapi/v1/order', params=order_params, headers=headers)
            
            if response.success:
                self.logger.info(f"订单提交成功: {response.data}")
            else:
                self.logger.error(f"订单提交失败: {response.error_message}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"下单异常: {e}")
            return APIResponse(
                success=False,
                data=None,
                error_message=f"下单异常: {e}",
                source=self.provider.value
            )
    
    async def close(self) -> None:
        """关闭客户端"""
        if self.session:
            await self.session.close()

class CoinGeckoAPIClient(BaseAPIClient):
    """CoinGecko API客户端（备份数据源）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(APIProvider.COINGECKO, config)
        
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.coingecko.com/api/v3')
        
        # CoinGecko符号映射
        self.symbol_mapping = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'BNBUSDT': 'binancecoin',
            'ADAUSDT': 'cardano',
            'SOLUSDT': 'solana',
            'DOTUSDT': 'polkadot',
            'DOGEUSDT': 'dogecoin',
            'AVAXUSDT': 'avalanche-2',
            'LINKUSDT': 'chainlink',
            'UNIUSDT': 'uniswap'
        }
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换符号到CoinGecko格式"""
        return self.symbol_mapping.get(symbol.upper(), symbol.lower().replace('usdt', ''))
    
    async def _make_request(self, method: str, endpoint: str, 
                           params: Dict[str, Any] = None, 
                           headers: Dict[str, str] = None) -> APIResponse:
        """发起CoinGecko API请求"""
        await self._rate_limit_check()
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            # 添加API密钥
            if self.api_key:
                headers = headers or {}
                headers['x-cg-demo-api-key'] = self.api_key
            
            # 发起请求
            proxy = 'http://127.0.0.1:7897'  # VPN代理
            async with self.session.request(
                method, 
                url, 
                params=params,
                headers=headers,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                data = await response.json()
                
                if response.status == 200:
                    self._update_stats(True)
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=response.status,
                        source=self.provider.value
                    )
                else:
                    error_message = data.get('error', f'HTTP {response.status}')
                    self.logger.error(f"CoinGecko API错误: {error_message}")
                    self._update_stats(False)
                    
                    return APIResponse(
                        success=False,
                        data=None,
                        error_message=error_message,
                        status_code=response.status,
                        source=self.provider.value
                    )
                    
        except Exception as e:
            error_message = f"请求异常: {str(e)}"
            self.logger.error(f"CoinGecko API异常: {e}")
            self._update_stats(False)
            
            return APIResponse(
                success=False,
                data=None,
                error_message=error_message,
                status_code=500,
                source=self.provider.value
            )
    
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 1000, start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> APIResponse:
        """获取K线数据（CoinGecko格式）"""
        # CoinGecko不支持标准的K线数据，返回价格历史
        coin_id = self._convert_symbol(symbol)
        
        # 检查缓存
        cached_data = self.cache.get(
            DataType.KLINES, 
            symbol, 
            interval, 
            max_age=300
        )
        
        if cached_data:
            self._update_stats(True, cached=True)
            return APIResponse(
                success=True,
                data=cached_data,
                source=self.provider.value,
                cached=True
            )
        
        # 构建请求参数
        params = {
            'vs_currency': 'usd',
            'days': '1',
            'interval': 'hourly' if interval in ['1h', '4h'] else 'daily'
        }
        
        # 发起请求
        response = await self._make_request(
            'GET', 
            f'/coins/{coin_id}/market_chart', 
            params
        )
        
        # 缓存成功的响应
        if response.success:
            self.cache.set(
                DataType.KLINES, 
                symbol, 
                response.data, 
                interval,
                ttl=300
            )
        
        return response
    
    async def get_ticker(self, symbol: str) -> APIResponse:
        """获取ticker数据"""
        coin_id = self._convert_symbol(symbol)
        
        # 检查缓存
        cached_data = self.cache.get(
            DataType.TICKER, 
            symbol, 
            max_age=60
        )
        
        if cached_data:
            self._update_stats(True, cached=True)
            return APIResponse(
                success=True,
                data=cached_data,
                source=self.provider.value,
                cached=True
            )
        
        # 发起请求
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true'
        }
        
        response = await self._make_request(
            'GET', 
            '/simple/price', 
            params
        )
        
        # 缓存成功的响应
        if response.success:
            self.cache.set(
                DataType.TICKER, 
                symbol, 
                response.data, 
                ttl=60
            )
        
        return response
    
    async def close(self) -> None:
        """关闭客户端"""
        if self.session:
            await self.session.close()

class APIClientManager:
    """API客户端管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ConfigManager().get_config()
        self.logger = get_logger(__name__)
        
        # 初始化客户端
        self.clients = {}
        self.primary_client = None
        self.backup_clients = []
        
        # 初始化Redis连接
        self.redis_client = None
        self._init_redis()
        
        # 初始化客户端
        self._init_clients()
        
        # 健康检查
        self.health_check_interval = 300  # 5分钟
        self.last_health_check = 0
    
    def _init_redis(self) -> None:
        """初始化Redis连接"""
        try:
            redis_url = self.config.redis_url
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.logger.info("Redis连接成功")
        except Exception as e:
            self.logger.warning(f"Redis连接失败: {e}")
            self.redis_client = None
    
    def _init_clients(self) -> None:
        """初始化API客户端"""
        # 初始化Binance客户端
        try:
            binance_config = self.config.api.get('binance', {})
            binance_client = BinanceAPIClient(binance_config.dict())
            binance_client.cache.redis_client = self.redis_client
            
            self.clients[APIProvider.BINANCE] = binance_client
            self.primary_client = binance_client
            
            self.logger.info("Binance客户端初始化成功")
        except Exception as e:
            self.logger.error(f"Binance客户端初始化失败: {e}")
        
        # 初始化CoinGecko客户端（备份）
        try:
            coingecko_config = self.config.api.get('coingecko', {})
            coingecko_client = CoinGeckoAPIClient(coingecko_config.dict())
            coingecko_client.cache.redis_client = self.redis_client
            
            self.clients[APIProvider.COINGECKO] = coingecko_client
            self.backup_clients.append(coingecko_client)
            
            self.logger.info("CoinGecko客户端初始化成功")
        except Exception as e:
            self.logger.error(f"CoinGecko客户端初始化失败: {e}")
    
    async def _health_check(self) -> Dict[APIProvider, bool]:
        """健康检查"""
        health_status = {}
        
        for provider, client in self.clients.items():
            try:
                # 简单的健康检查
                if provider == APIProvider.BINANCE:
                    response = await client.get_ticker('BTCUSDT')
                elif provider == APIProvider.COINGECKO:
                    response = await client.get_ticker('BTCUSDT')
                else:
                    response = APIResponse(success=True, data=None)
                
                health_status[provider] = response.success
                
            except Exception as e:
                self.logger.warning(f"{provider.value}健康检查失败: {e}")
                health_status[provider] = False
        
        self.last_health_check = time.time()
        return health_status
    
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 1000, start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> APIResponse:
        """获取K线数据（带备份）"""
        # 主客户端请求
        if self.primary_client:
            try:
                response = await self.primary_client.get_klines(
                    symbol, interval, limit, start_time, end_time
                )
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"主客户端请求失败: {e}")
        
        # 备份客户端请求
        for backup_client in self.backup_clients:
            try:
                response = await backup_client.get_klines(
                    symbol, interval, limit, start_time, end_time
                )
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"备份客户端请求失败: {e}")
        
        # 所有客户端都失败
        return APIResponse(
            success=False,
            data=None,
            error_message="所有API客户端请求失败",
            source="manager"
        )
    
    async def get_ticker(self, symbol: str) -> APIResponse:
        """获取ticker数据（带备份）"""
        # 主客户端请求
        if self.primary_client:
            try:
                response = await self.primary_client.get_ticker(symbol)
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"主客户端请求失败: {e}")
        
        # 备份客户端请求
        for backup_client in self.backup_clients:
            try:
                response = await backup_client.get_ticker(symbol)
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"备份客户端请求失败: {e}")
        
        # 所有客户端都失败
        return APIResponse(
            success=False,
            data=None,
            error_message="所有API客户端请求失败",
            source="manager"
        )
    
    async def get_funding_rate(self, symbol: str) -> APIResponse:
        """获取资金费率"""
        # 只有Binance支持资金费率
        if self.primary_client and hasattr(self.primary_client, 'get_funding_rate'):
            try:
                response = await self.primary_client.get_funding_rate(symbol)
                return response
            except Exception as e:
                self.logger.warning(f"获取资金费率失败: {e}")
        
        return APIResponse(
            success=False,
            data=None,
            error_message="资金费率获取失败",
            source="manager"
        )
    
    async def get_account_info(self) -> APIResponse:
        """获取账户信息"""
        if self.primary_client and hasattr(self.primary_client, 'get_account_info'):
            try:
                response = await self.primary_client.get_account_info()
                return response
            except Exception as e:
                self.logger.warning(f"获取账户信息失败: {e}")
        
        return APIResponse(
            success=False,
            data=None,
            error_message="账户信息获取失败",
            source="manager"
        )
    
    async def get_positions(self) -> APIResponse:
        """获取持仓信息"""
        if self.primary_client and hasattr(self.primary_client, 'get_positions'):
            try:
                response = await self.primary_client.get_positions()
                return response
            except Exception as e:
                self.logger.warning(f"获取持仓信息失败: {e}")
        
        return APIResponse(
            success=False,
            data=None,
            error_message="持仓信息获取失败",
            source="manager"
        )
    
    async def get_exchange_info(self) -> APIResponse:
        """获取交易所信息"""
        # 主客户端请求
        if self.primary_client:
            try:
                response = await self.primary_client._make_request('GET', '/fapi/v1/exchangeInfo')
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"主客户端获取交易所信息失败: {e}")
        
        # 返回失败响应
        return APIResponse(
            success=False,
            data=None,
            error_message="获取交易所信息失败",
            source="manager"
        )

    async def get_24hr_ticker_stats(self) -> APIResponse:
        """获取24小时价格变动统计"""
        # 主客户端请求
        if self.primary_client:
            try:
                response = await self.primary_client._make_request('GET', '/fapi/v1/ticker/24hr')
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"主客户端获取24小时统计失败: {e}")
        
        # 备用客户端请求 (如果有的话)
        for backup_client in self.backup_clients:
            try:
                # CoinGecko等备用源可能需要不同的处理
                if backup_client.provider == APIProvider.COINGECKO:
                    # 这里可以实现CoinGecko的等效API调用
                    continue
                response = await backup_client._make_request('GET', '/fapi/v1/ticker/24hr')
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"备份客户端获取24小时统计失败: {e}")
        
        # 返回失败响应
        return APIResponse(
            success=False,
            data=None,
            error_message="获取24小时价格统计失败",
            source="manager"
        )
    
    async def place_order(self, order_params: dict) -> APIResponse:
        """下单 - 仅使用主客户端"""
        if self.primary_client and hasattr(self.primary_client, 'place_order'):
            try:
                response = await self.primary_client.place_order(order_params)
                return response
            except Exception as e:
                self.logger.error(f"下单失败: {e}")
                return APIResponse(
                    success=False,
                    data=None,
                    error_message=f"下单失败: {e}",
                    source="manager"
                )
        
        return APIResponse(
            success=False,
            data=None,
            error_message="主客户端不支持下单功能",
            source="manager"
        )
    
    def get_client_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取客户端统计信息"""
        stats = {}
        for provider, client in self.clients.items():
            stats[provider.value] = client.get_stats()
        return stats
    
    async def close(self) -> None:
        """关闭所有客户端"""
        for client in self.clients.values():
            try:
                await client.close()
            except Exception as e:
                self.logger.warning(f"关闭客户端失败: {e}")
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                self.logger.warning(f"关闭Redis连接失败: {e}")

# 全局API客户端管理器
api_client_manager = None

def get_api_client() -> APIClientManager:
    """获取API客户端管理器"""
    global api_client_manager
    if api_client_manager is None:
        api_client_manager = APIClientManager()
    return api_client_manager

async def close_api_client() -> None:
    """关闭API客户端管理器"""
    global api_client_manager
    if api_client_manager:
        await api_client_manager.close()
        api_client_manager = None

if __name__ == "__main__":
    # 测试API客户端
    async def test_api_client():
        client_manager = get_api_client()
        
        # 测试获取K线数据
        klines_response = await client_manager.get_klines('BTCUSDT', '1h', 100)
        print(f"K线数据: {klines_response.success}")
        
        # 测试获取ticker数据
        ticker_response = await client_manager.get_ticker('BTCUSDT')
        print(f"Ticker数据: {ticker_response.success}")
        
        # 测试获取资金费率
        funding_response = await client_manager.get_funding_rate('BTCUSDT')
        print(f"资金费率: {funding_response.success}")
        
        # 获取统计信息
        stats = client_manager.get_client_stats()
        print(f"客户端统计: {stats}")
        
        # 关闭客户端
        await client_manager.close()
    
    asyncio.run(test_api_client()) 