"""
真实数据提供器
使用币安公开API获取真实市场数据（不需要API密钥）
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time

from utils.logger import get_logger

class RealDataProvider:
    """真实数据提供器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.base_url = "https://fapi.binance.com"
        self.session = None
        self.is_running = False
        
        self.logger.info("真实数据提供器初始化完成")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def get_server_time(self) -> int:
        """获取服务器时间"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/fapi/v1/time") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('serverTime', int(time.time() * 1000))
                else:
                    self.logger.error(f"获取服务器时间失败: {response.status}")
                    return int(time.time() * 1000)
                    
        except Exception as e:
            self.logger.error(f"获取服务器时间失败: {e}")
            return int(time.time() * 1000)
    
    async def get_historical_klines(self, symbol: str = "BTCUSDT", 
                                  interval: str = "1h", limit: int = 500) -> List[List]:
        """获取真实历史K线数据"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1500)  # 币安API限制
            }
            
            self.logger.info(f"获取{symbol}真实历史K线数据，周期：{interval}，数量：{limit}")
            
            async with self.session.get(f"{self.base_url}/fapi/v1/klines", params=params) as response:
                if response.status == 200:
                    klines = await response.json()
                    self.logger.info(f"成功获取{len(klines)}条真实K线数据")
                    return klines
                else:
                    error_text = await response.text()
                    self.logger.error(f"获取历史K线数据失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"获取历史K线数据失败: {e}")
            raise
    
    async def get_symbol_ticker(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """获取真实价格信息"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/fapi/v1/ticker/price", 
                                      params={'symbol': symbol}) as response:
                if response.status == 200:
                    ticker = await response.json()
                    ticker['time'] = int(time.time() * 1000)
                    return ticker
                else:
                    error_text = await response.text()
                    self.logger.error(f"获取价格失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"获取价格失败: {e}")
            raise
    
    async def get_24hr_ticker(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """获取24小时价格变动统计"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/fapi/v1/ticker/24hr",
                                      params={'symbol': symbol}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"获取24h统计失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"获取24h统计失败: {e}")
            raise
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/fapi/v1/exchangeInfo") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"获取交易所信息失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"获取交易所信息失败: {e}")
            raise
    
    async def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 10) -> Dict[str, Any]:
        """获取订单簿"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            async with self.session.get(f"{self.base_url}/fapi/v1/depth", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"获取订单簿失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            raise
    
    async def get_recent_trades(self, symbol: str = "BTCUSDT", limit: int = 100) -> List[Dict]:
        """获取最近成交"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            async with self.session.get(f"{self.base_url}/fapi/v1/trades", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"获取最近成交失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"获取最近成交失败: {e}")
            raise
    
    async def start_realtime_data(self, symbols: List[str], callback):
        """启动实时数据推送（使用WebSocket）"""
        try:
            self.is_running = True
            self.logger.info("启动真实实时数据推送")
            
            # 构建WebSocket URL
            streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
            ws_url = f"wss://fstream.binance.com/ws/{'/'.join(streams)}"
            
            import websockets
            async with websockets.connect(ws_url) as websocket:
                while self.is_running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # 调用回调函数
                        await callback(data)
                        
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.warning("WebSocket连接断开，尝试重连...")
                        break
                    except Exception as e:
                        self.logger.error(f"WebSocket数据处理失败: {e}")
                        await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"实时数据推送失败: {e}")
        finally:
            self.is_running = False
    
    def stop_realtime_data(self):
        """停止实时数据推送"""
        self.is_running = False
        self.logger.info("停止真实实时数据推送")
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            server_time = await self.get_server_time()
            ticker = await self.get_symbol_ticker("BTCUSDT")
            
            self.logger.info(f"连接测试成功 - 服务器时间: {server_time}, BTC价格: {ticker.get('price')}")
            return True
            
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False

# 全局实例
real_data_provider = RealDataProvider() 