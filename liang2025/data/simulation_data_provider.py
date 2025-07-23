"""
模拟数据提供器
在API权限问题时提供模拟数据进行系统测试
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import random
import json

from utils.logger import get_logger

class SimulationDataProvider:
    """模拟数据提供器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.current_price = 43000.0  # BTC起始价格
        self.price_volatility = 0.02  # 价格波动率
        self.is_running = False
        
        # 模拟账户
        self.account_balance = 1000.0  # USDT
        self.positions = {}
        
        self.logger.info("模拟数据提供器初始化完成")
    
    async def get_historical_klines(self, symbol: str = "BTCUSDT", 
                                  interval: str = "1h", limit: int = 500) -> List[List]:
        """获取历史K线数据（模拟）"""
        try:
            self.logger.info(f"生成{symbol}历史K线数据，周期：{interval}，数量：{limit}")
            
            # 生成历史数据
            end_time = datetime.now()
            
            # 根据interval计算时间间隔
            interval_minutes = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "4h": 240, "1d": 1440
            }.get(interval, 60)
            
            klines = []
            base_price = 43000.0  # 起始价格
            
            for i in range(limit):
                # 计算时间戳
                timestamp = end_time - timedelta(minutes=interval_minutes * (limit - i))
                open_time = int(timestamp.timestamp() * 1000)
                close_time = open_time + (interval_minutes * 60 * 1000) - 1
                
                # 生成价格数据（随机游走）
                price_change = random.gauss(0, base_price * 0.01)
                open_price = base_price
                high_price = open_price + random.uniform(0, abs(price_change) + base_price * 0.005)
                low_price = open_price - random.uniform(0, abs(price_change) + base_price * 0.005)
                close_price = open_price + price_change
                
                # 确保high >= max(open, close)，low <= min(open, close)
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                volume = random.uniform(100, 1000)
                
                kline = [
                    open_time,                    # 开盘时间
                    f"{open_price:.2f}",         # 开盘价
                    f"{high_price:.2f}",         # 最高价  
                    f"{low_price:.2f}",          # 最低价
                    f"{close_price:.2f}",        # 收盘价
                    f"{volume:.2f}",             # 成交量
                    close_time,                   # 收盘时间
                    f"{volume * close_price:.2f}", # 成交额
                    random.randint(50, 200),      # 成交笔数
                    f"{volume * 0.6:.2f}",       # 主动买入成交量
                    f"{volume * 0.6 * close_price:.2f}", # 主动买入成交额
                    "0"                          # 忽略
                ]
                
                klines.append(kline)
                base_price = close_price  # 更新基础价格
            
            self.current_price = float(klines[-1][4])  # 更新当前价格
            self.logger.info(f"生成{len(klines)}条K线数据，当前价格：{self.current_price}")
            return klines
            
        except Exception as e:
            self.logger.error(f"生成历史K线数据失败: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息（模拟）"""
        try:
            account_info = {
                "totalWalletBalance": f"{self.account_balance:.8f}",
                "totalUnrealizedProfit": "0.00000000",
                "totalMarginBalance": f"{self.account_balance:.8f}",
                "totalPositionInitialMargin": "0.00000000",
                "totalOpenOrderInitialMargin": "0.00000000",
                "assets": [
                    {
                        "asset": "USDT",
                        "walletBalance": f"{self.account_balance:.8f}",
                        "unrealizedProfit": "0.00000000",
                        "marginBalance": f"{self.account_balance:.8f}",
                        "maintMargin": "0.00000000",
                        "initialMargin": "0.00000000",
                        "positionInitialMargin": "0.00000000",
                        "openOrderInitialMargin": "0.00000000",
                        "crossWalletBalance": f"{self.account_balance:.8f}",
                        "maxWithdrawAmount": f"{self.account_balance:.8f}",
                        "marginAvailable": True
                    }
                ],
                "positions": []
            }
            
            self.logger.info(f"模拟账户余额: {self.account_balance} USDT")
            return account_info
            
        except Exception as e:
            self.logger.error(f"获取模拟账户信息失败: {e}")
            raise
    
    async def get_symbol_ticker(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """获取交易对实时价格（模拟）"""
        try:
            # 模拟价格小幅波动
            price_change = random.gauss(0, self.current_price * 0.001)
            self.current_price += price_change
            
            ticker = {
                "symbol": symbol,
                "price": f"{self.current_price:.2f}",
                "time": int(datetime.now().timestamp() * 1000)
            }
            
            return ticker
            
        except Exception as e:
            self.logger.error(f"获取模拟价格失败: {e}")
            raise
    
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = "MARKET", **kwargs) -> Dict[str, Any]:
        """下单（模拟）"""
        try:
            order_id = random.randint(1000000, 9999999)
            
            # 计算订单价值
            order_value = quantity * self.current_price
            commission = order_value * 0.0004  # 0.04% 手续费
            
            # 模拟订单成功
            order_result = {
                "orderId": order_id,
                "symbol": symbol,
                "status": "FILLED",
                "executedQty": f"{quantity:.6f}",
                "cummulativeQuoteQty": f"{order_value:.8f}",
                "avgPrice": f"{self.current_price:.2f}",
                "origQty": f"{quantity:.6f}",
                "side": side,
                "type": order_type,
                "timeInForce": "GTC",
                "updateTime": int(datetime.now().timestamp() * 1000)
            }
            
            # 更新模拟账户
            if side == "BUY":
                self.account_balance -= (order_value + commission)
            else:
                self.account_balance += (order_value - commission)
            
            self.logger.info(f"模拟订单执行成功: {side} {quantity} {symbol} @ {self.current_price}")
            return order_result
            
        except Exception as e:
            self.logger.error(f"模拟下单失败: {e}")
            raise
    
    async def start_realtime_data(self, symbols: List[str], callback):
        """启动实时数据推送（模拟）"""
        try:
            self.is_running = True
            self.logger.info("启动模拟实时数据推送")
            
            while self.is_running:
                for symbol in symbols:
                    # 模拟价格变化
                    price_change = random.gauss(0, self.current_price * 0.001)
                    self.current_price += price_change
                    
                    # 构建ticker数据
                    ticker_data = {
                        "e": "24hrTicker",
                        "s": symbol,
                        "c": f"{self.current_price:.2f}",
                        "P": f"{price_change/self.current_price*100:.4f}",
                        "v": f"{random.uniform(1000, 5000):.2f}",
                        "E": int(datetime.now().timestamp() * 1000)
                    }
                    
                    # 调用回调函数
                    await callback(ticker_data)
                
                await asyncio.sleep(1)  # 每秒更新一次
                
        except Exception as e:
            self.logger.error(f"模拟实时数据推送失败: {e}")
        finally:
            self.is_running = False
    
    def stop_realtime_data(self):
        """停止实时数据推送"""
        self.is_running = False
        self.logger.info("停止模拟实时数据推送")
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息（模拟）"""
        try:
            exchange_info = {
                "timezone": "UTC",
                "serverTime": int(datetime.now().timestamp() * 1000),
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "TRADING",
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                        "pricePrecision": 2,
                        "quantityPrecision": 6,
                        "filters": [
                            {
                                "filterType": "PRICE_FILTER",
                                "minPrice": "0.01",
                                "maxPrice": "1000000.00",
                                "tickSize": "0.01"
                            },
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.000001",
                                "maxQty": "9000.000000",
                                "stepSize": "0.000001"
                            }
                        ]
                    }
                ]
            }
            
            return exchange_info
            
        except Exception as e:
            self.logger.error(f"获取模拟交易所信息失败: {e}")
            raise

# 全局实例
simulation_provider = SimulationDataProvider() 