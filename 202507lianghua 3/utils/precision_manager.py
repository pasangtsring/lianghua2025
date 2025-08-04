"""
智能精度管理器 - 自动获取和管理币安交易对精度信息
基于用户需求的改进方案1：启动时批量获取 + 智能缓存更新
"""

import time
from typing import Dict, Optional, Any
from utils.logger import get_logger


class SmartPrecisionManager:
    """智能精度管理器"""
    
    def __init__(self):
        """初始化精度管理器"""
        self.logger = get_logger(__name__)
        self.symbol_precision = {}  # 本地缓存 {symbol: precision_info}
        self.last_update = 0        # 最后更新时间戳
        self.update_interval = 3600 # 更新间隔：1小时（3600秒）
        self.api_client = None      # API客户端引用
        
        self.logger.info("智能精度管理器初始化完成")
    
    async def init(self, api_client):
        """
        初始化精度管理器
        
        Args:
            api_client: API客户端实例
        """
        try:
            self.api_client = api_client
            self.logger.info("开始初始化精度数据...")
            
            # 启动时获取精度数据
            await self.refresh_precision_data()
            
            if self.symbol_precision:
                self.logger.info(f"✅ 精度管理器初始化成功，已加载 {len(self.symbol_precision)} 个币种精度信息")
            else:
                self.logger.warning("⚠️ 精度数据获取失败，将使用默认精度")
                
        except Exception as e:
            self.logger.error(f"精度管理器初始化失败: {e}")
    
    async def refresh_precision_data(self):
        """刷新精度数据（从币安获取最新信息）"""
        try:
            self.logger.info("🔄 正在从币安获取最新精度信息...")
            
            # 调用API获取交易所信息
            exchange_info_response = await self.api_client.get_exchange_info()
            
            if not exchange_info_response or not exchange_info_response.success:
                self.logger.error(f"获取交易所信息失败: {exchange_info_response.error_message if exchange_info_response else 'No response'}")
                return
            
            exchange_data = exchange_info_response.data
            if not exchange_data or 'symbols' not in exchange_data:
                self.logger.error("交易所信息格式错误，缺少symbols数据")
                return
            
            # 解析所有USDT交易对的精度信息
            usdt_count = 0
            for symbol_info in exchange_data['symbols']:
                try:
                    symbol = symbol_info.get('symbol', '')
                    status = symbol_info.get('status', '')
                    
                    # 只处理活跃的USDT交易对
                    if symbol.endswith('USDT') and status == 'TRADING':
                        # 基础精度信息
                        quantity_precision = symbol_info.get('quantityPrecision', 2)
                        price_precision = symbol_info.get('pricePrecision', 2)
                        
                        # 从filters中获取更详细的信息
                        min_qty = 0.001
                        max_qty = 1000000.0
                        step_size = None
                        
                        for filter_info in symbol_info.get('filters', []):
                            if filter_info.get('filterType') == 'LOT_SIZE':
                                min_qty = float(filter_info.get('minQty', '0.001'))
                                max_qty = float(filter_info.get('maxQty', '1000000'))
                                step_size = float(filter_info.get('stepSize', '0.001'))
                                
                                # 根据stepSize重新计算精度
                                if step_size:
                                    if step_size >= 1:
                                        quantity_precision = 0
                                    elif step_size >= 0.1:
                                        quantity_precision = 1
                                    elif step_size >= 0.01:
                                        quantity_precision = 2
                                    elif step_size >= 0.001:
                                        quantity_precision = 3
                                    elif step_size >= 0.0001:
                                        quantity_precision = 4
                                    else:
                                        # 计算小数位数
                                        step_str = f"{step_size:.10f}".rstrip('0')
                                        if '.' in step_str:
                                            quantity_precision = len(step_str.split('.')[1])
                                        else:
                                            quantity_precision = 0
                                break
                        
                        # 存储精度信息
                        self.symbol_precision[symbol] = {
                            'quantity_precision': quantity_precision,
                            'price_precision': price_precision,
                            'min_qty': min_qty,
                            'max_qty': max_qty,
                            'step_size': step_size
                        }
                        usdt_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"解析 {symbol} 精度信息失败: {e}")
                    continue
            
            # 更新时间戳
            self.last_update = time.time()
            
            self.logger.info(f"✅ 精度数据更新完成，共处理 {usdt_count} 个USDT交易对")
            
            # 记录一些示例精度信息（用于调试）
            sample_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT']
            for symbol in sample_symbols:
                if symbol in self.symbol_precision:
                    info = self.symbol_precision[symbol]
                    self.logger.info(f"  📋 {symbol}: 数量精度={info['quantity_precision']}, 最小={info['min_qty']}, 最大={info['max_qty']}")
                    
        except Exception as e:
            self.logger.error(f"刷新精度数据失败: {e}")
    
    async def get_quantity_precision(self, symbol: str) -> int:
        """
        获取币种的数量精度（智能缓存）
        
        Args:
            symbol: 交易币种符号（如 'BTCUSDT'）
            
        Returns:
            数量精度（小数位数）
        """
        try:
            # 检查是否需要更新缓存
            if time.time() - self.last_update > self.update_interval:
                self.logger.info("精度数据缓存过期，正在更新...")
                await self.refresh_precision_data()
            
            # 返回精度信息
            precision_info = self.symbol_precision.get(symbol, {})
            return precision_info.get('quantity_precision', 2)  # 默认2位小数
            
        except Exception as e:
            self.logger.error(f"获取 {symbol} 数量精度失败: {e}")
            return 2  # 默认精度
    
    def adjust_quantity(self, symbol: str, quantity: float) -> float:
        """
        调整数量精度（核心方法）
        
        Args:
            symbol: 交易币种符号
            quantity: 原始数量
            
        Returns:  
            调整后的数量
        """
        try:
            # 获取精度信息
            precision_info = self.symbol_precision.get(symbol, {})
            quantity_precision = precision_info.get('quantity_precision', 2)
            min_qty = precision_info.get('min_qty', 0.001)
            max_qty = precision_info.get('max_qty', 1000000.0)
            
            # 精度调整
            if quantity_precision == 0:
                adjusted_qty = int(quantity)
            else:
                adjusted_qty = round(quantity, quantity_precision)
            
            # 范围检查和调整
            if adjusted_qty < min_qty:
                adjusted_qty = min_qty
                self.logger.warning(f"⚠️ {symbol} 数量 {quantity} 小于最小值 {min_qty}，已调整为 {adjusted_qty}")
            elif adjusted_qty > max_qty:
                adjusted_qty = max_qty
                self.logger.warning(f"⚠️ {symbol} 数量 {quantity} 大于最大值 {max_qty}，已调整为 {adjusted_qty}")
            
            # 记录调整信息（仅在有变化时）
            if abs(adjusted_qty - quantity) > 1e-8:  # 避免浮点数精度问题
                self.logger.info(f"📏 {symbol} 数量精度调整: {quantity} → {adjusted_qty} (精度:{quantity_precision}位)")
            
            return adjusted_qty
            
        except Exception as e:
            self.logger.error(f"调整 {symbol} 数量precision失败: {e}")
            # 降级处理：使用简单的2位小数精度
            return round(quantity, 2)
    
    def get_precision_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取币种的完整精度信息
        
        Args:
            symbol: 交易币种符号
            
        Returns:
            精度信息字典
        """
        return self.symbol_precision.get(symbol, {
            'quantity_precision': 2,
            'price_precision': 2,
            'min_qty': 0.001,
            'max_qty': 1000000.0,
            'step_size': 0.001
        })
    
    def is_precision_data_fresh(self) -> bool:
        """检查精度数据是否新鲜"""
        return (time.time() - self.last_update) < self.update_interval
    
    def get_loaded_symbols_count(self) -> int:
        """获取已加载的币种数量"""
        return len(self.symbol_precision)
    
    def has_symbol_precision(self, symbol: str) -> bool:
        """检查是否有指定币种的精度信息"""
        return symbol in self.symbol_precision 