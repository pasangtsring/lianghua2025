"""
数据获取模块包
"""

# 条件导入 - 避免循环依赖和缺失依赖
try:
    from .api_client import APIClient
    _api_client_available = True
except ImportError:
    _api_client_available = False

from .advanced_data_fetcher import (
    XHeatData, 
    LiquidityData, 
    CoinGeckoData, 
    DataValidationResult,
    XHeatFetcher,
    LiquidityDataFetcher,
    EnhancedCoinGeckoFetcher,
    DataValidator,
    AdvancedDataFetcher,
    advanced_data_context
)

__all__ = [
    'XHeatData',
    'LiquidityData',
    'CoinGeckoData',
    'DataValidationResult',
    'XHeatFetcher',
    'LiquidityDataFetcher',
    'EnhancedCoinGeckoFetcher',
    'DataValidator',
    'AdvancedDataFetcher',
    'advanced_data_context'
]

# 如果APIClient可用，则添加到__all__
if _api_client_available:
    __all__.insert(0, 'APIClient') 