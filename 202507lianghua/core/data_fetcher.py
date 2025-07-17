"""
数据获取模块（兼容性文件）
为了保持架构一致性，这里导入已有的高级数据获取器
"""

from ..data.advanced_data_fetcher import AdvancedDataFetcher as DataFetcher

__all__ = ['DataFetcher'] 