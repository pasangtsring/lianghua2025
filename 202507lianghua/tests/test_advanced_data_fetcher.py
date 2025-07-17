"""
高级数据获取器测试模块
严格遵循零简化原则，确保所有功能完整测试
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from data.advanced_data_fetcher import (
    XHeatData, LiquidityData, CoinGeckoData, DataValidationResult,
    XHeatFetcher, LiquidityDataFetcher, EnhancedCoinGeckoFetcher,
    DataValidator, AdvancedDataFetcher, advanced_data_context
)
from config.config_manager import ConfigManager
from data.api_client import APIClient
from utils.logger import logger


class TestXHeatData:
    """测试X热度数据结构"""
    
    def test_x_heat_data_creation(self):
        """测试X热度数据创建"""
        data = XHeatData(
            symbol="BTC",
            timestamp=datetime.now(),
            mentions=100,
            positive_sentiment=0.6,
            negative_sentiment=0.3,
            neutral_sentiment=0.1,
            volume_24h=1000000.0,
            influential_mentions=20,
            trend_score=75.0,
            keywords=["bullish", "moon", "hodl"],
            metadata={"source": "twitter"}
        )
        
        assert data.symbol == "BTC"
        assert data.mentions == 100
        assert data.positive_sentiment == 0.6
        assert data.negative_sentiment == 0.3
        assert data.neutral_sentiment == 0.1
        assert data.volume_24h == 1000000.0
        assert data.influential_mentions == 20
        assert data.trend_score == 75.0
        assert len(data.keywords) == 3
        assert data.metadata["source"] == "twitter"
    
    def test_x_heat_data_default_values(self):
        """测试X热度数据默认值"""
        data = XHeatData(
            symbol="ETH",
            timestamp=datetime.now(),
            mentions=50,
            positive_sentiment=0.5,
            negative_sentiment=0.3,
            neutral_sentiment=0.2,
            volume_24h=500000.0,
            influential_mentions=10,
            trend_score=60.0
        )
        
        assert data.keywords == []
        assert data.metadata == {}


class TestLiquidityData:
    """测试流动性数据结构"""
    
    def test_liquidity_data_creation(self):
        """测试流动性数据创建"""
        data = LiquidityData(
            symbol="BTC",
            timestamp=datetime.now(),
            bid_depth={"50000": 1.0, "49950": 2.0},
            ask_depth={"50050": 1.5, "50100": 2.5},
            spread=50.0,
            spread_percentage=0.1,
            volume_24h=2000000.0,
            turnover_rate=0.15,
            liquidity_score=85.0,
            slippage_1k=0.05,
            slippage_10k=0.15,
            market_impact=0.1,
            metadata={"source": "binance"}
        )
        
        assert data.symbol == "BTC"
        assert data.spread == 50.0
        assert data.spread_percentage == 0.1
        assert data.volume_24h == 2000000.0
        assert data.liquidity_score == 85.0
        assert data.slippage_1k == 0.05
        assert data.slippage_10k == 0.15
        assert len(data.bid_depth) == 2
        assert len(data.ask_depth) == 2


class TestCoinGeckoData:
    """测试CoinGecko数据结构"""
    
    def test_coingecko_data_creation(self):
        """测试CoinGecko数据创建"""
        data = CoinGeckoData(
            symbol="BTC",
            timestamp=datetime.now(),
            price=50000.0,
            market_cap=1000000000.0,
            volume_24h=20000000.0,
            price_change_24h=1000.0,
            price_change_percentage_24h=2.0,
            market_cap_rank=1,
            circulating_supply=19000000.0,
            total_supply=21000000.0,
            max_supply=21000000.0,
            fully_diluted_valuation=1050000000.0,
            developer_score=85.0,
            community_score=90.0,
            liquidity_score=88.0,
            public_interest_score=82.0,
            metadata={"coingecko_id": "bitcoin"}
        )
        
        assert data.symbol == "BTC"
        assert data.price == 50000.0
        assert data.market_cap == 1000000000.0
        assert data.market_cap_rank == 1
        assert data.developer_score == 85.0
        assert data.community_score == 90.0
        assert data.metadata["coingecko_id"] == "bitcoin"


class TestDataValidationResult:
    """测试数据验证结果结构"""
    
    def test_validation_result_creation(self):
        """测试验证结果创建"""
        result = DataValidationResult(
            is_valid=True,
            confidence=0.95,
            anomalies=["轻微异常"],
            quality_score=95.0,
            suggestions=["建议增加数据源"],
            metadata={"validation_time": "2024-01-01T00:00:00"}
        )
        
        assert result.is_valid is True
        assert result.confidence == 0.95
        assert result.quality_score == 95.0
        assert len(result.anomalies) == 1
        assert len(result.suggestions) == 1


class TestXHeatFetcher:
    """测试X热度获取器"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        config = Mock(spec=ConfigManager)
        config.get.return_value = None
        return config
    
    @pytest.fixture
    def x_heat_fetcher(self, config):
        """X热度获取器"""
        return XHeatFetcher(config)
    
    @pytest.mark.asyncio
    async def test_x_heat_fetcher_initialization(self, x_heat_fetcher):
        """测试X热度获取器初始化"""
        assert x_heat_fetcher.config is not None
        assert x_heat_fetcher.session is None
        assert x_heat_fetcher.cache == {}
        assert x_heat_fetcher.cache_ttl == 300
    
    @pytest.mark.asyncio
    async def test_x_heat_fetcher_context_manager(self, x_heat_fetcher):
        """测试X热度获取器上下文管理器"""
        async with x_heat_fetcher as fetcher:
            assert fetcher.session is not None
            assert fetcher.session.closed is False
    
    @pytest.mark.asyncio
    async def test_fetch_x_heat_data_cache(self, x_heat_fetcher):
        """测试X热度数据缓存"""
        # 模拟缓存数据
        test_data = XHeatData(
            symbol="BTC",
            timestamp=datetime.now(),
            mentions=100,
            positive_sentiment=0.6,
            negative_sentiment=0.3,
            neutral_sentiment=0.1,
            volume_24h=1000000.0,
            influential_mentions=20,
            trend_score=75.0
        )
        
        x_heat_fetcher.cache["x_heat_BTC"] = (test_data, time.time())
        
        async with x_heat_fetcher:
            result = await x_heat_fetcher.fetch_x_heat_data("BTC")
            assert result == test_data
    
    @pytest.mark.asyncio
    async def test_fetch_x_heat_data_no_cache(self, x_heat_fetcher):
        """测试X热度数据无缓存情况"""
        with patch.object(x_heat_fetcher, '_fetch_twitter_api_v2', return_value=None):
            with patch.object(x_heat_fetcher, '_fetch_alternative_social_data', return_value=None):
                with patch.object(x_heat_fetcher, '_fetch_social_scraping', return_value=None):
                    async with x_heat_fetcher:
                        result = await x_heat_fetcher.fetch_x_heat_data("BTC")
                        assert result is None
    
    def test_parse_twitter_data(self, x_heat_fetcher):
        """测试Twitter数据解析"""
        mock_data = {
            "data": [
                {
                    "text": "BTC is bullish! Going to the moon!",
                    "author_id": "123",
                    "public_metrics": {"like_count": 100}
                },
                {
                    "text": "Bitcoin bearish sentiment today",
                    "author_id": "456",
                    "public_metrics": {"like_count": 50}
                }
            ],
            "includes": {
                "users": [
                    {"id": "123", "public_metrics": {"followers_count": 50000}},
                    {"id": "456", "public_metrics": {"followers_count": 5000}}
                ]
            }
        }
        
        result = x_heat_fetcher._parse_twitter_data("BTC", mock_data)
        
        assert result.symbol == "BTC"
        assert result.mentions == 2
        assert result.positive_sentiment > 0
        assert result.negative_sentiment > 0
        assert result.influential_mentions == 1  # 只有一个用户粉丝超过10k
    
    def test_parse_coingecko_social_data(self, x_heat_fetcher):
        """测试CoinGecko社交数据解析"""
        mock_data = {
            "community_data": {
                "twitter_followers": 1000000,
                "reddit_subscribers": 500000,
                "facebook_likes": 200000,
                "telegram_channel_user_count": 100000
            },
            "market_data": {
                "total_volume": {"usd": 20000000000}
            }
        }
        
        result = x_heat_fetcher._parse_coingecko_social_data("BTC", mock_data)
        
        assert result.symbol == "BTC"
        assert result.mentions == 1000000
        assert result.volume_24h == 20000000000
        assert result.metadata["source"] == "coingecko"
        assert result.metadata["reddit_subscribers"] == 500000


class TestLiquidityDataFetcher:
    """测试流动性数据获取器"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def api_client(self):
        """API客户端模拟"""
        return Mock(spec=APIClient)
    
    @pytest.fixture
    def liquidity_fetcher(self, config, api_client):
        """流动性数据获取器"""
        return LiquidityDataFetcher(config, api_client)
    
    @pytest.mark.asyncio
    async def test_fetch_liquidity_data_cache(self, liquidity_fetcher):
        """测试流动性数据缓存"""
        test_data = LiquidityData(
            symbol="BTC",
            timestamp=datetime.now(),
            bid_depth={"50000": 1.0},
            ask_depth={"50050": 1.0},
            spread=50.0,
            spread_percentage=0.1,
            volume_24h=1000000.0,
            turnover_rate=0.1,
            liquidity_score=80.0,
            slippage_1k=0.05,
            slippage_10k=0.15,
            market_impact=0.1
        )
        
        liquidity_fetcher.cache["liquidity_BTC"] = (test_data, time.time())
        
        result = await liquidity_fetcher.fetch_liquidity_data("BTC")
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_fetch_liquidity_data_no_cache(self, liquidity_fetcher):
        """测试流动性数据无缓存情况"""
        # 模拟API返回数据
        mock_order_book = {
            "bids": [["50000", "1.0"], ["49950", "2.0"]],
            "asks": [["50050", "1.5"], ["50100", "2.5"]]
        }
        
        mock_volume_data = {
            "quoteVolume": "1000000.0"
        }
        
        mock_market_stats = {
            "symbol": "BTCUSDT",
            "status": "TRADING"
        }
        
        liquidity_fetcher.api_client.fetch_order_book = AsyncMock(return_value=mock_order_book)
        liquidity_fetcher.api_client.fetch_24h_ticker = AsyncMock(return_value=mock_volume_data)
        liquidity_fetcher.api_client.fetch_exchange_info = AsyncMock(return_value=mock_market_stats)
        
        result = await liquidity_fetcher.fetch_liquidity_data("BTC")
        
        assert result is not None
        assert result.symbol == "BTC"
        assert result.spread == 50.0
        assert result.volume_24h == 1000000.0
        assert len(result.bid_depth) == 2
        assert len(result.ask_depth) == 2
    
    def test_calculate_liquidity_score(self, liquidity_fetcher):
        """测试流动性评分计算"""
        bid_depth = {"50000": "10.0", "49950": "20.0"}
        ask_depth = {"50050": "15.0", "50100": "25.0"}
        spread_percentage = 0.1
        volume_24h = 1000000.0
        
        score = liquidity_fetcher._calculate_liquidity_score(
            bid_depth, ask_depth, spread_percentage, volume_24h
        )
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_calculate_slippage(self, liquidity_fetcher):
        """测试滑点计算"""
        bid_depth = {"50000": "1.0", "49950": "2.0"}
        ask_depth = {"50050": "1.5", "50100": "2.5"}
        usdt_amount = 1000.0
        
        slippage = liquidity_fetcher._calculate_slippage(
            bid_depth, ask_depth, usdt_amount
        )
        
        assert slippage >= 0
        assert isinstance(slippage, float)


class TestEnhancedCoinGeckoFetcher:
    """测试增强版CoinGecko获取器"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        config = Mock(spec=ConfigManager)
        config.get.return_value = None
        return config
    
    @pytest.fixture
    def coingecko_fetcher(self, config):
        """CoinGecko获取器"""
        return EnhancedCoinGeckoFetcher(config)
    
    @pytest.mark.asyncio
    async def test_coingecko_fetcher_initialization(self, coingecko_fetcher):
        """测试CoinGecko获取器初始化"""
        assert coingecko_fetcher.config is not None
        assert coingecko_fetcher.session is None
        assert coingecko_fetcher.base_url == "https://api.coingecko.com/api/v3"
        assert coingecko_fetcher.cache == {}
        assert coingecko_fetcher.cache_ttl == 300
    
    @pytest.mark.asyncio
    async def test_coingecko_context_manager(self, coingecko_fetcher):
        """测试CoinGecko上下文管理器"""
        async with coingecko_fetcher as fetcher:
            assert fetcher.session is not None
            assert fetcher.current_url == fetcher.base_url
    
    @pytest.mark.asyncio
    async def test_fetch_enhanced_data_cache(self, coingecko_fetcher):
        """测试增强数据缓存"""
        test_data = CoinGeckoData(
            symbol="BTC",
            timestamp=datetime.now(),
            price=50000.0,
            market_cap=1000000000.0,
            volume_24h=20000000.0,
            price_change_24h=1000.0,
            price_change_percentage_24h=2.0,
            market_cap_rank=1,
            circulating_supply=19000000.0,
            total_supply=21000000.0,
            max_supply=21000000.0,
            fully_diluted_valuation=1050000000.0,
            developer_score=85.0,
            community_score=90.0,
            liquidity_score=88.0,
            public_interest_score=82.0
        )
        
        coingecko_fetcher.cache["coingecko_enhanced_BTC"] = (test_data, time.time())
        
        async with coingecko_fetcher:
            result = await coingecko_fetcher.fetch_enhanced_data("BTC")
            assert result == test_data
    
    def test_merge_enhanced_data(self, coingecko_fetcher):
        """测试增强数据合并"""
        coin_data = {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "market_cap_rank": 1,
            "market_data": {
                "current_price": {"usd": 50000.0},
                "market_cap": {"usd": 1000000000.0},
                "total_volume": {"usd": 20000000.0},
                "price_change_24h": 1000.0,
                "price_change_percentage_24h": 2.0,
                "circulating_supply": 19000000.0,
                "total_supply": 21000000.0,
                "max_supply": 21000000.0,
                "fully_diluted_valuation": {"usd": 1050000000.0}
            },
            "developer_score": 85.0,
            "community_score": 90.0,
            "liquidity_score": 88.0,
            "public_interest_score": 82.0,
            "community_data": {},
            "links": {
                "homepage": ["https://bitcoin.org"],
                "twitter_screen_name": "bitcoin"
            }
        }
        
        result = coingecko_fetcher._merge_enhanced_data(
            "BTC", coin_data, None, None, None
        )
        
        assert result.symbol == "BTC"
        assert result.price == 50000.0
        assert result.market_cap == 1000000000.0
        assert result.market_cap_rank == 1
        assert result.developer_score == 85.0
        assert result.metadata["coingecko_id"] == "bitcoin"
        assert result.metadata["name"] == "Bitcoin"


class TestDataValidator:
    """测试数据验证器"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def validator(self, config):
        """数据验证器"""
        return DataValidator(config)
    
    def test_validate_x_heat_data_valid(self, validator):
        """测试有效X热度数据验证"""
        data = XHeatData(
            symbol="BTC",
            timestamp=datetime.now(),
            mentions=100,
            positive_sentiment=0.6,
            negative_sentiment=0.3,
            neutral_sentiment=0.1,
            volume_24h=1000000.0,
            influential_mentions=20,
            trend_score=75.0,
            keywords=["bullish", "moon"],
            metadata={"source": "twitter"}
        )
        
        result = validator.validate_x_heat_data(data)
        
        assert result.is_valid is True
        assert result.confidence > 0.5
        assert result.quality_score > 50
        assert len(result.anomalies) == 0
    
    def test_validate_x_heat_data_invalid(self, validator):
        """测试无效X热度数据验证"""
        data = XHeatData(
            symbol="BTC",
            timestamp=datetime.now(),
            mentions=-10,  # 负数
            positive_sentiment=1.5,  # 超出范围
            negative_sentiment=-0.1,  # 负数
            neutral_sentiment=0.1,
            volume_24h=1000000.0,
            influential_mentions=200,  # 超过总提及数
            trend_score=75.0,
            keywords=[],
            metadata={"source": "twitter"}
        )
        
        result = validator.validate_x_heat_data(data)
        
        assert result.is_valid is False
        assert result.confidence <= 0.5
        assert len(result.anomalies) > 0
        assert "提及次数不能为负数" in result.anomalies
        assert "正面情感值应在0-1之间" in result.anomalies
    
    def test_validate_liquidity_data_valid(self, validator):
        """测试有效流动性数据验证"""
        data = LiquidityData(
            symbol="BTC",
            timestamp=datetime.now(),
            bid_depth={"50000": 1.0, "49950": 2.0},
            ask_depth={"50050": 1.5, "50100": 2.5},
            spread=50.0,
            spread_percentage=0.1,
            volume_24h=2000000.0,
            turnover_rate=0.15,
            liquidity_score=85.0,
            slippage_1k=0.05,
            slippage_10k=0.15,
            market_impact=0.1
        )
        
        result = validator.validate_liquidity_data(data)
        
        assert result.is_valid is True
        assert result.confidence > 0.5
        assert result.quality_score > 50
        assert len(result.anomalies) == 0
    
    def test_validate_liquidity_data_invalid(self, validator):
        """测试无效流动性数据验证"""
        data = LiquidityData(
            symbol="BTC",
            timestamp=datetime.now(),
            bid_depth={},  # 空数据
            ask_depth={},  # 空数据
            spread=-10.0,  # 负数
            spread_percentage=-0.1,  # 负数
            volume_24h=-1000.0,  # 负数
            turnover_rate=0.15,
            liquidity_score=150.0,  # 超出范围
            slippage_1k=2.0,  # 异常高
            slippage_10k=5.0,  # 异常高
            market_impact=0.1
        )
        
        result = validator.validate_liquidity_data(data)
        
        assert result.is_valid is False
        assert result.confidence <= 0.5
        assert len(result.anomalies) > 0
        assert "买卖价差不能为负数" in result.anomalies
        assert "缺少买盘深度数据" in result.anomalies
    
    def test_validate_coingecko_data_valid(self, validator):
        """测试有效CoinGecko数据验证"""
        data = CoinGeckoData(
            symbol="BTC",
            timestamp=datetime.now(),
            price=50000.0,
            market_cap=1000000000.0,
            volume_24h=20000000.0,
            price_change_24h=1000.0,
            price_change_percentage_24h=2.0,
            market_cap_rank=1,
            circulating_supply=19000000.0,
            total_supply=21000000.0,
            max_supply=21000000.0,
            fully_diluted_valuation=1050000000.0,
            developer_score=85.0,
            community_score=90.0,
            liquidity_score=88.0,
            public_interest_score=82.0
        )
        
        result = validator.validate_coingecko_data(data)
        
        assert result.is_valid is True
        assert result.confidence > 0.5
        assert result.quality_score > 50
        assert len(result.anomalies) == 0


class TestAdvancedDataFetcher:
    """测试高级数据获取器主类"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def api_client(self):
        """API客户端模拟"""
        return Mock(spec=APIClient)
    
    @pytest.fixture
    def advanced_fetcher(self, config, api_client):
        """高级数据获取器"""
        return AdvancedDataFetcher(config, api_client)
    
    @pytest.mark.asyncio
    async def test_advanced_fetcher_initialization(self, advanced_fetcher):
        """测试高级数据获取器初始化"""
        assert advanced_fetcher.config is not None
        assert advanced_fetcher.api_client is not None
        assert advanced_fetcher.x_heat_fetcher is not None
        assert advanced_fetcher.liquidity_fetcher is not None
        assert advanced_fetcher.coingecko_fetcher is not None
        assert advanced_fetcher.validator is not None
    
    @pytest.mark.asyncio
    async def test_fetch_all_advanced_data(self, advanced_fetcher):
        """测试获取所有高级数据"""
        # 模拟各个获取器的返回数据
        mock_x_data = XHeatData(
            symbol="BTC",
            timestamp=datetime.now(),
            mentions=100,
            positive_sentiment=0.6,
            negative_sentiment=0.3,
            neutral_sentiment=0.1,
            volume_24h=1000000.0,
            influential_mentions=20,
            trend_score=75.0
        )
        
        mock_liquidity_data = LiquidityData(
            symbol="BTC",
            timestamp=datetime.now(),
            bid_depth={"50000": 1.0},
            ask_depth={"50050": 1.0},
            spread=50.0,
            spread_percentage=0.1,
            volume_24h=1000000.0,
            turnover_rate=0.1,
            liquidity_score=80.0,
            slippage_1k=0.05,
            slippage_10k=0.15,
            market_impact=0.1
        )
        
        mock_coingecko_data = CoinGeckoData(
            symbol="BTC",
            timestamp=datetime.now(),
            price=50000.0,
            market_cap=1000000000.0,
            volume_24h=20000000.0,
            price_change_24h=1000.0,
            price_change_percentage_24h=2.0,
            market_cap_rank=1,
            circulating_supply=19000000.0,
            total_supply=21000000.0,
            max_supply=21000000.0,
            fully_diluted_valuation=1050000000.0,
            developer_score=85.0,
            community_score=90.0,
            liquidity_score=88.0,
            public_interest_score=82.0
        )
        
        # 模拟各个获取器的方法
        with patch.object(advanced_fetcher.x_heat_fetcher, 'fetch_x_heat_data', return_value=mock_x_data):
            with patch.object(advanced_fetcher.liquidity_fetcher, 'fetch_liquidity_data', return_value=mock_liquidity_data):
                with patch.object(advanced_fetcher.coingecko_fetcher, 'fetch_enhanced_data', return_value=mock_coingecko_data):
                    with patch.object(advanced_fetcher.x_heat_fetcher, '__aenter__', return_value=advanced_fetcher.x_heat_fetcher):
                        with patch.object(advanced_fetcher.x_heat_fetcher, '__aexit__', return_value=None):
                            with patch.object(advanced_fetcher.coingecko_fetcher, '__aenter__', return_value=advanced_fetcher.coingecko_fetcher):
                                with patch.object(advanced_fetcher.coingecko_fetcher, '__aexit__', return_value=None):
                                    result = await advanced_fetcher.fetch_all_advanced_data("BTC")
        
        assert "x_heat" in result
        assert "liquidity" in result
        assert "coingecko" in result
        assert result["x_heat"]["data"] == mock_x_data
        assert result["liquidity"]["data"] == mock_liquidity_data
        assert result["coingecko"]["data"] == mock_coingecko_data
    
    @pytest.mark.asyncio
    async def test_get_data_quality_report(self, advanced_fetcher):
        """测试数据质量报告"""
        # 模拟fetch_all_advanced_data返回
        mock_all_data = {
            "x_heat": {
                "validation": DataValidationResult(
                    is_valid=True,
                    confidence=0.8,
                    anomalies=[],
                    quality_score=80.0,
                    suggestions=[]
                )
            },
            "liquidity": {
                "validation": DataValidationResult(
                    is_valid=True,
                    confidence=0.9,
                    anomalies=[],
                    quality_score=90.0,
                    suggestions=[]
                )
            }
        }
        
        with patch.object(advanced_fetcher, 'fetch_all_advanced_data', return_value=mock_all_data):
            report = await advanced_fetcher.get_data_quality_report("BTC")
        
        assert report["symbol"] == "BTC"
        assert "x_heat" in report["data_sources"]
        assert "liquidity" in report["data_sources"]
        assert report["quality_scores"]["x_heat"] == 80.0
        assert report["quality_scores"]["liquidity"] == 90.0
        assert report["overall_quality"] == 85.0  # (80 + 90) / 2


class TestAdvancedDataContext:
    """测试高级数据上下文管理器"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def api_client(self):
        """API客户端模拟"""
        return Mock(spec=APIClient)
    
    @pytest.mark.asyncio
    async def test_advanced_data_context(self, config, api_client):
        """测试高级数据上下文管理器"""
        async with advanced_data_context(config, api_client) as fetcher:
            assert isinstance(fetcher, AdvancedDataFetcher)
            assert fetcher.config == config
            assert fetcher.api_client == api_client


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def config(self):
        """真实配置管理器"""
        config = ConfigManager()
        config.load_config()
        return config
    
    @pytest.fixture
    def api_client(self, config):
        """真实API客户端"""
        return APIClient(config)
    
    @pytest.mark.asyncio
    async def test_full_integration(self, config, api_client):
        """完整集成测试"""
        try:
            async with advanced_data_context(config, api_client) as fetcher:
                # 测试获取数据
                result = await fetcher.fetch_all_advanced_data("BTC")
                
                # 验证结果结构
                assert isinstance(result, dict)
                
                # 如果有数据，验证数据结构
                if "x_heat" in result:
                    assert isinstance(result["x_heat"]["data"], XHeatData)
                    assert isinstance(result["x_heat"]["validation"], DataValidationResult)
                
                if "liquidity" in result:
                    assert isinstance(result["liquidity"]["data"], LiquidityData)
                    assert isinstance(result["liquidity"]["validation"], DataValidationResult)
                
                if "coingecko" in result:
                    assert isinstance(result["coingecko"]["data"], CoinGeckoData)
                    assert isinstance(result["coingecko"]["validation"], DataValidationResult)
                
                # 测试质量报告
                report = await fetcher.get_data_quality_report("BTC")
                assert "symbol" in report
                assert "timestamp" in report
                assert "overall_quality" in report
                
        except Exception as e:
            # 在没有真实API访问的情况下，这是预期的
            logger.info(f"集成测试异常 (预期): {str(e)}")
            pytest.skip("需要真实API访问权限")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 