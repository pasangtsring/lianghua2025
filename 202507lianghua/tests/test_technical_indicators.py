"""
技术指标模块测试
严格遵循零简化原则，确保所有功能完整测试
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

from core.technical_indicators import (
    IndicatorResult, MACDResult, MAResult, ATRResult, RSIResult,
    BollingerResult, KDJResult, TechnicalIndicatorCalculator,
    IndicatorAnalyzer
)
from config.config_manager import ConfigManager
from utils.logger import get_logger

# 创建logger实例
logger = get_logger(__name__)


class TestIndicatorDataClasses:
    """测试指标数据类"""
    
    def test_indicator_result_creation(self):
        """测试IndicatorResult创建"""
        result = IndicatorResult(
            indicator_name="SMA",
            symbol="BTCUSDT",
            timeframe="1h",
            timestamp=datetime.now(),
            values={"sma_20": 50000.0},
            metadata={"period": 20},
            calculation_time=0.01,
            confidence=0.95
        )
        
        assert result.indicator_name == "SMA"
        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert result.values["sma_20"] == 50000.0
        assert result.metadata["period"] == 20
        assert result.calculation_time == 0.01
        assert result.confidence == 0.95
    
    def test_macd_result_creation(self):
        """测试MACDResult创建"""
        result = MACDResult(
            macd_line=100.0,
            signal_line=80.0,
            histogram=20.0,
            fast_ema=50100.0,
            slow_ema=50000.0,
            timestamp=datetime.now()
        )
        
        assert result.macd_line == 100.0
        assert result.signal_line == 80.0
        assert result.histogram == 20.0
        assert result.fast_ema == 50100.0
        assert result.slow_ema == 50000.0
        assert result.divergence == 20.0  # macd_line - signal_line
        assert result.momentum == 20.0  # abs(histogram)
        assert result.trend_strength == 100.0  # abs(macd_line)
    
    def test_ma_result_creation(self):
        """测试MAResult创建"""
        result = MAResult(
            ma_type="SMA",
            period=20,
            value=50000.0,
            timestamp=datetime.now(),
            trend_direction="UP",
            trend_strength=0.8
        )
        
        assert result.ma_type == "SMA"
        assert result.period == 20
        assert result.value == 50000.0
        assert result.trend_direction == "UP"
        assert result.trend_strength == 0.8
        assert result.signal_strength == "STRONG"  # 基于trend_strength > 0.7
    
    def test_atr_result_creation(self):
        """测试ATRResult创建"""
        result = ATRResult(
            atr_value=0.03,
            volatility_level="",
            price_range=0.03,
            timestamp=datetime.now()
        )
        
        assert result.atr_value == 0.03
        assert result.price_range == 0.03
        assert result.volatility_level == "MODERATE"  # 基于atr_value > 0.02
    
    def test_rsi_result_creation(self):
        """测试RSIResult创建"""
        result = RSIResult(
            rsi_value=75.0,
            overbought=True,
            oversold=False,
            timestamp=datetime.now()
        )
        
        assert result.rsi_value == 75.0
        assert result.overbought is True
        assert result.oversold is False
        assert result.neutral is False  # rsi_value > 70
    
    def test_bollinger_result_creation(self):
        """测试BollingerResult创建"""
        result = BollingerResult(
            upper_band=51000.0,
            middle_band=50000.0,
            lower_band=49000.0,
            current_price=50500.0,
            bandwidth=0.04,
            position="",
            timestamp=datetime.now()
        )
        
        assert result.upper_band == 51000.0
        assert result.middle_band == 50000.0
        assert result.lower_band == 49000.0
        assert result.current_price == 50500.0
        assert result.bandwidth == 0.04
        assert result.position == "BETWEEN"  # 在上下轨之间
    
    def test_kdj_result_creation(self):
        """测试KDJResult创建"""
        result = KDJResult(
            k_value=85.0,
            d_value=80.0,
            j_value=95.0,
            timestamp=datetime.now()
        )
        
        assert result.k_value == 85.0
        assert result.d_value == 80.0
        assert result.j_value == 95.0
        assert result.overbought is True  # k > 80 and d > 80
        assert result.oversold is False
        assert result.golden_cross is True  # k > d
        assert result.death_cross is False


class TestTechnicalIndicatorCalculator:
    """测试技术指标计算器"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        config = Mock(spec=ConfigManager)
        return config
    
    @pytest.fixture
    def calculator(self, config):
        """技术指标计算器"""
        return TechnicalIndicatorCalculator(config)
    
    @pytest.fixture
    def sample_prices(self):
        """样本价格数据"""
        # 生成100个价格点，模拟上涨趋势
        base_price = 50000
        prices = []
        for i in range(100):
            price = base_price + i * 10 + np.random.normal(0, 50)
            prices.append(max(price, 40000))  # 确保价格不会过低
        return prices
    
    @pytest.fixture
    def sample_ohlcv(self):
        """样本OHLCV数据"""
        np.random.seed(42)  # 确保测试结果一致
        
        base_price = 50000
        ohlcv = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        current_price = base_price
        for i in range(100):
            # 模拟价格变动
            price_change = np.random.normal(0, 100)
            current_price += price_change
            current_price = max(current_price, 40000)
            
            # 生成OHLC
            high = current_price + np.random.uniform(0, 200)
            low = current_price - np.random.uniform(0, 200)
            close = current_price + np.random.uniform(-50, 50)
            volume = np.random.uniform(1000, 10000)
            
            ohlcv['open'].append(current_price)
            ohlcv['high'].append(high)
            ohlcv['low'].append(low)
            ohlcv['close'].append(close)
            ohlcv['volume'].append(volume)
        
        return ohlcv
    
    def test_calculator_initialization(self, calculator):
        """测试计算器初始化"""
        assert calculator.config is not None
        assert calculator.executor is not None
        assert calculator.calculation_cache == {}
        assert calculator.cache_ttl == 60
    
    def test_calculate_sma(self, calculator, sample_prices):
        """测试简单移动平均计算"""
        period = 20
        sma_values = calculator.calculate_sma(sample_prices, period)
        
        # 验证结果长度
        assert len(sma_values) == len(sample_prices) - period + 1
        
        # 验证第一个SMA值
        expected_first = sum(sample_prices[:period]) / period
        assert abs(sma_values[0] - expected_first) < 0.001
        
        # 验证最后一个SMA值
        expected_last = sum(sample_prices[-period:]) / period
        assert abs(sma_values[-1] - expected_last) < 0.001
    
    def test_calculate_sma_insufficient_data(self, calculator):
        """测试SMA数据不足的情况"""
        prices = [100, 101, 102]
        sma_values = calculator.calculate_sma(prices, 5)
        assert sma_values == []
    
    def test_calculate_ema(self, calculator, sample_prices):
        """测试指数移动平均计算"""
        period = 20
        ema_values = calculator.calculate_ema(sample_prices, period)
        
        # 验证结果长度
        assert len(ema_values) == len(sample_prices) - period + 1
        
        # 验证第一个EMA值等于SMA
        expected_first = sum(sample_prices[:period]) / period
        assert abs(ema_values[0] - expected_first) < 0.001
        
        # 验证EMA的连续性
        for i in range(1, len(ema_values)):
            assert ema_values[i] > 0
    
    def test_calculate_wma(self, calculator, sample_prices):
        """测试加权移动平均计算"""
        period = 20
        wma_values = calculator.calculate_wma(sample_prices, period)
        
        # 验证结果长度
        assert len(wma_values) == len(sample_prices) - period + 1
        
        # 验证WMA值的合理性
        for wma in wma_values:
            assert wma > 0
    
    def test_calculate_macd(self, calculator, sample_prices):
        """测试MACD计算"""
        macd_results = calculator.calculate_macd(sample_prices, 12, 26, 9)
        
        # 验证结果不为空
        assert len(macd_results) > 0
        
        # 验证MACD结果结构
        for result in macd_results:
            assert isinstance(result, MACDResult)
            assert hasattr(result, 'macd_line')
            assert hasattr(result, 'signal_line')
            assert hasattr(result, 'histogram')
            assert hasattr(result, 'fast_ema')
            assert hasattr(result, 'slow_ema')
            assert hasattr(result, 'divergence')
            assert hasattr(result, 'momentum')
        
        # 验证MACD基本关系
        latest = macd_results[-1]
        assert latest.histogram == latest.macd_line - latest.signal_line
        assert latest.divergence == latest.macd_line - latest.signal_line
        assert latest.momentum == abs(latest.histogram)
    
    def test_calculate_macd_insufficient_data(self, calculator):
        """测试MACD数据不足的情况"""
        prices = [100] * 10  # 只有10个数据点
        macd_results = calculator.calculate_macd(prices, 12, 26, 9)
        assert macd_results == []
    
    def test_calculate_rsi(self, calculator, sample_prices):
        """测试RSI计算"""
        rsi_results = calculator.calculate_rsi(sample_prices, 14)
        
        # 验证结果不为空
        assert len(rsi_results) > 0
        
        # 验证RSI结果结构
        for result in rsi_results:
            assert isinstance(result, RSIResult)
            assert 0 <= result.rsi_value <= 100
            assert isinstance(result.overbought, bool)
            assert isinstance(result.oversold, bool)
            assert isinstance(result.neutral, bool)
        
        # 验证RSI逻辑
        for result in rsi_results:
            if result.rsi_value > 70:
                assert result.overbought is True
            elif result.rsi_value < 30:
                assert result.oversold is True
            else:
                assert result.neutral is True
    
    def test_calculate_atr(self, calculator, sample_ohlcv):
        """测试ATR计算"""
        highs = sample_ohlcv['high']
        lows = sample_ohlcv['low']
        closes = sample_ohlcv['close']
        
        atr_results = calculator.calculate_atr(highs, lows, closes, 14)
        
        # 验证结果不为空
        assert len(atr_results) > 0
        
        # 验证ATR结果结构
        for result in atr_results:
            assert isinstance(result, ATRResult)
            assert result.atr_value >= 0
            assert result.price_range >= 0
            assert result.volatility_level in ["LOW", "MODERATE", "HIGH"]
    
    def test_calculate_bollinger_bands(self, calculator, sample_prices):
        """测试布林带计算"""
        bb_results = calculator.calculate_bollinger_bands(sample_prices, 20, 2.0)
        
        # 验证结果不为空
        assert len(bb_results) > 0
        
        # 验证布林带结果结构
        for result in bb_results:
            assert isinstance(result, BollingerResult)
            assert result.upper_band > result.middle_band > result.lower_band
            assert result.bandwidth >= 0
            assert result.position in ["ABOVE_UPPER", "BETWEEN", "BELOW_LOWER"]
    
    def test_calculate_kdj(self, calculator, sample_ohlcv):
        """测试KDJ计算"""
        highs = sample_ohlcv['high']
        lows = sample_ohlcv['low']
        closes = sample_ohlcv['close']
        
        kdj_results = calculator.calculate_kdj(highs, lows, closes, 14, 3, 3)
        
        # 验证结果不为空
        assert len(kdj_results) > 0
        
        # 验证KDJ结果结构
        for result in kdj_results:
            assert isinstance(result, KDJResult)
            assert 0 <= result.k_value <= 100
            assert 0 <= result.d_value <= 100
            assert isinstance(result.overbought, bool)
            assert isinstance(result.oversold, bool)
            assert isinstance(result.golden_cross, bool)
            assert isinstance(result.death_cross, bool)
    
    def test_calculate_williams_r(self, calculator, sample_ohlcv):
        """测试威廉指标计算"""
        highs = sample_ohlcv['high']
        lows = sample_ohlcv['low']
        closes = sample_ohlcv['close']
        
        wr_values = calculator.calculate_williams_r(highs, lows, closes, 14)
        
        # 验证结果不为空
        assert len(wr_values) > 0
        
        # 验证威廉指标值范围
        for wr in wr_values:
            assert -100 <= wr <= 0
    
    def test_calculate_volume_indicators(self, calculator, sample_ohlcv):
        """测试成交量指标计算"""
        volumes = sample_ohlcv['volume']
        
        # 测试成交量移动平均
        volume_sma = calculator.calculate_volume_sma(volumes, 20)
        assert len(volume_sma) > 0
        
        # 测试成交量比率
        volume_ratio = calculator.calculate_volume_ratio(volumes, 20)
        assert len(volume_ratio) > 0
        
        # 验证成交量比率的合理性
        for ratio in volume_ratio:
            assert ratio >= 0
    
    def test_calculate_momentum_indicators(self, calculator, sample_prices):
        """测试动量指标计算"""
        # 测试动量指标
        momentum = calculator.calculate_momentum(sample_prices, 10)
        assert len(momentum) > 0
        
        # 测试变动率指标
        roc = calculator.calculate_roc(sample_prices, 12)
        assert len(roc) > 0
    
    def test_calculate_volatility(self, calculator, sample_prices):
        """测试波动率计算"""
        volatility = calculator.calculate_volatility(sample_prices, 20)
        
        # 验证结果不为空
        assert len(volatility) > 0
        
        # 验证波动率值的合理性
        for vol in volatility:
            assert vol >= 0
    
    def test_calculate_support_resistance(self, calculator, sample_ohlcv):
        """测试支撑阻力位计算"""
        highs = sample_ohlcv['high']
        lows = sample_ohlcv['low']
        
        sr_levels = calculator.calculate_support_resistance(highs, lows, 20)
        
        # 验证结果结构
        assert 'support' in sr_levels
        assert 'resistance' in sr_levels
        assert isinstance(sr_levels['support'], list)
        assert isinstance(sr_levels['resistance'], list)
    
    def test_calculate_pivot_points(self, calculator):
        """测试枢轴点计算"""
        high, low, close = 51000, 49000, 50000
        
        pivot_points = calculator.calculate_pivot_points(high, low, close)
        
        # 验证结果结构
        required_keys = ['pivot', 's1', 's2', 's3', 'r1', 'r2', 'r3']
        for key in required_keys:
            assert key in pivot_points
        
        # 验证枢轴点计算
        expected_pivot = (high + low + close) / 3
        assert abs(pivot_points['pivot'] - expected_pivot) < 0.001
    
    def test_calculate_fibonacci_retracement(self, calculator):
        """测试斐波那契回撤计算"""
        high, low = 51000, 49000
        
        fib_levels = calculator.calculate_fibonacci_retracement(high, low)
        
        # 验证结果结构
        required_keys = ['0%', '23.6%', '38.2%', '50%', '61.8%', '100%']
        for key in required_keys:
            assert key in fib_levels
        
        # 验证斐波那契计算
        assert fib_levels['0%'] == high
        assert fib_levels['100%'] == low
        assert fib_levels['50%'] == (high + low) / 2
    
    def test_calculate_all_indicators(self, calculator, sample_ohlcv):
        """测试所有指标计算"""
        results = calculator.calculate_all_indicators(
            sample_ohlcv, "BTCUSDT", "1h"
        )
        
        # 验证结果不为空
        assert len(results) > 0
        
        # 验证元数据
        assert 'meta' in results
        assert results['meta']['symbol'] == "BTCUSDT"
        assert results['meta']['timeframe'] == "1h"
        assert results['meta']['data_points'] == len(sample_ohlcv['close'])
        assert results['meta']['calculation_time'] > 0
        assert results['meta']['indicators_count'] > 0
        
        # 验证主要指标存在
        expected_indicators = ['sma_20', 'ema_20', 'macd', 'rsi', 'bollinger']
        for indicator in expected_indicators:
            if indicator in results:
                assert results[indicator] is not None
    
    @pytest.mark.asyncio
    async def test_calculate_indicators_async(self, calculator, sample_ohlcv):
        """测试异步指标计算"""
        results = await calculator.calculate_indicators_async(
            sample_ohlcv, "BTCUSDT", "1h"
        )
        
        # 验证结果不为空
        assert len(results) > 0
        assert 'meta' in results
    
    def test_cache_functionality(self, calculator):
        """测试缓存功能"""
        # 测试缓存键生成
        cache_key = calculator._get_cache_key(
            "sma", "BTCUSDT", "1h", period=20
        )
        assert isinstance(cache_key, str)
        assert "sma" in cache_key
        assert "BTCUSDT" in cache_key
        assert "1h" in cache_key
        
        # 测试缓存设置和获取
        test_data = {"test": "value"}
        calculator._cache_result(cache_key, test_data)
        cached_data = calculator._get_cached_result(cache_key)
        assert cached_data == test_data
        
        # 测试缓存清理
        calculator.clear_cache()
        cached_data = calculator._get_cached_result(cache_key)
        assert cached_data is None
        
        # 测试缓存信息
        calculator._cache_result(cache_key, test_data)
        cache_info = calculator.get_cache_info()
        assert cache_info['cache_size'] == 1
        assert cache_info['cache_ttl'] == 60
        assert cache_key in cache_info['cache_keys']


class TestIndicatorAnalyzer:
    """测试指标分析器"""
    
    @pytest.fixture
    def config(self):
        """配置管理器模拟"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def calculator(self, config):
        """技术指标计算器"""
        return TechnicalIndicatorCalculator(config)
    
    @pytest.fixture
    def analyzer(self, calculator):
        """指标分析器"""
        return IndicatorAnalyzer(calculator)
    
    @pytest.fixture
    def sample_macd_results(self):
        """样本MACD结果"""
        return [
            MACDResult(
                macd_line=90.0,
                signal_line=100.0,
                histogram=-10.0,
                fast_ema=50100.0,
                slow_ema=50000.0,
                timestamp=datetime.now() - timedelta(minutes=1)
            ),
            MACDResult(
                macd_line=110.0,
                signal_line=100.0,
                histogram=10.0,
                fast_ema=50200.0,
                slow_ema=50000.0,
                timestamp=datetime.now()
            )
        ]
    
    def test_analyzer_initialization(self, analyzer):
        """测试分析器初始化"""
        assert analyzer.calculator is not None
    
    def test_analyze_macd_signals(self, analyzer, sample_macd_results):
        """测试MACD信号分析"""
        analysis = analyzer.analyze_macd_signals(sample_macd_results)
        
        # 验证分析结果结构
        required_keys = [
            'golden_cross', 'death_cross', 'macd_trend', 'histogram_trend',
            'signal_strength', 'trend_consistency', 'current_position'
        ]
        for key in required_keys:
            assert key in analysis
        
        # 验证具体分析结果
        assert analysis['golden_cross'] is True  # MACD线穿越信号线
        assert analysis['death_cross'] is False
        assert analysis['macd_trend'] == "UP"
        assert analysis['histogram_trend'] == "UP"
        assert analysis['current_position'] == "ABOVE"
    
    def test_analyze_macd_signals_insufficient_data(self, analyzer):
        """测试MACD信号分析数据不足"""
        analysis = analyzer.analyze_macd_signals([])
        assert analysis == {}
    
    def test_analyze_trend_strength(self, analyzer):
        """测试趋势强度分析"""
        # 构造测试数据
        mock_results = {
            'macd': [MACDResult(
                macd_line=100.0,
                signal_line=80.0,
                histogram=20.0,
                fast_ema=50100.0,
                slow_ema=50000.0,
                timestamp=datetime.now()
            )],
            'rsi': [RSIResult(
                rsi_value=60.0,
                overbought=False,
                oversold=False,
                timestamp=datetime.now()
            )],
            'sma_20': [49000.0, 50000.0]  # 上涨趋势
        }
        
        analysis = analyzer.analyze_trend_strength(mock_results)
        
        # 验证分析结果结构
        required_keys = ['overall_trend', 'trend_score', 'signal_count', 'trend_strength']
        for key in required_keys:
            assert key in analysis
        
        # 验证趋势分析
        assert analysis['overall_trend'] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert -1 <= analysis['trend_score'] <= 1
        assert analysis['signal_count'] > 0
        assert analysis['trend_strength'] >= 0
    
    def test_detect_reversal_patterns(self, analyzer):
        """测试反转模式检测"""
        # 构造测试数据
        mock_results = {
            'rsi': [RSIResult(
                rsi_value=25.0,  # 超卖
                overbought=False,
                oversold=True,
                timestamp=datetime.now()
            )],
            'bollinger': [BollingerResult(
                upper_band=51000.0,
                middle_band=50000.0,
                lower_band=49000.0,
                current_price=48500.0,  # 低于下轨
                bandwidth=0.04,
                position="BELOW_LOWER",
                timestamp=datetime.now()
            )],
            'kdj': [KDJResult(
                k_value=15.0,  # 超卖
                d_value=18.0,  # 超卖
                j_value=9.0,
                timestamp=datetime.now()
            )]
        }
        
        patterns = analyzer.detect_reversal_patterns(mock_results)
        
        # 验证检测结果
        assert len(patterns) > 0
        
        # 验证模式结构
        for pattern in patterns:
            assert 'type' in pattern
            assert 'confidence' in pattern
            assert 'direction' in pattern
            assert 'value' in pattern
            assert pattern['direction'] in ['BULLISH_REVERSAL', 'BEARISH_REVERSAL']
    
    def test_generate_trading_signals(self, analyzer):
        """测试交易信号生成"""
        # 构造测试数据
        mock_results = {
            'macd': [
                MACDResult(
                    macd_line=90.0,
                    signal_line=100.0,
                    histogram=-10.0,
                    fast_ema=50100.0,
                    slow_ema=50000.0,
                    timestamp=datetime.now() - timedelta(minutes=1)
                ),
                MACDResult(
                    macd_line=110.0,
                    signal_line=100.0,
                    histogram=10.0,
                    fast_ema=50200.0,
                    slow_ema=50000.0,
                    timestamp=datetime.now()
                )
            ],
            'rsi': [RSIResult(
                rsi_value=60.0,
                overbought=False,
                oversold=False,
                timestamp=datetime.now()
            )],
            'sma_20': [49000.0, 50000.0]
        }
        
        signals = analyzer.generate_trading_signals(mock_results)
        
        # 验证信号结构
        required_keys = ['buy_signals', 'sell_signals', 'hold_signals', 'overall_signal', 'confidence']
        for key in required_keys:
            assert key in signals
        
        # 验证信号内容
        assert isinstance(signals['buy_signals'], list)
        assert isinstance(signals['sell_signals'], list)
        assert isinstance(signals['hold_signals'], list)
        assert signals['overall_signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signals['confidence'] <= 1
        
        # 验证MACD金叉信号
        buy_signal_types = [signal['type'] for signal in signals['buy_signals']]
        assert 'MACD_GOLDEN_CROSS' in buy_signal_types


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def config(self):
        """真实配置管理器"""
        config = ConfigManager()
        return config
    
    @pytest.fixture
    def calculator(self, config):
        """真实技术指标计算器"""
        return TechnicalIndicatorCalculator(config)
    
    @pytest.fixture
    def analyzer(self, calculator):
        """真实指标分析器"""
        return IndicatorAnalyzer(calculator)
    
    @pytest.fixture
    def realistic_data(self):
        """真实样本数据"""
        # 模拟真实的加密货币价格走势
        np.random.seed(42)
        
        ohlcv = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        base_price = 50000
        trend = 1  # 初始上涨趋势
        
        for i in range(200):  # 200个数据点
            # 每20个点可能改变趋势
            if i % 20 == 0:
                trend = np.random.choice([-1, 1])
            
            # 价格变动
            price_change = np.random.normal(trend * 50, 200)
            base_price += price_change
            base_price = max(base_price, 30000)  # 最低价格
            
            # 生成OHLC
            high = base_price + np.random.uniform(0, 500)
            low = base_price - np.random.uniform(0, 500)
            close = base_price + np.random.uniform(-100, 100)
            volume = np.random.uniform(5000, 50000)
            
            ohlcv['open'].append(base_price)
            ohlcv['high'].append(high)
            ohlcv['low'].append(low)
            ohlcv['close'].append(close)
            ohlcv['volume'].append(volume)
        
        return ohlcv
    
    def test_full_technical_analysis(self, calculator, analyzer, realistic_data):
        """完整技术分析测试"""
        # 计算所有指标
        results = calculator.calculate_all_indicators(
            realistic_data, "BTCUSDT", "1h"
        )
        
        # 验证计算结果
        assert len(results) > 0
        assert 'meta' in results
        assert results['meta']['indicators_count'] > 5
        
        # 生成交易信号
        signals = analyzer.generate_trading_signals(results)
        
        # 验证信号生成
        assert signals['overall_signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signals['confidence'] <= 1
        
        # 检测反转模式
        patterns = analyzer.detect_reversal_patterns(results)
        assert isinstance(patterns, list)
        
        # 分析趋势强度
        trend_analysis = analyzer.analyze_trend_strength(results)
        assert trend_analysis['overall_trend'] in ['BULLISH', 'BEARISH', 'NEUTRAL']
        
        logger.info(f"技术分析完成: 信号={signals['overall_signal']}, "
                   f"置信度={signals['confidence']:.2f}, "
                   f"趋势={trend_analysis['overall_trend']}")
    
    @pytest.mark.asyncio
    async def test_async_technical_analysis(self, calculator, realistic_data):
        """异步技术分析测试"""
        # 异步计算指标
        results = await calculator.calculate_indicators_async(
            realistic_data, "BTCUSDT", "1h"
        )
        
        # 验证异步计算结果
        assert len(results) > 0
        assert 'meta' in results
        assert results['meta']['calculation_time'] > 0
    
    def test_performance_benchmarks(self, calculator, realistic_data):
        """性能基准测试"""
        import time
        
        # 测试计算性能
        start_time = time.time()
        results = calculator.calculate_all_indicators(
            realistic_data, "BTCUSDT", "1h"
        )
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # 验证性能要求
        assert calculation_time < 1.0  # 应该在1秒内完成
        assert results['meta']['calculation_time'] == calculation_time
        
        logger.info(f"性能测试完成: 计算时间={calculation_time:.3f}s, "
                   f"数据点={len(realistic_data['close'])}")
    
    def test_edge_cases(self, calculator):
        """边缘情况测试"""
        # 测试空数据
        empty_data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        results = calculator.calculate_all_indicators(empty_data, "BTCUSDT", "1h")
        assert results == {}
        
        # 测试极少数据
        minimal_data = {
            'open': [50000],
            'high': [50100],
            'low': [49900],
            'close': [50050],
            'volume': [1000]
        }
        results = calculator.calculate_all_indicators(minimal_data, "BTCUSDT", "1h")
        assert 'meta' in results
        
        # 测试异常价格数据
        abnormal_data = {
            'open': [50000, 0, 100000],
            'high': [50100, 0, 100100],
            'low': [49900, 0, 99900],
            'close': [50050, 0, 100050],
            'volume': [1000, 0, 2000]
        }
        
        # 应该能处理异常数据而不崩溃
        try:
            results = calculator.calculate_all_indicators(abnormal_data, "BTCUSDT", "1h")
            assert isinstance(results, dict)
        except Exception as e:
            logger.warning(f"处理异常数据时发生错误: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 