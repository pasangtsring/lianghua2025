"""
MACD背离检测器测试
严格遵循零简化原则，确保所有功能完整测试
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import math

from core.macd_divergence_detector import (
    DivergenceType, SignalStrength, Peak, TrendLine,
    DivergencePattern, MACDDivergenceSignal,
    PeakDetector, TrendLineAnalyzer, DivergenceValidator,
    MACDDivergenceDetector
)
from core.technical_indicators import MACDResult, TechnicalIndicatorCalculator
from config.config_manager import ConfigManager
from utils.logger import get_logger

# 创建logger实例
logger = get_logger(__name__)


class TestDivergenceEnums:
    """测试背离相关枚举"""
    
    def test_divergence_type_enum(self):
        """测试背离类型枚举"""
        assert DivergenceType.BULLISH_REGULAR.value == "bullish_regular"
        assert DivergenceType.BEARISH_REGULAR.value == "bearish_regular"
        assert DivergenceType.BULLISH_HIDDEN.value == "bullish_hidden"
        assert DivergenceType.BEARISH_HIDDEN.value == "bearish_hidden"
        assert DivergenceType.NO_DIVERGENCE.value == "no_divergence"
    
    def test_signal_strength_enum(self):
        """测试信号强度枚举"""
        assert SignalStrength.VERY_STRONG.value == "very_strong"
        assert SignalStrength.STRONG.value == "strong"
        assert SignalStrength.MODERATE.value == "moderate"
        assert SignalStrength.WEAK.value == "weak"
        assert SignalStrength.VERY_WEAK.value == "very_weak"


class TestPeakDataClass:
    """测试峰值数据类"""
    
    def test_peak_creation(self):
        """测试峰值创建"""
        peak = Peak(
            index=10,
            value=100.0,
            timestamp=datetime.now(),
            peak_type='high',
            confirmation_candles=3,
            volume=1000.0
        )
        
        assert peak.index == 10
        assert peak.value == 100.0
        assert peak.peak_type == 'high'
        assert peak.confirmation_candles == 3
        assert peak.volume == 1000.0
        assert peak.strength == 0.6  # 3/5 = 0.6
    
    def test_peak_strength_calculation(self):
        """测试峰值强度计算"""
        # 测试不同的确认K线数量
        test_cases = [
            (0, 0.0),
            (1, 0.2),
            (3, 0.6),
            (5, 1.0),
            (10, 1.0)  # 超过5的情况，应该限制在1.0
        ]
        
        for confirmation_candles, expected_strength in test_cases:
            peak = Peak(
                index=10,
                value=100.0,
                timestamp=datetime.now(),
                peak_type='high',
                confirmation_candles=confirmation_candles
            )
            assert peak.strength == expected_strength


class TestTrendLineDataClass:
    """测试趋势线数据类"""
    
    def test_trend_line_creation(self):
        """测试趋势线创建"""
        points = [
            Peak(5, 100.0, datetime.now(), 'high'),
            Peak(10, 105.0, datetime.now(), 'high')
        ]
        
        trend_line = TrendLine(
            points=points,
            slope=1.0,
            intercept=95.0,
            r_squared=0.95,
            start_index=5,
            end_index=10
        )
        
        assert trend_line.points == points
        assert trend_line.slope == 1.0
        assert trend_line.intercept == 95.0
        assert trend_line.r_squared == 0.95
        assert trend_line.start_index == 5
        assert trend_line.end_index == 10
        assert trend_line.equation == "y = 1.000000x + 95.000000"


class TestDivergencePatternDataClass:
    """测试背离模式数据类"""
    
    def test_divergence_pattern_creation(self):
        """测试背离模式创建"""
        price_points = [
            Peak(5, 100.0, datetime.now(), 'high'),
            Peak(10, 95.0, datetime.now(), 'high')
        ]
        price_trend = TrendLine(price_points, -1.0, 105.0, 0.9, 5, 10)
        
        macd_points = [
            Peak(5, 2.0, datetime.now(), 'high'),
            Peak(10, 2.5, datetime.now(), 'high')
        ]
        macd_trend = TrendLine(macd_points, 0.1, 1.5, 0.8, 5, 10)
        
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        pattern = DivergencePattern(
            divergence_type=DivergenceType.BEARISH_REGULAR,
            price_trend=price_trend,
            macd_trend=macd_trend,
            start_time=start_time,
            end_time=end_time,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            risk_reward_ratio=2.0,
            expected_move=0.05
        )
        
        assert pattern.divergence_type == DivergenceType.BEARISH_REGULAR
        assert pattern.confidence == 0.8
        assert pattern.risk_reward_ratio == 2.0
        assert pattern.expected_move == 0.05
        assert pattern.duration == timedelta(hours=24)


class TestMACDDivergenceSignalDataClass:
    """测试MACD背离信号数据类"""
    
    def test_signal_creation(self):
        """测试信号创建"""
        # 创建模拟的背离模式
        price_points = [Peak(5, 100.0, datetime.now(), 'low')]
        price_trend = TrendLine(price_points, -1.0, 105.0, 0.9, 5, 10)
        
        macd_points = [Peak(5, -1.0, datetime.now(), 'low')]
        macd_trend = TrendLine(macd_points, 0.1, -1.5, 0.8, 5, 10)
        
        pattern = DivergencePattern(
            divergence_type=DivergenceType.BULLISH_REGULAR,
            price_trend=price_trend,
            macd_trend=macd_trend,
            start_time=datetime.now() - timedelta(hours=12),
            end_time=datetime.now(),
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            risk_reward_ratio=2.0,
            expected_move=0.05
        )
        
        signal = MACDDivergenceSignal(
            symbol="BTCUSDT",
            timeframe="1h",
            timestamp=datetime.now(),
            divergence_pattern=pattern,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            position_size=0.02,
            risk_percentage=0.02,
            expected_duration=timedelta(hours=24),
            invalidation_level=48500.0
        )
        
        assert signal.symbol == "BTCUSDT"
        assert signal.timeframe == "1h"
        assert signal.entry_price == 50000.0
        assert signal.stop_loss == 49000.0
        assert signal.take_profit == 52000.0
        assert signal.position_size == 0.02
        assert signal.risk_percentage == 0.02
        assert signal.risk_reward_ratio == 2.0  # (52000-50000)/(50000-49000)


class TestPeakDetector:
    """测试峰值检测器"""
    
    @pytest.fixture
    def config(self):
        """配置fixture"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def peak_detector(self, config):
        """峰值检测器fixture"""
        config.get_setting = Mock(side_effect=lambda key, default: default)
        return PeakDetector(config)
    
    @pytest.fixture
    def sample_data(self):
        """样本数据fixture"""
        # 创建一个包含明显峰值的数据
        base_data = [100, 102, 105, 108, 110, 107, 104, 102, 105, 108, 112, 109, 106, 103, 101]
        volumes = [1000 + i * 100 for i in range(len(base_data))]
        return base_data, volumes
    
    def test_detect_peaks_basic(self, peak_detector, sample_data):
        """测试基本峰值检测"""
        data, volumes = sample_data
        
        peaks, valleys = peak_detector.detect_peaks(data, volumes)
        
        # 验证返回的是Peak对象列表
        assert isinstance(peaks, list)
        assert isinstance(valleys, list)
        
        # 验证峰值对象的属性
        for peak in peaks:
            assert isinstance(peak, Peak)
            assert peak.peak_type == 'high'
            assert peak.value > 0
        
        for valley in valleys:
            assert isinstance(valley, Peak)
            assert valley.peak_type == 'low'
            assert valley.value > 0
    
    def test_detect_peaks_insufficient_data(self, peak_detector):
        """测试数据不足的情况"""
        short_data = [100, 101, 102]
        peaks, valleys = peak_detector.detect_peaks(short_data)
        
        assert peaks == []
        assert valleys == []
    
    def test_confirm_peak_high(self, peak_detector):
        """测试高点确认"""
        data = np.array([100, 102, 105, 108, 110, 107, 104, 102, 105])
        
        # 测试有效的高点
        assert peak_detector._confirm_peak(data, 4, 'high') == True
        
        # 测试无效的高点（边界情况）
        assert peak_detector._confirm_peak(data, 1, 'high') == False
    
    def test_confirm_peak_low(self, peak_detector):
        """测试低点确认"""
        data = np.array([110, 107, 104, 102, 99, 102, 105, 108, 110])
        
        # 测试有效的低点
        assert peak_detector._confirm_peak(data, 4, 'low') == True
        
        # 测试无效的低点（边界情况）
        assert peak_detector._confirm_peak(data, 1, 'low') == False
    
    def test_filter_significant_peaks(self, peak_detector):
        """测试显著峰值过滤"""
        peaks = [
            Peak(0, 100.0, datetime.now(), 'high'),
            Peak(1, 100.5, datetime.now(), 'high'),  # 变化不显著
            Peak(2, 103.0, datetime.now(), 'high'),  # 变化显著
            Peak(3, 104.0, datetime.now(), 'high'),  # 变化不显著
            Peak(4, 106.0, datetime.now(), 'high')   # 变化显著
        ]
        
        filtered = peak_detector.filter_significant_peaks(peaks, 0.02)
        
        # 应该过滤掉变化不显著的峰值
        assert len(filtered) < len(peaks)
        assert filtered[0].value == 100.0
        assert 103.0 in [p.value for p in filtered]


class TestTrendLineAnalyzer:
    """测试趋势线分析器"""
    
    @pytest.fixture
    def config(self):
        """配置fixture"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def trend_analyzer(self, config):
        """趋势线分析器fixture"""
        config.get_setting = Mock(side_effect=lambda key, default: default)
        return TrendLineAnalyzer(config)
    
    def test_create_trend_line_valid(self, trend_analyzer):
        """测试创建有效趋势线"""
        peaks = [
            Peak(0, 100.0, datetime.now(), 'high'),
            Peak(10, 110.0, datetime.now(), 'high'),
            Peak(20, 120.0, datetime.now(), 'high')
        ]
        
        trend_line = trend_analyzer.create_trend_line(peaks)
        
        assert trend_line is not None
        assert isinstance(trend_line, TrendLine)
        assert trend_line.slope > 0  # 上升趋势
        assert trend_line.r_squared > 0
        assert len(trend_line.points) == 3
    
    def test_create_trend_line_insufficient_points(self, trend_analyzer):
        """测试点数不足的情况"""
        peaks = [Peak(0, 100.0, datetime.now(), 'high')]
        
        trend_line = trend_analyzer.create_trend_line(peaks)
        
        assert trend_line is None
    
    def test_analyze_trend_divergence_bearish(self, trend_analyzer):
        """测试看跌背离分析"""
        # 价格上升趋势
        price_peaks = [
            Peak(0, 100.0, datetime.now(), 'high'),
            Peak(10, 110.0, datetime.now(), 'high')
        ]
        price_trend = TrendLine(price_peaks, 1.0, 100.0, 0.9, 0, 10)
        
        # MACD下降趋势
        macd_peaks = [
            Peak(0, 2.0, datetime.now(), 'high'),
            Peak(10, 1.5, datetime.now(), 'high')
        ]
        macd_trend = TrendLine(macd_peaks, -0.05, 2.0, 0.8, 0, 10)
        
        divergence_type, strength = trend_analyzer.analyze_trend_divergence(
            price_trend, macd_trend
        )
        
        assert divergence_type == DivergenceType.BEARISH_REGULAR
        assert strength > 0
    
    def test_analyze_trend_divergence_bullish(self, trend_analyzer):
        """测试看涨背离分析"""
        # 价格下降趋势
        price_peaks = [
            Peak(0, 110.0, datetime.now(), 'low'),
            Peak(10, 100.0, datetime.now(), 'low')
        ]
        price_trend = TrendLine(price_peaks, -1.0, 110.0, 0.9, 0, 10)
        
        # MACD上升趋势
        macd_peaks = [
            Peak(0, -2.0, datetime.now(), 'low'),
            Peak(10, -1.5, datetime.now(), 'low')
        ]
        macd_trend = TrendLine(macd_peaks, 0.05, -2.0, 0.8, 0, 10)
        
        divergence_type, strength = trend_analyzer.analyze_trend_divergence(
            price_trend, macd_trend
        )
        
        assert divergence_type == DivergenceType.BULLISH_REGULAR
        assert strength > 0
    
    def test_analyze_trend_divergence_no_divergence(self, trend_analyzer):
        """测试无背离情况"""
        # 价格和MACD都上升
        price_peaks = [
            Peak(0, 100.0, datetime.now(), 'high'),
            Peak(10, 110.0, datetime.now(), 'high')
        ]
        price_trend = TrendLine(price_peaks, 1.0, 100.0, 0.9, 0, 10)
        
        macd_peaks = [
            Peak(0, 1.0, datetime.now(), 'high'),
            Peak(10, 1.5, datetime.now(), 'high')
        ]
        macd_trend = TrendLine(macd_peaks, 0.05, 1.0, 0.8, 0, 10)
        
        divergence_type, strength = trend_analyzer.analyze_trend_divergence(
            price_trend, macd_trend
        )
        
        assert divergence_type == DivergenceType.NO_DIVERGENCE
        assert strength == 0.0


class TestDivergenceValidator:
    """测试背离验证器"""
    
    @pytest.fixture
    def config(self):
        """配置fixture"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def validator(self, config):
        """背离验证器fixture"""
        config.get_setting = Mock(side_effect=lambda key, default: default)
        return DivergenceValidator(config)
    
    @pytest.fixture
    def valid_pattern(self):
        """有效背离模式fixture"""
        price_points = [
            Peak(0, 100.0, datetime.now(), 'high'),
            Peak(10, 110.0, datetime.now(), 'high')
        ]
        price_trend = TrendLine(price_points, 1.0, 100.0, 0.9, 0, 10)
        
        macd_points = [
            Peak(0, 2.0, datetime.now(), 'high'),
            Peak(10, 1.5, datetime.now(), 'high')
        ]
        macd_trend = TrendLine(macd_points, -0.05, 2.0, 0.8, 0, 10)
        
        return DivergencePattern(
            divergence_type=DivergenceType.BEARISH_REGULAR,
            price_trend=price_trend,
            macd_trend=macd_trend,
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now(),
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            risk_reward_ratio=2.0,
            expected_move=0.05
        )
    
    def test_validate_divergence_valid(self, validator, valid_pattern):
        """测试有效背离验证"""
        is_valid, confidence, issues = validator.validate_divergence(valid_pattern)
        
        assert is_valid == True
        assert confidence > 0.5
        assert len(issues) == 0
    
    def test_validate_divergence_no_divergence(self, validator, valid_pattern):
        """测试无背离情况"""
        valid_pattern.divergence_type = DivergenceType.NO_DIVERGENCE
        
        is_valid, confidence, issues = validator.validate_divergence(valid_pattern)
        
        assert is_valid == False
        assert confidence == 0.0
        assert "无背离模式" in issues
    
    def test_validate_divergence_poor_trend_quality(self, validator, valid_pattern):
        """测试趋势线质量差的情况"""
        valid_pattern.price_trend.r_squared = 0.3
        valid_pattern.macd_trend.r_squared = 0.4
        
        is_valid, confidence, issues = validator.validate_divergence(valid_pattern)
        
        assert confidence < 0.8  # 置信度应该降低
        assert any("R²过低" in issue for issue in issues)
    
    def test_validate_divergence_short_duration(self, validator, valid_pattern):
        """测试持续时间过短的情况"""
        valid_pattern.start_time = datetime.now() - timedelta(hours=6)
        valid_pattern.end_time = datetime.now()
        
        is_valid, confidence, issues = validator.validate_divergence(valid_pattern)
        
        assert confidence < 0.8  # 置信度应该降低
        assert any("持续时间过短" in issue for issue in issues)
    
    def test_calculate_signal_strength(self, validator):
        """测试信号强度计算"""
        test_cases = [
            (0.9, 3.0, SignalStrength.VERY_STRONG),
            (0.8, 2.0, SignalStrength.STRONG),
            (0.6, 1.5, SignalStrength.MODERATE),
            (0.4, 1.0, SignalStrength.WEAK),
            (0.2, 0.5, SignalStrength.VERY_WEAK)
        ]
        
        for confidence, risk_reward, expected_strength in test_cases:
            strength = validator.calculate_signal_strength(confidence, risk_reward)
            assert strength == expected_strength


class TestMACDDivergenceDetector:
    """测试MACD背离检测器主类"""
    
    @pytest.fixture
    def config(self):
        """配置fixture"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def detector(self, config):
        """MACD背离检测器fixture"""
        config.get_setting = Mock(side_effect=lambda key, default: default)
        return MACDDivergenceDetector(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """样本市场数据fixture"""
        # 创建模拟的市场数据，包含明显的背离模式
        prices = []
        macd_results = []
        
        # 生成150个数据点
        for i in range(150):
            # 创建一个看跌背离的模式：价格上升，MACD下降
            if i < 75:
                price = 50000 + i * 100  # 价格上升
                macd_line = 100 - i * 1.2  # MACD下降
            else:
                price = 50000 + (150 - i) * 80  # 价格继续上升但斜率减小
                macd_line = 100 - (150 - i) * 0.8  # MACD继续下降
            
            prices.append(price)
            
            macd_result = MACDResult(
                macd_line=macd_line,
                signal_line=macd_line - 20,
                histogram=20,
                fast_ema=price * 1.001,
                slow_ema=price * 0.999,
                timestamp=datetime.now()
            )
            macd_results.append(macd_result)
        
        volumes = [1000 + i * 10 for i in range(150)]
        return prices, macd_results, volumes
    
    def test_detect_divergence_basic(self, detector, sample_market_data):
        """测试基本背离检测"""
        prices, macd_results, volumes = sample_market_data
        
        signals = detector.detect_divergence(
            prices, macd_results, volumes, "BTCUSDT", "1h"
        )
        
        # 验证返回的是信号列表
        assert isinstance(signals, list)
        
        # 验证信号对象的属性
        for signal in signals:
            assert isinstance(signal, MACDDivergenceSignal)
            assert signal.symbol == "BTCUSDT"
            assert signal.timeframe == "1h"
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.take_profit > 0
            assert signal.position_size > 0
            assert signal.risk_percentage > 0
    
    def test_detect_divergence_insufficient_data(self, detector):
        """测试数据不足的情况"""
        prices = [50000, 50100, 50200]
        macd_results = [
            MACDResult(100, 80, 20, 50100, 50000, datetime.now()),
            MACDResult(90, 70, 20, 50200, 50100, datetime.now()),
            MACDResult(80, 60, 20, 50300, 50200, datetime.now())
        ]
        
        signals = detector.detect_divergence(
            prices, macd_results, None, "BTCUSDT", "1h"
        )
        
        assert signals == []
    
    def test_find_closest_peak(self, detector):
        """测试寻找最接近的峰值"""
        peaks = [
            Peak(5, 100.0, datetime.now(), 'high'),
            Peak(15, 110.0, datetime.now(), 'high'),
            Peak(25, 120.0, datetime.now(), 'high')
        ]
        
        # 测试找到最接近的峰值
        closest = detector._find_closest_peak(peaks, 12)
        assert closest.index == 15
        
        # 测试超出最大距离的情况
        closest = detector._find_closest_peak(peaks, 50, max_distance=10)
        assert closest is None
    
    def test_calculate_risk_reward_ratio(self, detector):
        """测试风险回报比计算"""
        peaks = [
            Peak(0, 100.0, datetime.now(), 'low'),
            Peak(10, 95.0, datetime.now(), 'low')
        ]
        prices = [100.0] * 50 + [102.0] * 50
        
        # 测试看涨背离
        ratio = detector._calculate_risk_reward_ratio(
            DivergenceType.BULLISH_REGULAR, peaks, prices
        )
        assert ratio > 0
        
        # 测试看跌背离
        ratio = detector._calculate_risk_reward_ratio(
            DivergenceType.BEARISH_REGULAR, peaks, prices
        )
        assert ratio > 0
    
    def test_calculate_expected_move(self, detector):
        """测试预期移动幅度计算"""
        peaks = [
            Peak(0, 100.0, datetime.now(), 'high'),
            Peak(20, 110.0, datetime.now(), 'high')
        ]
        prices = [100 + i * 0.1 for i in range(50)]
        macd_histograms = [0.5 + i * 0.01 for i in range(50)]
        
        expected_move = detector._calculate_expected_move(
            DivergenceType.BEARISH_REGULAR, peaks, prices, macd_histograms
        )
        
        assert expected_move > 0
        assert expected_move <= 0.15  # 应该限制在最大15%
    
    @pytest.mark.asyncio
    async def test_detect_divergence_async(self, detector, sample_market_data):
        """测试异步背离检测"""
        prices, macd_results, volumes = sample_market_data
        
        signals = await detector.detect_divergence_async(
            prices, macd_results, volumes, "BTCUSDT", "1h"
        )
        
        assert isinstance(signals, list)
        # 验证异步结果与同步结果一致
        sync_signals = detector.detect_divergence(
            prices, macd_results, volumes, "BTCUSDT", "1h"
        )
        assert len(signals) == len(sync_signals)
    
    def test_get_detection_stats(self, detector):
        """测试获取检测统计信息"""
        stats = detector.get_detection_stats()
        
        assert isinstance(stats, dict)
        assert 'lookback_period' in stats
        assert 'min_peak_distance' in stats
        assert 'dynamic_threshold' in stats
        assert 'cache_size' in stats
        assert 'detector_components' in stats
    
    def test_clear_cache(self, detector):
        """测试清空缓存"""
        # 添加一些缓存数据
        detector._cache['test_key'] = 'test_value'
        
        detector.clear_cache()
        
        assert len(detector._cache) == 0


class TestIntegrationTests:
    """集成测试"""
    
    @pytest.fixture
    def config(self):
        """配置fixture"""
        return Mock(spec=ConfigManager)
    
    @pytest.fixture
    def full_detector(self, config):
        """完整的背离检测器fixture"""
        config.get_setting = Mock(side_effect=lambda key, default: default)
        return MACDDivergenceDetector(config)
    
    def test_full_divergence_detection_workflow(self, full_detector):
        """测试完整的背离检测工作流"""
        # 创建一个完整的背离模式数据
        prices = []
        macd_results = []
        volumes = []
        
        # 生成200个数据点，包含明显的看跌背离
        for i in range(200):
            # 第一阶段：价格上升，MACD也上升
            if i < 100:
                price = 50000 + i * 50
                macd_line = i * 0.5
            # 第二阶段：价格继续上升但斜率减小，MACD开始下降（背离）
            else:
                price = 50000 + i * 30
                macd_line = 50 - (i - 100) * 0.8
            
            prices.append(price)
            volumes.append(1000 + i * 20)
            
            macd_result = MACDResult(
                macd_line=macd_line,
                signal_line=macd_line - 10,
                histogram=10,
                fast_ema=price * 1.001,
                slow_ema=price * 0.999,
                timestamp=datetime.now()
            )
            macd_results.append(macd_result)
        
        # 执行背离检测
        signals = full_detector.detect_divergence(
            prices, macd_results, volumes, "BTCUSDT", "1h"
        )
        
        # 验证结果
        assert isinstance(signals, list)
        
        # 检查信号质量
        for signal in signals:
            assert isinstance(signal, MACDDivergenceSignal)
            assert signal.symbol == "BTCUSDT"
            assert signal.timeframe == "1h"
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.take_profit > 0
            assert signal.position_size > 0
            assert signal.risk_percentage > 0
            assert signal.risk_reward_ratio > 0
            assert isinstance(signal.divergence_pattern, DivergencePattern)
            assert signal.divergence_pattern.divergence_type != DivergenceType.NO_DIVERGENCE
    
    def test_performance_benchmarks(self, full_detector):
        """测试性能基准"""
        # 创建大量数据进行性能测试
        large_prices = [50000 + i * 10 for i in range(1000)]
        large_macd_results = [
            MACDResult(i * 0.1, i * 0.05, i * 0.05, 50000 + i * 10, 50000 + i * 9, datetime.now())
            for i in range(1000)
        ]
        large_volumes = [1000 + i * 50 for i in range(1000)]
        
        import time
        start_time = time.time()
        
        signals = full_detector.detect_divergence(
            large_prices, large_macd_results, large_volumes, "BTCUSDT", "1h"
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证性能要求（应该在合理时间内完成）
        assert execution_time < 10.0  # 10秒内完成
        assert isinstance(signals, list)
        
        logger.info(f"性能测试结果: {execution_time:.3f}秒处理1000个数据点")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 