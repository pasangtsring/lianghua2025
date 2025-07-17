#!/usr/bin/env python3
"""
真实币安数据MACD背离检测验证脚本
使用ETH历史数据验证优化后的MACD背离检测器
"""

import os
import sys
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from core.macd_divergence_detector import (
    MACDDivergenceDetector, 
    DivergenceDetectionConfig,
    DivergenceType,
    SignalStrength,
    MACDDivergenceSignal
)
from core.technical_indicators import MACDResult, TechnicalIndicatorCalculator
from utils.logger import get_logger
from config.config_manager import ConfigManager

# 导入币安API客户端
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("警告：未安装python-binance，将使用模拟数据")

# 导入requests作为备用
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = get_logger(__name__)


class BinanceDataFetcher:
    """币安数据获取器"""
    
    def __init__(self, use_testnet: bool = False):
        self.use_testnet = use_testnet
        self.binance_client = None
        self.base_url = "https://fapi.binance.com"
        
        # 尝试初始化币安客户端
        if BINANCE_AVAILABLE:
            try:
                self.binance_client = Client()
                self.binance_client.API_URL = 'https://fapi.binance.com'
                logger.info("币安客户端初始化成功")
            except Exception as e:
                logger.warning(f"币安客户端初始化失败: {e}")
                self.binance_client = None
    
    async def fetch_historical_klines(self, symbol: str, interval: str, 
                                    limit: int = 1000, 
                                    start_time: Optional[datetime] = None,
                                    end_time: Optional[datetime] = None) -> List[Dict]:
        """获取历史K线数据"""
        
        # 方法1：使用python-binance客户端
        if self.binance_client:
            try:
                return await self._fetch_with_binance_client(
                    symbol, interval, limit, start_time, end_time
                )
            except Exception as e:
                logger.warning(f"使用binance客户端获取数据失败: {e}")
        
        # 方法2：使用requests直接调用API
        if REQUESTS_AVAILABLE:
            try:
                return await self._fetch_with_requests(
                    symbol, interval, limit, start_time, end_time
                )
            except Exception as e:
                logger.warning(f"使用requests获取数据失败: {e}")
        
        # 方法3：使用模拟数据
        logger.info("使用模拟数据进行测试")
        return self._generate_realistic_simulation_data(symbol, limit)
    
    async def _fetch_with_binance_client(self, symbol: str, interval: str, 
                                       limit: int, start_time: Optional[datetime],
                                       end_time: Optional[datetime]) -> List[Dict]:
        """使用币安客户端获取数据"""
        try:
            # 转换时间格式
            start_str = None
            end_str = None
            
            if start_time:
                start_str = str(int(start_time.timestamp() * 1000))
            if end_time:
                end_str = str(int(end_time.timestamp() * 1000))
            
            # 获取K线数据
            klines = self.binance_client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_str,
                endTime=end_str
            )
            
            # 转换为标准格式
            formatted_data = []
            for kline in klines:
                formatted_data.append({
                    'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': datetime.fromtimestamp(int(kline[6]) / 1000),
                    'quote_volume': float(kline[7]),
                    'trade_count': int(kline[8]),
                    'taker_buy_base': float(kline[9]),
                    'taker_buy_quote': float(kline[10])
                })
            
            logger.info(f"成功获取 {len(formatted_data)} 条K线数据 (binance客户端)")
            return formatted_data
            
        except Exception as e:
            logger.error(f"binance客户端获取数据失败: {e}")
            raise
    
    async def _fetch_with_requests(self, symbol: str, interval: str, 
                                 limit: int, start_time: Optional[datetime],
                                 end_time: Optional[datetime]) -> List[Dict]:
        """使用requests获取数据"""
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            klines = response.json()
            
            # 转换为标准格式
            formatted_data = []
            for kline in klines:
                formatted_data.append({
                    'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': datetime.fromtimestamp(int(kline[6]) / 1000),
                    'quote_volume': float(kline[7]),
                    'trade_count': int(kline[8]),
                    'taker_buy_base': float(kline[9]),
                    'taker_buy_quote': float(kline[10])
                })
            
            logger.info(f"成功获取 {len(formatted_data)} 条K线数据 (requests)")
            return formatted_data
            
        except Exception as e:
            logger.error(f"requests获取数据失败: {e}")
            raise
    
    def _generate_realistic_simulation_data(self, symbol: str, limit: int) -> List[Dict]:
        """生成更真实的模拟数据"""
        logger.info(f"生成 {limit} 条模拟数据用于测试")
        
        # 基于真实ETH价格范围生成数据
        base_price = 3500.0  # ETH基础价格
        current_time = datetime.now()
        
        data = []
        price = base_price
        
        for i in range(limit):
            # 生成更真实的价格波动
            # 添加趋势、波动性和随机性
            trend = np.sin(i * 0.02) * 50  # 长期趋势
            volatility = np.random.normal(0, 20)  # 随机波动
            momentum = np.random.normal(0, 10)  # 短期动量
            
            # 价格变化
            price_change = trend + volatility + momentum
            price += price_change
            
            # 确保价格在合理范围内
            price = max(2000, min(5000, price))
            
            # 生成OHLCV数据
            spread = price * 0.002  # 0.2% spread
            open_price = price + np.random.normal(0, spread)
            high_price = max(open_price, price + np.random.exponential(spread))
            low_price = min(open_price, price - np.random.exponential(spread))
            close_price = price + np.random.normal(0, spread/2)
            volume = np.random.lognormal(10, 1)
            
            timestamp = current_time - timedelta(hours=limit - i)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2),
                'close_time': timestamp + timedelta(hours=1),
                'quote_volume': round(volume * close_price, 2),
                'trade_count': int(np.random.lognormal(8, 1)),
                'taker_buy_base': round(volume * 0.6, 2),
                'taker_buy_quote': round(volume * close_price * 0.6, 2)
            })
        
        return data


class RealDataMACDTester:
    """真实数据MACD测试器"""
    
    def __init__(self):
        self.data_fetcher = BinanceDataFetcher()
        
        # 初始化配置管理器
        config_manager = ConfigManager()
        self.indicator_calculator = TechnicalIndicatorCalculator(config_manager)
        self.logger = get_logger(__name__)
        
        # 使用专家建议的最优参数
        self.config = DivergenceDetectionConfig(
            lookback_period=70,
            min_peak_distance=3,
            prominence_multiplier=0.2,
            min_divergence_gap=0.03,
            min_consecutive_count=2,
            time_alignment_tolerance=5
        )
        
        self.detector = MACDDivergenceDetector(self.config)
    
    async def test_eth_divergence_detection(self) -> Dict[str, Any]:
        """测试ETH背离检测"""
        print("=== ETH MACD背离检测测试 ===")
        
        # 获取ETH历史数据
        symbol = "ETHUSDT"
        interval = "1h"
        limit = 500  # 获取500小时数据
        
        print(f"获取 {symbol} {interval} 数据...")
        
        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=limit)
        
        try:
            # 获取历史数据
            klines_data = await self.data_fetcher.fetch_historical_klines(
                symbol, interval, limit, start_time, end_time
            )
            
            if not klines_data:
                raise ValueError("未能获取到数据")
            
            # 转换为DataFrame便于处理
            df = pd.DataFrame(klines_data)
            df = df.sort_values('timestamp')
            
            print(f"成功获取 {len(df)} 条数据")
            print(f"数据时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            print(f"价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # 计算MACD指标
            print("\n计算MACD指标...")
            macd_results = self._calculate_macd_indicators(df)
            
            # 执行背离检测
            print("\n执行背离检测...")
            divergence_signals = self._detect_divergences(df, macd_results)
            
            # 分析结果
            analysis_result = self._analyze_detection_results(
                df, macd_results, divergence_signals
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"ETH背离检测测试失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'interval': interval
            }
    
    def _calculate_macd_indicators(self, df: pd.DataFrame) -> List[MACDResult]:
        """计算MACD指标"""
        closes = df['close'].tolist()
        macd_results = []
        
        # 使用指标计算器
        for i in range(len(closes)):
            if i < 34:  # MACD需要至少34根K线
                # 填充空值
                macd_results.append(MACDResult(
                    macd_line=0.0,
                    signal_line=0.0,
                    histogram=0.0,
                    fast_ema=closes[i],
                    slow_ema=closes[i],
                    timestamp=df.iloc[i]['timestamp']
                ))
            else:
                # 计算MACD
                recent_closes = closes[max(0, i-100):i+1]
                macd_data_list = self.indicator_calculator.calculate_macd(
                    recent_closes, 
                    fast_period=12, 
                    slow_period=26, 
                    signal_period=9
                )
                
                if macd_data_list:
                    # 取最后一个MACD结果
                    macd_data = macd_data_list[-1]
                    macd_results.append(MACDResult(
                        macd_line=macd_data.macd_line,
                        signal_line=macd_data.signal_line,
                        histogram=macd_data.histogram,
                        fast_ema=macd_data.fast_ema,
                        slow_ema=macd_data.slow_ema,
                        timestamp=df.iloc[i]['timestamp']
                    ))
                else:
                    macd_results.append(MACDResult(
                        macd_line=0.0,
                        signal_line=0.0,
                        histogram=0.0,
                        fast_ema=closes[i],
                        slow_ema=closes[i],
                        timestamp=df.iloc[i]['timestamp']
                    ))
        
        return macd_results
    
    def _detect_divergences(self, df: pd.DataFrame, 
                          macd_results: List[MACDResult]) -> List[MACDDivergenceSignal]:
        """检测背离"""
        prices = df['close'].tolist()
        volumes = df['volume'].tolist()
        
        # 使用增强版检测
        signals = self.detector.detect_divergence_enhanced(
            prices, macd_results, volumes, 
            symbol="ETHUSDT", timeframe="1h"
        )
        
        return signals
    
    def _analyze_detection_results(self, df: pd.DataFrame, 
                                 macd_results: List[MACDResult],
                                 signals: List[MACDDivergenceSignal]) -> Dict[str, Any]:
        """分析检测结果"""
        
        # 基础统计
        total_candles = len(df)
        signal_count = len(signals)
        
        # 信号类型统计
        signal_types = {}
        signal_strengths = {}
        
        for signal in signals:
            signal_type = signal.divergence_type.value
            signal_strength = signal.signal_strength.value
            
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            signal_strengths[signal_strength] = signal_strengths.get(signal_strength, 0) + 1
        
        # 价格统计
        price_stats = {
            'min': float(df['close'].min()),
            'max': float(df['close'].max()),
            'mean': float(df['close'].mean()),
            'std': float(df['close'].std()),
            'change': float(df['close'].iloc[-1] - df['close'].iloc[0]),
            'change_pct': float((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
        }
        
        # MACD统计
        macd_values = [r.macd_line for r in macd_results if r.macd_line != 0]
        macd_stats = {
            'min': float(min(macd_values)) if macd_values else 0,
            'max': float(max(macd_values)) if macd_values else 0,
            'mean': float(np.mean(macd_values)) if macd_values else 0,
            'std': float(np.std(macd_values)) if macd_values else 0
        }
        
        # 信号详情
        signal_details = []
        for i, signal in enumerate(signals):
            signal_details.append({
                'signal_id': i + 1,
                'type': signal.divergence_type.value,
                'strength': signal.signal_strength.value,
                'confidence': signal.confidence,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                'expected_return': signal.expected_return,
                'max_risk': signal.max_risk,
                'metadata': signal.metadata
            })
        
        # 构建结果
        result = {
            'success': True,
            'symbol': 'ETHUSDT',
            'interval': '1h',
            'test_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_candles': total_candles,
                'time_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'price_stats': price_stats,
                'macd_stats': macd_stats
            },
            'detection_summary': {
                'total_signals': signal_count,
                'signal_rate': signal_count / total_candles * 100,
                'signal_types': signal_types,
                'signal_strengths': signal_strengths
            },
            'signals': signal_details,
            'config_used': {
                'lookback_period': self.config.lookback_period,
                'min_peak_distance': self.config.min_peak_distance,
                'prominence_multiplier': self.config.prominence_multiplier,
                'min_divergence_gap': self.config.min_divergence_gap,
                'min_consecutive_count': self.config.min_consecutive_count,
                'time_alignment_tolerance': self.config.time_alignment_tolerance
            }
        }
        
        return result
    
    def print_test_results(self, result: Dict[str, Any]) -> None:
        """打印测试结果"""
        if not result['success']:
            print(f"❌ 测试失败: {result['error']}")
            return
        
        print(f"\n✅ 测试成功完成")
        print(f"🔍 数据概览:")
        print(f"  - 交易对: {result['symbol']}")
        print(f"  - 时间周期: {result['interval']}")
        print(f"  - 数据量: {result['data_summary']['total_candles']} 根K线")
        print(f"  - 时间范围: {result['data_summary']['time_range']['start']} 到 {result['data_summary']['time_range']['end']}")
        
        price_stats = result['data_summary']['price_stats']
        print(f"  - 价格范围: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
        print(f"  - 价格变化: {price_stats['change_pct']:.2f}% (${price_stats['change']:.2f})")
        
        print(f"\n📊 检测结果:")
        detection = result['detection_summary']
        print(f"  - 总信号数: {detection['total_signals']}")
        print(f"  - 信号频率: {detection['signal_rate']:.2f}%")
        
        if detection['signal_types']:
            print(f"  - 信号类型分布:")
            for sig_type, count in detection['signal_types'].items():
                print(f"    • {sig_type}: {count} 个")
        
        if detection['signal_strengths']:
            print(f"  - 信号强度分布:")
            for strength, count in detection['signal_strengths'].items():
                print(f"    • {strength}: {count} 个")
        
        print(f"\n🎯 参数配置:")
        config = result['config_used']
        print(f"  - 检测窗口: {config['lookback_period']}")
        print(f"  - 峰值间距: {config['min_peak_distance']}")
        print(f"  - 噪音过滤: {config['prominence_multiplier']}")
        print(f"  - 背离阈值: {config['min_divergence_gap']}")
        print(f"  - 连续次数: {config['min_consecutive_count']}")
        print(f"  - 时间容忍: {config['time_alignment_tolerance']}")
        
        # 打印信号详情
        if result['signals']:
            print(f"\n📈 信号详情:")
            for signal in result['signals']:
                print(f"  信号 {signal['signal_id']}:")
                print(f"    类型: {signal['type']}")
                print(f"    强度: {signal['strength']}")
                print(f"    置信度: {signal['confidence']:.3f}")
                print(f"    风险回报比: {signal['risk_reward_ratio']:.2f}")
                print(f"    入场价: ${signal['entry_price']:.2f}")
                print(f"    止损价: ${signal['stop_loss']:.2f}")
                print(f"    止盈价: ${signal['take_profit']:.2f}")
                print(f"    预期收益: {signal['expected_return']:.2f}%")
                print(f"    最大风险: {signal['max_risk']:.2f}%")
                print()
        else:
            print(f"\n📈 未检测到背离信号")
    
    def save_results_to_file(self, result: Dict[str, Any], 
                           filename: str = None) -> None:
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eth_macd_test_results_{timestamp}.json"
        
        try:
            # 确保目录存在
            os.makedirs('test_results', exist_ok=True)
            filepath = os.path.join('test_results', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"📁 结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


async def main():
    """主函数"""
    print("🚀 启动真实币安数据MACD背离检测验证")
    print("="*60)
    
    # 创建测试器
    tester = RealDataMACDTester()
    
    # 执行测试
    try:
        result = await tester.test_eth_divergence_detection()
        
        # 打印结果
        tester.print_test_results(result)
        
        # 保存结果
        tester.save_results_to_file(result)
        
        print("\n🎉 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 