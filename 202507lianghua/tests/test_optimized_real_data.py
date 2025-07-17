#!/usr/bin/env python3
"""
优化版真实数据MACD背离检测测试
使用更宽松的参数设置，适合真实市场数据
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

# 导入优化后的检测器
from core.complete_macd_divergence_detector import (
    CompleteMACDDivergenceDetector,
    DivergenceDetectionConfig,
    MACDResult,
    create_optimized_config,
    detect_macd_divergence
)

# 导入币安数据获取器
from tests.test_real_binance_data import BinanceDataFetcher

# 导入技术指标计算器
from core.technical_indicators import TechnicalIndicatorCalculator
from config.config_manager import ConfigManager

import logging
logging.basicConfig(level=logging.INFO)


class OptimizedRealDataTester:
    """优化版真实数据测试器"""
    
    def __init__(self):
        self.data_fetcher = BinanceDataFetcher()
        config_manager = ConfigManager()
        self.indicator_calculator = TechnicalIndicatorCalculator(config_manager)
        
        # 使用优化的加密货币配置
        self.config = create_optimized_config("crypto")
        self.detector = CompleteMACDDivergenceDetector(self.config)
        
        print("使用优化配置:")
        print(f"  检测窗口: {self.config.lookback_period}")
        print(f"  峰值间距: {self.config.min_peak_distance}")
        print(f"  噪音过滤: {self.config.prominence_multiplier}")
        print(f"  背离阈值: {self.config.min_divergence_gap}")
        print(f"  时间容忍: {self.config.time_alignment_tolerance}")
        print()
    
    async def test_multiple_symbols(self, symbols: List[str] = None, 
                                   intervals: List[str] = None) -> Dict[str, Any]:
        """测试多个交易对"""
        if symbols is None:
            symbols = ["ETHUSDT", "BTCUSDT", "ADAUSDT"]
        if intervals is None:
            intervals = ["1h", "4h"]
        
        results = {}
        
        for symbol in symbols:
            for interval in intervals:
                print(f"\n=== 测试 {symbol} {interval} ===")
                
                try:
                    result = await self.test_single_symbol(symbol, interval)
                    results[f"{symbol}_{interval}"] = result
                    
                    # 打印简要结果
                    if result['success']:
                        signal_count = result['detection_summary']['total_signals']
                        print(f"✅ {symbol} {interval}: {signal_count} 个信号")
                    else:
                        print(f"❌ {symbol} {interval}: {result['error']}")
                        
                except Exception as e:
                    print(f"❌ {symbol} {interval} 测试失败: {e}")
                    results[f"{symbol}_{interval}"] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    async def test_single_symbol(self, symbol: str, interval: str) -> Dict[str, Any]:
        """测试单个交易对"""
        
        # 根据时间间隔调整数据量
        if interval == "1h":
            limit = 200  # 8天数据
        elif interval == "4h":
            limit = 150  # 25天数据
        elif interval == "1d":
            limit = 100  # 100天数据
        else:
            limit = 200
        
        # 获取历史数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=limit * self._get_interval_hours(interval))
        
        klines_data = await self.data_fetcher.fetch_historical_klines(
            symbol, interval, limit, start_time, end_time
        )
        
        if not klines_data:
            return {
                'success': False,
                'error': '无法获取数据',
                'symbol': symbol,
                'interval': interval
            }
        
        # 转换数据
        df = pd.DataFrame(klines_data)
        df = df.sort_values('timestamp')
        
        # 计算MACD
        macd_results = self._calculate_macd_indicators(df)
        
        # 执行背离检测
        signals = self.detector.detect_divergence(
            df['close'].tolist(), 
            macd_results, 
            df['volume'].tolist(),
            symbol, 
            interval
        )
        
        # 分析结果
        result = self._analyze_results(df, macd_results, signals, symbol, interval)
        
        return result
    
    def _get_interval_hours(self, interval: str) -> int:
        """获取时间间隔的小时数"""
        if interval == "1h":
            return 1
        elif interval == "4h":
            return 4
        elif interval == "1d":
            return 24
        else:
            return 1
    
    def _calculate_macd_indicators(self, df: pd.DataFrame) -> List[MACDResult]:
        """计算MACD指标"""
        closes = df['close'].tolist()
        macd_results = []
        
        for i in range(len(closes)):
            if i < 34:  # MACD需要至少34根K线
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
    
    def _analyze_results(self, df: pd.DataFrame, macd_results: List[MACDResult],
                        signals: List, symbol: str, interval: str) -> Dict[str, Any]:
        """分析结果"""
        
        # 价格统计
        price_stats = {
            'min': float(df['close'].min()),
            'max': float(df['close'].max()),
            'mean': float(df['close'].mean()),
            'change_pct': float((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
        }
        
        # 信号统计
        signal_types = {}
        signal_strengths = {}
        
        for signal in signals:
            signal_type = signal.divergence_type.value
            signal_strength = signal.signal_strength.value
            
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            signal_strengths[signal_strength] = signal_strengths.get(signal_strength, 0) + 1
        
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
                'expected_return': signal.expected_return,
                'entry_time': signal.entry_time.isoformat(),
                'metadata': signal.metadata
            })
        
        return {
            'success': True,
            'symbol': symbol,
            'interval': interval,
            'test_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_candles': len(df),
                'time_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'price_stats': price_stats
            },
            'detection_summary': {
                'total_signals': len(signals),
                'signal_rate': len(signals) / len(df) * 100,
                'signal_types': signal_types,
                'signal_strengths': signal_strengths
            },
            'signals': signal_details,
            'detector_stats': self.detector.get_statistics()
        }
    
    def print_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """打印综合结果"""
        print("\n" + "="*80)
        print("📊 综合测试结果")
        print("="*80)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.get('success', False))
        total_signals = sum(r.get('detection_summary', {}).get('total_signals', 0) 
                           for r in results.values() if r.get('success', False))
        
        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests}")
        print(f"总信号数: {total_signals}")
        print(f"平均信号率: {total_signals / max(successful_tests, 1):.2f} 个/测试")
        
        # 按交易对分组
        by_symbol = {}
        for key, result in results.items():
            if result.get('success'):
                symbol = result['symbol']
                if symbol not in by_symbol:
                    by_symbol[symbol] = []
                by_symbol[symbol].append(result)
        
        print(f"\n📈 按交易对统计:")
        for symbol, symbol_results in by_symbol.items():
            symbol_signals = sum(r['detection_summary']['total_signals'] for r in symbol_results)
            print(f"  {symbol}: {symbol_signals} 个信号 (共{len(symbol_results)}个时间周期)")
        
        # 信号类型统计
        all_signal_types = {}
        all_signal_strengths = {}
        
        for result in results.values():
            if result.get('success'):
                for sig_type, count in result['detection_summary']['signal_types'].items():
                    all_signal_types[sig_type] = all_signal_types.get(sig_type, 0) + count
                for strength, count in result['detection_summary']['signal_strengths'].items():
                    all_signal_strengths[strength] = all_signal_strengths.get(strength, 0) + count
        
        if all_signal_types:
            print(f"\n🎯 信号类型分布:")
            for sig_type, count in all_signal_types.items():
                print(f"  {sig_type}: {count} 个")
        
        if all_signal_strengths:
            print(f"\n💪 信号强度分布:")
            for strength, count in all_signal_strengths.items():
                print(f"  {strength}: {count} 个")
        
        # 详细信号信息
        print(f"\n🔍 详细信号信息:")
        for key, result in results.items():
            if result.get('success') and result['detection_summary']['total_signals'] > 0:
                symbol = result['symbol']
                interval = result['interval']
                signals = result['signals']
                
                print(f"\n  {symbol} {interval}:")
                for signal in signals:
                    print(f"    📍 {signal['type']} | 强度: {signal['strength']} | "
                          f"置信度: {signal['confidence']:.3f} | 风险回报比: {signal['risk_reward_ratio']:.2f}")
                    print(f"       入场: ${signal['entry_price']:.2f} | 止损: ${signal['stop_loss']:.2f} | "
                          f"止盈: ${signal['take_profit']:.2f}")
    
    def save_comprehensive_results(self, results: Dict[str, Any], filename: str = None) -> None:
        """保存综合结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_divergence_test_{timestamp}.json"
        
        try:
            os.makedirs('test_results', exist_ok=True)
            filepath = os.path.join('test_results', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n📁 综合结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


async def main():
    """主函数"""
    print("🚀 启动优化版真实数据MACD背离检测测试")
    print("="*80)
    
    # 创建测试器
    tester = OptimizedRealDataTester()
    
    # 测试多个交易对和时间周期
    symbols = ["ETHUSDT", "BTCUSDT", "ADAUSDT", "BNBUSDT"]
    intervals = ["1h", "4h"]
    
    try:
        # 执行综合测试
        results = await tester.test_multiple_symbols(symbols, intervals)
        
        # 打印结果
        tester.print_comprehensive_results(results)
        
        # 保存结果
        tester.save_comprehensive_results(results)
        
        print("\n🎉 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 