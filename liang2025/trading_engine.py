"""
主交易引擎
整合所有模块，实现主交易逻辑
支持多币种选币和分析
"""

import asyncio
import signal
import sys
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import numpy as np # Added for numpy

from config.config_manager import ConfigManager
from utils.logger import get_logger
from data.advanced_data_fetcher import AdvancedDataFetcher
from data.api_client import get_api_client
from core.technical_indicators import TechnicalIndicatorCalculator
from core.complete_macd_divergence_detector import CompleteMACDDivergenceDetector
from core.coin_scanner import CoinScanner
from core.signal_generator import SignalGeneratorWithEnhancedFilter
from risk.risk_manager import RiskManager
from utils.telegram_bot import TelegramBot
# 移除复杂的家庭财务管理导入

class TradingEngine:
    """主交易引擎 - 多币种自动选币交易系统"""
    
    def __init__(self):
        """初始化交易引擎"""
        # 初始化配置
        self.config = ConfigManager()
        self.logger = get_logger(__name__)
        
        # 初始化API客户端
        self.api_client_manager = get_api_client()
        
        # 初始化核心组件
        self.data_fetcher = AdvancedDataFetcher(self.config, self.api_client_manager)
        self.technical_indicators = TechnicalIndicatorCalculator(self.config)
        self.macd_detector = CompleteMACDDivergenceDetector(self.config)  # 修复：传入配置管理器
        self.coin_scanner = CoinScanner(self.config, self.api_client_manager)
        self.signal_generator = SignalGeneratorWithEnhancedFilter(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # 简化初始化，专注核心功能
        
        # 初始化通知组件
        self.telegram_bot = TelegramBot(self.config)
        
        # 运行状态
        self.is_running = False
        self.should_stop = False
        
        # 交易配置
        self.trading_config = self.config.get_trading_config()
        self.multi_symbol_mode = getattr(self.trading_config, 'multi_symbol_mode', True)
        
        # 多币种数据存储
        self.selected_symbols = []  # 选中的币种列表
        self.coin_data = {}       # 币种K线数据 {symbol: klines}
        self.coin_signals = {}    # 币种信号 {symbol: signal}
        self.current_positions = {}  # 当前持仓 {symbol: position_info}
        self.trade_history = []
        self.consecutive_losses = 0  # 连续亏损次数
        
        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("交易引擎初始化完成")

    def get_dynamic_leverage(self, market_condition: str, confidence: float) -> float:
        """
        简单的动态杠杆计算 - 基于大佬建议
        
        Args:
            market_condition: 市场条件 ('bullish', 'bearish', 'neutral')
            confidence: 信号置信度 (0-1)
            
        Returns:
            建议杠杆倍数
        """
        try:
            trading_config = self.config.get_trading_config()
            risk_config = trading_config.risk
            
            # 大佬建议：牛市和熊市不同杠杆
            if 'bullish' in market_condition.lower() or 'spring' in market_condition.lower() or 'summer' in market_condition.lower():
                # 牛市杠杆 - 基于置信度在范围内调整
                base_leverage = getattr(risk_config, 'leverage_bull', 5.0)
                max_leverage = min(base_leverage * 1.5, getattr(risk_config, 'max_leverage', 10.0))
            else:
                # 熊市杠杆 - 更保守
                base_leverage = getattr(risk_config, 'leverage_bear', 3.0) 
                max_leverage = min(base_leverage * 1.2, getattr(risk_config, 'max_leverage', 10.0))
            
            # 根据置信度调整杠杆
            min_leverage = 1.0
            leverage = min_leverage + (max_leverage - min_leverage) * confidence
            
            # 简单的连续亏损保护
            consecutive_losses = getattr(self, 'consecutive_losses', 0)
            if consecutive_losses >= 3:
                leverage *= 0.5  # 连续亏损降低杠杆
            
            leverage = max(1.0, min(leverage, getattr(risk_config, 'max_leverage', 10.0)))
            
            self.logger.info(f"动态杠杆: 市场={market_condition}, 置信度={confidence:.2f}, 杠杆={leverage:.1f}x")
            return leverage
            
        except Exception as e:
            self.logger.error(f"动态杠杆计算失败: {e}")
            return 3.0  # 安全默认值

    def calculate_dynamic_position_size(self, signal: Dict, symbol: str) -> float:
        """
        简单的动态仓位计算 - 基于大佬杠杆建议
        
        Args:
            signal: 交易信号
            symbol: 交易品种
            
        Returns:
            建议仓位大小
        """
        try:
            confidence = signal.get('confidence', 0.5)
            entry_price = signal.get('price', 0)
            market_condition = signal.get('market_condition', 'neutral')
            
            if entry_price <= 0:
                return 0
            
            # 获取动态杠杆
            leverage = self.get_dynamic_leverage(market_condition, confidence)
            
            # 简单的仓位计算
            base_position_size = self.calculate_position_size_advanced(entry_price, symbol)
            
            # 应用杠杆和置信度调整
            adjusted_position = base_position_size * leverage * confidence
            
            self.logger.info(f"动态仓位: {symbol} - 杠杆{leverage:.1f}x, 置信度{confidence:.2f}, 仓位{adjusted_position:.6f}")
            
            return adjusted_position
            
        except Exception as e:
            self.logger.error(f"动态仓位计算失败: {e}")
            return 0
    
    def signal_handler(self, signum, frame):
        """处理系统信号"""
        self.logger.info(f"收到信号 {signum}，准备关闭交易引擎")
        self.should_stop = True
        self.is_running = False
    
    async def start(self):
        """启动交易引擎"""
        try:
            self.is_running = True
            self.logger.info("🚀 启动多币种自动选币交易引擎...")
            
            # 发送启动通知
            await self.telegram_bot.send_message("🚀 多币种交易引擎已启动")
            
            # 启动主循环
            await self.main_loop()
            
            # 主循环退出后的清理工作
            self.logger.info("🛑 主循环已退出，正在清理资源...")
            await self.telegram_bot.send_message("🛑 交易引擎主循环已退出")
            
        except Exception as e:
            self.logger.error(f"启动交易引擎失败: {e}")
            await self.telegram_bot.send_message(f"❌ 交易引擎启动失败: {e}")
            raise
    
    async def stop(self):
        """停止交易引擎"""
        try:
            self.should_stop = True
            self.is_running = False
            
            self.logger.info("交易引擎停止完成")
            await self.telegram_bot.send_message("🛑 交易引擎已停止")
            
        except Exception as e:
            self.logger.error(f"停止交易引擎失败: {e}")
    
    async def main_loop(self):
        """主循环 - 多币种交易流程"""
        last_coin_selection_time = None
        coin_selection_interval = 3600  # 1小时执行一次选币
        
        while not self.should_stop:
            try:
                current_time = time.time()
                
                # 1. 币种选择阶段（每小时执行一次）
                if (self.multi_symbol_mode and 
                    (last_coin_selection_time is None or 
                     current_time - last_coin_selection_time > coin_selection_interval)):
                    await self.execute_coin_selection()
                    last_coin_selection_time = current_time
                
                # 2. 数据获取阶段
                await self.fetch_all_coin_data()
                
                # 3. 技术分析阶段
                await self.analyze_all_coins()
                
                # 4. 信号检测阶段
                await self.detect_all_signals()
                
                # 5. 交易执行阶段
                await self.execute_all_trades()
                
                # 6. 持仓监控阶段
                await self.monitor_positions()
                
                # 7. 风险管理阶段
                await self.update_risk_management()
                
                # 8. 报告生成阶段
                await self.generate_reports()
                
                # 等待下一个周期（可中断的睡眠）
                for _ in range(30):  # 30秒检查一次，每秒检查一次停止标志
                    if self.should_stop:
                        break
                    await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"主循环执行失败: {e}")
                await self.telegram_bot.send_message(f"⚠️ 主循环错误: {e}")
                # 可中断的错误恢复等待
                for _ in range(60):
                    if self.should_stop:
                        break
                    await asyncio.sleep(1)
    
    async def execute_coin_selection(self):
        """执行币种选择"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🎯 阶段1: 执行币种选择")
            self.logger.info("=" * 80)
            
            # 使用大佬版选币逻辑
            selected_symbols = await self.coin_scanner.scan_and_select_coins()
            
            if selected_symbols:
                self.selected_symbols = selected_symbols
                self.logger.info(f"✅ 选币完成，选择了 {len(selected_symbols)} 个币种: {selected_symbols}")
            else:
                # 备用：使用主流币
                self.selected_symbols = ['BTCUSDT', 'ETHUSDT']
                self.logger.warning("⚠️ 选币失败，使用默认主流币")
                
        except Exception as e:
            self.logger.error(f"选币阶段失败: {e}")
            # 备用：使用主流币
            self.selected_symbols = ['BTCUSDT', 'ETHUSDT']
    
    async def fetch_all_coin_data(self):
        """阶段2: 获取选中币种的K线数据"""
        try:
            self.logger.info("📊 阶段2: 获取选中币种的K线数据")
            
            self.selected_symbols_data = {}
            
            # 获取每个选中币种的数据
            for symbol in self.selected_symbols:
                self.logger.info(f"📈 正在获取 {symbol} 的K线数据...")
                
                # 获取K线数据
                klines_response = await self.api_client_manager.get_klines(
                    symbol, '1h', limit=200
                )
                
                if klines_response and klines_response.success:
                    klines = klines_response.data
                    
                    # 转换为DataFrame
                    import pandas as pd
                    ohlcv_data = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    # 转换数据类型
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        ohlcv_data[col] = pd.to_numeric(ohlcv_data[col])
                    
                    self.selected_symbols_data[symbol] = ohlcv_data
                    self.logger.info(f"✅ {symbol} K线数据获取成功: {len(ohlcv_data)} 条")
                else:
                    self.logger.error(f"❌ {symbol} K线数据获取失败")
                    
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
    
    async def analyze_all_coins(self):
        """阶段3: 对选中币种执行技术分析"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🔬 阶段3: 对选中币种执行技术分析")
            self.logger.info("=" * 80)
            
            # 修复：使用正确的变量名
            if not hasattr(self, 'selected_symbols_data') or not self.selected_symbols_data:
                self.logger.warning("⚠️ 没有币种数据可供分析")
                return
            
            for symbol in self.selected_symbols_data:
                ohlcv_data = self.selected_symbols_data[symbol]
                
                if ohlcv_data is None or len(ohlcv_data) < 50:
                    self.logger.warning(f"⚠️ {symbol} 数据不足，跳过分析")
                    continue
                
                self.logger.info(f"📊 正在对 {symbol} 执行技术分析...")
                
                try:
                    # 准备技术指标数据格式
                    indicator_data = {
                        'open': ohlcv_data['open'].tolist(),
                        'high': ohlcv_data['high'].tolist(), 
                        'low': ohlcv_data['low'].tolist(),
                        'close': ohlcv_data['close'].tolist(),
                        'volume': ohlcv_data['volume'].tolist()
                    }
                    
                    # 计算所有技术指标
                    indicators = self.technical_indicators.calculate_all_indicators(
                        indicator_data, symbol, '1h'
                    )
                    
                    # MACD背离检测
                    closes = ohlcv_data['close'].tolist()
                    macd_result = None
                    
                    if 'macd' in indicators and len(indicators['macd']) > 0:
                        try:
                            macd_result = self.macd_detector.detect_divergence(
                                closes, 
                                indicators['macd'],
                                symbol=symbol,
                                timeframe='1h'
                            )
                        except Exception as e:
                            self.logger.warning(f"⚠️ {symbol} MACD背离检测失败: {e}")
                    
                    # 详细输出分析结果
                    self.log_detailed_analysis_results(symbol, indicators, macd_result, ohlcv_data)
                    
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 技术分析失败: {e}")
            
        except Exception as e:
            self.logger.error(f"技术分析阶段失败: {e}")
    
    def log_detailed_analysis_results(self, symbol: str, indicators: Dict, macd_result, ohlcv_data):
        """详细输出分析结果（按用户要求）"""
        try:
            current_price = float(ohlcv_data['close'].iloc[-1])
            self.logger.info(f"📋 {symbol} 技术分析详细结果:")
            self.logger.info(f"   💰 当前价格: {current_price:.4f}")
            
            # 移动平均线分析
            if 'sma_20' in indicators and len(indicators['sma_20']) > 0:
                sma_20 = indicators['sma_20'][-1]
                sma_50 = indicators.get('sma_50', [sma_20])[-1] if 'sma_50' in indicators else sma_20
                self.logger.info(f"   📈 SMA20: {sma_20:.4f}, SMA50: {sma_50:.4f}")
                
                # 价格与均线关系（添加除零保护）
                if sma_20 > 0:
                    price_diff_pct = ((current_price/sma_20-1)*100)
                    if current_price > sma_20:
                        self.logger.info(f"   ✅ 价格在SMA20上方 (+{price_diff_pct:.2f}%)")
                    else:
                        self.logger.info(f"   ❌ 价格在SMA20下方 ({price_diff_pct:.2f}%)")
                else:
                    self.logger.info(f"   ⚠️ SMA20数据异常，无法计算价格偏离度")
            
            # RSI分析
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                rsi_data = indicators['rsi'][-1]
                rsi_value = rsi_data.rsi_value if hasattr(rsi_data, 'rsi_value') else rsi_data
                
                if rsi_value < 30:
                    rsi_status = "🔥 超卖"
                elif rsi_value > 70:
                    rsi_status = "⚠️ 超买"
                else:
                    rsi_status = "✅ 正常"
                    
                self.logger.info(f"   📊 RSI(14): {rsi_value:.2f} {rsi_status}")
            
            # MACD分析
            if 'macd' in indicators and len(indicators['macd']) > 0:
                macd_data = indicators['macd'][-1]
                if hasattr(macd_data, 'macd_line'):
                    macd_line = macd_data.macd_line
                    signal_line = macd_data.signal_line
                    histogram = macd_data.histogram
                    
                    trend = "看涨" if macd_line > signal_line else "看跌"
                    momentum = "增强" if histogram > 0 else "减弱"
                    
                    self.logger.info(f"   📈 MACD: {macd_line:.6f}, 信号线: {signal_line:.6f}")
                    self.logger.info(f"   📊 柱状图: {histogram:.6f} (趋势{trend}, 动量{momentum})")
            
            # MACD背离分析
            if macd_result:
                if hasattr(macd_result, 'has_divergence') and macd_result.has_divergence:
                    div_type = getattr(macd_result, 'divergence_type', '未知')
                    confidence = getattr(macd_result, 'confidence', 0)
                    self.logger.info(f"   🔍 MACD背离: {div_type}背离 (置信度: {confidence:.1%})")
                else:
                    self.logger.info(f"   📊 MACD背离: 无明显背离信号")
            
            # 布林带分析
            if 'bollinger' in indicators and len(indicators['bollinger']) > 0:
                bb_data = indicators['bollinger'][-1]
                if hasattr(bb_data, 'upper_band'):
                    upper = bb_data.upper_band
                    lower = bb_data.lower_band
                    position = bb_data.position
                    
                    self.logger.info(f"   📊 布林带: 上轨{upper:.4f}, 下轨{lower:.4f}")
                    self.logger.info(f"   📍 位置: {position}")
            
            # 成交量分析（添加除零保护）
            current_volume = float(ohlcv_data['volume'].iloc[-1])
            avg_volume = float(ohlcv_data['volume'].tail(20).mean())
            
            # 除零保护：如果平均成交量为0或过小，设为默认值
            if avg_volume <= 0:
                volume_ratio = 1.0
                volume_status = "⚠️ 数据异常"
                self.logger.info(f"   📊 成交量: {current_volume:.0f} (平均成交量数据异常)")
            else:
                volume_ratio = current_volume / avg_volume
                volume_status = "🔥 放量" if volume_ratio > 1.5 else "📉 缩量" if volume_ratio < 0.7 else "✅ 正常"
                self.logger.info(f"   📊 成交量: {current_volume:.0f} (20期均值比: {volume_ratio:.2f}x {volume_status})")
            
            # 综合结论
            signals = []
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                rsi_value = indicators['rsi'][-1].rsi_value if hasattr(indicators['rsi'][-1], 'rsi_value') else indicators['rsi'][-1]
                if rsi_value < 30:
                    signals.append("RSI超卖")
                elif rsi_value > 70:
                    signals.append("RSI超买")
            
            if current_price > indicators.get('sma_20', [current_price])[-1]:
                signals.append("价格强势")
            
            conclusion = "、".join(signals) if signals else "中性"
            self.logger.info(f"   💡 技术分析结论: {conclusion}")
            self.logger.info(f"   " + "="*50)
            
        except Exception as e:
            self.logger.error(f"输出分析结果失败: {e}")
    
    async def detect_all_signals(self):
        """阶段4: 检测所有币种的交易信号 - 使用专业SignalGenerator"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("📡 阶段4: 检测交易信号（使用专业SignalGenerator）")
            self.logger.info("=" * 80)
            
            # 修复：使用正确的变量名
            if not hasattr(self, 'selected_symbols_data') or not self.selected_symbols_data:
                self.logger.warning("⚠️ 没有币种数据可供信号检测")
                return
            
            buy_signals = []
            sell_signals = []
            
            for symbol in self.selected_symbols_data:
                ohlcv_data = self.selected_symbols_data[symbol]
                
                if ohlcv_data is None or len(ohlcv_data) < 100:  # SignalGenerator需要至少100条数据
                    self.logger.info(f"⚠️ {symbol} 数据不足，跳过信号检测")
                    continue
                
                self.logger.info(f"📡 正在使用专业SignalGenerator检测 {symbol} 的交易信号...")
                
                try:
                    # 转换DataFrame为SignalGenerator需要的格式
                    kline_data = []
                    for i, row in ohlcv_data.iterrows():
                        kline = {
                            'open_time': int(i) if isinstance(i, (int, float)) else int(time.time() * 1000),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        }
                        kline_data.append(kline)
                    
                    # 使用专业SignalGenerator生成信号（包含HEAD_SHOULDER、双向交易、置信度过滤）
                    trading_signal = self.signal_generator.generate_signal(kline_data)
                    
                    if trading_signal:
                        signal_data = {
                            'symbol': symbol,
                            'type': trading_signal.signal_type.value.upper(),
                            'price': trading_signal.entry_price,
                            'confidence': trading_signal.confidence,
                            'strength': trading_signal.signal_strength.value,
                            'stop_loss': trading_signal.stop_loss_price,
                            'take_profit': trading_signal.take_profit_price,
                            'risk_reward_ratio': trading_signal.risk_reward_ratio,
                            'reasons': trading_signal.reasons,
                            'market_condition': trading_signal.market_condition,
                            'timestamp': trading_signal.timestamp
                        }
                        
                        # 分类信号（支持双向交易）
                        if trading_signal.signal_type.value.upper() == 'BUY':
                            buy_signals.append(signal_data)
                            self.logger.info(f"🟢 {symbol} 买入信号 - 置信度: {trading_signal.confidence:.2f}, 强度: {trading_signal.signal_strength.value}")
                            self.logger.info(f"   📋 理由: {', '.join(trading_signal.reasons)}")
                            self.logger.info(f"   📊 风险回报比: {trading_signal.risk_reward_ratio:.2f}, 市场条件: {trading_signal.market_condition}")
                        elif trading_signal.signal_type.value.upper() == 'SELL':
                            sell_signals.append(signal_data)
                            self.logger.info(f"🔴 {symbol} 卖出信号 - 置信度: {trading_signal.confidence:.2f}, 强度: {trading_signal.signal_strength.value}")
                            self.logger.info(f"   📋 理由: {', '.join(trading_signal.reasons)}")
                            self.logger.info(f"   📊 风险回报比: {trading_signal.risk_reward_ratio:.2f}, 市场条件: {trading_signal.market_condition}")
                    else:
                        current_price = float(ohlcv_data['close'].iloc[-1])
                        self.logger.info(f"➡️ {symbol} 观望 - 当前价格: {current_price:.4f} (未达到信号生成条件)")
                        
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 专业信号检测失败: {e}")
            
            # 统计和存储信号
            self.current_signals = {
                'buy': buy_signals,
                'sell': sell_signals,
                'timestamp': datetime.now()
            }
            
            # 输出信号汇总
            self.logger.info(f"📊 专业信号汇总: {len(buy_signals)} 个买入信号, {len(sell_signals)} 个卖出信号")
            self.logger.info("🎯 使用功能: HEAD_SHOULDER形态识别 + 双向交易 + 置信度过滤 + 周期分析")
            
            # 详细输出信号
            if buy_signals:
                self.logger.info("🟢 买入信号详情:")
                for signal in buy_signals:
                    self.logger.info(f"   • {signal['symbol']}: {signal['price']:.4f} (置信度:{signal['confidence']:.2f})")
                    
            if sell_signals:
                self.logger.info("🔴 卖出信号详情:")
                for signal in sell_signals:
                    self.logger.info(f"   • {signal['symbol']}: {signal['price']:.4f} (置信度:{signal['confidence']:.2f})")
            
        except Exception as e:
            self.logger.error(f"专业信号检测阶段失败: {e}")
    
    async def execute_all_trades(self):
        """阶段5: 执行交易决策和订单管理"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("⚡ 阶段5: 执行交易决策")
            self.logger.info("=" * 80)
            
            # 检查是否有信号数据
            if not hasattr(self, 'current_signals') or not self.current_signals:
                self.logger.info("📭 当前无交易信号，跳过交易执行")
                return
            
            # 从current_signals中获取买入和卖出信号
            buy_signals = self.current_signals.get('buy', [])
            sell_signals = self.current_signals.get('sell', [])
            
            if not buy_signals and not sell_signals:
                self.logger.info("📭 当前无有效交易信号，跳过交易执行")
                return
            
            # 处理买入信号
            if buy_signals:
                self.logger.info(f"📈 处理 {len(buy_signals)} 个买入信号:")
                for signal in buy_signals:
                    await self.execute_buy_order(signal)
            
            # 处理卖出信号
            if sell_signals:
                self.logger.info(f"📉 处理 {len(sell_signals)} 个卖出信号:")
                for signal in sell_signals:
                    await self.execute_sell_order(signal)
            
            # 当前市场条件下没有符合执行标准的信号
            if not buy_signals and not sell_signals:
                self.logger.info("💡 当前市场条件：")
                for symbol in self.selected_symbols:
                    if symbol in self.selected_symbols_data:
                        current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                        self.logger.info(f"   📊 {symbol}: 当前价格 {current_price:.4f}, 状态: 观望中")
                
        except Exception as e:
            self.logger.error(f"交易执行阶段失败: {e}")
    
    async def execute_buy_order(self, signal):
        """执行买入订单（使用专业信号的完整信息）"""
        try:
            symbol = signal['symbol']
            entry_price = signal['price']
            confidence = signal['confidence']
            strength = signal['strength']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            risk_reward_ratio = signal['risk_reward_ratio']
            reasons = signal.get('reasons', [])
            market_condition = signal.get('market_condition', 'unknown')
            
            self.logger.info(f"🎯 执行 {symbol} 专业买入订单:")
            self.logger.info(f"   💰 入场价格: {entry_price:.6f}")
            self.logger.info(f"   📊 信号置信度: {confidence:.2f}, 强度: {strength}")
            self.logger.info(f"   🎯 风险回报比: {risk_reward_ratio:.2f}")
            self.logger.info(f"   🌍 市场条件: {market_condition}")
            self.logger.info(f"   💡 买入理由: {', '.join(reasons)}")
            
            # 计算仓位大小（基于大佬的动态杠杆建议）
            position_size = self.calculate_dynamic_position_size(signal, symbol)
            # 动态杠杆已包含置信度和市场条件调整
            
            self.logger.info(f"   💼 仓位大小: {position_size:.6f} {symbol.replace('USDT', '')} (动态杠杆已调整)")
            
            self.logger.info(f"   🛡️ 专业风险管理设置:")
            self.logger.info(f"      🔴 止损点位: {stop_loss:.6f} ({((stop_loss/entry_price-1)*100):.2f}%)")
            self.logger.info(f"      🟢 止盈点位: {take_profit:.6f} ({((take_profit/entry_price-1)*100):.2f}%)")
            
            # 生成订单ID
            order_id = f"BUY_{symbol}_{int(datetime.now().timestamp())}"
            
            # 根据配置判断交易模式
            raw_config = self.config.get_config()
            config_dict = raw_config.dict()
            
            # 详细调试配置读取过程
            self.logger.info(f"   🔧 配置调试 - 完整配置结构存在api: {'api' in config_dict}")
            if 'api' in config_dict:
                api_config = config_dict.get('api', {})
                self.logger.info(f"   🔧 配置调试 - API配置存在binance: {'binance' in api_config}")
                if 'binance' in api_config:
                    binance_config = api_config.get('binance', {})
                    self.logger.info(f"   🔧 配置调试 - Binance配置: {binance_config}")
                    simulation_mode = binance_config.get('simulation_mode', True)
                else:
                    simulation_mode = True
                    self.logger.warning("   ⚠️ 配置警告 - 没有找到binance配置，使用默认模拟模式")
            else:
                simulation_mode = True  
                self.logger.warning("   ⚠️ 配置警告 - 没有找到api配置，使用默认模拟模式")
            
            self.logger.info(f"   🔧 配置调试: simulation_mode = {simulation_mode}")
            
            if simulation_mode:
                self.logger.info(f"   ✅ 买入订单已提交（模拟模式）")
            else:
                self.logger.info(f"   ✅ 买入订单已提交（实盘交易）")
                # 实盘交易逻辑
                try:
                    # 调用币安期货API下单
                    order_result = await self._place_real_buy_order(
                        symbol=symbol,
                        quantity=position_size,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if order_result:
                        self.logger.info(f"   🎉 实盘买入订单成功执行！")
                        self.logger.info(f"      📋 币安订单ID: {order_result.get('orderId', 'N/A')}")
                        self.logger.info(f"      💰 实际成交价: {order_result.get('price', entry_price)}")
                        self.logger.info(f"      📊 实际成交量: {order_result.get('executedQty', position_size)}")
                        
                        # 更新订单ID为真实订单ID
                        order_id = f"BINANCE_{order_result.get('orderId', order_id)}"
                    else:
                        self.logger.error(f"   ❌ 实盘买入订单失败！")
                        return  # 下单失败，不创建本地持仓记录
                        
                except Exception as api_error:
                    self.logger.error(f"   ❌ 币安API下单失败: {api_error}")
                    return  # API调用失败，不创建本地持仓记录
            
            # 更新持仓记录（包含专业信号信息）
            if not hasattr(self, 'current_positions'):
                self.current_positions = {}
                
            self.current_positions[symbol] = {
                'type': 'LONG',
                'entry_price': entry_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'strength': strength,
                'risk_reward_ratio': risk_reward_ratio,
                'market_condition': market_condition,
                'order_id': order_id,
                'timestamp': datetime.now(),
                'reasons': reasons
            }
            
        except Exception as e:
            self.logger.error(f"执行专业买入订单失败: {e}")
    
    async def execute_sell_order(self, signal):
        """执行卖出订单（使用专业信号的完整信息）"""
        try:
            symbol = signal['symbol']
            entry_price = signal['price']
            confidence = signal['confidence']
            strength = signal['strength']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            risk_reward_ratio = signal['risk_reward_ratio']
            reasons = signal.get('reasons', [])
            market_condition = signal.get('market_condition', 'unknown')
            
            self.logger.info(f"🎯 执行 {symbol} 专业卖出订单:")
            self.logger.info(f"   💰 入场价格: {entry_price:.6f}")
            self.logger.info(f"   📊 信号置信度: {confidence:.2f}, 强度: {strength}")
            self.logger.info(f"   🎯 风险回报比: {risk_reward_ratio:.2f}")
            self.logger.info(f"   🌍 市场条件: {market_condition}")
            self.logger.info(f"   💡 卖出理由: {', '.join(reasons)}")
            
            # 计算仓位大小（基于大佬的动态杠杆建议）
            position_size = self.calculate_dynamic_position_size(signal, symbol)
            # 动态杠杆已包含置信度和市场条件调整
            
            self.logger.info(f"   💼 仓位大小: {position_size:.6f} {symbol.replace('USDT', '')} (动态杠杆已调整)")
            
            self.logger.info(f"   🛡️ 专业风险管理设置:")
            self.logger.info(f"      🔴 止损点位: {stop_loss:.6f} ({((stop_loss/entry_price-1)*100):.2f}%)")
            self.logger.info(f"      🟢 止盈点位: {take_profit:.6f} ({((take_profit/entry_price-1)*100):.2f}%)")
            
            # 生成订单ID
            order_id = f"SELL_{symbol}_{int(datetime.now().timestamp())}"
            
            # 根据配置判断交易模式
            raw_config = self.config.get_config()
            config_dict = raw_config.dict()
            
            # 详细调试配置读取过程
            self.logger.info(f"   🔧 配置调试 - 完整配置结构存在api: {'api' in config_dict}")
            if 'api' in config_dict:
                api_config = config_dict.get('api', {})
                self.logger.info(f"   🔧 配置调试 - API配置存在binance: {'binance' in api_config}")
                if 'binance' in api_config:
                    binance_config = api_config.get('binance', {})
                    self.logger.info(f"   🔧 配置调试 - Binance配置: {binance_config}")
                    simulation_mode = binance_config.get('simulation_mode', True)
                else:
                    simulation_mode = True
                    self.logger.warning("   ⚠️ 配置警告 - 没有找到binance配置，使用默认模拟模式")
            else:
                simulation_mode = True  
                self.logger.warning("   ⚠️ 配置警告 - 没有找到api配置，使用默认模拟模式")
            
            self.logger.info(f"   🔧 配置调试: simulation_mode = {simulation_mode}")
            
            if simulation_mode:
                self.logger.info(f"   ✅ 卖出订单已提交（模拟模式）")
            else:
                self.logger.info(f"   ✅ 卖出订单已提交（实盘交易）")
                # 实盘交易逻辑
                try:
                    # 调用币安期货API下单
                    order_result = await self._place_real_sell_order(
                        symbol=symbol,
                        quantity=position_size,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if order_result:
                        self.logger.info(f"   🎉 实盘卖出订单成功执行！")
                        self.logger.info(f"      📋 币安订单ID: {order_result.get('orderId', 'N/A')}")
                        self.logger.info(f"      💰 实际成交价: {order_result.get('price', entry_price)}")
                        self.logger.info(f"      📊 实际成交量: {order_result.get('executedQty', position_size)}")
                        
                        # 更新订单ID为真实订单ID
                        order_id = f"BINANCE_{order_result.get('orderId', order_id)}"
                    else:
                        self.logger.error(f"   ❌ 实盘卖出订单失败！")
                        return  # 下单失败，不创建本地持仓记录
                        
                except Exception as api_error:
                    self.logger.error(f"   ❌ 币安API下单失败: {api_error}")
                    return  # API调用失败，不创建本地持仓记录
            
            # 更新持仓记录（包含专业信号信息）
            if not hasattr(self, 'current_positions'):
                self.current_positions = {}
                
            self.current_positions[symbol] = {
                'type': 'SHORT',
                'entry_price': entry_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'strength': strength,
                'risk_reward_ratio': risk_reward_ratio,
                'market_condition': market_condition,
                'order_id': order_id,
                'timestamp': datetime.now(),
                'reasons': reasons
            }
            
        except Exception as e:
            self.logger.error(f"执行专业卖出订单失败: {e}")
    
    def calculate_position_size_advanced(self, current_price: float, symbol: str) -> float:
        """高级仓位大小计算"""
        try:
            risk_config = self.config.get_risk_config()
            trading_config = self.config.get_trading_config()
            
            # 基础资金
            base_capital = trading_config.initial_capital
            risk_per_trade = risk_config.risk_per_trade
            max_position_size = risk_config.max_position_size
            
            # 风险金额
            risk_amount = base_capital * risk_per_trade
            
            # 基于价格计算基础仓位
            base_position = risk_amount / current_price
            
            # 应用最大仓位限制
            max_allowed = base_capital * max_position_size / current_price
            
            # 返回较小值
            return min(base_position, max_allowed)
            
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0.01  # 默认最小仓位
    
    def calculate_stop_loss_take_profit(self, entry_price: float, direction: str, symbol: str) -> tuple:
        """计算止盈止损点位"""
        try:
            risk_config = self.config.get_risk_config()
            
            # 基础止损比例
            stop_loss_pct = risk_config.stop_loss_pct
            take_profit_ratio = risk_config.take_profit_ratio
            
            if direction == 'BUY':
                # 多头：止损在下方，止盈在上方
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + stop_loss_pct * take_profit_ratio)
            else:  # SELL
                # 空头：止损在上方，止盈在下方
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - stop_loss_pct * take_profit_ratio)
                
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"计算止盈止损失败: {e}")
            if direction == 'BUY':
                return entry_price * 0.98, entry_price * 1.06
            else:
                return entry_price * 1.02, entry_price * 0.94
    
    async def monitor_positions(self):
        """阶段6: 持仓监控 - 同步币安真实持仓数据"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("👁️ 阶段6: 持仓监控")
            self.logger.info("=" * 80)
            
            # 🚀 关键修复：从币安API获取真实持仓数据
            self.logger.info("🔄 正在从币安API同步真实持仓数据...")
            
            try:
                account_response = await self.api_client_manager.get_account_info()
                
                if account_response and account_response.success:
                    account_data = account_response.data
                    positions_data = account_data.get('positions', [])
                    
                    # 更新系统持仓数据
                    self.current_positions = {}
                    active_positions = []
                    
                    for pos in positions_data:
                        symbol = pos.get('symbol', '')
                        position_amt = float(pos.get('positionAmt', 0))
                        entry_price = float(pos.get('entryPrice', 0))
                        unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                        
                        # 只处理有仓位的品种
                        if abs(position_amt) > 0 and entry_price > 0:
                            side = 'LONG' if position_amt > 0 else 'SHORT'
                            position_info = {
                                'symbol': symbol,
                                'side': side,
                                'size': abs(position_amt),
                                'entry_price': entry_price,
                                'unrealized_pnl': unrealized_pnl,
                                'position_amt': position_amt,
                                'timestamp': datetime.now()
                            }
                            self.current_positions[symbol] = position_info
                            active_positions.append(f"{symbol}: {side} ${abs(position_amt):.2f}")
                    
                    if active_positions:
                        self.logger.info(f"✅ 币安API持仓同步成功")
                        self.logger.info(f"💼 持仓数量: {len(self.current_positions)}")
                        self.logger.info(f"📋 持仓详情:")
                        for pos_detail in active_positions:
                            self.logger.info(f"   • {pos_detail}")
                    else:
                        self.logger.info("📭 币安账户当前无持仓")
                        return
                        
                else:
                    self.logger.error("❌ 获取币安账户信息失败")
                    # 回退到内存持仓检查
                    if not hasattr(self, 'current_positions') or not self.current_positions:
                        self.logger.info("📭 当前无持仓，跳过持仓监控")
                        return
                        
            except Exception as e:
                self.logger.error(f"币安持仓同步异常: {e}")
                # 回退到内存持仓检查
                if not hasattr(self, 'current_positions') or not self.current_positions:
                    self.logger.info("📭 当前无持仓，跳过持仓监控")
                    return
            
            # 监控每个持仓（修复线程安全问题）
            # 创建持仓的副本避免在迭代时修改字典
            current_positions_copy = dict(self.current_positions.items())
            for symbol, position in current_positions_copy.items():
                # 再次检查该持仓是否仍然存在（可能已被平仓）
                if symbol not in self.current_positions:
                    self.logger.info(f"⚠️ {symbol} 持仓已被平仓，跳过监控")
                    continue
                    
                self.logger.info(f"👁️ 监控 {symbol} 持仓...")
                await self.check_position_exit_signals(symbol, position)
                
        except Exception as e:
            self.logger.error(f"持仓监控阶段失败: {e}")
    
    async def check_position_exit_signals(self, symbol: str, position: dict):
        """检查持仓退出信号（详细的拐点和反转分析）"""
        try:
            # 获取实时当前价格（关键修复）
            self.logger.info(f"   🔄 正在获取 {symbol} 实时价格...")
            
            # 使用API获取实时价格
            try:
                ticker_response = await self.api_client_manager.get_ticker(symbol)
                if ticker_response and ticker_response.success:
                    current_price = float(ticker_response.data.get('lastPrice', 0))
                else:
                    # 备用：从K线数据获取
                    if symbol in self.selected_symbols_data:
                        current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                    else:
                        self.logger.error(f"❌ {symbol} 无法获取价格数据")
                        return
            except Exception as e:
                self.logger.warning(f"⚠️ {symbol} 实时价格获取失败，使用K线价格: {e}")
                if symbol in self.selected_symbols_data:
                    current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                else:
                    return
                    
            entry_price = position['entry_price']
            position_type = position['side']  # 修复：使用'side'而不是'type'
            position_size = position['size']
            
            # 动态计算止损止盈（关键修复）
            direction = 'BUY' if position_type == 'LONG' else 'SELL'
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(entry_price, direction, symbol)
            
            # 计算当前盈亏
            if position_type == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
            self.logger.info(f"   📊 {symbol} 持仓状态:")
            self.logger.info(f"      📈 持仓类型: {position_type}")
            self.logger.info(f"      💰 入场价格: {entry_price:.6f}")
            self.logger.info(f"      💱 当前价格: {current_price:.6f} (实时获取)")
            self.logger.info(f"      📊 当前盈亏: {pnl_pct:+.2f}%")
            self.logger.info(f"      🛡️ 止损价格: {stop_loss:.6f}")
            self.logger.info(f"      🎯 止盈价格: {take_profit:.6f}")
            
            # 检查止损止盈
            exit_signal = None
            if position_type == 'LONG':
                if current_price <= stop_loss:
                    exit_signal = "STOP_LOSS"
                    self.logger.warning(f"🚨 {symbol} 触发止损! 价格{current_price:.6f} <= 止损{stop_loss:.6f}")
                elif current_price >= take_profit:
                    exit_signal = "TAKE_PROFIT" 
                    self.logger.info(f"🎉 {symbol} 触发止盈! 价格{current_price:.6f} >= 止盈{take_profit:.6f}")
            else:  # SHORT
                if current_price >= stop_loss:
                    exit_signal = "STOP_LOSS"
                    self.logger.warning(f"🚨 {symbol} 触发止损! 价格{current_price:.6f} >= 止损{stop_loss:.6f}")
                elif current_price <= take_profit:
                    exit_signal = "TAKE_PROFIT"
                    self.logger.info(f"🎉 {symbol} 触发止盈! 价格{current_price:.6f} <= 止盈{take_profit:.6f}")
            
            # 技术拐点检测（使用实时价格数据）
            if symbol in self.selected_symbols_data:
                await self.check_technical_reversal(symbol, self.selected_symbols_data[symbol], position_type, current_price)
            
            # 如果有退出信号，执行平仓
            if exit_signal:
                await self.close_position(symbol, exit_signal, current_price)
                
        except Exception as e:
            self.logger.error(f"检查 {symbol} 退出信号失败: {e}")
    
    async def check_technical_reversal(self, symbol: str, price_data, position_type: str, current_price: float = None):
        """检测技术反转信号"""
        try:
            # 获取最近价格数据
            closes = price_data['close'].values
            if len(closes) < 20:
                return
                
            # 使用实时价格（如果提供的话）
            if current_price is not None:
                # 使用实时价格作为最新价格
                recent_closes = closes[-19:]  # 取前19个历史价格
                recent_closes = list(recent_closes) + [current_price]  # 加上实时价格
                current_price_for_analysis = current_price
            else:
                recent_closes = closes[-20:]
                current_price_for_analysis = recent_closes[-1]
            
            # 短期和中期价格变化
            price_change_5 = (current_price_for_analysis - recent_closes[-5]) / recent_closes[-5] * 100
            price_change_10 = (current_price_for_analysis - recent_closes[-10]) / recent_closes[-10] * 100
            
            # 拐点检测
            local_max = max(recent_closes[-10:-5])
            local_min = min(recent_closes[-10:-5])
            
            self.logger.info(f"   🔍 {symbol} 技术分析:")
            self.logger.info(f"      📈 短期变化(5周期): {price_change_5:+.2f}%")
            self.logger.info(f"      📊 中期变化(10周期): {price_change_10:+.2f}%")
            self.logger.info(f"      🔺 近期高点: {local_max:.6f}")
            self.logger.info(f"      🔻 近期低点: {local_min:.6f}")
            
            # 反转信号检测
            reversal_signals = []
            
            if abs(price_change_5) > 3:  # 短期剧烈变化
                if price_change_5 > 0 and price_change_10 < -2:
                    reversal_signals.append("短期反弹，中期仍跌")
                elif price_change_5 < 0 and price_change_10 > 2:
                    reversal_signals.append("短期回调，中期仍涨")
            
            # 拐点信号
            if current_price_for_analysis < local_max * 0.95:  # 从高点下跌5%
                reversal_signals.append(f"从高点{local_max:.6f}下跌{((1-current_price_for_analysis/local_max)*100):.1f}%")
            elif current_price_for_analysis > local_min * 1.05:  # 从低点上涨5%
                reversal_signals.append(f"从低点{local_min:.6f}上涨{((current_price_for_analysis/local_min-1)*100):.1f}%")
                
            if reversal_signals:
                self.logger.info(f"   🔄 {symbol} 反转信号: {'; '.join(reversal_signals)}")
                
                # 对于持仓的反转警告
                if position_type == 'LONG' and any('下跌' in s for s in reversal_signals):
                    self.logger.warning(f"⚠️ {symbol} 多头持仓面临反转风险")
                elif position_type == 'SHORT' and any('上涨' in s for s in reversal_signals):
                    self.logger.warning(f"⚠️ {symbol} 空头持仓面临反转风险")
            else:
                self.logger.info(f"   ✅ {symbol} 暂无明显反转信号")
                
        except Exception as e:
            self.logger.error(f"{symbol} 技术反转检测失败: {e}")
    
    async def close_position(self, symbol: str, reason: str, current_price: float):
        """平仓操作"""
        try:
            position = self.current_positions[symbol]
            
            self.logger.info(f"⚡ 执行 {symbol} 平仓操作:")
            self.logger.info(f"   🎯 平仓原因: {reason}")
            self.logger.info(f"   💰 平仓价格: {current_price:.6f}")
            
            # 计算最终盈亏
            entry_price = position['entry_price']
            if position['side'] == 'LONG':
                final_pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                final_pnl = ((entry_price - current_price) / entry_price) * 100
                
            self.logger.info(f"   📊 最终盈亏: {final_pnl:+.2f}%")
            
            # 模拟平仓订单
            close_order_id = f"CLOSE_{symbol}_{int(datetime.now().timestamp())}"
            self.logger.info(f"   📋 平仓订单ID: {close_order_id}")
            
            # 根据配置判断交易模式
            # 直接从原始配置字典获取，避免APIConfig对象的.get()方法问题
            raw_config = self.config.get_config()
            config_dict = raw_config.dict()
            
            # 详细调试配置读取过程
            self.logger.info(f"   🔧 配置调试 - 完整配置结构存在api: {'api' in config_dict}")
            if 'api' in config_dict:
                api_config = config_dict.get('api', {})
                self.logger.info(f"   🔧 配置调试 - API配置存在binance: {'binance' in api_config}")
                if 'binance' in api_config:
                    binance_config = api_config.get('binance', {})
                    self.logger.info(f"   🔧 配置调试 - Binance配置: {binance_config}")
                    simulation_mode = binance_config.get('simulation_mode', True)
                else:
                    simulation_mode = True
                    self.logger.warning("   ⚠️ 配置警告 - 没有找到binance配置，使用默认模拟模式")
            else:
                simulation_mode = True  
                self.logger.warning("   ⚠️ 配置警告 - 没有找到api配置，使用默认模拟模式")
            
            self.logger.info(f"   🔧 配置调试: simulation_mode = {simulation_mode}")
            
            if simulation_mode:
                self.logger.info(f"   ✅ 平仓订单已提交（模拟模式）")
            else:
                # 实盘交易模式：调用真实的币安API执行平仓
                try:
                    # 构建平仓订单参数（智能适配单/双向持仓模式）
                    close_side = "SELL" if position['side'] == 'LONG' else "BUY"
                    
                    # 基础订单参数（移除reduceOnly参数，币安API当前不需要）
                    order_params = {
                        'symbol': symbol,
                        'side': close_side,
                        'type': 'MARKET',
                        'quantity': str(abs(position['size']))  # 使用绝对值
                    }
                    
                    # 先尝试不带positionSide参数（单向持仓模式）
                    self.logger.info(f"   🎯 尝试单向持仓模式平仓: {order_params}")
                    
                    # 调用币安API执行平仓
                    api_response = await self.api_client_manager.place_order(order_params)
                    
                    if not api_response.success and "position side does not match" in str(api_response.data).lower():
                        # 如果单向模式失败，尝试双向持仓模式
                        self.logger.info("   🔄 单向持仓模式失败，尝试双向持仓模式...")
                        position_side = "LONG" if position['side'] == 'LONG' else "SHORT"
                        order_params['positionSide'] = position_side
                        self.logger.info(f"   🎯 尝试双向持仓模式平仓: {order_params}")
                        
                        # 再次调用API
                        api_response = await self.api_client_manager.place_order(order_params)
                    
                    if api_response.success:
                        self.logger.info(f"   ✅ 币安API平仓成功: {api_response.data}")
                        order_id = api_response.data.get('orderId', 'N/A')
                        self.logger.info(f"   📋 币安订单ID: {order_id}")
                    else:
                        self.logger.error(f"   ❌ 币安API平仓失败: {api_response.error_message}")
                        # 发送Telegram警告
                        await self.telegram_bot.send_message(
                            f"⚠️ {symbol} 平仓失败！\n"
                            f"错误: {api_response.error_message}\n"
                            f"请手动检查并平仓！"
                        )
                        return  # 平仓失败，不删除持仓记录
                        
                except Exception as api_error:
                    self.logger.error(f"   ❌ 调用币安API平仓异常: {api_error}")
                    await self.telegram_bot.send_message(
                        f"⚠️ {symbol} 平仓API调用异常！\n"
                        f"错误: {api_error}\n"
                        f"请手动检查并平仓！"
                    )
                    return  # API调用失败，不删除持仓记录
            
            # 移除持仓记录
            del self.current_positions[symbol]
            
        except Exception as e:
            self.logger.error(f"平仓 {symbol} 失败: {e}")
    
    async def update_risk_management(self):
        """阶段7: 风险管理和系统状态检查"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🛡️ 阶段7: 风险管理")
            self.logger.info("=" * 80)
            
            risk_config = self.config.get_risk_config()
            trading_config = self.config.get_trading_config()
            
            # 获取真实的币安账户余额
            try:
                account_response = await self.api_client_manager.get_account_info()
                if account_response and account_response.success:
                    account_data = account_response.data
                    # 获取USDT余额（总余额）
                    total_wallet_balance = float(account_data.get('totalWalletBalance', 0))
                    available_balance = float(account_data.get('availableBalance', 0))
                    used_margin = float(account_data.get('totalInitialMargin', 0))
                    
                    self.logger.info(f"📊 风险管理状态（真实币安数据）:")
                    self.logger.info(f"   💼 当前持仓数: {len(self.current_positions) if hasattr(self, 'current_positions') else 0}")
                    self.logger.info(f"   🎯 最大总敞口: {risk_config.max_total_exposure:.1%}")
                    self.logger.info(f"   💰 总资金: ${total_wallet_balance:,.2f} USDT")
                    self.logger.info(f"   💵 可用余额: ${available_balance:,.2f} USDT")
                    self.logger.info(f"   🔒 已用保证金: ${used_margin:,.2f} USDT")
                    self.logger.info(f"   ⚖️ 单笔风险: {risk_config.risk_per_trade:.1%}")
                    self.logger.info(f"   🛡️ 止损比例: {risk_config.stop_loss_pct:.1%}")
                    
                    # 使用真实资金计算风险
                    current_capital = total_wallet_balance
                else:
                    # API调用失败，使用配置值作为备用
                    self.logger.warning("⚠️ 无法获取币安账户信息，使用配置默认值")
                    current_capital = trading_config.initial_capital
                    self.logger.info(f"📊 风险管理状态（配置默认值）:")
                    self.logger.info(f"   💼 当前持仓数: {len(self.current_positions) if hasattr(self, 'current_positions') else 0}")
                    self.logger.info(f"   🎯 最大总敞口: {risk_config.max_total_exposure:.1%}")
                    self.logger.info(f"   💰 初始资金: ${current_capital:,.2f}")
                    self.logger.info(f"   ⚖️ 单笔风险: {risk_config.risk_per_trade:.1%}")
                    self.logger.info(f"   🛡️ 止损比例: {risk_config.stop_loss_pct:.1%}")
                    
            except Exception as e:
                # API调用异常，使用配置值作为备用
                self.logger.error(f"获取币安账户信息失败: {e}")
                current_capital = trading_config.initial_capital
                self.logger.info(f"📊 风险管理状态（备用配置）:")
                self.logger.info(f"   💼 当前持仓数: {len(self.current_positions) if hasattr(self, 'current_positions') else 0}")
                self.logger.info(f"   🎯 最大总敞口: {risk_config.max_total_exposure:.1%}")
                self.logger.info(f"   💰 初始资金: ${current_capital:,.2f}")
                self.logger.info(f"   ⚖️ 单笔风险: {risk_config.risk_per_trade:.1%}")
                self.logger.info(f"   🛡️ 止损比例: {risk_config.stop_loss_pct:.1%}")
                
            # 系统风险状态检查
            total_exposure = 0.0
            
            # 计算当前总风险敞口
            if hasattr(self, 'current_positions') and self.current_positions:
                self.logger.info(f"📋 持仓详情:")
                for symbol, position in self.current_positions.items():
                    entry_price = position['entry_price']
                    size = position['size']
                    position_value = entry_price * size
                    total_exposure += position_value
                    
                    self.logger.info(f"   • {symbol}: {position['side']} ${position_value:,.2f}")
                    
                exposure_pct = (total_exposure / current_capital) * 100
                self.logger.info(f"   📊 总敞口: ${total_exposure:,.2f} ({exposure_pct:.1f}%)")
                
                # 风险警告
                if exposure_pct > risk_config.max_total_exposure * 100:
                    self.logger.warning(f"⚠️ 总敞口超限! {exposure_pct:.1f}% > {risk_config.max_total_exposure*100:.1f}%")
                else:
                    self.logger.info(f"✅ 总敞口正常")
            else:
                self.logger.info("📭 当前无持仓，风险敞口为0")
                
        except Exception as e:
            self.logger.error(f"风险管理阶段失败: {e}")
    
    async def generate_reports(self):
        """阶段8: 生成系统运行报告"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("📋 阶段8: 系统运行摘要")
            self.logger.info("=" * 80)
            
            # 统计信息
            selected_count = len(self.selected_symbols) if hasattr(self, 'selected_symbols') else 0
            signal_count = 0
            position_count = len(self.current_positions) if hasattr(self, 'current_positions') else 0
            
            if hasattr(self, 'current_signals') and self.current_signals:
                signal_count = len(self.current_signals.get('buy', [])) + len(self.current_signals.get('sell', []))
            
            self.logger.info(f"📊 运行统计:")
            self.logger.info(f"   🎯 选中币种数: {selected_count}")
            self.logger.info(f"   📡 活跃信号数: {signal_count}")
            self.logger.info(f"   💼 持仓数量: {position_count}")
            
            # 当前市场状况摘要
            if hasattr(self, 'selected_symbols') and self.selected_symbols:
                self.logger.info(f"📈 市场状况摘要:")
                for symbol in self.selected_symbols:
                    if hasattr(self, 'selected_symbols_data') and symbol in self.selected_symbols_data:
                        current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                        status = "持仓中" if hasattr(self, 'current_positions') and symbol in self.current_positions else "观望中"
                        self.logger.info(f"   • {symbol}: {current_price:.4f} ({status})")
            
            # 下次执行时间提示
            next_scan_time = datetime.now() + timedelta(seconds=30)
            self.logger.info(f"⏰ 下次扫描时间: {next_scan_time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"报告生成阶段失败: {e}") 

    async def _place_real_buy_order(self, symbol: str, quantity: float, price: float, stop_loss: float, take_profit: float):
        """执行真实的币安期货买入订单"""
        try:
            self.logger.info(f"   🔄 正在向币安提交 {symbol} 买入订单...")
            
            # 根据交易对调整数量精度（基于币安期货实际规则）
            if symbol in ['XTZUSDT', 'ADAUSDT', 'DOGEUSDT']:
                # 小价格币种，精度为整数
                adjusted_quantity = int(quantity)
            elif symbol in ['SOLUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']:
                # SOL类中价格币种，精度为1位小数
                adjusted_quantity = round(quantity, 1)
            elif symbol in ['LTCUSDT']:
                # LTC，精度为2位小数
                adjusted_quantity = round(quantity, 2)
            elif symbol in ['ETHUSDT']:
                # ETH，精度为3位小数
                adjusted_quantity = round(quantity, 3)
            elif symbol in ['BTCUSDT']:
                # BTC，精度为4位小数
                adjusted_quantity = round(quantity, 4)
            else:
                # 默认精度为2位小数（更安全的默认值）
                adjusted_quantity = round(quantity, 2)
            
            self.logger.info(f"   📏 数量精度调整: {quantity} → {adjusted_quantity}")
            
            # 使用市价单快速成交（兼容双向持仓模式）
            order_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',  # 市价单
                'quantity': adjusted_quantity,
                'positionSide': 'LONG',  # 明确指定多头持仓（双向持仓模式）
                'timestamp': int(time.time() * 1000)
            }
            
            self.logger.info(f"   📋 下单参数: {order_params}")
            
            # 调用API客户端下单
            response = await self.api_client_manager.place_order(order_params)
            
            if response and response.success:
                order_data = response.data
                self.logger.info(f"   ✅ 币安下单API调用成功")
                return order_data
            else:
                error_msg = response.error_message if response else "未知错误"
                self.logger.error(f"   ❌ 币安下单失败: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"   ❌ 币安下单异常: {e}")
            return None 