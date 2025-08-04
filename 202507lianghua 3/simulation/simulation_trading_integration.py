"""
模拟交易集成模块
将30天模拟交易系统与现有交易引擎集成，支持实时信号处理和模拟交易执行
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from simulation.simulation_trading_manager import (
    SimulationTradingManager, TradeType, SimulationConfig
)
from config.config_manager import ConfigManager
from data.api_client import get_api_client
from core.ultimate_multi_timeframe_signal_generator import UltimateMultiTimeframeSignalGenerator
from core.signal_generator import SignalType, SignalStrength
from utils.logger import get_logger
import pandas as pd

class SimulationTradingIntegration:
    """模拟交易集成系统"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.logger = get_logger(__name__)
        self.config = config_manager or ConfigManager()
        
        # 初始化组件
        self.simulation_manager = SimulationTradingManager(config_manager)
        self.api_client = get_api_client()
        self.signal_generator = UltimateMultiTimeframeSignalGenerator(self.config)
        
        # 运行状态
        self.is_running = False
        self.should_stop = False
        
        # 配置参数
        self.test_symbols = self.simulation_manager.sim_config.test_symbols
        self.check_interval = 30  # 30秒检查一次
        
        # 价格缓存
        self.price_cache: Dict[str, float] = {}
        
        self.logger.info("🔄 模拟交易集成系统初始化完成")
        self.logger.info(f"   📊 测试币种: {', '.join(self.test_symbols)}")
        self.logger.info(f"   ⏰ 检查间隔: {self.check_interval}秒")
    
    async def start_simulation(self, duration_days: int = 30):
        """启动模拟交易系统"""
        try:
            if self.is_running:
                self.logger.warning("模拟交易系统已在运行中")
                return
            
            self.is_running = True
            self.should_stop = False
            
            self.logger.info(f"🚀 启动30天模拟交易系统（计划运行{duration_days}天）")
            self.logger.info("="*80)
            
            # 显示初始状态
            await self._log_system_status()
            
            # 主循环
            start_time = datetime.now()
            end_time = start_time + timedelta(days=duration_days)
            cycle_count = 0
            
            while not self.should_stop and datetime.now() < end_time:
                try:
                    cycle_count += 1
                    self.logger.info(f"🔄 执行第{cycle_count}次交易周期检查...")
                    
                    # 执行一个完整的交易周期
                    await self._execute_trading_cycle()
                    
                    # 生成日报告（如果是新的一天）
                    await self._generate_daily_report_if_needed()
                    
                    # 等待下一个周期
                    await self._wait_for_next_cycle()
                    
                except Exception as e:
                    self.logger.error(f"交易周期执行失败: {e}")
                    await asyncio.sleep(self.check_interval)
            
            # 生成最终报告
            await self._generate_final_report()
            
        except Exception as e:
            self.logger.error(f"模拟交易系统运行失败: {e}")
        finally:
            self.is_running = False
            self.logger.info("🏁 模拟交易系统已停止")
    
    async def _execute_trading_cycle(self):
        """执行一个完整的交易周期"""
        try:
            # 1. 更新价格缓存
            await self._update_price_cache()
            
            # 2. 检查现有持仓的止损止盈
            await self._check_existing_positions()
            
            # 3. 为每个币种生成信号
            await self._process_trading_signals()
            
            # 4. 执行风险管理检查
            await self._risk_management_check()
            
        except Exception as e:
            self.logger.error(f"交易周期执行失败: {e}")
    
    async def _update_price_cache(self):
        """更新价格缓存"""
        try:
            for symbol in self.test_symbols:
                try:
                    # 获取实时价格（使用K线最新收盘价）
                    response = await self.api_client.get_klines(
                        symbol=symbol,
                        interval='1m',
                        limit=1
                    )
                    
                    if response and response.success and response.data:
                        latest_kline = response.data[0]
                        current_price = float(latest_kline[4])  # 收盘价
                        self.price_cache[symbol] = current_price
                        
                    await asyncio.sleep(0.1)  # 避免API频率限制
                    
                except Exception as e:
                    self.logger.error(f"获取{symbol}价格失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"更新价格缓存失败: {e}")
    
    async def _check_existing_positions(self):
        """检查现有持仓的止损止盈"""
        try:
            for symbol in list(self.simulation_manager.open_positions.keys()):
                if symbol in self.price_cache:
                    current_price = self.price_cache[symbol]
                    
                    # 检查止损止盈
                    position_closed = await self.simulation_manager.check_stop_loss_take_profit(
                        symbol, current_price
                    )
                    
                    if position_closed:
                        self.logger.info(f"   📍 {symbol} 持仓已自动平仓")
                        
        except Exception as e:
            self.logger.error(f"检查持仓失败: {e}")
    
    async def _process_trading_signals(self):
        """处理交易信号"""
        try:
            for symbol in self.test_symbols:
                # 跳过已有持仓的币种
                if symbol in self.simulation_manager.open_positions:
                    continue
                
                try:
                    # 获取多时间周期数据
                    multi_data = await self._fetch_multi_timeframe_data(symbol)
                    if not multi_data:
                        continue
                    
                    # 生成信号
                    signal = await self.signal_generator.generate_ultimate_signal(symbol, multi_data)
                    
                    if signal and symbol in self.price_cache:
                        await self._execute_signal(symbol, signal, self.price_cache[symbol])
                    
                    await asyncio.sleep(0.2)  # 避免频率限制
                    
                except Exception as e:
                    self.logger.error(f"处理{symbol}信号失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"信号处理失败: {e}")
    
    async def _fetch_multi_timeframe_data(self, symbol: str) -> Optional[Dict]:
        """获取多时间周期数据"""
        try:
            timeframes_mapping = {
                'trend': '4h',
                'signal': '1h', 
                'entry': '15m',
                'confirm': '5m'
            }
            
            multi_data = {}
            for tf_name, interval in timeframes_mapping.items():
                response = await self.api_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=100
                )
                
                if response and response.success:
                    klines_data = response.data
                    df = pd.DataFrame(klines_data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    # 转换数据类型
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col])
                        
                    multi_data[tf_name] = df
                else:
                    self.logger.warning(f"获取{symbol} {interval}数据失败")
                    return None
                    
                await asyncio.sleep(0.1)
                
            return multi_data
            
        except Exception as e:
            self.logger.error(f"获取{symbol}多时间周期数据失败: {e}")
            return None
    
    async def _execute_signal(self, symbol: str, signal, current_price: float):
        """执行交易信号"""
        try:
            # 只处理明确的买入或卖出信号
            if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
                return
            
            # 检查信号强度阈值
            if signal.confidence < 0.6:  # 置信度阈值60%
                self.logger.debug(f"   ⚪ {symbol} 信号置信度不足: {signal.confidence:.3f}")
                return
            
            # 确定交易类型
            trade_type = TradeType.BUY if signal.signal_type == SignalType.BUY else TradeType.SELL
            
            # 计算仓位大小和杠杆
            leverage = self._calculate_leverage(signal)
            quantity = self._calculate_position_size(current_price, leverage)
            
            # 计算止损止盈
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                current_price, trade_type, signal
            )
            
            # 准备信号信息
            signal_info = {
                'confidence': signal.confidence,
                'strength': signal.signal_strength.value if hasattr(signal.signal_strength, 'value') else str(signal.signal_strength),
                'reasons': getattr(signal, 'reasons', [])
            }
            
            # 执行开仓
            success = await self.simulation_manager.open_position(
                symbol=symbol,
                trade_type=trade_type,
                entry_price=current_price,
                quantity=quantity,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_info=signal_info
            )
            
            if success:
                self.logger.info(f"   ✅ {symbol} 模拟开仓成功")
                self.logger.info(f"      🎯 信号: {signal.signal_type.value}")
                self.logger.info(f"      📊 置信度: {signal.confidence:.3f}")
                self.logger.info(f"      💪 强度: {signal_info['strength']}")
            
        except Exception as e:
            self.logger.error(f"执行{symbol}交易信号失败: {e}")
    
    def _calculate_leverage(self, signal) -> float:
        """根据信号强度计算杠杆倍数"""
        try:
            base_leverage = 10.0  # 基础杠杆
            
            # 根据信号强度调整
            if hasattr(signal, 'signal_strength'):
                strength = signal.signal_strength
                if hasattr(strength, 'value'):
                    strength_value = strength.value
                else:
                    strength_value = str(strength)
                
                if 'VERY_STRONG' in strength_value:
                    leverage_multiplier = 1.5
                elif 'STRONG' in strength_value:
                    leverage_multiplier = 1.2
                elif 'MEDIUM' in strength_value:
                    leverage_multiplier = 1.0
                else:
                    leverage_multiplier = 0.8
            else:
                leverage_multiplier = 1.0
            
            # 根据置信度调整
            confidence_multiplier = signal.confidence
            
            # 计算最终杠杆
            final_leverage = base_leverage * leverage_multiplier * confidence_multiplier
            
            # 限制在合理范围内
            final_leverage = max(2.0, min(final_leverage, self.simulation_manager.sim_config.max_leverage))
            
            return round(final_leverage, 1)
            
        except Exception as e:
            self.logger.error(f"计算杠杆失败: {e}")
            return 5.0  # 默认杠杆
    
    def _calculate_position_size(self, price: float, leverage: float) -> float:
        """计算仓位大小"""
        try:
            # 使用固定风险金额计算仓位
            risk_amount = self.simulation_manager.current_capital * 0.02  # 2%风险
            position_value = risk_amount * leverage / 0.02  # 假设2%止损
            quantity = position_value / price / leverage
            
            # 确保仓位不超过限制
            max_position_value = self.simulation_manager.current_capital * self.simulation_manager.sim_config.max_position_size_pct
            if position_value > max_position_value:
                quantity = max_position_value / price / leverage
            
            return round(quantity, 6)
            
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0.001  # 默认最小仓位
    
    def _calculate_stop_loss_take_profit(self, entry_price: float, trade_type: TradeType, signal) -> tuple:
        """计算止损止盈价位"""
        try:
            # 基础止损距离（2%）
            stop_distance_pct = 0.02
            
            # 根据信号强度调整
            if hasattr(signal, 'signal_strength'):
                strength = str(signal.signal_strength)
                if 'VERY_STRONG' in strength:
                    stop_distance_pct = 0.015  # 1.5%
                elif 'STRONG' in strength:
                    stop_distance_pct = 0.018  # 1.8%
                elif 'MEDIUM' in strength:
                    stop_distance_pct = 0.02   # 2%
                else:
                    stop_distance_pct = 0.025  # 2.5%
            
            # 计算止损止盈
            if trade_type == TradeType.BUY:
                stop_loss = entry_price * (1 - stop_distance_pct)
                take_profit = entry_price * (1 + stop_distance_pct * 2.5)  # 2.5:1 风险回报比
            else:  # SELL
                stop_loss = entry_price * (1 + stop_distance_pct)
                take_profit = entry_price * (1 - stop_distance_pct * 2.5)
            
            return round(stop_loss, 6), round(take_profit, 6)
            
        except Exception as e:
            self.logger.error(f"计算止损止盈失败: {e}")
            return None, None
    
    async def _risk_management_check(self):
        """风险管理检查"""
        try:
            summary = self.simulation_manager.get_performance_summary()
            
            # 检查最大回撤
            if summary['max_drawdown'] > self.simulation_manager.sim_config.max_drawdown_pct:
                self.logger.warning(f"⚠️ 最大回撤超限: {summary['max_drawdown']*100:.1f}%")
            
            # 检查连续亏损
            if summary['current_consecutive_losses'] >= 5:
                self.logger.warning(f"⚠️ 连续亏损过多: {summary['current_consecutive_losses']}次")
            
            # 检查资金状况
            if summary['current_capital'] < self.simulation_manager.sim_config.initial_capital * 0.8:
                self.logger.warning(f"⚠️ 资金亏损严重: {summary['current_capital']:.2f} USDT")
            
        except Exception as e:
            self.logger.error(f"风险管理检查失败: {e}")
    
    async def _generate_daily_report_if_needed(self):
        """如果需要则生成日报告"""
        try:
            # 每天晚上23:30生成日报告
            now = datetime.now()
            if now.hour == 23 and now.minute >= 30:
                await self.simulation_manager.generate_daily_report()
                
        except Exception as e:
            self.logger.error(f"生成日报告失败: {e}")
    
    async def _wait_for_next_cycle(self):
        """等待下一个周期"""
        for _ in range(self.check_interval):
            if self.should_stop:
                break
            await asyncio.sleep(1)
    
    async def _log_system_status(self):
        """记录系统状态"""
        try:
            summary = self.simulation_manager.get_performance_summary()
            
            self.logger.info("📊 当前系统状态:")
            self.logger.info(f"   💰 当前资金: {summary['current_capital']:.4f} USDT")
            self.logger.info(f"   📈 总交易数: {summary['total_trades']}")
            self.logger.info(f"   🎯 胜率: {summary['win_rate']*100:.1f}%")
            self.logger.info(f"   📊 盈利因子: {summary['profit_factor']:.2f}")
            self.logger.info(f"   📉 最大回撤: {summary['max_drawdown']*100:.1f}%")
            self.logger.info(f"   🔄 持仓数: {summary['open_positions']}")
            
        except Exception as e:
            self.logger.error(f"记录系统状态失败: {e}")
    
    async def _generate_final_report(self):
        """生成最终报告"""
        try:
            self.logger.info("="*80)
            self.logger.info("📋 30天模拟交易最终报告")
            self.logger.info("="*80)
            
            summary = self.simulation_manager.get_performance_summary()
            
            # 基础统计
            self.logger.info(f"📊 交易统计:")
            self.logger.info(f"   📈 总交易数: {summary['total_trades']}")
            self.logger.info(f"   ✅ 盈利交易: {summary['winning_trades']}")
            self.logger.info(f"   ❌ 亏损交易: {summary['losing_trades']}")
            self.logger.info(f"   🎯 胜率: {summary['win_rate']*100:.1f}%")
            
            # 盈利统计
            self.logger.info(f"💰 盈利统计:")
            self.logger.info(f"   💵 总盈亏: {summary['total_pnl']:+.4f} USDT")
            self.logger.info(f"   📊 收益率: {summary['return_percentage']:+.2f}%")
            self.logger.info(f"   💪 盈利因子: {summary['profit_factor']:.2f}")
            self.logger.info(f"   📈 平均交易盈亏: {summary['avg_trade_pnl']:+.4f} USDT")
            
            # 风险指标
            self.logger.info(f"🛡️ 风险指标:")
            self.logger.info(f"   📉 最大回撤: {summary['max_drawdown']*100:.1f}%")
            self.logger.info(f"   🔄 最大连续亏损: {summary['max_consecutive_losses']}")
            self.logger.info(f"   💼 最终资金: {summary['current_capital']:.4f} USDT")
            
            # 交易频率
            self.logger.info(f"⏰ 交易频率:")
            self.logger.info(f"   📅 交易天数: {summary['trading_days']}")
            self.logger.info(f"   📊 日均交易: {summary['trades_per_day']:.1f} 单/天")
            
            # 系统评估
            self.logger.info(f"🎯 系统评估: {summary['status']}")
            
            # 导出详细报告
            csv_file = await self.simulation_manager.export_trades_csv()
            if csv_file:
                self.logger.info(f"📄 详细交易记录已导出: {csv_file}")
            
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"生成最终报告失败: {e}")
    
    def stop(self):
        """停止模拟交易系统"""
        self.should_stop = True
        self.logger.info("🛑 正在停止模拟交易系统...")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'is_running': self.is_running,
            'should_stop': self.should_stop,
            'test_symbols': self.test_symbols,
            'check_interval': self.check_interval,
            'price_cache': self.price_cache,
            'performance': self.simulation_manager.get_performance_summary()
        } 