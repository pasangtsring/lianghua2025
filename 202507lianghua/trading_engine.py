"""
主交易引擎
整合所有模块，实现主交易逻辑
"""

import asyncio
import signal
import sys
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading

from config.config_manager import ConfigManager
from utils.logger import Logger
from data.advanced_data_fetcher import AdvancedDataFetcher
from core.technical_indicators import TechnicalIndicators
from core.complete_macd_divergence_detector import CompleteMACDDivergenceDetector
from risk.risk_manager import RiskManager
from utils.telegram_bot import TelegramBot

class TradingEngine:
    """主交易引擎"""
    
    def __init__(self):
        """初始化交易引擎"""
        # 初始化配置
        self.config = ConfigManager()
        self.logger = Logger(__name__)
        
        # 初始化核心组件
        self.data_fetcher = AdvancedDataFetcher(self.config)
        self.technical_indicators = TechnicalIndicators(self.config)
        self.macd_detector = CompleteMACDDivergenceDetector(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # 初始化通知组件
        self.telegram_bot = TelegramBot(self.config)
        
        # 运行状态
        self.is_running = False
        self.should_stop = False
        
        # 交易配置
        self.trading_config = self.config.get_trading_config()
        self.symbol = self.trading_config.get('symbol', 'BTCUSDT')
        self.interval = self.trading_config.get('interval', '1h')
        
        # 数据存储
        self.kline_data = []
        self.current_positions = {}
        self.trade_history = []
        
        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("交易引擎初始化完成")
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，准备关闭交易引擎")
        self.should_stop = True
    
    async def start(self):
        """启动交易引擎"""
        try:
            self.logger.info("启动交易引擎...")
            self.is_running = True
            
            # 发送启动通知
            await self.telegram_bot.send_message("🚀 交易引擎已启动")
            
            # 启动主循环
            await self.main_loop()
            
        except Exception as e:
            self.logger.error(f"启动交易引擎失败: {e}")
            await self.telegram_bot.send_message(f"❌ 交易引擎启动失败: {e}")
            raise
    
    async def stop(self):
        """停止交易引擎"""
        try:
            self.logger.info("停止交易引擎...")
            self.should_stop = True
            self.is_running = False
            
            # 关闭所有仓位
            await self.close_all_positions()
            
            # 发送停止通知
            await self.telegram_bot.send_message("🛑 交易引擎已停止")
            
        except Exception as e:
            self.logger.error(f"停止交易引擎失败: {e}")
    
    async def main_loop(self):
        """主循环"""
        while not self.should_stop:
            try:
                # 获取最新数据
                await self.fetch_latest_data()
                
                # 计算技术指标
                await self.calculate_indicators()
                
                # 检测交易信号
                await self.detect_signals()
                
                # 执行交易
                await self.execute_trades()
                
                # 更新风险管理
                await self.update_risk_management()
                
                # 生成报告
                await self.generate_reports()
                
                # 等待下一个周期
                await asyncio.sleep(60)  # 1分钟检查一次
                
            except Exception as e:
                self.logger.error(f"主循环执行失败: {e}")
                await self.telegram_bot.send_message(f"⚠️ 主循环错误: {e}")
                await asyncio.sleep(60)
    
    async def fetch_latest_data(self):
        """获取最新数据"""
        try:
            # 获取K线数据
            klines = await self.data_fetcher.get_klines(self.symbol, self.interval, limit=200)
            
            if klines:
                self.kline_data = klines
                self.logger.debug(f"获取K线数据: {len(klines)} 条")
            
            # 获取当前仓位
            positions = await self.data_fetcher.get_positions()
            if positions:
                self.current_positions = positions
            
        except Exception as e:
            self.logger.error(f"获取最新数据失败: {e}")
    
    async def calculate_indicators(self):
        """计算技术指标"""
        try:
            if len(self.kline_data) < 50:
                return
            
            # 计算MACD
            macd_data = self.technical_indicators.calculate_macd(self.kline_data)
            
            # 计算其他指标
            sma_data = self.technical_indicators.calculate_sma(self.kline_data, 20)
            rsi_data = self.technical_indicators.calculate_rsi(self.kline_data, 14)
            
            self.logger.debug("技术指标计算完成")
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
    
    async def detect_signals(self):
        """检测交易信号"""
        try:
            if len(self.kline_data) < 100:
                return
            
            # 检测MACD背离
            divergence_result = self.macd_detector.detect_divergence(self.kline_data)
            
            if divergence_result['has_divergence']:
                await self.process_divergence_signal(divergence_result)
            
        except Exception as e:
            self.logger.error(f"检测交易信号失败: {e}")
    
    async def process_divergence_signal(self, divergence_result):
        """处理背离信号"""
        try:
            signal_type = divergence_result['signal_type']
            confidence = divergence_result['confidence']
            
            self.logger.info(f"检测到背离信号: {signal_type}, 置信度: {confidence:.2f}")
            
            # 发送信号通知
            message = f"🔍 检测到{signal_type}背离信号\n"
            message += f"置信度: {confidence:.2f}\n"
            message += f"品种: {self.symbol}\n"
            message += f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            await self.telegram_bot.send_message(message)
            
            # 如果置信度足够高，考虑交易
            if confidence > 0.7:
                await self.consider_trade(signal_type, confidence)
            
        except Exception as e:
            self.logger.error(f"处理背离信号失败: {e}")
    
    async def consider_trade(self, signal_type, confidence):
        """考虑交易"""
        try:
            current_price = float(self.kline_data[-1]['close'])
            
            # 确定交易方向
            if signal_type == 'bearish_divergence':
                side = 'SELL'
                stop_loss_price = current_price * 1.02  # 2% 止损
                take_profit_price = current_price * 0.96  # 4% 止盈
            else:  # bullish_divergence
                side = 'BUY'
                stop_loss_price = current_price * 0.98  # 2% 止损
                take_profit_price = current_price * 1.04  # 4% 止盈
            
            # 风险检查
            quantity = self.calculate_position_size(current_price, stop_loss_price)
            
            risk_check, risk_msg = self.risk_manager.check_pre_trade_risk(
                self.symbol, side, quantity, current_price
            )
            
            if risk_check:
                await self.place_order(side, quantity, current_price, stop_loss_price, take_profit_price)
            else:
                self.logger.warning(f"风险检查不通过: {risk_msg}")
                await self.telegram_bot.send_message(f"⚠️ 风险检查不通过: {risk_msg}")
            
        except Exception as e:
            self.logger.error(f"考虑交易失败: {e}")
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """计算仓位大小"""
        try:
            # 简化版本，实际应该使用风险管理模块
            risk_per_trade = self.config.get_risk_config().get('risk_per_trade', 0.02)
            initial_capital = self.config.get_trading_config().get('initial_capital', 10000)
            
            risk_amount = initial_capital * risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff > 0:
                return risk_amount / price_diff
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0
    
    async def place_order(self, side, quantity, price, stop_loss, take_profit):
        """下单"""
        try:
            # 这里应该调用交易所API下单
            # 现在只是记录日志
            self.logger.info(f"下单: {side} {quantity} @ {price}")
            self.logger.info(f"止损: {stop_loss}, 止盈: {take_profit}")
            
            # 记录交易历史
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'pending'
            }
            
            self.trade_history.append(trade_record)
            
            # 发送交易通知
            message = f"📊 交易信号\n"
            message += f"品种: {self.symbol}\n"
            message += f"方向: {side}\n"
            message += f"数量: {quantity:.4f}\n"
            message += f"价格: {price:.2f}\n"
            message += f"止损: {stop_loss:.2f}\n"
            message += f"止盈: {take_profit:.2f}"
            
            await self.telegram_bot.send_message(message)
            
        except Exception as e:
            self.logger.error(f"下单失败: {e}")
    
    async def execute_trades(self):
        """执行交易"""
        try:
            # 检查挂单状态
            # 检查止损止盈
            # 更新仓位
            pass
            
        except Exception as e:
            self.logger.error(f"执行交易失败: {e}")
    
    async def update_risk_management(self):
        """更新风险管理"""
        try:
            # 更新仓位风险
            self.risk_manager.update_position_risk(self.current_positions)
            
            # 获取风险指标
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # 检查风险限制
            if risk_metrics.current_drawdown > 0.1:  # 10%回撤警告
                await self.telegram_bot.send_message(f"⚠️ 当前回撤: {risk_metrics.current_drawdown:.2%}")
            
        except Exception as e:
            self.logger.error(f"更新风险管理失败: {e}")
    
    async def generate_reports(self):
        """生成报告"""
        try:
            # 每小时生成一次报告
            if datetime.now().minute == 0:
                report = self.generate_hourly_report()
                if report:
                    await self.telegram_bot.send_message(report)
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
    
    def generate_hourly_report(self):
        """生成小时报告"""
        try:
            if not self.kline_data:
                return None
            
            current_price = float(self.kline_data[-1]['close'])
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            report = f"📈 小时报告\n"
            report += f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            report += f"品种: {self.symbol}\n"
            report += f"当前价格: {current_price:.2f}\n"
            report += f"当前回撤: {risk_metrics.current_drawdown:.2%}\n"
            report += f"今日盈亏: {risk_metrics.daily_pnl:.2f}\n"
            report += f"总盈亏: {risk_metrics.total_pnl:.2f}\n"
            report += f"仓位数量: {len(self.current_positions)}"
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成小时报告失败: {e}")
            return None
    
    async def close_all_positions(self):
        """关闭所有仓位"""
        try:
            if self.current_positions:
                self.logger.info("关闭所有仓位...")
                # 这里应该调用交易所API关闭仓位
                # 现在只是清空本地记录
                self.current_positions.clear()
                
        except Exception as e:
            self.logger.error(f"关闭所有仓位失败: {e}")
    
    def get_status(self):
        """获取引擎状态"""
        return {
            'is_running': self.is_running,
            'symbol': self.symbol,
            'interval': self.interval,
            'positions': len(self.current_positions),
            'trades_today': len([t for t in self.trade_history if t['timestamp'].date() == datetime.now().date()]),
            'last_update': datetime.now().isoformat()
        } 