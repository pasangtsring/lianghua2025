"""
30天模拟交易管理器
负责模拟交易的完整生命周期管理，包括资金管理、交易记录、性能统计等
"""

import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os

from utils.logger import get_logger
from config.config_manager import ConfigManager

class TradeType(Enum):
    """交易类型"""
    BUY = "BUY"
    SELL = "SELL"

class TradeStatus(Enum):
    """交易状态"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class SimulatedTrade:
    """模拟交易记录"""
    trade_id: str
    symbol: str
    trade_type: TradeType
    entry_price: float
    quantity: float
    leverage: float
    entry_time: datetime
    
    # 平仓信息
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    
    # 止损止盈
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # 交易结果
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    commission: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    
    # 信号信息
    signal_confidence: float = 0.0
    signal_strength: str = ""
    entry_reasons: List[str] = None
    
    def __post_init__(self):
        if self.entry_reasons is None:
            self.entry_reasons = []

@dataclass
class SimulationConfig:
    """模拟交易配置"""
    initial_capital: float = 65.0
    max_leverage: float = 20.0
    target_trades_per_day: int = 8
    test_symbols: List[str] = None
    commission_rate: float = 0.0004  # 0.04%
    
    # 风险控制
    max_position_size_pct: float = 0.15  # 单笔最大仓位15%
    max_daily_loss_pct: float = 0.10     # 日最大亏损10%
    max_drawdown_pct: float = 0.20       # 最大回撤20%
    
    # 目标指标
    target_win_rate: float = 0.65        # 目标胜率65%
    target_profit_factor: float = 2.0    # 目标盈利因子2.0
    target_sharpe_ratio: float = 1.5     # 目标夏普比率1.5
    
    def __post_init__(self):
        if self.test_symbols is None:
            self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']

class SimulationTradingManager:
    """30天模拟交易管理器"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.logger = get_logger(__name__)
        self.config_manager = config_manager or ConfigManager()
        
        # 模拟交易配置
        self.sim_config = SimulationConfig()
        
        # 账户状态
        self.current_capital = self.sim_config.initial_capital
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak_capital = self.sim_config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # 交易记录
        self.trades: List[SimulatedTrade] = []
        self.open_positions: Dict[str, SimulatedTrade] = {}
        self.daily_stats: List[Dict] = []
        
        # 统计指标
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.sharpe_ratio = 0.0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        
        # 文件路径
        self.data_dir = "simulation/data"
        self.trades_file = f"{self.data_dir}/trades_history.json"
        self.stats_file = f"{self.data_dir}/daily_stats.json"
        self.report_file = f"{self.data_dir}/simulation_report.json"
        
        self._ensure_data_directory()
        self._load_existing_data()
        
        self.logger.info("🎯 30天模拟交易管理器初始化完成")
        self.logger.info(f"   💰 初始资金: {self.sim_config.initial_capital} USDT")
        self.logger.info(f"   🎯 目标胜率: {self.sim_config.target_win_rate*100:.1f}%")
        self.logger.info(f"   📊 目标盈利因子: {self.sim_config.target_profit_factor}")
        self.logger.info(f"   📈 目标交易频率: {self.sim_config.target_trades_per_day} 单/天")
    
    def _ensure_data_directory(self):
        """确保数据目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_existing_data(self):
        """加载已有的交易数据（用于恢复状态）"""
        try:
            # 加载交易历史
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r', encoding='utf-8') as f:
                    trades_data = json.load(f)
                
                for trade_data in trades_data:
                    trade = SimulatedTrade(**trade_data)
                    # 转换时间字符串回datetime对象
                    trade.entry_time = datetime.fromisoformat(trade_data['entry_time'])
                    if trade_data.get('exit_time'):
                        trade.exit_time = datetime.fromisoformat(trade_data['exit_time'])
                    trade.trade_type = TradeType(trade_data['trade_type'])
                    trade.status = TradeStatus(trade_data['status'])
                    
                    self.trades.append(trade)
                
                self.logger.info(f"   📚 加载历史交易记录: {len(self.trades)} 笔")
            
            # 加载日统计
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.daily_stats = json.load(f)
                self.logger.info(f"   📊 加载日统计数据: {len(self.daily_stats)} 天")
            
            # 重新计算当前状态
            self._recalculate_stats()
            
        except Exception as e:
            self.logger.warning(f"加载历史数据失败: {e}")
    
    def _recalculate_stats(self):
        """重新计算所有统计指标"""
        try:
            if not self.trades:
                return
            
            # 计算基础统计
            closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
            self.total_trades = len(closed_trades)
            
            if self.total_trades == 0:
                return
            
            # 计算盈亏统计
            total_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
            total_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
            self.total_pnl = sum(t.pnl for t in closed_trades)
            
            # 计算胜率
            self.winning_trades = len([t for t in closed_trades if t.pnl > 0])
            self.losing_trades = len([t for t in closed_trades if t.pnl < 0])
            self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            # 计算盈利因子
            self.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # 更新资金状态
            self.current_capital = self.sim_config.initial_capital + self.total_pnl
            self.peak_capital = max(self.peak_capital, self.current_capital)
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # 计算连续亏损
            consecutive_losses = 0
            max_consecutive = 0
            for trade in reversed(closed_trades):
                if trade.pnl < 0:
                    consecutive_losses += 1
                    max_consecutive = max(max_consecutive, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            self.current_consecutive_losses = consecutive_losses
            self.max_consecutive_losses = max_consecutive
            
        except Exception as e:
            self.logger.error(f"重新计算统计指标失败: {e}")
    
    async def open_position(self, symbol: str, trade_type: TradeType, 
                           entry_price: float, quantity: float, leverage: float,
                           stop_loss: float = None, take_profit: float = None,
                           signal_info: Dict = None) -> bool:
        """开仓"""
        try:
            # 生成交易ID
            trade_id = f"{symbol}_{trade_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 检查是否已有该币种的持仓
            if symbol in self.open_positions:
                self.logger.warning(f"   ⚠️ {symbol} 已有持仓，跳过开仓")
                return False
            
            # 风险检查
            position_value = quantity * entry_price * leverage
            position_size_pct = position_value / self.current_capital
            
            if position_size_pct > self.sim_config.max_position_size_pct:
                self.logger.warning(f"   ⚠️ {symbol} 仓位过大 ({position_size_pct*100:.1f}% > {self.sim_config.max_position_size_pct*100:.1f}%)")
                return False
            
            # 创建交易记录
            trade = SimulatedTrade(
                trade_id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                entry_price=entry_price,
                quantity=quantity,
                leverage=leverage,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_confidence=signal_info.get('confidence', 0.0) if signal_info else 0.0,
                signal_strength=signal_info.get('strength', '') if signal_info else '',
                entry_reasons=signal_info.get('reasons', []) if signal_info else []
            )
            
            # 计算手续费
            trade.commission = position_value * self.sim_config.commission_rate
            
            # 添加到持仓
            self.open_positions[symbol] = trade
            self.trades.append(trade)
            
            self.logger.info(f"   📈 开仓成功: {symbol} {trade_type.value}")
            self.logger.info(f"      💰 价格: {entry_price:.6f}")
            self.logger.info(f"      📏 数量: {quantity:.6f}")
            self.logger.info(f"      ⚡ 杠杆: {leverage}x")
            self.logger.info(f"      💼 仓位占比: {position_size_pct*100:.1f}%")
            
            await self._save_data()
            return True
            
        except Exception as e:
            self.logger.error(f"开仓失败: {e}")
            return False
    
    async def close_position(self, symbol: str, exit_price: float, 
                           exit_reason: str = "manual") -> bool:
        """平仓"""
        try:
            if symbol not in self.open_positions:
                self.logger.warning(f"   ⚠️ {symbol} 无持仓可平")
                return False
            
            trade = self.open_positions[symbol]
            
            # 计算盈亏
            if trade.trade_type == TradeType.BUY:
                pnl_per_unit = exit_price - trade.entry_price
            else:  # SELL (做空)
                pnl_per_unit = trade.entry_price - exit_price
            
            trade.pnl = pnl_per_unit * trade.quantity * trade.leverage - trade.commission
            trade.pnl_percentage = (pnl_per_unit / trade.entry_price) * trade.leverage * 100
            
            # 更新交易记录
            trade.exit_price = exit_price
            trade.exit_time = datetime.now()
            trade.exit_reason = exit_reason
            trade.status = TradeStatus.CLOSED
            
            # 从持仓中移除
            del self.open_positions[symbol]
            
            # 更新资金
            self.current_capital += trade.pnl
            self.total_pnl += trade.pnl
            
            # 更新连续亏损
            if trade.pnl < 0:
                self.current_consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.current_consecutive_losses)
            else:
                self.current_consecutive_losses = 0
            
            # 更新统计
            self.total_trades += 1
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.win_rate = self.winning_trades / self.total_trades
            
            # 更新最大回撤
            self.peak_capital = max(self.peak_capital, self.current_capital)
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            self.logger.info(f"   📉 平仓成功: {symbol}")
            self.logger.info(f"      💰 平仓价格: {exit_price:.6f}")
            self.logger.info(f"      📊 盈亏: {trade.pnl:+.4f} USDT ({trade.pnl_percentage:+.2f}%)")
            self.logger.info(f"      🎯 平仓原因: {exit_reason}")
            self.logger.info(f"      💼 当前资金: {self.current_capital:.4f} USDT")
            
            await self._save_data()
            return True
            
        except Exception as e:
            self.logger.error(f"平仓失败: {e}")
            return False
    
    async def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> bool:
        """检查止损止盈"""
        try:
            if symbol not in self.open_positions:
                return False
            
            trade = self.open_positions[symbol]
            should_close = False
            exit_reason = ""
            
            if trade.trade_type == TradeType.BUY:
                # 多头：价格下跌到止损或上涨到止盈
                if trade.stop_loss and current_price <= trade.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
                elif trade.take_profit and current_price >= trade.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
            else:  # SELL
                # 空头：价格上涨到止损或下跌到止盈
                if trade.stop_loss and current_price >= trade.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
                elif trade.take_profit and current_price <= trade.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                await self.close_position(symbol, current_price, exit_reason)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"检查止损止盈失败: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
            
            if not closed_trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_pnl": 0.0,
                    "current_capital": self.current_capital,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "avg_trade_pnl": 0.0,
                    "trading_days": 0,
                    "trades_per_day": 0.0,
                    "status": "STARTING"
                }
            
            # 计算统计指标
            total_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
            total_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
            avg_trade_pnl = sum(t.pnl for t in closed_trades) / len(closed_trades)
            
            # 计算交易天数
            first_trade = min(closed_trades, key=lambda x: x.entry_time)
            last_trade = max(closed_trades, key=lambda x: x.entry_time)
            trading_days = (last_trade.entry_time - first_trade.entry_time).days + 1
            trades_per_day = len(closed_trades) / trading_days if trading_days > 0 else 0
            
            # 评估状态
            status = "EXCELLENT"
            if self.win_rate < self.sim_config.target_win_rate:
                status = "NEEDS_IMPROVEMENT"
            elif self.max_drawdown > self.sim_config.max_drawdown_pct:
                status = "HIGH_RISK"
            elif self.profit_factor < self.sim_config.target_profit_factor:
                status = "UNDERPERFORMING"
            elif trades_per_day < self.sim_config.target_trades_per_day * 0.8:
                status = "LOW_FREQUENCY"
            
            return {
                "total_trades": len(closed_trades),
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "total_pnl": self.total_pnl,
                "current_capital": self.current_capital,
                "initial_capital": self.sim_config.initial_capital,
                "return_percentage": (self.current_capital / self.sim_config.initial_capital - 1) * 100,
                "max_drawdown": self.max_drawdown,
                "current_drawdown": self.current_drawdown,
                "avg_trade_pnl": avg_trade_pnl,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "trading_days": trading_days,
                "trades_per_day": trades_per_day,
                "max_consecutive_losses": self.max_consecutive_losses,
                "current_consecutive_losses": self.current_consecutive_losses,
                "open_positions": len(self.open_positions),
                "status": status,
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取性能摘要失败: {e}")
            return {"error": str(e)}
    
    async def generate_daily_report(self) -> Dict[str, Any]:
        """生成日报告"""
        try:
            today = datetime.now().date()
            today_trades = [t for t in self.trades 
                          if t.entry_time.date() == today and t.status == TradeStatus.CLOSED]
            
            # 计算今日统计
            daily_pnl = sum(t.pnl for t in today_trades)
            daily_trades_count = len(today_trades)
            daily_wins = len([t for t in today_trades if t.pnl > 0])
            daily_win_rate = daily_wins / daily_trades_count if daily_trades_count > 0 else 0
            
            # 保存日统计
            daily_stat = {
                "date": today.isoformat(),
                "trades_count": daily_trades_count,
                "pnl": daily_pnl,
                "win_rate": daily_win_rate,
                "capital": self.current_capital,
                "drawdown": self.current_drawdown,
                "open_positions": len(self.open_positions)
            }
            
            # 检查是否已存在今日记录
            existing_index = -1
            for i, stat in enumerate(self.daily_stats):
                if stat['date'] == today.isoformat():
                    existing_index = i
                    break
            
            if existing_index >= 0:
                self.daily_stats[existing_index] = daily_stat
            else:
                self.daily_stats.append(daily_stat)
            
            await self._save_data()
            
            self.logger.info(f"📊 今日交易报告:")
            self.logger.info(f"   📈 交易次数: {daily_trades_count}")
            self.logger.info(f"   💰 今日盈亏: {daily_pnl:+.4f} USDT")
            self.logger.info(f"   🎯 今日胜率: {daily_win_rate*100:.1f}%")
            self.logger.info(f"   💼 当前资金: {self.current_capital:.4f} USDT")
            
            return daily_stat
            
        except Exception as e:
            self.logger.error(f"生成日报告失败: {e}")
            return {}
    
    async def _save_data(self):
        """保存数据到文件"""
        try:
            # 保存交易历史
            trades_data = []
            for trade in self.trades:
                trade_dict = asdict(trade)
                # 转换datetime对象为字符串
                trade_dict['entry_time'] = trade.entry_time.isoformat()
                if trade.exit_time:
                    trade_dict['exit_time'] = trade.exit_time.isoformat()
                # 转换Enum为字符串
                trade_dict['trade_type'] = trade.trade_type.value
                trade_dict['status'] = trade.status.value
                trades_data.append(trade_dict)
            
            with open(self.trades_file, 'w', encoding='utf-8') as f:
                json.dump(trades_data, f, indent=2, ensure_ascii=False)
            
            # 保存日统计
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.daily_stats, f, indent=2, ensure_ascii=False)
            
            # 保存性能报告
            summary = self.get_performance_summary()
            with open(self.report_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
    
    async def export_trades_csv(self, filename: str = None) -> str:
        """导出交易记录为CSV"""
        try:
            if not filename:
                filename = f"{self.data_dir}/trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            trades_data = []
            for trade in self.trades:
                if trade.status == TradeStatus.CLOSED:
                    trades_data.append({
                        'Trade ID': trade.trade_id,
                        'Symbol': trade.symbol,
                        'Type': trade.trade_type.value,
                        'Entry Price': trade.entry_price,
                        'Exit Price': trade.exit_price,
                        'Quantity': trade.quantity,
                        'Leverage': trade.leverage,
                        'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'Exit Time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'PnL': trade.pnl,
                        'PnL %': trade.pnl_percentage,
                        'Commission': trade.commission,
                        'Exit Reason': trade.exit_reason,
                        'Signal Confidence': trade.signal_confidence,
                        'Signal Strength': trade.signal_strength
                    })
            
            df = pd.DataFrame(trades_data)
            df.to_csv(filename, index=False, encoding='utf-8')
            
            self.logger.info(f"交易记录已导出: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"导出交易记录失败: {e}")
            return "" 