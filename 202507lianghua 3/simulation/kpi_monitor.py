"""
关键指标监控管理器
负责监控模拟交易系统的核心KPI指标，确保系统性能符合预期目标
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import os

from utils.logger import get_logger
from simulation.simulation_trading_manager import SimulationTradingManager

class AlertLevel(Enum):
    """警报级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class KPIStatus(Enum):
    """KPI状态"""
    EXCELLENT = "EXCELLENT"    # 超出目标
    GOOD = "GOOD"             # 达到目标
    WARNING = "WARNING"       # 接近目标
    CRITICAL = "CRITICAL"     # 未达目标

@dataclass
class KPITarget:
    """KPI目标设置"""
    # 交易频率目标
    min_trades_per_day: float = 5.0
    max_trades_per_day: float = 8.0
    
    # 胜率目标
    min_win_rate: float = 0.65          # 65%
    excellent_win_rate: float = 0.75    # 75%
    
    # 回撤控制
    max_drawdown: float = 0.20          # 20%
    warning_drawdown: float = 0.15      # 15%
    
    # 盈利因子目标
    min_profit_factor: float = 2.0
    excellent_profit_factor: float = 3.0
    
    # 信号质量
    min_signal_confidence: float = 0.6  # 60%
    min_signal_frequency: float = 0.3   # 30%信号转化率

@dataclass
class KPIMetrics:
    """KPI指标数据"""
    # 时间信息
    timestamp: datetime
    period_days: int
    
    # 交易频率指标
    total_trades: int
    trades_per_day: float
    
    # 盈利指标
    win_rate: float
    profit_factor: float
    total_pnl: float
    return_percentage: float
    
    # 风险指标
    max_drawdown: float
    current_drawdown: float
    consecutive_losses: int
    
    # 信号质量指标
    total_signals: int
    executed_signals: int
    signal_conversion_rate: float
    avg_signal_confidence: float
    
    # 资金状况
    current_capital: float
    peak_capital: float
    
    # 状态评估
    overall_status: KPIStatus
    detailed_status: Dict[str, KPIStatus]

@dataclass
class KPIAlert:
    """KPI警报"""
    timestamp: datetime
    level: AlertLevel
    metric: str
    current_value: float
    target_value: float
    message: str
    suggestion: str

class KPIMonitor:
    """关键指标监控管理器"""
    
    def __init__(self, simulation_manager: SimulationTradingManager):
        self.logger = get_logger(__name__)
        self.simulation_manager = simulation_manager
        
        # KPI配置
        self.targets = KPITarget()
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_start_time = None
        
        # 历史数据
        self.metrics_history: List[KPIMetrics] = []
        self.alerts_history: List[KPIAlert] = []
        
        # 文件路径
        self.data_dir = "simulation/monitoring"
        self.metrics_file = f"{self.data_dir}/kpi_metrics.json"
        self.alerts_file = f"{self.data_dir}/kpi_alerts.json"
        self.reports_dir = f"{self.data_dir}/reports"
        
        self._ensure_directories()
        self._load_historical_data()
        
        self.logger.info("📊 KPI监控管理器初始化完成")
        self.logger.info(f"   🎯 目标交易频率: {self.targets.min_trades_per_day}-{self.targets.max_trades_per_day} 单/天")
        self.logger.info(f"   📈 目标胜率: ≥{self.targets.min_win_rate*100:.0f}%")
        self.logger.info(f"   📉 最大回撤限制: ≤{self.targets.max_drawdown*100:.0f}%")
        self.logger.info(f"   💪 目标盈利因子: ≥{self.targets.min_profit_factor}")
    
    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def _load_historical_data(self):
        """加载历史数据"""
        try:
            # 加载指标历史
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                
                for data in metrics_data:
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    data['overall_status'] = KPIStatus(data['overall_status'])
                    data['detailed_status'] = {k: KPIStatus(v) for k, v in data['detailed_status'].items()}
                    self.metrics_history.append(KPIMetrics(**data))
                
                self.logger.info(f"   📚 加载历史指标数据: {len(self.metrics_history)} 条")
            
            # 加载警报历史
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r', encoding='utf-8') as f:
                    alerts_data = json.load(f)
                
                for data in alerts_data:
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    data['level'] = AlertLevel(data['level'])
                    self.alerts_history.append(KPIAlert(**data))
                
                self.logger.info(f"   🚨 加载历史警报数据: {len(self.alerts_history)} 条")
                
        except Exception as e:
            self.logger.warning(f"加载历史数据失败: {e}")
    
    async def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitoring_start_time = datetime.now()
        
        self.logger.info("🚀 开始KPI监控")
        self.logger.info("="*60)
        
        # 生成初始报告
        await self.generate_current_metrics()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        self.logger.info("🛑 KPI监控已停止")
    
    async def generate_current_metrics(self) -> KPIMetrics:
        """生成当前指标"""
        try:
            # 获取模拟交易性能数据
            performance = self.simulation_manager.get_performance_summary()
            
            # 计算运行天数
            if self.monitoring_start_time:
                period_days = max(1, (datetime.now() - self.monitoring_start_time).days)
            else:
                period_days = max(1, performance.get('trading_days', 1))
            
            # 计算信号质量指标
            signal_metrics = await self._calculate_signal_metrics()
            
            # 创建指标对象
            metrics = KPIMetrics(
                timestamp=datetime.now(),
                period_days=period_days,
                
                # 交易频率
                total_trades=performance['total_trades'],
                trades_per_day=performance.get('trades_per_day', performance['total_trades'] / period_days),
                
                # 盈利指标
                win_rate=performance['win_rate'],
                profit_factor=performance['profit_factor'],
                total_pnl=performance['total_pnl'],
                return_percentage=performance.get('return_percentage', 0),
                
                # 风险指标
                max_drawdown=performance['max_drawdown'],
                current_drawdown=performance['current_drawdown'],
                consecutive_losses=performance['current_consecutive_losses'],
                
                # 信号质量
                total_signals=signal_metrics['total_signals'],
                executed_signals=signal_metrics['executed_signals'],
                signal_conversion_rate=signal_metrics['conversion_rate'],
                avg_signal_confidence=signal_metrics['avg_confidence'],
                
                # 资金状况
                current_capital=performance['current_capital'],
                peak_capital=performance.get('peak_capital', performance['current_capital']),
                
                # 状态评估（稍后填充）
                overall_status=KPIStatus.GOOD,
                detailed_status={}
            )
            
            # 评估状态
            metrics.detailed_status = self._evaluate_detailed_status(metrics)
            metrics.overall_status = self._evaluate_overall_status(metrics.detailed_status)
            
            # 检查警报
            await self._check_alerts(metrics)
            
            # 保存到历史
            self.metrics_history.append(metrics)
            
            # 保存数据
            await self._save_data()
            
            # 记录日志
            self._log_metrics_summary(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"生成当前指标失败: {e}")
            return None
    
    async def _calculate_signal_metrics(self) -> Dict[str, Any]:
        """计算信号质量指标"""
        try:
            # 从交易记录中分析信号转化情况
            trades = self.simulation_manager.trades
            
            # 统计信号相关信息
            total_signals = len(trades) * 2  # 假设每笔交易对应2个信号（开仓+平仓）
            executed_signals = len(trades)
            
            if trades:
                confidences = [t.signal_confidence for t in trades if t.signal_confidence > 0]
                avg_confidence = statistics.mean(confidences) if confidences else 0.0
            else:
                avg_confidence = 0.0
            
            conversion_rate = executed_signals / total_signals if total_signals > 0 else 0.0
            
            return {
                'total_signals': total_signals,
                'executed_signals': executed_signals,
                'conversion_rate': conversion_rate,
                'avg_confidence': avg_confidence
            }
            
        except Exception as e:
            self.logger.error(f"计算信号指标失败: {e}")
            return {
                'total_signals': 0,
                'executed_signals': 0,
                'conversion_rate': 0.0,
                'avg_confidence': 0.0
            }
    
    def _evaluate_detailed_status(self, metrics: KPIMetrics) -> Dict[str, KPIStatus]:
        """评估详细状态"""
        status = {}
        
        # 交易频率状态
        if metrics.trades_per_day < self.targets.min_trades_per_day:
            status['trading_frequency'] = KPIStatus.CRITICAL
        elif metrics.trades_per_day > self.targets.max_trades_per_day:
            status['trading_frequency'] = KPIStatus.WARNING
        else:
            status['trading_frequency'] = KPIStatus.GOOD
        
        # 胜率状态
        if metrics.win_rate >= self.targets.excellent_win_rate:
            status['win_rate'] = KPIStatus.EXCELLENT
        elif metrics.win_rate >= self.targets.min_win_rate:
            status['win_rate'] = KPIStatus.GOOD
        elif metrics.win_rate >= self.targets.min_win_rate * 0.9:
            status['win_rate'] = KPIStatus.WARNING
        else:
            status['win_rate'] = KPIStatus.CRITICAL
        
        # 回撤状态
        if metrics.max_drawdown <= self.targets.warning_drawdown:
            status['drawdown'] = KPIStatus.EXCELLENT
        elif metrics.max_drawdown <= self.targets.max_drawdown:
            status['drawdown'] = KPIStatus.GOOD
        elif metrics.max_drawdown <= self.targets.max_drawdown * 1.1:
            status['drawdown'] = KPIStatus.WARNING
        else:
            status['drawdown'] = KPIStatus.CRITICAL
        
        # 盈利因子状态
        if metrics.profit_factor >= self.targets.excellent_profit_factor:
            status['profit_factor'] = KPIStatus.EXCELLENT
        elif metrics.profit_factor >= self.targets.min_profit_factor:
            status['profit_factor'] = KPIStatus.GOOD
        elif metrics.profit_factor >= self.targets.min_profit_factor * 0.8:
            status['profit_factor'] = KPIStatus.WARNING
        else:
            status['profit_factor'] = KPIStatus.CRITICAL
        
        # 信号质量状态
        if (metrics.avg_signal_confidence >= self.targets.min_signal_confidence and 
            metrics.signal_conversion_rate >= self.targets.min_signal_frequency):
            status['signal_quality'] = KPIStatus.GOOD
        elif (metrics.avg_signal_confidence >= self.targets.min_signal_confidence * 0.9 or
              metrics.signal_conversion_rate >= self.targets.min_signal_frequency * 0.8):
            status['signal_quality'] = KPIStatus.WARNING
        else:
            status['signal_quality'] = KPIStatus.CRITICAL
        
        return status
    
    def _evaluate_overall_status(self, detailed_status: Dict[str, KPIStatus]) -> KPIStatus:
        """评估整体状态"""
        status_counts = {}
        for status in detailed_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 如果有任何CRITICAL，整体为CRITICAL
        if status_counts.get(KPIStatus.CRITICAL, 0) > 0:
            return KPIStatus.CRITICAL
        
        # 如果有超过一半WARNING，整体为WARNING
        total_metrics = len(detailed_status)
        if status_counts.get(KPIStatus.WARNING, 0) > total_metrics / 2:
            return KPIStatus.WARNING
        
        # 如果有任何EXCELLENT且其他都是GOOD以上，整体为EXCELLENT
        if (status_counts.get(KPIStatus.EXCELLENT, 0) > 0 and 
            status_counts.get(KPIStatus.WARNING, 0) == 0 and
            status_counts.get(KPIStatus.CRITICAL, 0) == 0):
            return KPIStatus.EXCELLENT
        
        # 否则为GOOD
        return KPIStatus.GOOD
    
    async def _check_alerts(self, metrics: KPIMetrics):
        """检查警报条件"""
        alerts = []
        
        # 交易频率警报
        if metrics.trades_per_day < self.targets.min_trades_per_day:
            alerts.append(KPIAlert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                metric="trading_frequency",
                current_value=metrics.trades_per_day,
                target_value=self.targets.min_trades_per_day,
                message=f"交易频率过低: {metrics.trades_per_day:.1f} < {self.targets.min_trades_per_day}",
                suggestion="检查信号生成器参数，降低信号阈值或增加币种"
            ))
        elif metrics.trades_per_day > self.targets.max_trades_per_day:
            alerts.append(KPIAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                metric="trading_frequency",
                current_value=metrics.trades_per_day,
                target_value=self.targets.max_trades_per_day,
                message=f"交易频率过高: {metrics.trades_per_day:.1f} > {self.targets.max_trades_per_day}",
                suggestion="提高信号阈值或减少币种数量"
            ))
        
        # 胜率警报
        if metrics.win_rate < self.targets.min_win_rate:
            level = AlertLevel.CRITICAL if metrics.win_rate < self.targets.min_win_rate * 0.9 else AlertLevel.WARNING
            alerts.append(KPIAlert(
                timestamp=datetime.now(),
                level=level,
                metric="win_rate",
                current_value=metrics.win_rate,
                target_value=self.targets.min_win_rate,
                message=f"胜率未达目标: {metrics.win_rate*100:.1f}% < {self.targets.min_win_rate*100:.0f}%",
                suggestion="检查信号质量，优化止损止盈策略"
            ))
        
        # 回撤警报
        if metrics.max_drawdown > self.targets.max_drawdown:
            alerts.append(KPIAlert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                metric="max_drawdown",
                current_value=metrics.max_drawdown,
                target_value=self.targets.max_drawdown,
                message=f"最大回撤超限: {metrics.max_drawdown*100:.1f}% > {self.targets.max_drawdown*100:.0f}%",
                suggestion="降低杠杆倍数，减小仓位大小，优化风险控制"
            ))
        elif metrics.max_drawdown > self.targets.warning_drawdown:
            alerts.append(KPIAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                metric="max_drawdown",
                current_value=metrics.max_drawdown,
                target_value=self.targets.warning_drawdown,
                message=f"回撤接近警戒线: {metrics.max_drawdown*100:.1f}% > {self.targets.warning_drawdown*100:.0f}%",
                suggestion="密切关注风险，考虑调整交易策略"
            ))
        
        # 保存警报
        for alert in alerts:
            self.alerts_history.append(alert)
            self._log_alert(alert)
    
    def _log_metrics_summary(self, metrics: KPIMetrics):
        """记录指标摘要日志"""
        self.logger.info("📊 KPI指标当前状态:")
        self.logger.info(f"   📈 总体状态: {metrics.overall_status.value}")
        self.logger.info(f"   🔄 交易频率: {metrics.trades_per_day:.1f} 单/天 ({metrics.detailed_status['trading_frequency'].value})")
        self.logger.info(f"   🎯 胜率: {metrics.win_rate*100:.1f}% ({metrics.detailed_status['win_rate'].value})")
        self.logger.info(f"   📉 最大回撤: {metrics.max_drawdown*100:.1f}% ({metrics.detailed_status['drawdown'].value})")
        self.logger.info(f"   💪 盈利因子: {metrics.profit_factor:.2f} ({metrics.detailed_status['profit_factor'].value})")
        self.logger.info(f"   📡 信号质量: {metrics.avg_signal_confidence*100:.1f}% ({metrics.detailed_status['signal_quality'].value})")
        self.logger.info(f"   💰 当前资金: {metrics.current_capital:.4f} USDT")
    
    def _log_alert(self, alert: KPIAlert):
        """记录警报日志"""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️", 
            AlertLevel.CRITICAL: "🚨"
        }
        
        emoji = level_emoji.get(alert.level, "📢")
        
        if alert.level == AlertLevel.CRITICAL:
            self.logger.error(f"{emoji} {alert.level.value}: {alert.message}")
            self.logger.error(f"   💡 建议: {alert.suggestion}")
        elif alert.level == AlertLevel.WARNING:
            self.logger.warning(f"{emoji} {alert.level.value}: {alert.message}")
            self.logger.warning(f"   💡 建议: {alert.suggestion}")
        else:
            self.logger.info(f"{emoji} {alert.level.value}: {alert.message}")
    
    async def _save_data(self):
        """保存数据"""
        try:
            # 保存指标历史
            metrics_data = []
            for metrics in self.metrics_history:
                data = asdict(metrics)
                data['timestamp'] = metrics.timestamp.isoformat()
                data['overall_status'] = metrics.overall_status.value
                data['detailed_status'] = {k: v.value for k, v in metrics.detailed_status.items()}
                metrics_data.append(data)
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # 保存警报历史
            alerts_data = []
            for alert in self.alerts_history:
                data = asdict(alert)
                data['timestamp'] = alert.timestamp.isoformat()
                data['level'] = alert.level.value
                alerts_data.append(data)
            
            with open(self.alerts_file, 'w', encoding='utf-8') as f:
                json.dump(alerts_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存监控数据失败: {e}")
    
    def get_recent_metrics(self, days: int = 7) -> List[KPIMetrics]:
        """获取最近几天的指标"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_date]
    
    def get_recent_alerts(self, days: int = 7) -> List[KPIAlert]:
        """获取最近几天的警报"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [a for a in self.alerts_history if a.timestamp >= cutoff_date]
    
    async def generate_monitoring_report(self) -> str:
        """生成监控报告"""
        try:
            recent_metrics = self.get_recent_metrics(7)
            recent_alerts = self.get_recent_alerts(7)
            
            if not recent_metrics:
                return "暂无监控数据"
            
            latest = recent_metrics[-1]
            
            report = f"""
# 📊 KPI监控报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 当前状态总览
- **整体状态**: {latest.overall_status.value}
- **监控周期**: {latest.period_days} 天
- **总交易数**: {latest.total_trades}
- **当前资金**: {latest.current_capital:.4f} USDT

## 📈 关键指标详情

### 交易频率 ({latest.detailed_status['trading_frequency'].value})
- 当前: {latest.trades_per_day:.1f} 单/天
- 目标: {self.targets.min_trades_per_day}-{self.targets.max_trades_per_day} 单/天

### 胜率指标 ({latest.detailed_status['win_rate'].value})
- 当前: {latest.win_rate*100:.1f}%
- 目标: ≥{self.targets.min_win_rate*100:.0f}%

### 风险控制 ({latest.detailed_status['drawdown'].value})
- 最大回撤: {latest.max_drawdown*100:.1f}%
- 当前回撤: {latest.current_drawdown*100:.1f}%
- 限制: ≤{self.targets.max_drawdown*100:.0f}%

### 盈利能力 ({latest.detailed_status['profit_factor'].value})
- 盈利因子: {latest.profit_factor:.2f}
- 总盈亏: {latest.total_pnl:+.4f} USDT
- 收益率: {latest.return_percentage:+.2f}%

### 信号质量 ({latest.detailed_status['signal_quality'].value})
- 平均置信度: {latest.avg_signal_confidence*100:.1f}%
- 信号转化率: {latest.signal_conversion_rate*100:.1f}%

## 🚨 最近警报 ({len(recent_alerts)} 条)
"""
            
            if recent_alerts:
                for alert in recent_alerts[-5:]:  # 显示最近5条
                    report += f"- {alert.level.value}: {alert.message}\n"
            else:
                report += "- 无警报\n"
            
            report += f"""
## 📊 趋势分析
"""
            
            if len(recent_metrics) >= 2:
                first = recent_metrics[0]
                trend_win_rate = latest.win_rate - first.win_rate
                trend_drawdown = latest.max_drawdown - first.max_drawdown
                trend_frequency = latest.trades_per_day - first.trades_per_day
                
                report += f"- 胜率趋势: {trend_win_rate*100:+.1f}%\n"
                report += f"- 回撤趋势: {trend_drawdown*100:+.1f}%\n"
                report += f"- 频率趋势: {trend_frequency:+.1f} 单/天\n"
            else:
                report += "- 数据不足，无法分析趋势\n"
            
            # 保存报告
            report_file = f"{self.reports_dir}/kpi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"📄 监控报告已生成: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成监控报告失败: {e}")
            return f"报告生成失败: {e}" 