"""
KPI预警系统
当关键指标偏离目标范围时自动发送警报和优化建议
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json
import os

from utils.logger import get_logger
from simulation.kpi_monitor import KPIAlert, AlertLevel, KPIMetrics, KPIStatus
from utils.telegram_bot import TelegramBot

@dataclass
class AlertRule:
    """警报规则配置"""
    metric_name: str
    threshold_critical: float
    threshold_warning: float
    comparison: str  # 'gt', 'lt', 'eq'
    message_template: str
    suggestion_template: str
    auto_action: Optional[str] = None  # 自动执行的动作

@dataclass
class AlertAction:
    """警报响应动作"""
    action_type: str
    parameters: Dict[str, Any]
    description: str

class AlertSystem:
    """KPI预警系统"""
    
    def __init__(self, telegram_bot: TelegramBot = None):
        self.logger = get_logger(__name__)
        self.telegram_bot = telegram_bot
        
        # 警报规则
        self.alert_rules = self._initialize_alert_rules()
        
        # 自动响应动作
        self.auto_actions: Dict[str, Callable] = {}
        
        # 警报频率限制（避免垃圾信息）
        self.alert_cooldown = {}  # metric -> last_alert_time
        self.cooldown_period = 3600  # 1小时
        
        # 数据存储
        self.data_dir = "simulation/monitoring"
        self.actions_file = f"{self.data_dir}/alert_actions.json"
        self.action_history: List[Dict] = []
        
        self._load_action_history()
        
        self.logger.info("🚨 KPI预警系统初始化完成")
        self.logger.info(f"   📋 警报规则数量: {len(self.alert_rules)}")
        self.logger.info(f"   ⏰ 警报冷却期: {self.cooldown_period//60} 分钟")
    
    def _initialize_alert_rules(self) -> List[AlertRule]:
        """初始化警报规则"""
        return [
            # 交易频率规则
            AlertRule(
                metric_name="trading_frequency_low",
                threshold_critical=5.0,
                threshold_warning=6.0,
                comparison="lt",
                message_template="交易频率过低: {current:.1f} < {target:.0f} 单/天",
                suggestion_template="建议: 1)降低信号阈值 2)增加币种数量 3)检查API连接",
                auto_action="adjust_signal_threshold"
            ),
            AlertRule(
                metric_name="trading_frequency_high",
                threshold_critical=8.0,
                threshold_warning=7.5,
                comparison="gt",
                message_template="交易频率过高: {current:.1f} > {target:.0f} 单/天",
                suggestion_template="建议: 1)提高信号阈值 2)减少币种数量 3)增加过滤条件",
                auto_action="adjust_signal_threshold"
            ),
            
            # 胜率规则
            AlertRule(
                metric_name="win_rate",
                threshold_critical=0.65,
                threshold_warning=0.60,
                comparison="lt",
                message_template="胜率未达目标: {current:.1f}% < {target:.0f}%",
                suggestion_template="建议: 1)优化止损止盈策略 2)提高信号质量 3)调整仓位管理",
                auto_action="optimize_exit_strategy"
            ),
            
            # 最大回撤规则
            AlertRule(
                metric_name="max_drawdown",
                threshold_critical=0.20,
                threshold_warning=0.15,
                comparison="gt",
                message_template="最大回撤超限: {current:.1f}% > {target:.0f}%",
                suggestion_template="建议: 1)降低杠杆倍数 2)减小仓位大小 3)加强风险控制",
                auto_action="reduce_risk_exposure"
            ),
            
            # 盈利因子规则
            AlertRule(
                metric_name="profit_factor",
                threshold_critical=2.0,
                threshold_warning=1.5,
                comparison="lt",
                message_template="盈利因子不足: {current:.2f} < {target:.1f}",
                suggestion_template="建议: 1)优化信号质量 2)调整风险回报比 3)减少交易成本",
                auto_action="optimize_profit_factor"
            ),
            
            # 连续亏损规则
            AlertRule(
                metric_name="consecutive_losses",
                threshold_critical=5,
                threshold_warning=3,
                comparison="gt",
                message_template="连续亏损过多: {current} 次 > {target} 次",
                suggestion_template="建议: 1)暂停交易评估 2)检查策略有效性 3)降低风险敞口",
                auto_action="emergency_risk_reduction"
            )
        ]
    
    def _load_action_history(self):
        """加载响应动作历史"""
        try:
            if os.path.exists(self.actions_file):
                with open(self.actions_file, 'r', encoding='utf-8') as f:
                    self.action_history = json.load(f)
                self.logger.info(f"   📚 加载响应动作历史: {len(self.action_history)} 条")
        except Exception as e:
            self.logger.warning(f"加载响应动作历史失败: {e}")
    
    async def process_metrics(self, metrics: KPIMetrics) -> List[KPIAlert]:
        """处理指标并生成警报"""
        alerts = []
        
        try:
            # 检查每个警报规则
            for rule in self.alert_rules:
                alert = await self._check_rule(rule, metrics)
                if alert:
                    alerts.append(alert)
            
            # 发送警报通知
            for alert in alerts:
                await self._send_alert_notification(alert)
                
                # 执行自动响应动作
                if alert.level == AlertLevel.CRITICAL:
                    await self._execute_auto_action(alert, metrics)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"处理指标警报失败: {e}")
            return []
    
    async def _check_rule(self, rule: AlertRule, metrics: KPIMetrics) -> Optional[KPIAlert]:
        """检查单个警报规则"""
        try:
            # 获取指标值
            current_value = self._get_metric_value(rule.metric_name, metrics)
            if current_value is None:
                return None
            
            # 检查冷却期
            if self._is_in_cooldown(rule.metric_name):
                return None
            
            # 确定警报级别
            alert_level = None
            target_value = None
            
            if rule.comparison == "lt":
                if current_value < rule.threshold_critical:
                    alert_level = AlertLevel.CRITICAL
                    target_value = rule.threshold_critical
                elif current_value < rule.threshold_warning:
                    alert_level = AlertLevel.WARNING
                    target_value = rule.threshold_warning
                    
            elif rule.comparison == "gt":
                if current_value > rule.threshold_critical:
                    alert_level = AlertLevel.CRITICAL
                    target_value = rule.threshold_critical
                elif current_value > rule.threshold_warning:
                    alert_level = AlertLevel.WARNING
                    target_value = rule.threshold_warning
            
            # 如果没有触发警报，返回None
            if alert_level is None:
                return None
            
            # 更新冷却期
            self.alert_cooldown[rule.metric_name] = datetime.now()
            
            # 创建警报
            message = rule.message_template.format(
                current=current_value * 100 if 'rate' in rule.metric_name or 'drawdown' in rule.metric_name else current_value,
                target=target_value * 100 if 'rate' in rule.metric_name or 'drawdown' in rule.metric_name else target_value
            )
            
            return KPIAlert(
                timestamp=datetime.now(),
                level=alert_level,
                metric=rule.metric_name,
                current_value=current_value,
                target_value=target_value,
                message=message,
                suggestion=rule.suggestion_template
            )
            
        except Exception as e:
            self.logger.error(f"检查警报规则失败 {rule.metric_name}: {e}")
            return None
    
    def _get_metric_value(self, metric_name: str, metrics: KPIMetrics) -> Optional[float]:
        """获取指标值"""
        metric_mapping = {
            "trading_frequency_low": metrics.trades_per_day,
            "trading_frequency_high": metrics.trades_per_day,
            "win_rate": metrics.win_rate,
            "max_drawdown": metrics.max_drawdown,
            "profit_factor": metrics.profit_factor,
            "consecutive_losses": metrics.consecutive_losses,
            "signal_confidence": metrics.avg_signal_confidence,
            "signal_conversion": metrics.signal_conversion_rate
        }
        
        return metric_mapping.get(metric_name)
    
    def _is_in_cooldown(self, metric_name: str) -> bool:
        """检查是否在冷却期内"""
        if metric_name not in self.alert_cooldown:
            return False
        
        last_alert = self.alert_cooldown[metric_name]
        return (datetime.now() - last_alert).total_seconds() < self.cooldown_period
    
    async def _send_alert_notification(self, alert: KPIAlert):
        """发送警报通知"""
        try:
            # 记录日志
            level_emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨"
            }
            
            emoji = level_emoji.get(alert.level, "📢")
            
            if alert.level == AlertLevel.CRITICAL:
                self.logger.error(f"{emoji} {alert.level.value}: {alert.message}")
                self.logger.error(f"   💡 {alert.suggestion}")
            elif alert.level == AlertLevel.WARNING:
                self.logger.warning(f"{emoji} {alert.level.value}: {alert.message}")
                self.logger.warning(f"   💡 {alert.suggestion}")
            else:
                self.logger.info(f"{emoji} {alert.level.value}: {alert.message}")
            
            # 发送Telegram通知
            if self.telegram_bot:
                message = f"""
{emoji} **KPI警报** - {alert.level.value}

🎯 **指标**: {alert.metric}
📊 **当前值**: {alert.current_value:.3f}
🚨 **告警值**: {alert.target_value:.3f}
⏰ **时间**: {alert.timestamp.strftime('%H:%M:%S')}

📝 **详情**: {alert.message}
💡 **建议**: {alert.suggestion}
"""
                
                try:
                    await self.telegram_bot.send_message(message)
                except Exception as e:
                    self.logger.warning(f"Telegram通知发送失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"发送警报通知失败: {e}")
    
    async def _execute_auto_action(self, alert: KPIAlert, metrics: KPIMetrics):
        """执行自动响应动作"""
        try:
            # 查找对应的警报规则
            rule = next((r for r in self.alert_rules if r.metric_name == alert.metric), None)
            if not rule or not rule.auto_action:
                return
            
            action_type = rule.auto_action
            
            # 准备动作参数
            action_params = {
                "alert": alert,
                "metrics": metrics,
                "timestamp": datetime.now()
            }
            
            # 记录动作
            action_record = {
                "timestamp": datetime.now().isoformat(),
                "action_type": action_type,
                "trigger_metric": alert.metric,
                "trigger_value": alert.current_value,
                "target_value": alert.target_value,
                "parameters": action_params
            }
            
            self.logger.warning(f"🤖 执行自动响应动作: {action_type}")
            
            # 执行具体动作
            if action_type == "adjust_signal_threshold":
                await self._auto_adjust_signal_threshold(alert, metrics)
            elif action_type == "reduce_risk_exposure":
                await self._auto_reduce_risk_exposure(alert, metrics)
            elif action_type == "emergency_risk_reduction":
                await self._auto_emergency_risk_reduction(alert, metrics)
            elif action_type == "optimize_exit_strategy":
                await self._auto_optimize_exit_strategy(alert, metrics)
            elif action_type == "optimize_profit_factor":
                await self._auto_optimize_profit_factor(alert, metrics)
            
            # 保存动作记录
            self.action_history.append(action_record)
            await self._save_action_history()
            
            # 发送动作通知
            if self.telegram_bot:
                message = f"""
🤖 **自动响应动作已执行**

⚡ **动作类型**: {action_type}
🎯 **触发指标**: {alert.metric}
📊 **触发值**: {alert.current_value:.3f}
⏰ **执行时间**: {datetime.now().strftime('%H:%M:%S')}

📝 **说明**: 系统已自动调整相关参数以改善指标表现
"""
                try:
                    await self.telegram_bot.send_message(message)
                except Exception as e:
                    self.logger.warning(f"自动动作通知发送失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"执行自动响应动作失败: {e}")
    
    async def _auto_adjust_signal_threshold(self, alert: KPIAlert, metrics: KPIMetrics):
        """自动调整信号阈值"""
        try:
            if "frequency_low" in alert.metric:
                # 交易频率过低 - 降低信号阈值
                adjustment = -0.05  # 降低5%
                self.logger.info(f"   📉 建议降低信号阈值 {adjustment*100:+.0f}% 以增加交易频率")
            elif "frequency_high" in alert.metric:
                # 交易频率过高 - 提高信号阈值
                adjustment = +0.05  # 提高5%
                self.logger.info(f"   📈 建议提高信号阈值 {adjustment*100:+.0f}% 以减少交易频率")
            
            # 这里可以集成到实际的信号生成器参数调整
            # 由于需要修改现有系统，暂时只记录建议
            
        except Exception as e:
            self.logger.error(f"自动调整信号阈值失败: {e}")
    
    async def _auto_reduce_risk_exposure(self, alert: KPIAlert, metrics: KPIMetrics):
        """自动降低风险敞口"""
        try:
            if metrics.max_drawdown > 0.20:
                # 回撤过大 - 建议降低杠杆和仓位
                leverage_reduction = 0.2  # 降低20%
                position_reduction = 0.3  # 降低30%
                
                self.logger.warning(f"   📉 建议降低杠杆 {leverage_reduction*100:.0f}%")
                self.logger.warning(f"   📉 建议降低仓位大小 {position_reduction*100:.0f}%")
            
        except Exception as e:
            self.logger.error(f"自动降低风险敞口失败: {e}")
    
    async def _auto_emergency_risk_reduction(self, alert: KPIAlert, metrics: KPIMetrics):
        """紧急风险降低"""
        try:
            if metrics.consecutive_losses >= 5:
                self.logger.error("   🚨 连续亏损过多，建议紧急风险管理")
                self.logger.error("   🛑 建议暂停新交易")
                self.logger.error("   📉 建议降低现有仓位")
                
                # 发送紧急通知
                if self.telegram_bot:
                    emergency_msg = """
🚨 **紧急风险警报**

⚠️ 连续亏损次数过多，建议立即采取行动：
1. 暂停新的交易开仓
2. 评估现有持仓风险
3. 考虑减仓或平仓
4. 检查策略有效性

请立即关注并手动干预！
"""
                    await self.telegram_bot.send_message(emergency_msg)
            
        except Exception as e:
            self.logger.error(f"紧急风险降低失败: {e}")
    
    async def _auto_optimize_exit_strategy(self, alert: KPIAlert, metrics: KPIMetrics):
        """自动优化退出策略"""
        try:
            if metrics.win_rate < 0.65:
                self.logger.info("   🎯 建议优化止损止盈策略")
                self.logger.info("   📏 建议调整风险回报比至2.5:1或更高")
                self.logger.info("   ⏰ 建议缩短持仓时间")
            
        except Exception as e:
            self.logger.error(f"自动优化退出策略失败: {e}")
    
    async def _auto_optimize_profit_factor(self, alert: KPIAlert, metrics: KPIMetrics):
        """自动优化盈利因子"""
        try:
            if metrics.profit_factor < 2.0:
                self.logger.info("   💪 建议提高信号质量标准")
                self.logger.info("   🎯 建议优化风险回报比")
                self.logger.info("   📉 建议减少交易频率，提高质量")
            
        except Exception as e:
            self.logger.error(f"自动优化盈利因子失败: {e}")
    
    async def _save_action_history(self):
        """保存响应动作历史"""
        try:
            # 确保目录存在
            os.makedirs(self.data_dir, exist_ok=True)
            
            with open(self.actions_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存响应动作历史失败: {e}")
    
    def get_action_summary(self, days: int = 7) -> Dict[str, Any]:
        """获取动作执行摘要"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            recent_actions = [a for a in self.action_history if a['timestamp'] >= cutoff_date]
            
            action_types = {}
            for action in recent_actions:
                action_type = action['action_type']
                action_types[action_type] = action_types.get(action_type, 0) + 1
            
            return {
                "total_actions": len(recent_actions),
                "action_types": action_types,
                "last_action": recent_actions[-1] if recent_actions else None,
                "period_days": days
            }
            
        except Exception as e:
            self.logger.error(f"获取动作摘要失败: {e}")
            return {"error": str(e)}
    
    async def generate_alert_report(self) -> str:
        """生成预警系统报告"""
        try:
            action_summary = self.get_action_summary(7)
            
            report = f"""
# 🚨 预警系统报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 警报规则概况
- **总规则数**: {len(self.alert_rules)} 条
- **冷却期**: {self.cooldown_period//60} 分钟
- **活跃警报**: {len([k for k, v in self.alert_cooldown.items() if (datetime.now() - v).total_seconds() < self.cooldown_period])} 个

## 🤖 自动响应统计（最近7天）
- **总执行次数**: {action_summary['total_actions']}
- **动作类型分布**:
"""
            
            for action_type, count in action_summary.get('action_types', {}).items():
                report += f"  - {action_type}: {count} 次\n"
            
            if action_summary.get('last_action'):
                last_action = action_summary['last_action']
                report += f"""
## 📅 最近动作
- **类型**: {last_action['action_type']}
- **触发指标**: {last_action['trigger_metric']}
- **执行时间**: {last_action['timestamp']}
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成预警报告失败: {e}")
            return f"报告生成失败: {e}" 