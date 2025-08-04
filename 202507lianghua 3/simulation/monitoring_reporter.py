"""
KPI监控报告系统
生成详细的性能分析报告、趋势分析和优化建议
"""

import asyncio
import json
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os
import statistics
import numpy as np

from utils.logger import get_logger
from simulation.kpi_monitor import KPIMonitor, KPIMetrics, KPIStatus
from simulation.alert_system import AlertSystem

class MonitoringReporter:
    """KPI监控报告系统"""
    
    def __init__(self, kpi_monitor: KPIMonitor, alert_system: AlertSystem):
        self.logger = get_logger(__name__)
        self.kpi_monitor = kpi_monitor
        self.alert_system = alert_system
        
        # 配置
        self.data_dir = "simulation/monitoring"
        self.reports_dir = f"{self.data_dir}/reports"
        self.charts_dir = f"{self.data_dir}/charts"
        
        # 确保目录存在
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # 配置图表样式
        if HAS_MATPLOTLIB:
            plt.style.use('default')
            sns.set_palette("husl")
        else:
            self.logger.warning("matplotlib未安装，图表生成功能将被禁用")
        
        self.logger.info("📊 监控报告系统初始化完成")
        self.logger.info(f"   📂 报告目录: {self.reports_dir}")
        self.logger.info(f"   📈 图表目录: {self.charts_dir}")
    
    async def generate_comprehensive_report(self, period_days: int = 7) -> str:
        """生成综合监控报告"""
        try:
            self.logger.info(f"📋 生成{period_days}天综合监控报告...")
            
            # 获取数据
            recent_metrics = self.kpi_monitor.get_recent_metrics(period_days)
            recent_alerts = self.kpi_monitor.get_recent_alerts(period_days)
            action_summary = self.alert_system.get_action_summary(period_days)
            
            if not recent_metrics:
                return "❌ 无监控数据，无法生成报告"
            
            # 分析数据
            latest_metrics = recent_metrics[-1]
            trend_analysis = self._analyze_trends(recent_metrics)
            performance_analysis = self._analyze_performance(recent_metrics)
            risk_analysis = self._analyze_risk(recent_metrics)
            
            # 生成图表
            charts = await self._generate_charts(recent_metrics, period_days)
            
            # 生成报告内容
            report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_content = f"""
# 📊 KPI监控综合报告

**报告编号**: {report_timestamp}  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**监控周期**: {period_days} 天  
**数据点数**: {len(recent_metrics)} 个  

---

## 🎯 执行摘要

### 当前状态
- **整体状态**: {latest_metrics.overall_status.value}
- **监控周期**: {latest_metrics.period_days} 天
- **总交易数**: {latest_metrics.total_trades}
- **当前资金**: {latest_metrics.current_capital:.4f} USDT
- **总收益率**: {latest_metrics.return_percentage:+.2f}%

### 核心指标概览
| 指标 | 当前值 | 目标值 | 状态 | 趋势 |
|------|--------|--------|------|------|
| 交易频率 | {latest_metrics.trades_per_day:.1f} 单/天 | 5-8 单/天 | {latest_metrics.detailed_status['trading_frequency'].value} | {trend_analysis['trading_frequency']['direction']} |
| 胜率 | {latest_metrics.win_rate*100:.1f}% | ≥65% | {latest_metrics.detailed_status['win_rate'].value} | {trend_analysis['win_rate']['direction']} |
| 最大回撤 | {latest_metrics.max_drawdown*100:.1f}% | ≤20% | {latest_metrics.detailed_status['drawdown'].value} | {trend_analysis['max_drawdown']['direction']} |
| 盈利因子 | {latest_metrics.profit_factor:.2f} | ≥2.0 | {latest_metrics.detailed_status['profit_factor'].value} | {trend_analysis['profit_factor']['direction']} |
| 信号质量 | {latest_metrics.avg_signal_confidence*100:.1f}% | ≥60% | {latest_metrics.detailed_status['signal_quality'].value} | {trend_analysis['signal_quality']['direction']} |

---

## 📈 趋势分析

### 交易频率趋势
- **当前**: {latest_metrics.trades_per_day:.1f} 单/天
- **趋势**: {trend_analysis['trading_frequency']['direction']} ({trend_analysis['trading_frequency']['change']:+.1f} 单/天)
- **分析**: {trend_analysis['trading_frequency']['analysis']}

### 盈利能力趋势
- **胜率变化**: {trend_analysis['win_rate']['change']*100:+.1f}%
- **盈利因子变化**: {trend_analysis['profit_factor']['change']:+.2f}
- **收益率趋势**: {trend_analysis['return']['direction']}

### 风险指标趋势
- **回撤变化**: {trend_analysis['max_drawdown']['change']*100:+.1f}%
- **风险趋势**: {risk_analysis['trend']}
- **风险评级**: {risk_analysis['level']}

---

## 📊 性能分析

### 交易统计
- **总交易数**: {performance_analysis['total_trades']}
- **平均日交易**: {performance_analysis['avg_daily_trades']:.1f} 单
- **交易成功率**: {performance_analysis['success_rate']*100:.1f}%
- **平均持仓时间**: {performance_analysis['avg_holding_time']} 小时

### 盈利分析
- **总盈亏**: {performance_analysis['total_pnl']:+.4f} USDT
- **平均单笔盈亏**: {performance_analysis['avg_trade_pnl']:+.4f} USDT
- **最大单笔盈利**: {performance_analysis['max_profit']:+.4f} USDT
- **最大单笔亏损**: {performance_analysis['max_loss']:+.4f} USDT

### 稳定性指标
- **胜率标准差**: {performance_analysis['win_rate_std']:.3f}
- **收益波动率**: {performance_analysis['return_volatility']:.3f}
- **夏普比率**: {performance_analysis['sharpe_ratio']:.2f}

---

## 🛡️ 风险分析

### 风险控制效果
- **最大回撤**: {latest_metrics.max_drawdown*100:.1f}% (目标: ≤20%)
- **当前回撤**: {latest_metrics.current_drawdown*100:.1f}%
- **连续亏损**: {latest_metrics.consecutive_losses} 次 (最大: {risk_analysis['max_consecutive_losses']} 次)

### 资金管理
- **资金利用率**: {risk_analysis['capital_utilization']*100:.1f}%
- **风险敞口**: {risk_analysis['risk_exposure']*100:.1f}%
- **安全边际**: {risk_analysis['safety_margin']*100:.1f}%

### 风险建议
{risk_analysis['recommendations']}

---

## 🚨 警报统计

### 警报概况
- **总警报数**: {len(recent_alerts)}
- **严重警报**: {len([a for a in recent_alerts if a.level.value == 'CRITICAL'])} 个
- **警告数量**: {len([a for a in recent_alerts if a.level.value == 'WARNING'])} 个

### 自动响应
- **执行动作数**: {action_summary['total_actions']}
- **动作类型**: {', '.join(action_summary.get('action_types', {}).keys())}

### 最近警报
"""

            # 添加最近5个警报
            if recent_alerts:
                report_content += "\n| 时间 | 级别 | 指标 | 描述 |\n|------|------|------|------|\n"
                for alert in recent_alerts[-5:]:
                    report_content += f"| {alert.timestamp.strftime('%m-%d %H:%M')} | {alert.level.value} | {alert.metric} | {alert.message} |\n"
            else:
                report_content += "\n- 无警报记录\n"

            report_content += f"""

---

## 📊 图表分析

### 可视化图表
{self._format_charts_section(charts)}

---

## 💡 优化建议

### 立即行动项
{self._generate_immediate_actions(latest_metrics, trend_analysis)}

### 中期优化
{self._generate_medium_term_optimizations(performance_analysis, risk_analysis)}

### 长期规划
{self._generate_long_term_planning(trend_analysis)}

---

## 📋 总结

### 系统健康度评估
{self._assess_system_health(latest_metrics, trend_analysis, risk_analysis)}

### 下期关注重点
{self._highlight_next_focus(latest_metrics, recent_alerts)}

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*数据来源: KPI监控系统*  
*报告周期: {period_days} 天*
"""

            # 保存报告
            report_file = f"{self.reports_dir}/comprehensive_report_{report_timestamp}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"📄 综合报告已生成: {report_file}")
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"生成综合报告失败: {e}")
            return f"❌ 报告生成失败: {e}"
    
    def _analyze_trends(self, metrics_list: List[KPIMetrics]) -> Dict[str, Any]:
        """分析趋势"""
        try:
            if len(metrics_list) < 2:
                return {"error": "数据不足，无法分析趋势"}
            
            first = metrics_list[0]
            latest = metrics_list[-1]
            
            def trend_direction(change):
                if abs(change) < 0.01:
                    return "稳定"
                return "上升" if change > 0 else "下降"
            
            trading_freq_change = latest.trades_per_day - first.trades_per_day
            win_rate_change = latest.win_rate - first.win_rate
            drawdown_change = latest.max_drawdown - first.max_drawdown
            profit_factor_change = latest.profit_factor - first.profit_factor
            return_change = latest.return_percentage - first.return_percentage
            
            return {
                "trading_frequency": {
                    "change": trading_freq_change,
                    "direction": trend_direction(trading_freq_change),
                    "analysis": "交易频率保持稳定" if abs(trading_freq_change) < 0.5 else 
                               "交易频率显著增加" if trading_freq_change > 0.5 else "交易频率明显下降"
                },
                "win_rate": {
                    "change": win_rate_change,
                    "direction": trend_direction(win_rate_change)
                },
                "max_drawdown": {
                    "change": drawdown_change,
                    "direction": trend_direction(drawdown_change)
                },
                "profit_factor": {
                    "change": profit_factor_change,
                    "direction": trend_direction(profit_factor_change)
                },
                "return": {
                    "change": return_change,
                    "direction": trend_direction(return_change)
                },
                "signal_quality": {
                    "change": latest.avg_signal_confidence - first.avg_signal_confidence,
                    "direction": trend_direction(latest.avg_signal_confidence - first.avg_signal_confidence)
                }
            }
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            return {"error": str(e)}
    
    def _analyze_performance(self, metrics_list: List[KPIMetrics]) -> Dict[str, Any]:
        """分析性能"""
        try:
            if not metrics_list:
                return {"error": "无数据"}
            
            latest = metrics_list[-1]
            
            # 计算统计指标
            win_rates = [m.win_rate for m in metrics_list]
            returns = [m.return_percentage for m in metrics_list]
            
            return {
                "total_trades": latest.total_trades,
                "avg_daily_trades": latest.trades_per_day,
                "success_rate": latest.win_rate,
                "total_pnl": latest.total_pnl,
                "avg_trade_pnl": latest.total_pnl / latest.total_trades if latest.total_trades > 0 else 0,
                "max_profit": max([m.total_pnl for m in metrics_list] + [0]),
                "max_loss": min([m.total_pnl for m in metrics_list] + [0]),
                "avg_holding_time": 24,  # 假设平均持仓24小时
                "win_rate_std": statistics.stdev(win_rates) if len(win_rates) > 1 else 0,
                "return_volatility": statistics.stdev(returns) if len(returns) > 1 else 0,
                "sharpe_ratio": latest.return_percentage / (statistics.stdev(returns) if len(returns) > 1 and statistics.stdev(returns) > 0 else 1)
            }
            
        except Exception as e:
            self.logger.error(f"性能分析失败: {e}")
            return {"error": str(e)}
    
    def _analyze_risk(self, metrics_list: List[KPIMetrics]) -> Dict[str, Any]:
        """分析风险"""
        try:
            if not metrics_list:
                return {"error": "无数据"}
            
            latest = metrics_list[-1]
            max_consecutive = max([m.consecutive_losses for m in metrics_list] + [0])
            
            # 风险等级评估
            risk_score = 0
            if latest.max_drawdown > 0.15:
                risk_score += 2
            if latest.consecutive_losses > 3:
                risk_score += 2
            if latest.win_rate < 0.6:
                risk_score += 1
            
            risk_levels = ["低风险", "中等风险", "高风险", "极高风险"]
            risk_level = risk_levels[min(risk_score, 3)]
            
            # 资金利用率
            capital_utilization = (latest.current_capital - 65) / 65 if latest.current_capital > 65 else 0
            
            recommendations = []
            if latest.max_drawdown > 0.15:
                recommendations.append("- 考虑降低杠杆倍数")
            if latest.consecutive_losses > 3:
                recommendations.append("- 评估策略有效性")
            if latest.win_rate < 0.65:
                recommendations.append("- 优化信号质量")
            if not recommendations:
                recommendations.append("- 继续保持当前策略")
            
            return {
                "trend": "风险上升" if latest.max_drawdown > 0.1 else "风险可控",
                "level": risk_level,
                "max_consecutive_losses": max_consecutive,
                "capital_utilization": capital_utilization,
                "risk_exposure": latest.max_drawdown,
                "safety_margin": max(0, 0.2 - latest.max_drawdown),
                "recommendations": "\n".join(recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"风险分析失败: {e}")
            return {"error": str(e)}
    
    async def _generate_charts(self, metrics_list: List[KPIMetrics], period_days: int) -> Dict[str, str]:
        """生成图表"""
        charts = {}
        
        try:
            # 准备数据
            dates = [m.timestamp for m in metrics_list]
            win_rates = [m.win_rate * 100 for m in metrics_list]
            drawdowns = [m.max_drawdown * 100 for m in metrics_list]
            profit_factors = [m.profit_factor for m in metrics_list]
            capitals = [m.current_capital for m in metrics_list]
            
            # 1. 资金曲线图
            plt.figure(figsize=(12, 6))
            plt.plot(dates, capitals, marker='o', linewidth=2, markersize=4)
            plt.title('资金曲线', fontsize=14, fontweight='bold')
            plt.xlabel('时间')
            plt.ylabel('资金 (USDT)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_file = f"{self.charts_dir}/capital_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            charts['capital_curve'] = chart_file
            
            # 2. 关键指标趋势图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 胜率趋势
            ax1.plot(dates, win_rates, color='green', marker='o', linewidth=2)
            ax1.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='目标65%')
            ax1.set_title('胜率趋势')
            ax1.set_ylabel('胜率 (%)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 最大回撤趋势
            ax2.plot(dates, drawdowns, color='red', marker='o', linewidth=2)
            ax2.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='限制20%')
            ax2.set_title('最大回撤趋势')
            ax2.set_ylabel('回撤 (%)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 盈利因子趋势
            ax3.plot(dates, profit_factors, color='blue', marker='o', linewidth=2)
            ax3.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='目标2.0')
            ax3.set_title('盈利因子趋势')
            ax3.set_ylabel('盈利因子')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 交易频率
            trades_per_day = [m.trades_per_day for m in metrics_list]
            ax4.plot(dates, trades_per_day, color='orange', marker='o', linewidth=2)
            ax4.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='最低5单/天')
            ax4.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='最高8单/天')
            ax4.set_title('交易频率趋势')
            ax4.set_ylabel('交易频率 (单/天)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # 格式化x轴
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            chart_file = f"{self.charts_dir}/kpi_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            charts['kpi_trends'] = chart_file
            
            self.logger.info(f"📈 图表已生成: {len(charts)} 个")
            
        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")
            charts['error'] = str(e)
        
        return charts
    
    def _format_charts_section(self, charts: Dict[str, str]) -> str:
        """格式化图表部分"""
        if 'error' in charts:
            return f"❌ 图表生成失败: {charts['error']}"
        
        content = ""
        for chart_name, chart_path in charts.items():
            if chart_name == 'capital_curve':
                content += f"- **资金曲线图**: `{chart_path}`\n"
            elif chart_name == 'kpi_trends':
                content += f"- **KPI趋势图**: `{chart_path}`\n"
        
        return content if content else "- 无图表生成"
    
    def _generate_immediate_actions(self, latest: KPIMetrics, trends: Dict) -> str:
        """生成立即行动建议"""
        actions = []
        
        if latest.detailed_status.get('trading_frequency') == KPIStatus.CRITICAL:
            actions.append("- 🚨 **紧急**: 调整交易频率 - 检查信号生成参数")
        
        if latest.detailed_status.get('win_rate') == KPIStatus.CRITICAL:
            actions.append("- 🚨 **紧急**: 胜率过低 - 暂停交易，评估策略")
        
        if latest.detailed_status.get('drawdown') == KPIStatus.CRITICAL:
            actions.append("- 🚨 **紧急**: 回撤超限 - 减少仓位，降低杠杆")
        
        if latest.consecutive_losses >= 5:
            actions.append("- 🚨 **紧急**: 连续亏损过多 - 停止新交易")
        
        if not actions:
            actions.append("- ✅ 暂无紧急行动项，系统运行正常")
        
        return "\n".join(actions)
    
    def _generate_medium_term_optimizations(self, performance: Dict, risk: Dict) -> str:
        """生成中期优化建议"""
        optimizations = []
        
        if performance.get('sharpe_ratio', 0) < 1.0:
            optimizations.append("- 📈 优化风险调整收益 - 提高夏普比率")
        
        if risk.get('level') == '高风险':
            optimizations.append("- 🛡️ 降低系统风险等级 - 优化风险管理")
        
        if performance.get('win_rate_std', 0) > 0.1:
            optimizations.append("- 📊 提高胜率稳定性 - 减少策略波动")
        
        if not optimizations:
            optimizations.append("- ✅ 系统表现稳定，继续当前策略")
        
        return "\n".join(optimizations)
    
    def _generate_long_term_planning(self, trends: Dict) -> str:
        """生成长期规划建议"""
        planning = []
        
        if trends.get('trading_frequency', {}).get('direction') == '下降':
            planning.append("- 📉 制定交易频率提升计划")
        
        if trends.get('win_rate', {}).get('direction') == '下降':
            planning.append("- 🎯 制定胜率改进长期计划")
        
        planning.append("- 📈 考虑增加资金规模")
        planning.append("- 🔄 定期评估和优化策略参数")
        
        return "\n".join(planning)
    
    def _assess_system_health(self, latest: KPIMetrics, trends: Dict, risk: Dict) -> str:
        """评估系统健康度"""
        health_score = 0
        
        # 基于当前状态评分
        if latest.overall_status == KPIStatus.EXCELLENT:
            health_score += 30
        elif latest.overall_status == KPIStatus.GOOD:
            health_score += 20
        elif latest.overall_status == KPIStatus.WARNING:
            health_score += 10
        
        # 基于趋势评分
        positive_trends = sum(1 for t in trends.values() if isinstance(t, dict) and t.get('direction') == '上升')
        health_score += positive_trends * 5
        
        # 基于风险评分
        if risk.get('level') == '低风险':
            health_score += 20
        elif risk.get('level') == '中等风险':
            health_score += 10
        
        # 健康度评级
        if health_score >= 80:
            return "🟢 **优秀** - 系统运行状态极佳，各项指标表现优异"
        elif health_score >= 60:
            return "🟡 **良好** - 系统运行稳定，大部分指标达标"
        elif health_score >= 40:
            return "🟠 **一般** - 系统存在改进空间，需要关注部分指标"
        else:
            return "🔴 **需要改进** - 系统表现不佳，需要立即优化"
    
    def _highlight_next_focus(self, latest: KPIMetrics, alerts: List) -> str:
        """突出下期关注重点"""
        focus_areas = []
        
        # 基于当前状态
        critical_metrics = [k for k, v in latest.detailed_status.items() if v == KPIStatus.CRITICAL]
        warning_metrics = [k for k, v in latest.detailed_status.items() if v == KPIStatus.WARNING]
        
        if critical_metrics:
            focus_areas.append(f"- 🚨 **关键关注**: {', '.join(critical_metrics)}")
        
        if warning_metrics:
            focus_areas.append(f"- ⚠️ **密切监控**: {', '.join(warning_metrics)}")
        
        # 基于最近警报
        if len(alerts) > 3:
            focus_areas.append("- 📢 **警报频繁**: 加强系统监控频率")
        
        if not focus_areas:
            focus_areas.append("- ✅ **保持现状**: 继续监控各项指标稳定性")
        
        return "\n".join(focus_areas) 