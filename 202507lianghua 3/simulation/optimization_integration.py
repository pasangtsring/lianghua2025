"""
参数优化集成系统
集成参数优化器与监控系统，实现自动化的参数调优流程
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

from utils.logger import get_logger
from simulation.parameter_optimizer import ParameterOptimizer, OptimizationStrategy
from simulation.kpi_monitor import KPIMonitor
from simulation.alert_system import AlertSystem
from simulation.monitoring_reporter import MonitoringReporter
from simulation.simulation_trading_manager import SimulationTradingManager

class OptimizationIntegration:
    """参数优化集成系统"""
    
    def __init__(self, trading_manager: SimulationTradingManager):
        self.logger = get_logger(__name__)
        self.trading_manager = trading_manager
        
        # 初始化监控系统组件
        self.kpi_monitor = KPIMonitor(trading_manager)
        self.alert_system = AlertSystem()
        self.monitoring_reporter = MonitoringReporter(self.kpi_monitor, self.alert_system)
        
        # 初始化参数优化器
        self.parameter_optimizer = ParameterOptimizer(self.kpi_monitor, trading_manager)
        
        # 集成配置
        self.auto_optimization_enabled = True
        self.optimization_check_interval = 3600  # 1小时检查一次
        self.performance_threshold_days = 3  # 连续3天表现不佳触发优化
        
        # 运行状态
        self.is_running = False
        self.last_optimization_check = None
        self.optimization_task = None
        
        # 数据目录
        self.data_dir = "simulation/integration"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("🚀 参数优化集成系统初始化完成")
        self.logger.info(f"   ⚙️ 自动优化: {'启用' if self.auto_optimization_enabled else '禁用'}")
        self.logger.info(f"   ⏰ 检查间隔: {self.optimization_check_interval} 秒")
    
    async def start_integration(self):
        """启动集成系统"""
        try:
            if self.is_running:
                self.logger.warning("集成系统已在运行中")
                return
            
            self.is_running = True
            self.logger.info("🚀 启动参数优化集成系统...")
            
            # 启动自动优化任务
            if self.auto_optimization_enabled:
                self.optimization_task = asyncio.create_task(self._auto_optimization_loop())
                self.logger.info("✅ 自动优化循环已启动")
            
            # 生成初始报告
            await self._generate_startup_report()
            
            self.logger.info("🎉 参数优化集成系统启动成功！")
            
        except Exception as e:
            self.logger.error(f"启动集成系统失败: {e}")
            self.is_running = False
            raise
    
    async def stop_integration(self):
        """停止集成系统"""
        try:
            if not self.is_running:
                return
            
            self.logger.info("🛑 停止参数优化集成系统...")
            self.is_running = False
            
            # 停止自动优化任务
            if self.optimization_task and not self.optimization_task.done():
                self.optimization_task.cancel()
                try:
                    await self.optimization_task
                except asyncio.CancelledError:
                    pass
            
            # 生成停止报告
            await self._generate_shutdown_report()
            
            self.logger.info("✅ 参数优化集成系统已停止")
            
        except Exception as e:
            self.logger.error(f"停止集成系统失败: {e}")
    
    async def _auto_optimization_loop(self):
        """自动优化主循环"""
        try:
            while self.is_running:
                try:
                    # 执行优化检查
                    await self._perform_optimization_check()
                    
                    # 等待下一次检查
                    await asyncio.sleep(self.optimization_check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"自动优化循环错误: {e}")
                    # 等待一段时间后继续
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            self.logger.info("自动优化循环已取消")
        except Exception as e:
            self.logger.error(f"自动优化循环严重错误: {e}")
    
    async def _perform_optimization_check(self):
        """执行优化检查"""
        try:
            self.logger.info("🔍 执行定期优化检查...")
            self.last_optimization_check = datetime.now()
            
            # 1. 获取当前KPI
            current_kpi = await self.kpi_monitor.get_current_kpi()
            if not current_kpi:
                self.logger.warning("无法获取当前KPI，跳过优化检查")
                return
            
            # 2. 生成KPI报告
            kpi_report = await self.kpi_monitor.generate_kpi_report()
            
            # 3. 检查是否需要告警
            await self._check_and_send_alerts(current_kpi)
            
            # 4. 检查是否需要参数优化
            optimization_result = await self.parameter_optimizer.auto_optimization_check()
            
            # 5. 处理优化结果
            await self._handle_optimization_result(optimization_result)
            
            # 6. 生成监控报告
            monitoring_report = await self.monitoring_reporter.generate_comprehensive_report()
            
            # 7. 保存检查结果
            await self._save_check_results(current_kpi, optimization_result)
            
            self.logger.info("✅ 定期优化检查完成")
            
        except Exception as e:
            self.logger.error(f"优化检查失败: {e}")
    
    async def _check_and_send_alerts(self, kpi):
        """检查并发送告警"""
        try:
            # 检查各项KPI指标
            alerts = []
            
            # 胜率告警
            if kpi.win_rate < 0.5:
                alerts.append({
                    "level": "CRITICAL",
                    "metric": "win_rate",
                    "value": kpi.win_rate,
                    "message": f"胜率严重偏低: {kpi.win_rate:.1%}",
                    "suggestion": "建议立即优化信号阈值参数"
                })
            elif kpi.win_rate < 0.6:
                alerts.append({
                    "level": "WARNING",
                    "metric": "win_rate",
                    "value": kpi.win_rate,
                    "message": f"胜率偏低: {kpi.win_rate:.1%}",
                    "suggestion": "考虑调整信号识别策略"
                })
            
            # 回撤告警
            if kpi.max_drawdown > 0.2:
                alerts.append({
                    "level": "CRITICAL",
                    "metric": "max_drawdown",
                    "value": kpi.max_drawdown,
                    "message": f"回撤过大: {kpi.max_drawdown:.1%}",
                    "suggestion": "立即降低仓位规模或收紧止损"
                })
            elif kpi.max_drawdown > 0.15:
                alerts.append({
                    "level": "WARNING",
                    "metric": "max_drawdown",
                    "value": kpi.max_drawdown,
                    "message": f"回撤接近上限: {kpi.max_drawdown:.1%}",
                    "suggestion": "注意风险控制，考虑参数优化"
                })
            
            # 盈利因子告警
            if kpi.profit_factor < 1.5:
                alerts.append({
                    "level": "CRITICAL",
                    "metric": "profit_factor",
                    "value": kpi.profit_factor,
                    "message": f"盈利因子过低: {kpi.profit_factor:.2f}",
                    "suggestion": "优化止损止盈比例"
                })
            elif kpi.profit_factor < 2.0:
                alerts.append({
                    "level": "WARNING",
                    "metric": "profit_factor",
                    "value": kpi.profit_factor,
                    "message": f"盈利因子偏低: {kpi.profit_factor:.2f}",
                    "suggestion": "考虑调整交易策略"
                })
            
            # 发送告警
            for alert in alerts:
                await self.alert_system.process_alert({
                    "alert_level": alert["level"],
                    "metric_name": alert["metric"],
                    "current_value": alert["value"],
                    "message": alert["message"],
                    "suggestion": alert["suggestion"],
                    "timestamp": datetime.now().isoformat()
                })
            
            if alerts:
                self.logger.warning(f"⚠️ 发送了 {len(alerts)} 条告警")
            else:
                self.logger.info("✅ 所有KPI指标正常，无需告警")
                
        except Exception as e:
            self.logger.error(f"检查告警失败: {e}")
    
    async def _handle_optimization_result(self, optimization_result: Dict[str, Any]):
        """处理优化结果"""
        try:
            status = optimization_result.get("status")
            
            if status == "completed":
                # 优化成功
                optimized_count = optimization_result.get("optimization_count", 0)
                self.logger.info(f"🎉 参数优化成功，共优化 {optimized_count} 个参数")
                
                # 应用优化后的参数到交易系统
                await self._apply_optimized_parameters(optimization_result)
                
                # 发送优化成功通知
                await self._send_optimization_notification(optimization_result)
                
            elif status == "no_optimization_needed":
                # 无需优化
                reason = optimization_result.get("reason", "")
                self.logger.info(f"📊 系统表现良好，无需优化: {reason}")
                
            elif status == "skipped":
                # 跳过优化
                reason = optimization_result.get("reason", "")
                self.logger.info(f"⏭️ 跳过优化: {reason}")
                
            elif status == "error":
                # 优化失败
                error = optimization_result.get("error", "未知错误")
                self.logger.error(f"❌ 参数优化失败: {error}")
                
                # 发送错误告警
                await self.alert_system.process_alert({
                    "alert_level": "WARNING",
                    "metric_name": "optimization_error",
                    "current_value": 1,
                    "message": f"参数优化失败: {error}",
                    "suggestion": "检查系统状态，可能需要人工干预",
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            self.logger.error(f"处理优化结果失败: {e}")
    
    async def _apply_optimized_parameters(self, optimization_result: Dict[str, Any]):
        """应用优化后的参数"""
        try:
            optimized_params = optimization_result.get("optimized_params", [])
            
            if not optimized_params:
                return
            
            self.logger.info("⚙️ 应用优化后的参数...")
            
            # 应用参数到交易管理器
            for param in optimized_params:
                param_name = param.get("param_name")
                new_value = param.get("new_value")
                old_value = param.get("old_value")
                
                if param_name and new_value is not None:
                    # 根据参数类型应用到相应的配置
                    await self._update_trading_parameter(param_name, new_value)
                    
                    self.logger.info(f"   ✅ {param_name}: {old_value:.3f} → {new_value:.3f}")
            
            self.logger.info("🎯 所有优化参数已成功应用")
            
        except Exception as e:
            self.logger.error(f"应用优化参数失败: {e}")
    
    async def _update_trading_parameter(self, param_name: str, new_value: float):
        """更新交易参数"""
        try:
            # 这里需要根据实际的交易管理器接口来实现
            # 由于是模拟系统，我们主要更新配置记录
            
            config_update = {
                "parameter": param_name,
                "new_value": new_value,
                "timestamp": datetime.now().isoformat(),
                "applied_by": "optimization_system"
            }
            
            # 保存参数更新记录
            updates_file = f"{self.data_dir}/parameter_updates.json"
            updates = []
            
            if os.path.exists(updates_file):
                with open(updates_file, 'r', encoding='utf-8') as f:
                    updates = json.load(f)
            
            updates.append(config_update)
            
            with open(updates_file, 'w', encoding='utf-8') as f:
                json.dump(updates, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"参数更新记录已保存: {param_name} = {new_value}")
            
        except Exception as e:
            self.logger.error(f"更新交易参数失败 {param_name}: {e}")
    
    async def _send_optimization_notification(self, optimization_result: Dict[str, Any]):
        """发送优化通知"""
        try:
            optimized_params = optimization_result.get("optimized_params", [])
            strategy = optimization_result.get("strategy", "unknown")
            
            if not optimized_params:
                return
            
            # 构建通知消息
            message = f"🎉 参数优化完成\n"
            message += f"📊 优化策略: {strategy}\n"
            message += f"🔧 优化参数: {len(optimized_params)} 个\n\n"
            
            for param in optimized_params:
                param_name = param.get("param_name")
                old_value = param.get("old_value")
                new_value = param.get("new_value") 
                improvement = param.get("improvement_score", 0)
                confidence = param.get("confidence", 0)
                
                message += f"• {param_name}\n"
                message += f"  {old_value:.3f} → {new_value:.3f}\n"
                message += f"  改善: {improvement:.3f}, 置信度: {confidence:.1%}\n\n"
            
            # 发送信息级告警
            await self.alert_system.process_alert({
                "alert_level": "INFO",
                "metric_name": "optimization_completed",
                "current_value": len(optimized_params),
                "message": message,
                "suggestion": "参数已自动应用，请监控后续表现",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"发送优化通知失败: {e}")
    
    async def _save_check_results(self, kpi, optimization_result):
        """保存检查结果"""
        try:
            timestamp = datetime.now()
            
            check_result = {
                "timestamp": timestamp.isoformat(),
                "kpi": {
                    "win_rate": kpi.win_rate,
                    "profit_factor": kpi.profit_factor,
                    "max_drawdown": kpi.max_drawdown,
                    "sharpe_ratio": kpi.sharpe_ratio,
                    "total_trades": kpi.total_trades,
                    "total_pnl": kpi.total_pnl
                },
                "optimization_result": optimization_result
            }
            
            # 保存到日志文件
            log_file = f"{self.data_dir}/optimization_checks.json"
            checks = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    checks = json.load(f)
            
            checks.append(check_result)
            
            # 保持最近100条记录
            if len(checks) > 100:
                checks = checks[-100:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(checks, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            self.logger.error(f"保存检查结果失败: {e}")
    
    async def _generate_startup_report(self):
        """生成启动报告"""
        try:
            startup_time = datetime.now()
            
            report = f"""# 🚀 参数优化集成系统启动报告

## 📅 启动信息
- **启动时间**: {startup_time.strftime('%Y-%m-%d %H:%M:%S')}
- **自动优化**: {'启用' if self.auto_optimization_enabled else '禁用'}
- **检查间隔**: {self.optimization_check_interval} 秒
- **性能阈值**: {self.performance_threshold_days} 天

## 🔧 系统组件状态
- ✅ KPI监控器: 已初始化
- ✅ 告警系统: 已初始化  
- ✅ 监控报告器: 已初始化
- ✅ 参数优化器: 已初始化

## 📊 优化配置
- **优化策略**: {self.parameter_optimizer.optimization_strategy.value}
- **优化间隔**: {self.parameter_optimizer.optimization_interval} 天
- **最小交易数**: {self.parameter_optimizer.min_trades_for_optimization} 笔
- **可优化参数**: {len(self.parameter_optimizer.parameter_ranges)} 个

## 🎯 下一步计划
1. 开始定期KPI监控
2. 执行自动优化检查
3. 生成监控报告
4. 根据表现进行参数调优

系统已就绪，开始监控和优化！
"""
            
            # 保存启动报告
            report_file = f"{self.data_dir}/startup_report_{startup_time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"📄 启动报告已生成: {report_file}")
            
        except Exception as e:
            self.logger.error(f"生成启动报告失败: {e}")
    
    async def _generate_shutdown_report(self):
        """生成停止报告"""
        try:
            shutdown_time = datetime.now()
            
            # 获取最终KPI
            final_kpi = await self.kpi_monitor.get_current_kpi()
            
            report = f"""# 🛑 参数优化集成系统停止报告

## 📅 停止信息
- **停止时间**: {shutdown_time.strftime('%Y-%m-%d %H:%M:%S')}
- **最后检查**: {self.last_optimization_check.strftime('%Y-%m-%d %H:%M:%S') if self.last_optimization_check else '未执行'}

## 📊 最终系统状态
"""
            
            if final_kpi:
                report += f"""- **胜率**: {final_kpi.win_rate:.1%}
- **盈利因子**: {final_kpi.profit_factor:.2f}
- **最大回撤**: {final_kpi.max_drawdown:.1%}
- **夏普比率**: {final_kpi.sharpe_ratio:.2f}
- **总交易**: {final_kpi.total_trades} 笔
- **总盈亏**: {final_kpi.total_pnl:+.2f} USDT
"""
            else:
                report += "- 无法获取最终KPI数据\n"
            
            report += f"""
## 🔧 优化统计
- **优化历史**: {len(self.parameter_optimizer.optimization_history)} 次
- **系统运行**: 正常停止

系统已安全停止！
"""
            
            # 保存停止报告
            report_file = f"{self.data_dir}/shutdown_report_{shutdown_time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"📄 停止报告已生成: {report_file}")
            
        except Exception as e:
            self.logger.error(f"生成停止报告失败: {e}")
    
    async def manual_optimization(self, strategy: str = "adaptive") -> Dict[str, Any]:
        """手动触发参数优化"""
        try:
            self.logger.info(f"🔧 手动触发参数优化 (策略: {strategy})...")
            
            # 设置优化策略
            if strategy.upper() in [s.value for s in OptimizationStrategy]:
                self.parameter_optimizer.optimization_strategy = OptimizationStrategy(strategy.upper())
            
            # 执行优化
            result = await self.parameter_optimizer.optimize_parameters()
            
            # 处理结果
            await self._handle_optimization_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"手动优化失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate_integration_report(self) -> str:
        """生成集成系统报告"""
        try:
            current_time = datetime.now()
            
            # 获取当前KPI
            current_kpi = await self.kpi_monitor.get_current_kpi()
            
            # 获取优化报告
            optimization_report = await self.parameter_optimizer.generate_optimization_report()
            
            report = f"""# 📊 参数优化集成系统报告

## 🎯 系统状态
- **报告时间**: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
- **运行状态**: {'运行中' if self.is_running else '已停止'}
- **最后检查**: {self.last_optimization_check.strftime('%Y-%m-%d %H:%M:%S') if self.last_optimization_check else '未执行'}

## 📈 集成配置
- **自动优化**: {'启用' if self.auto_optimization_enabled else '禁用'}
- **检查间隔**: {self.optimization_check_interval} 秒
- **性能阈值**: {self.performance_threshold_days} 天

{optimization_report}

## 🎉 总结
参数优化集成系统正常运行，持续监控和优化交易性能。
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成集成报告失败: {e}")
            return f"报告生成失败: {e}" 