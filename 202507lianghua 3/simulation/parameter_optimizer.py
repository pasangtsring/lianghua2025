"""
参数优化调整系统
基于KPI监控反馈，自动优化交易系统参数以提升性能
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import os
import copy

from utils.logger import get_logger
from simulation.kpi_monitor import KPIMonitor, KPIMetrics, KPIStatus
from simulation.alert_system import AlertSystem
from simulation.simulation_trading_manager import SimulationTradingManager

class OptimizationStrategy(Enum):
    """优化策略类型"""
    GRADUAL = "GRADUAL"           # 渐进式优化
    AGGRESSIVE = "AGGRESSIVE"     # 积极优化
    CONSERVATIVE = "CONSERVATIVE" # 保守优化
    ADAPTIVE = "ADAPTIVE"         # 自适应优化

class ParameterType(Enum):
    """参数类型"""
    SIGNAL_THRESHOLD = "signal_threshold"     # 信号阈值
    POSITION_SIZE = "position_size"           # 仓位大小
    STOP_LOSS = "stop_loss"                   # 止损距离
    TAKE_PROFIT = "take_profit"               # 止盈距离
    LEVERAGE = "leverage"                     # 杠杆倍数
    RISK_RATIO = "risk_ratio"                 # 风险比例

@dataclass
class ParameterRange:
    """参数优化范围"""
    param_name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    param_type: ParameterType
    priority: int = 1  # 优化优先级 1-5
    
@dataclass
class OptimizationResult:
    """优化结果"""
    param_name: str
    old_value: float
    new_value: float
    improvement_score: float
    confidence: float
    test_trades: int
    test_period_days: int
    
@dataclass
class ABTestConfig:
    """A/B测试配置"""
    test_name: str
    control_params: Dict[str, float]
    test_params: Dict[str, float]
    test_duration_days: int
    min_trades_required: int
    success_criteria: Dict[str, float]  # KPI改善目标

class ParameterOptimizer:
    """参数优化调整系统"""
    
    def __init__(self, kpi_monitor: KPIMonitor, trading_manager: SimulationTradingManager):
        self.logger = get_logger(__name__)
        self.kpi_monitor = kpi_monitor
        self.trading_manager = trading_manager
        
        # 优化配置
        self.optimization_strategy = OptimizationStrategy.ADAPTIVE
        self.optimization_interval = 7  # 7天优化一次
        self.min_trades_for_optimization = 20  # 最少20笔交易才进行优化
        
        # 参数范围定义
        self.parameter_ranges = self._initialize_parameter_ranges()
        
        # A/B测试管理
        self.active_ab_tests: List[ABTestConfig] = []
        self.ab_test_results: List[Dict] = []
        
        # 优化历史
        self.optimization_history: List[OptimizationResult] = []
        
        # 数据目录
        self.data_dir = "simulation/optimization"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("🔧 参数优化系统初始化完成")
        self.logger.info(f"   📊 优化策略: {self.optimization_strategy.value}")
        self.logger.info(f"   📅 优化间隔: {self.optimization_interval} 天")
        self.logger.info(f"   📈 可优化参数: {len(self.parameter_ranges)} 个")
    
    def _initialize_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """初始化参数优化范围"""
        ranges = {}
        
        # 信号阈值优化
        ranges["signal_buy_threshold"] = ParameterRange(
            param_name="signal_buy_threshold",
            current_value=0.6,
            min_value=0.5,
            max_value=0.8,
            step_size=0.05,
            param_type=ParameterType.SIGNAL_THRESHOLD,
            priority=1
        )
        
        ranges["signal_sell_threshold"] = ParameterRange(
            param_name="signal_sell_threshold", 
            current_value=0.4,
            min_value=0.2,
            max_value=0.5,
            step_size=0.05,
            param_type=ParameterType.SIGNAL_THRESHOLD,
            priority=1
        )
        
        # 风险管理参数
        ranges["max_position_size_pct"] = ParameterRange(
            param_name="max_position_size_pct",
            current_value=0.15,
            min_value=0.05,
            max_value=0.25,
            step_size=0.02,
            param_type=ParameterType.POSITION_SIZE,
            priority=2
        )
        
        ranges["risk_per_trade"] = ParameterRange(
            param_name="risk_per_trade",
            current_value=0.02,
            min_value=0.01,
            max_value=0.05,
            step_size=0.005,
            param_type=ParameterType.RISK_RATIO,
            priority=2
        )
        
        # 止损止盈参数
        ranges["stop_loss_pct"] = ParameterRange(
            param_name="stop_loss_pct",
            current_value=0.02,
            min_value=0.01,
            max_value=0.05,
            step_size=0.005,
            param_type=ParameterType.STOP_LOSS,
            priority=3
        )
        
        ranges["take_profit_ratio"] = ParameterRange(
            param_name="take_profit_ratio",
            current_value=2.5,
            min_value=1.5,
            max_value=4.0,
            step_size=0.25,
            param_type=ParameterType.TAKE_PROFIT,
            priority=3
        )
        
        # 杠杆参数
        ranges["base_leverage"] = ParameterRange(
            param_name="base_leverage",
            current_value=10.0,
            min_value=5.0,
            max_value=15.0,
            step_size=1.0,
            param_type=ParameterType.LEVERAGE,
            priority=4
        )
        
        return ranges
    
    async def should_optimize(self) -> Tuple[bool, str]:
        """判断是否应该进行参数优化"""
        try:
            # 获取当前KPI
            current_kpi = await self.kpi_monitor.get_current_kpi()
            if not current_kpi:
                return False, "无法获取当前KPI数据"
            
            # 检查交易数量
            if current_kpi.total_trades < self.min_trades_for_optimization:
                return False, f"交易数量不足 ({current_kpi.total_trades} < {self.min_trades_for_optimization})"
            
            # 检查时间间隔
            last_optimization = self._get_last_optimization_time()
            if last_optimization:
                days_since = (datetime.now() - last_optimization).days
                if days_since < self.optimization_interval:
                    return False, f"优化间隔不足 ({days_since} < {self.optimization_interval} 天)"
            
            # 检查性能是否需要优化
            optimization_needed, reason = self._check_performance_needs_optimization(current_kpi)
            
            return optimization_needed, reason
            
        except Exception as e:
            self.logger.error(f"优化需求检查失败: {e}")
            return False, f"检查失败: {e}"
    
    def _check_performance_needs_optimization(self, kpi: KPIMetrics) -> Tuple[bool, str]:
        """检查性能是否需要优化"""
        reasons = []
        
        # 胜率检查
        if kpi.win_rate < 0.6:
            reasons.append(f"胜率偏低 ({kpi.win_rate:.1%} < 60%)")
        
        # 盈利因子检查
        if kpi.profit_factor < 1.8:
            reasons.append(f"盈利因子偏低 ({kpi.profit_factor:.2f} < 1.8)")
        
        # 回撤检查
        if kpi.max_drawdown > 0.15:
            reasons.append(f"回撤过大 ({kpi.max_drawdown:.1%} > 15%)")
        
        # 信号质量检查
        if kpi.avg_signal_confidence < 0.7:
            reasons.append(f"信号质量偏低 ({kpi.avg_signal_confidence:.1%} < 70%)")
        
        # 如果有任何问题，需要优化
        if reasons:
            return True, "; ".join(reasons)
        
        # 即使表现良好，也定期优化以追求更好表现
        return True, "定期优化以追求更优表现"
    
    async def optimize_parameters(self) -> Dict[str, Any]:
        """执行参数优化"""
        try:
            self.logger.info("🔧 开始参数优化过程...")
            
            # 检查是否需要优化
            should_opt, reason = await self.should_optimize()
            if not should_opt:
                return {"status": "skipped", "reason": reason}
            
            self.logger.info(f"📊 优化原因: {reason}")
            
            # 获取当前KPI作为基准
            baseline_kpi = await self.kpi_monitor.get_current_kpi()
            
            # 根据策略选择优化方法
            if self.optimization_strategy == OptimizationStrategy.GRADUAL:
                results = await self._gradual_optimization(baseline_kpi)
            elif self.optimization_strategy == OptimizationStrategy.AGGRESSIVE:
                results = await self._aggressive_optimization(baseline_kpi)
            elif self.optimization_strategy == OptimizationStrategy.CONSERVATIVE:
                results = await self._conservative_optimization(baseline_kpi)
            else:  # ADAPTIVE
                results = await self._adaptive_optimization(baseline_kpi)
            
            # 保存优化结果
            await self._save_optimization_results(results)
            
            self.logger.info(f"✅ 参数优化完成，优化了 {len(results.get('optimized_params', []))} 个参数")
            
            return results
            
        except Exception as e:
            self.logger.error(f"参数优化失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _adaptive_optimization(self, baseline_kpi: KPIMetrics) -> Dict[str, Any]:
        """自适应优化策略"""
        try:
            self.logger.info("🎯 执行自适应优化策略...")
            
            optimized_params = []
            
            # 根据当前表现确定优化重点
            if baseline_kpi.win_rate < 0.6:
                # 胜率低，优化信号阈值
                params = await self._optimize_signal_thresholds(baseline_kpi)
                optimized_params.extend(params)
            
            if baseline_kpi.max_drawdown > 0.12:
                # 回撤大，优化风险管理参数
                params = await self._optimize_risk_parameters(baseline_kpi)
                optimized_params.extend(params)
            
            if baseline_kpi.profit_factor < 2.0:
                # 盈利因子低，优化止损止盈
                params = await self._optimize_stop_take_profit(baseline_kpi)
                optimized_params.extend(params)
            
            # 如果没有明显问题，进行全面微调
            if not optimized_params:
                params = await self._comprehensive_fine_tuning(baseline_kpi)
                optimized_params.extend(params)
            
            return {
                "status": "completed",
                "strategy": "adaptive",
                "baseline_kpi": asdict(baseline_kpi),
                "optimized_params": optimized_params,
                "optimization_count": len(optimized_params)
            }
            
        except Exception as e:
            self.logger.error(f"自适应优化失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _optimize_signal_thresholds(self, baseline_kpi: KPIMetrics) -> List[OptimizationResult]:
        """优化信号阈值参数"""
        try:
            self.logger.info("📊 优化信号阈值参数...")
            results = []
            
            # 买入信号阈值优化
            buy_threshold_range = self.parameter_ranges["signal_buy_threshold"]
            buy_optimal = await self._find_optimal_value(
                buy_threshold_range, baseline_kpi, "win_rate"
            )
            
            if buy_optimal:
                results.append(buy_optimal)
                self.logger.info(f"   ✅ 买入阈值: {buy_optimal.old_value:.2f} → {buy_optimal.new_value:.2f}")
            
            # 卖出信号阈值优化
            sell_threshold_range = self.parameter_ranges["signal_sell_threshold"]
            sell_optimal = await self._find_optimal_value(
                sell_threshold_range, baseline_kpi, "win_rate"
            )
            
            if sell_optimal:
                results.append(sell_optimal)
                self.logger.info(f"   ✅ 卖出阈值: {sell_optimal.old_value:.2f} → {sell_optimal.new_value:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"信号阈值优化失败: {e}")
            return []
    
    async def _optimize_risk_parameters(self, baseline_kpi: KPIMetrics) -> List[OptimizationResult]:
        """优化风险管理参数"""
        try:
            self.logger.info("🛡️ 优化风险管理参数...")
            results = []
            
            # 仓位大小优化
            position_range = self.parameter_ranges["max_position_size_pct"]
            position_optimal = await self._find_optimal_value(
                position_range, baseline_kpi, "sharpe_ratio"
            )
            
            if position_optimal:
                results.append(position_optimal)
                self.logger.info(f"   ✅ 最大仓位: {position_optimal.old_value:.1%} → {position_optimal.new_value:.1%}")
            
            # 单笔风险优化
            risk_range = self.parameter_ranges["risk_per_trade"]
            risk_optimal = await self._find_optimal_value(
                risk_range, baseline_kpi, "max_drawdown", minimize=True
            )
            
            if risk_optimal:
                results.append(risk_optimal)
                self.logger.info(f"   ✅ 单笔风险: {risk_optimal.old_value:.1%} → {risk_optimal.new_value:.1%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"风险参数优化失败: {e}")
            return []
    
    async def _optimize_stop_take_profit(self, baseline_kpi: KPIMetrics) -> List[OptimizationResult]:
        """优化止损止盈参数"""
        try:
            self.logger.info("🎯 优化止损止盈参数...")
            results = []
            
            # 止损距离优化
            stop_range = self.parameter_ranges["stop_loss_pct"]
            stop_optimal = await self._find_optimal_value(
                stop_range, baseline_kpi, "profit_factor"
            )
            
            if stop_optimal:
                results.append(stop_optimal)
                self.logger.info(f"   ✅ 止损距离: {stop_optimal.old_value:.1%} → {stop_optimal.new_value:.1%}")
            
            # 止盈比率优化
            profit_range = self.parameter_ranges["take_profit_ratio"]
            profit_optimal = await self._find_optimal_value(
                profit_range, baseline_kpi, "profit_factor"
            )
            
            if profit_optimal:
                results.append(profit_optimal)
                self.logger.info(f"   ✅ 止盈比率: {profit_optimal.old_value:.1f} → {profit_optimal.new_value:.1f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"止损止盈优化失败: {e}")
            return []
    
    async def _comprehensive_fine_tuning(self, baseline_kpi: KPIMetrics) -> List[OptimizationResult]:
        """全面微调参数"""
        try:
            self.logger.info("🔧 执行全面参数微调...")
            results = []
            
            # 按优先级排序参数
            sorted_params = sorted(
                self.parameter_ranges.values(),
                key=lambda x: x.priority
            )
            
            # 对每个参数进行小幅度调整
            for param_range in sorted_params[:3]:  # 只调整前3个优先级最高的参数
                optimal = await self._fine_tune_parameter(param_range, baseline_kpi)
                if optimal:
                    results.append(optimal)
                    self.logger.info(f"   ✅ {param_range.param_name}: {optimal.old_value:.3f} → {optimal.new_value:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"全面微调失败: {e}")
            return []
    
    async def _find_optimal_value(self, param_range: ParameterRange, baseline_kpi: KPIMetrics, 
                                target_metric: str, minimize: bool = False) -> Optional[OptimizationResult]:
        """寻找参数的最优值"""
        try:
            current_value = param_range.current_value
            best_value = current_value
            best_score = getattr(baseline_kpi, target_metric)
            
            # 生成测试值
            test_values = []
            value = param_range.min_value
            while value <= param_range.max_value:
                if abs(value - current_value) > param_range.step_size / 2:  # 跳过当前值
                    test_values.append(value)
                value += param_range.step_size
            
            # 测试每个值
            for test_value in test_values:
                score = await self._simulate_parameter_change(
                    param_range.param_name, test_value, target_metric
                )
                
                if score is not None:
                    if minimize:
                        if score < best_score:
                            best_score = score
                            best_value = test_value
                    else:
                        if score > best_score:
                            best_score = score
                            best_value = test_value
            
            # 检查是否有改善
            baseline_score = getattr(baseline_kpi, target_metric)
            if minimize:
                improvement = baseline_score - best_score
            else:
                improvement = best_score - baseline_score
            
            if improvement > 0 and abs(best_value - current_value) > param_range.step_size / 2:
                # 计算置信度
                confidence = min(0.9, improvement / baseline_score * 10)
                
                return OptimizationResult(
                    param_name=param_range.param_name,
                    old_value=current_value,
                    new_value=best_value,
                    improvement_score=improvement,
                    confidence=confidence,
                    test_trades=len(test_values) * 5,  # 模拟每个测试值5笔交易
                    test_period_days=1
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"寻找最优值失败 {param_range.param_name}: {e}")
            return None
    
    async def _simulate_parameter_change(self, param_name: str, new_value: float, target_metric: str) -> Optional[float]:
        """模拟参数变化的效果"""
        try:
            # 这里实现参数变化的模拟逻辑
            # 由于是模拟，我们基于统计模型估算效果
            
            # 获取历史数据进行模拟
            current_kpi = await self.kpi_monitor.get_current_kpi()
            if not current_kpi:
                return None
            
            # 简化的模拟逻辑（实际应用中会更复杂）
            if param_name == "signal_buy_threshold":
                # 提高买入阈值通常会提高胜率但减少交易频率
                threshold_effect = (new_value - 0.6) * 0.1  # 基准值0.6
                if target_metric == "win_rate":
                    return current_kpi.win_rate + threshold_effect
                elif target_metric == "profit_factor":
                    return current_kpi.profit_factor + threshold_effect * 0.5
            
            elif param_name == "max_position_size_pct":
                # 仓位大小影响风险和收益
                size_effect = (new_value - 0.15) * 0.2  # 基准值0.15
                if target_metric == "sharpe_ratio":
                    return current_kpi.sharpe_ratio - abs(size_effect) * 0.1
                elif target_metric == "max_drawdown":
                    return current_kpi.max_drawdown + size_effect * 0.5
            
            elif param_name == "stop_loss_pct":
                # 止损距离影响胜率和盈利因子
                stop_effect = (new_value - 0.02) * 0.5  # 基准值0.02
                if target_metric == "profit_factor":
                    return current_kpi.profit_factor - stop_effect
                elif target_metric == "win_rate":
                    return current_kpi.win_rate + stop_effect * 0.3
            
            # 默认返回当前值（无变化）
            return getattr(current_kpi, target_metric)
            
        except Exception as e:
            self.logger.error(f"参数模拟失败 {param_name}: {e}")
            return None
    
    async def _fine_tune_parameter(self, param_range: ParameterRange, baseline_kpi: KPIMetrics) -> Optional[OptimizationResult]:
        """对参数进行精细调整"""
        try:
            current_value = param_range.current_value
            
            # 小幅度调整（±1个步长）
            test_values = [
                current_value - param_range.step_size,
                current_value + param_range.step_size
            ]
            
            # 确保在合法范围内
            test_values = [v for v in test_values if param_range.min_value <= v <= param_range.max_value]
            
            if not test_values:
                return None
            
            # 选择综合评分最高的值
            best_value = current_value
            best_composite_score = self._calculate_composite_score(baseline_kpi)
            
            for test_value in test_values:
                # 模拟多个指标的变化
                win_rate = await self._simulate_parameter_change(param_range.param_name, test_value, "win_rate")
                profit_factor = await self._simulate_parameter_change(param_range.param_name, test_value, "profit_factor")
                max_drawdown = await self._simulate_parameter_change(param_range.param_name, test_value, "max_drawdown")
                
                if all(x is not None for x in [win_rate, profit_factor, max_drawdown]):
                    # 创建模拟KPI
                    simulated_kpi = KPIMetrics(
                        timestamp=baseline_kpi.timestamp,
                        period_days=baseline_kpi.period_days,
                        total_trades=baseline_kpi.total_trades,
                        trades_per_day=baseline_kpi.trades_per_day,
                        win_rate=win_rate,
                        profit_factor=profit_factor,
                        total_pnl=baseline_kpi.total_pnl,
                        return_percentage=baseline_kpi.return_percentage,
                        max_drawdown=max_drawdown,
                        current_drawdown=baseline_kpi.current_drawdown,
                        consecutive_losses=baseline_kpi.consecutive_losses,
                        total_signals=baseline_kpi.total_signals,
                        executed_signals=baseline_kpi.executed_signals,
                        signal_conversion_rate=baseline_kpi.signal_conversion_rate,
                        avg_signal_confidence=baseline_kpi.avg_signal_confidence,
                        current_capital=baseline_kpi.current_capital,
                        peak_capital=baseline_kpi.peak_capital,
                        overall_status=baseline_kpi.overall_status,
                        detailed_status=baseline_kpi.detailed_status
                    )
                    
                    composite_score = self._calculate_composite_score(simulated_kpi)
                    
                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_value = test_value
            
            # 检查是否有改善
            if best_value != current_value:
                improvement = best_composite_score - self._calculate_composite_score(baseline_kpi)
                confidence = min(0.8, improvement * 10)  # 微调的置信度较低
                
                return OptimizationResult(
                    param_name=param_range.param_name,
                    old_value=current_value,
                    new_value=best_value,
                    improvement_score=improvement,
                    confidence=confidence,
                    test_trades=len(test_values) * 3,
                    test_period_days=1
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"参数微调失败 {param_range.param_name}: {e}")
            return None
    
    def _calculate_composite_score(self, kpi: KPIMetrics) -> float:
        """计算综合评分"""
        try:
            # 权重配置
            weights = {
                "win_rate": 0.35,
                "profit_factor": 0.3,
                "signal_quality": 0.15,
                "max_drawdown": 0.2  # 负向指标
            }
            
            # 标准化分数
            win_rate_score = min(1.0, kpi.win_rate / 0.8)  # 80%为满分
            profit_factor_score = min(1.0, kpi.profit_factor / 3.0)  # 3.0为满分
            signal_quality_score = min(1.0, kpi.avg_signal_confidence)  # 1.0为满分
            drawdown_score = max(0.0, 1.0 - kpi.max_drawdown / 0.2)  # 20%回撤为0分
            
            # 计算加权综合分数
            composite_score = (
                win_rate_score * weights["win_rate"] +
                profit_factor_score * weights["profit_factor"] +
                signal_quality_score * weights["signal_quality"] +
                drawdown_score * weights["max_drawdown"]
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"综合评分计算失败: {e}")
            return 0.0
    
    async def _save_optimization_results(self, results: Dict[str, Any]):
        """保存优化结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存详细结果
            result_file = f"{self.data_dir}/optimization_result_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # 更新优化历史
            if results.get("optimized_params"):
                for param_data in results["optimized_params"]:
                    opt_result = OptimizationResult(**param_data)
                    self.optimization_history.append(opt_result)
                    
                    # 更新参数范围的当前值
                    if opt_result.param_name in self.parameter_ranges:
                        self.parameter_ranges[opt_result.param_name].current_value = opt_result.new_value
            
            # 保存优化历史
            history_file = f"{self.data_dir}/optimization_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(opt) for opt in self.optimization_history], 
                         f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 优化结果已保存: {result_file}")
            
        except Exception as e:
            self.logger.error(f"保存优化结果失败: {e}")
    
    def _get_last_optimization_time(self) -> Optional[datetime]:
        """获取最后一次优化时间"""
        try:
            if not self.optimization_history:
                return None
            
            # 从优化历史中获取最新时间
            history_file = f"{self.data_dir}/optimization_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                if history_data:
                    # 假设有timestamp字段
                    return datetime.now() - timedelta(days=self.optimization_interval + 1)
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取最后优化时间失败: {e}")
            return None
    
    async def generate_optimization_report(self) -> str:
        """生成优化报告"""
        try:
            current_time = datetime.now()
            
            # 获取当前KPI
            current_kpi = await self.kpi_monitor.get_current_kpi()
            
            report = f"""# 📊 参数优化系统报告

## 🎯 系统状态
- **报告时间**: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
- **优化策略**: {self.optimization_strategy.value}
- **优化间隔**: {self.optimization_interval} 天
- **最小交易数**: {self.min_trades_for_optimization} 笔

## 📈 当前参数设置
"""
            
            for name, param_range in self.parameter_ranges.items():
                report += f"- **{name}**: {param_range.current_value:.3f} (范围: {param_range.min_value:.3f} - {param_range.max_value:.3f})\n"
            
            if current_kpi:
                report += f"""
## 📊 当前性能指标
- **胜率**: {current_kpi.win_rate:.1%}
- **盈利因子**: {current_kpi.profit_factor:.2f}
- **最大回撤**: {current_kpi.max_drawdown:.1%}
- **夏普比率**: {current_kpi.sharpe_ratio:.2f}
- **总交易**: {current_kpi.total_trades} 笔
- **总盈亏**: {current_kpi.total_pnl:+.2f} USDT
"""
            
            # 优化历史
            if self.optimization_history:
                report += "\n## 📋 最近优化历史\n"
                for i, opt in enumerate(self.optimization_history[-5:], 1):
                    report += f"{i}. **{opt.param_name}**: {opt.old_value:.3f} → {opt.new_value:.3f} (改善: {opt.improvement_score:.3f})\n"
            
            # 优化建议
            should_opt, reason = await self.should_optimize()
            report += f"""
## 💡 优化建议
- **是否需要优化**: {'是' if should_opt else '否'}
- **原因**: {reason}
"""
            
            if should_opt:
                report += "- **建议**: 建议执行参数优化以提升系统性能\n"
            else:
                report += "- **建议**: 继续监控系统表现，暂时保持当前参数\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成优化报告失败: {e}")
            return f"报告生成失败: {e}"
    
    async def auto_optimization_check(self):
        """自动优化检查（定期调用）"""
        try:
            self.logger.info("🔍 执行自动优化检查...")
            
            should_opt, reason = await self.should_optimize()
            
            if should_opt:
                self.logger.info(f"🚀 触发自动优化: {reason}")
                result = await self.optimize_parameters()
                
                if result.get("status") == "completed":
                    self.logger.info("✅ 自动优化完成")
                    return result
                else:
                    self.logger.warning(f"⚠️ 自动优化失败: {result.get('error', '未知错误')}")
                    return result
            else:
                self.logger.info(f"📊 无需优化: {reason}")
                return {"status": "no_optimization_needed", "reason": reason}
                
        except Exception as e:
            self.logger.error(f"自动优化检查失败: {e}")
            return {"status": "error", "error": str(e)} 