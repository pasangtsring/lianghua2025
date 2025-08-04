"""
🆕 任务4.3: 形态权重优化器
基于回测结果和实时表现动态调整形态权重和参数

主要功能：
1. 收集形态检测的表现数据
2. 基于成功率、风险回报比等指标计算最优权重
3. 动态调整检测参数
4. A/B测试框架支持
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

from utils.logger import get_logger

@dataclass
class PatternPerformance:
    """形态表现数据结构"""
    pattern_type: str
    total_signals: int = 0
    successful_signals: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    avg_holding_time: float = 0.0  # 小时
    success_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    def update_performance(self, success: bool, profit_loss: float, 
                         holding_time: float, confidence: float):
        """更新表现数据"""
        self.total_signals += 1
        self.confidence_scores.append(confidence)
        self.timestamps.append(datetime.now())
        
        if success:
            self.successful_signals += 1
            self.total_profit += abs(profit_loss)
            self.max_profit = max(self.max_profit, abs(profit_loss))
        else:
            self.total_loss += abs(profit_loss)
            self.max_loss = max(self.max_loss, abs(profit_loss))
        
        # 更新平均持仓时间
        total_time = self.avg_holding_time * (self.total_signals - 1) + holding_time
        self.avg_holding_time = total_time / self.total_signals
        
        # 重新计算指标
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """重新计算性能指标"""
        if self.total_signals > 0:
            self.success_rate = self.successful_signals / self.total_signals
        
        if self.total_loss > 0:
            self.profit_loss_ratio = self.total_profit / self.total_loss
        
        # 计算夏普比率（简化版）
        if len(self.confidence_scores) > 1:
            returns = np.array(self.confidence_scores)
            self.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)

@dataclass
class OptimizationResult:
    """优化结果数据结构"""
    pattern_weights: Dict[str, float]
    parameter_adjustments: Dict[str, Any]
    expected_improvement: float
    confidence_level: float
    optimization_timestamp: datetime = field(default_factory=datetime.now)

class PatternWeightOptimizer:
    """
    🔧 形态权重优化器
    
    基于历史表现数据动态调整形态权重和参数
    支持A/B测试和渐进式优化
    """
    
    def __init__(self, data_file: str = "pattern_performance_data.json"):
        self.logger = get_logger(__name__)
        self.data_file = Path(data_file)
        self.performance_data: Dict[str, PatternPerformance] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # 优化参数
        self.min_samples_for_optimization = 20  # 最少样本数
        self.optimization_interval_hours = 24   # 优化间隔
        self.weight_adjustment_rate = 0.1       # 权重调整速率
        self.confidence_threshold = 0.7         # 置信度阈值
        
        # 加载历史数据
        self._load_performance_data()
        
        self.logger.info("🔧 形态权重优化器初始化完成")
    
    def record_pattern_result(self, pattern_type: str, success: bool, 
                            profit_loss: float, holding_time: float, 
                            confidence: float):
        """
        记录形态检测结果
        
        Args:
            pattern_type: 形态类型
            success: 是否成功
            profit_loss: 盈亏金额
            holding_time: 持仓时间（小时）
            confidence: 信号置信度
        """
        try:
            if pattern_type not in self.performance_data:
                self.performance_data[pattern_type] = PatternPerformance(pattern_type=pattern_type)
            
            self.performance_data[pattern_type].update_performance(
                success, profit_loss, holding_time, confidence
            )
            
            self.logger.info(f"📊 记录{pattern_type}形态结果: 成功={success}, "
                           f"盈亏={profit_loss:.2f}, 置信度={confidence:.2f}")
            
            # 异步保存数据
            self._save_performance_data()
            
        except Exception as e:
            self.logger.error(f"❌ 记录形态结果失败: {e}")
    
    def should_optimize(self) -> bool:
        """
        判断是否应该进行优化
        
        Returns:
            是否应该优化
        """
        try:
            # 检查是否有足够的数据
            total_samples = sum(perf.total_signals for perf in self.performance_data.values())
            if total_samples < self.min_samples_for_optimization:
                return False
            
            # 检查时间间隔
            if self.optimization_history:
                last_optimization = self.optimization_history[-1].optimization_timestamp
                if datetime.now() - last_optimization < timedelta(hours=self.optimization_interval_hours):
                    return False
            
            # 检查是否有需要优化的形态
            patterns_needing_optimization = []
            for pattern_type, perf in self.performance_data.items():
                if (perf.total_signals >= 10 and 
                    (perf.success_rate < 0.4 or perf.profit_loss_ratio < 1.0)):
                    patterns_needing_optimization.append(pattern_type)
            
            return len(patterns_needing_optimization) > 0
            
        except Exception as e:
            self.logger.error(f"❌ 判断优化条件失败: {e}")
            return False
    
    def optimize_weights(self) -> OptimizationResult:
        """
        基于表现数据优化形态权重
        
        Returns:
            优化结果
        """
        try:
            self.logger.info("🔄 开始形态权重优化...")
            
            # 计算每个形态的综合得分
            pattern_scores = self._calculate_pattern_scores()
            
            # 基于得分计算新权重
            new_weights = self._calculate_optimal_weights(pattern_scores)
            
            # 计算参数调整建议
            parameter_adjustments = self._suggest_parameter_adjustments()
            
            # 评估预期改进
            expected_improvement = self._estimate_improvement(new_weights)
            
            # 计算置信度
            confidence_level = self._calculate_confidence(pattern_scores)
            
            # 创建优化结果
            optimization_result = OptimizationResult(
                pattern_weights=new_weights,
                parameter_adjustments=parameter_adjustments,
                expected_improvement=expected_improvement,
                confidence_level=confidence_level
            )
            
            # 记录优化历史
            self.optimization_history.append(optimization_result)
            
            self.logger.info(f"✅ 权重优化完成: 预期改进={expected_improvement:.2f}%, "
                           f"置信度={confidence_level:.2f}")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ 权重优化失败: {e}")
            return OptimizationResult(
                pattern_weights={},
                parameter_adjustments={},
                expected_improvement=0.0,
                confidence_level=0.0
            )
    
    def _calculate_pattern_scores(self) -> Dict[str, float]:
        """计算各形态的综合得分"""
        scores = {}
        
        for pattern_type, perf in self.performance_data.items():
            if perf.total_signals < 5:  # 样本太少，使用默认得分
                scores[pattern_type] = 0.5
                continue
            
            # 综合评分公式
            success_score = perf.success_rate * 0.4          # 成功率 40%
            profit_score = min(perf.profit_loss_ratio / 3.0, 1.0) * 0.3  # 盈亏比 30%
            sharpe_score = max(0, min(perf.sharpe_ratio / 2.0, 1.0)) * 0.2  # 夏普比率 20%
            frequency_score = min(perf.total_signals / 50.0, 1.0) * 0.1    # 信号频率 10%
            
            total_score = success_score + profit_score + sharpe_score + frequency_score
            scores[pattern_type] = max(0.1, min(1.0, total_score))  # 限制在0.1-1.0范围
            
            self.logger.debug(f"📊 {pattern_type}综合得分: {total_score:.3f} "
                            f"(成功率:{success_score:.3f}, 盈亏比:{profit_score:.3f})")
        
        return scores
    
    def _calculate_optimal_weights(self, pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """基于得分计算最优权重"""
        if not pattern_scores:
            return {}
        
        # 归一化得分
        total_score = sum(pattern_scores.values())
        if total_score == 0:
            # 均匀分配权重
            equal_weight = 1.0 / len(pattern_scores)
            return {pattern: equal_weight for pattern in pattern_scores.keys()}
        
        # 基于得分的权重分配
        base_weights = {
            pattern: score / total_score
            for pattern, score in pattern_scores.items()
        }
        
        # 应用调整速率，避免剧烈变化
        adjusted_weights = {}
        default_weight = 0.6  # 默认权重
        
        for pattern, new_weight in base_weights.items():
            if pattern in ['double_bottom_bull', 'double_top_bear']:
                # 对新增的双重形态使用保守调整
                current_weight = default_weight
                adjusted_weight = current_weight + (new_weight - current_weight) * self.weight_adjustment_rate
            else:
                # 对现有形态使用更积极的调整
                current_weight = default_weight
                adjusted_weight = current_weight + (new_weight - current_weight) * (self.weight_adjustment_rate * 1.5)
            
            adjusted_weights[pattern] = max(0.3, min(0.9, adjusted_weight))  # 限制权重范围
        
        return adjusted_weights
    
    def _suggest_parameter_adjustments(self) -> Dict[str, Any]:
        """建议参数调整"""
        adjustments = {}
        
        for pattern_type, perf in self.performance_data.items():
            if pattern_type in ['double_bottom_bull', 'double_top_bear'] and perf.total_signals >= 10:
                suggestions = {}
                
                # 基于成功率调整置信度阈值
                if perf.success_rate < 0.4:
                    suggestions['min_confidence'] = min(0.7, perf.confidence_scores[-10:] 
                                                      if perf.confidence_scores else 0.6)
                elif perf.success_rate > 0.7:
                    suggestions['min_confidence'] = max(0.4, np.percentile(perf.confidence_scores, 30)
                                                      if perf.confidence_scores else 0.5)
                
                # 基于信号频率调整相似度阈值
                if perf.total_signals < 5:  # 信号太少
                    suggestions['similarity_thresh'] = 0.12  # 放宽阈值
                elif perf.total_signals > 20:  # 信号太多
                    suggestions['similarity_thresh'] = 0.08  # 收紧阈值
                
                if suggestions:
                    adjustments[pattern_type] = suggestions
        
        return adjustments
    
    def _estimate_improvement(self, new_weights: Dict[str, float]) -> float:
        """估算预期改进幅度"""
        if not self.performance_data:
            return 0.0
        
        # 简化的改进估算：基于权重变化和历史表现
        total_improvement = 0.0
        weight_count = 0
        
        for pattern_type, new_weight in new_weights.items():
            if pattern_type in self.performance_data:
                perf = self.performance_data[pattern_type]
                if perf.total_signals > 0:
                    # 基于成功率和权重变化估算改进
                    success_factor = (perf.success_rate - 0.5) * 2  # -1 到 1
                    weight_change = new_weight - 0.6  # 假设原权重为0.6
                    improvement = success_factor * weight_change * 100
                    total_improvement += improvement
                    weight_count += 1
        
        return total_improvement / max(1, weight_count)
    
    def _calculate_confidence(self, pattern_scores: Dict[str, float]) -> float:
        """计算优化置信度"""
        if not pattern_scores:
            return 0.0
        
        # 基于样本数量和得分分布计算置信度
        total_samples = sum(perf.total_signals for perf in self.performance_data.values())
        sample_confidence = min(1.0, total_samples / 100.0)  # 100个样本达到满置信度
        
        # 基于得分稳定性
        scores = list(pattern_scores.values())
        score_stability = 1.0 - (np.std(scores) if len(scores) > 1 else 0.0)
        
        return min(1.0, sample_confidence * 0.7 + score_stability * 0.3)
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前推荐权重"""
        if self.optimization_history:
            return self.optimization_history[-1].pattern_weights
        else:
            # 返回默认权重
            return {
                "double_bottom_bull": 0.75,
                "double_top_bear": 0.75,
                "engulfing_bull": 0.6,
                "engulfing_bear": 0.6,
                "head_shoulder_bear": 0.65,
                "head_shoulder_bull": 0.65,
                "convergence_triangle_bull": 0.55,
                "convergence_triangle_bear": 0.55
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'total_patterns': len(self.performance_data),
            'total_signals': sum(perf.total_signals for perf in self.performance_data.values()),
            'overall_success_rate': 0.0,
            'patterns': {}
        }
        
        total_successful = sum(perf.successful_signals for perf in self.performance_data.values())
        total_signals = summary['total_signals']
        
        if total_signals > 0:
            summary['overall_success_rate'] = total_successful / total_signals
        
        for pattern_type, perf in self.performance_data.items():
            summary['patterns'][pattern_type] = {
                'signals': perf.total_signals,
                'success_rate': perf.success_rate,
                'profit_loss_ratio': perf.profit_loss_ratio,
                'avg_confidence': np.mean(perf.confidence_scores) if perf.confidence_scores else 0.0
            }
        
        return summary
    
    def _load_performance_data(self):
        """加载历史表现数据"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 重建PatternPerformance对象
                for pattern_type, perf_data in data.get('performance_data', {}).items():
                    perf = PatternPerformance(pattern_type=pattern_type)
                    perf.__dict__.update(perf_data)
                    # 转换时间戳
                    perf.timestamps = [datetime.fromisoformat(ts) for ts in perf_data.get('timestamps', [])]
                    self.performance_data[pattern_type] = perf
                
                self.logger.info(f"✅ 加载历史数据: {len(self.performance_data)}个形态")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 加载历史数据失败: {e}")
            self.performance_data = {}
    
    def _save_performance_data(self):
        """保存表现数据"""
        try:
            data = {
                'performance_data': {},
                'last_updated': datetime.now().isoformat()
            }
            
            for pattern_type, perf in self.performance_data.items():
                perf_dict = perf.__dict__.copy()
                # 转换时间戳为字符串
                perf_dict['timestamps'] = [ts.isoformat() for ts in perf.timestamps]
                data['performance_data'][pattern_type] = perf_dict
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"❌ 保存表现数据失败: {e}")


# 导出的优化器实例
pattern_optimizer = PatternWeightOptimizer() 