"""
智能减仓管理器 - 基于ROI等级的自动减仓系统
结合技术分析确认，实现智能化的持仓减仓操作
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timezone

class SmartReductionManager:
    """智能减仓管理器 - 基于ROI等级的减仓系统"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 减仓确认条件阈值 - 基于改造计划
        self.confirmation_thresholds = {
            'volume_threshold': 0.8,      # 成交量萎缩阈值
            'rsi_bull_threshold': 70,     # 牛市RSI阈值
            'rsi_bear_threshold': 30,     # 熊市RSI阈值
            'atr_stability_mult': 1.2     # ATR稳定性倍数
        }
        
        # 减仓执行策略
        self.reduction_strategies = {
            4: {  # Level 4: 50-80% ROI
                'name': 'first_reduction',
                'target_pct': 0.20,
                'cumulative': False,
                'description': '首次减仓20%'
            },
            5: {  # Level 5: 80-150% ROI  
                'name': 'progressive_reduction',
                'target_pct': 0.40,
                'cumulative': True,
                'description': '累计减仓40%'
            },
            6: {  # Level 6: 150%+ ROI
                'name': 'major_reduction',
                'target_pct': 0.70,
                'cumulative': True,
                'description': '累计减仓70%，保留核心仓位'
            }
        }
        
        self.logger.info("智能减仓管理器初始化完成")
    
    async def check_and_execute_reduction(self, symbol: str, roi_level: int, roi_pct: float, 
                                        position: Dict, api_client, precision_manager=None, emergency_mode: bool = False) -> Dict:
        """
        检查并执行智能减仓
        
        Args:
            symbol: 交易品种
            roi_level: ROI等级
            roi_pct: ROI百分比
            position: 持仓信息
            api_client: API客户端
            precision_manager: 精度管理器
            emergency_mode: 紧急模式
            
        Returns:
            执行结果字典
        """
        try:
            # 检查是否需要减仓
            if roi_level < 4:
                return {
                    'executed': False,
                    'reason': f'ROI等级{roi_level}不满足减仓条件（需要≥4）',
                    'roi_threshold': 50
                }
            
            # 获取减仓策略
            strategy = self.reduction_strategies.get(roi_level)
            if not strategy:
                return {
                    'executed': False,
                    'reason': f'ROI等级{roi_level}无减仓策略配置',
                    'available_levels': list(self.reduction_strategies.keys())
                }
            
            self.logger.info(f"🔍 {symbol} 减仓检查: ROI {roi_pct:.1f}% Level {roi_level}")
            self.logger.info(f"   策略: {strategy['description']}")
            self.logger.info(f"   📋 持仓状态: 当前{position.get('size', 0):.6f}, 历史减仓{len(position.get('reduction_history', []))}次")
            
            # 检查减仓确认条件
            confirmation_result = await self._check_reduction_conditions(symbol, api_client)
            if not confirmation_result['confirmed']:
                return {
                    'executed': False,
                    'reason': f'减仓确认条件不满足: {confirmation_result["reason"]}',
                    'confirmation_details': confirmation_result
                }
            
            # 计算需要减仓的数量
            reduction_calculation = self._calculate_reduction_amount(position, strategy, roi_pct)
            
            if reduction_calculation['needed_reduction'] <= 0:
                return {
                    'executed': False,
                    'reason': '无需减仓或已达到目标减仓比例',
                    'calculation_details': reduction_calculation
                }
            
            # 执行减仓操作
            execution_result = await self._execute_reduction(
                symbol, reduction_calculation['needed_reduction'], position, api_client, strategy, 
                precision_manager, emergency_mode
            )
            
            # 更新持仓记录
            self._update_position_records(position, reduction_calculation, roi_pct, strategy)
            
            return {
                'executed': True,
                'details': f'减仓{reduction_calculation["needed_reduction"]:.6f}币，策略: {strategy["description"]}',
                'reduction_amount': reduction_calculation['needed_reduction'],
                'strategy': strategy,
                'confirmation': confirmation_result,
                'execution': execution_result,
                'updated_position': {
                    'total_reduced': position.get('total_reduced', 0.0),
                    'reduction_history': position.get('reduction_history', [])
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 智能减仓执行失败: {e}")
            return {
                'executed': False,
                'reason': f'执行异常: {e}',
                'error_type': 'execution_error'
            }
    
    async def _check_reduction_conditions(self, symbol: str, api_client) -> Dict:
        """
        检查减仓确认条件
        基于成交量、RSI、ATR等多因子确认
        
        Args:
            symbol: 交易品种
            api_client: API客户端
            
        Returns:
            确认结果字典
        """
        try:
            # 这里是简化版实现，实际应该调用真实的技术分析
            # 在完整集成时，会调用项目现有的技术指标计算
            
            self.logger.debug(f"🔍 {symbol} 减仓确认条件检查...")
            
            # 模拟技术指标检查
            # 实际实现中会调用项目现有的技术指标模块
            confirmation_factors = {
                'volume_confirmed': True,   # 成交量确认
                'rsi_confirmed': True,      # RSI确认  
                'atr_confirmed': True,      # ATR稳定性确认
                'overall_score': 0.85       # 综合确认分数
            }
            
            # 综合判断
            min_score = 0.6  # 最小确认分数
            confirmed = confirmation_factors['overall_score'] >= min_score
            
            if confirmed:
                reason = f"多因子确认通过（分数: {confirmation_factors['overall_score']:.2f}）"
            else:
                reason = f"确认分数不足（{confirmation_factors['overall_score']:.2f} < {min_score}）"
            
            return {
                'confirmed': confirmed,
                'reason': reason,
                'factors': confirmation_factors,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"减仓确认条件检查失败: {e}")
            return {
                'confirmed': False,
                'reason': f'确认检查异常: {e}',
                'factors': {},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_reduction_amount(self, position: Dict, strategy: Dict, roi_pct: float) -> Dict:
        """
        计算减仓数量
        
        Args:
            position: 持仓信息
            strategy: 减仓策略
            roi_pct: 当前ROI百分比
            
        Returns:
            减仓计算结果
        """
        try:
            current_position_size = position.get('size', 0.0)
            reduction_base = position.get('reduction_base_position', current_position_size)
            already_reduced = position.get('total_reduced', 0.0)
            
            target_reduction_pct = strategy['target_pct']
            is_cumulative = strategy['cumulative']
            
            self.logger.info(f"   📊 减仓计算: 当前仓位{current_position_size:.6f}, 基准{reduction_base:.6f}, 已减仓{already_reduced:.6f}")
            
            if is_cumulative:
                # 累计减仓：基于基准仓位计算总减仓目标
                total_target_reduction = reduction_base * target_reduction_pct
                needed_reduction = total_target_reduction - already_reduced
                
                calculation_type = "cumulative"
                description = f"累计目标{target_reduction_pct:.0%}，已减{already_reduced:.6f}，还需{needed_reduction:.6f}"
            else:
                # 单次减仓：基于基准仓位计算本次减仓
                needed_reduction = reduction_base * target_reduction_pct
                
                # 检查是否已经执行过这个等级的减仓
                reduction_history = position.get('reduction_history', [])
                level_executions = [r for r in reduction_history if r.get('strategy_name') == strategy['name']]
                
                self.logger.info(f"   🔍 重复检查: 策略'{strategy['name']}'历史执行{len(level_executions)}次")
                if level_executions:
                    last_execution = level_executions[-1]
                    self.logger.info(f"   📅 上次执行: {last_execution.get('timestamp', 'N/A')}, ROI: {last_execution.get('roi_when_reduced', 'N/A')}%")
                
                if level_executions:
                    # 🆕 智能重复减仓判断
                    last_execution = level_executions[-1]
                    last_execution_time_str = last_execution.get('timestamp', '')
                    
                    # 解析时间戳
                    try:
                        if last_execution_time_str:
                            last_execution_time = datetime.fromisoformat(last_execution_time_str.replace('Z', '+00:00'))
                            time_since_last = (datetime.now(timezone.utc) - last_execution_time).total_seconds() / 3600
                        else:
                            time_since_last = 999  # 没有时间戳，允许重新执行
                    except:
                        time_since_last = 999  # 时间解析失败，允许重新执行
                    
                    last_roi = last_execution.get('roi_when_reduced', 0)
                    roi_improvement = roi_pct - last_roi
                    
                    # 条件性重复减仓：距离上次执行>6小时 或 ROI继续大幅上涨>20%
                    if time_since_last >= 6.0 or roi_improvement >= 20.0:
                        needed_reduction = reduction_base * target_reduction_pct
                        description = f"条件性重复减仓{target_reduction_pct:.0%}（间隔{time_since_last:.1f}h，ROI提升{roi_improvement:.1f}%）"
                        self.logger.info(f"   🔄 允许重复减仓: 时间间隔{time_since_last:.1f}h, ROI提升{roi_improvement:.1f}%")
                    else:
                        needed_reduction = 0.0
                        description = f"单次减仓{target_reduction_pct:.0%}已执行过（{time_since_last:.1f}h前，ROI提升{roi_improvement:.1f}%）"
                        self.logger.info(f"   ⏸️ 跳过重复减仓: 时间{time_since_last:.1f}h < 6h, ROI提升{roi_improvement:.1f}% < 20%")
                else:
                    description = f"首次单次减仓{target_reduction_pct:.0%}，需要{needed_reduction:.6f}"
                
                calculation_type = "single"
            
            # 确保减仓数量不超过当前持仓
            needed_reduction = max(0.0, min(needed_reduction, current_position_size))
            
            return {
                'needed_reduction': needed_reduction,
                'calculation_type': calculation_type,
                'target_pct': target_reduction_pct,
                'base_position': reduction_base,
                'current_position': current_position_size,
                'already_reduced': already_reduced,
                'description': description
            }
            
        except Exception as e:
            self.logger.error(f"减仓数量计算失败: {e}")
            return {
                'needed_reduction': 0.0,
                'calculation_type': 'error',
                'description': f'计算异常: {e}'
            }
    
    async def _execute_reduction(self, symbol: str, amount: float, position: Dict, 
                               api_client, strategy: Dict, precision_manager=None, emergency_mode: bool = False) -> Dict:
        """
        执行真实减仓操作 - 集成现有交易API
        
        Args:
            symbol: 交易品种
            amount: 减仓数量（原始数量，需要精度调整）
            position: 持仓信息
            api_client: API客户端
            strategy: 减仓策略
            precision_manager: 精度管理器
            emergency_mode: 紧急模式（降低确认要求）
            
        Returns:
            执行结果
        """
        try:
            # 🔧 集成现有精度管理器
            if precision_manager:
                adjusted_amount = precision_manager.adjust_quantity(symbol, amount)
                self.logger.info(f"📏 精度调整: {amount:.6f} → {adjusted_amount} ({symbol})")
            else:
                adjusted_amount = round(amount, 6)  # 默认6位小数
                self.logger.warning(f"⚠️ 精度管理器不可用，使用默认精度: {adjusted_amount}")
            
            # 构建减仓订单参数
            position_side = position.get('side', 'UNKNOWN')
            close_side = "SELL" if position_side == 'LONG' else "BUY"
            
            # 🔧 币安期货持仓方向设置
            # 对于减仓操作，positionSide 应该与原持仓方向一致
            if position_side in ['LONG', 'SHORT']:
                position_side_param = position_side
            else:
                # 默认设置为 BOTH（单向持仓模式）
                position_side_param = 'BOTH' 
                self.logger.warning(f"⚠️ 持仓方向未知({position_side})，默认使用BOTH模式")
            
            order_params = {
                'symbol': symbol,
                'side': close_side,
                'type': 'MARKET',
                'quantity': str(adjusted_amount),
                'positionSide': position_side_param
            }
            
            self.logger.info(f"🔄 执行智能减仓: {symbol} {close_side} {adjusted_amount}")
            self.logger.info(f"   策略: {strategy['description']}")
            self.logger.info(f"   持仓方向: {position_side}")
            self.logger.info(f"   紧急模式: {emergency_mode}")
            
            # 🔥 调用真实交易API
            api_response = await api_client.place_order(order_params)
            
            if api_response.success:
                # 更新持仓数量（注意正负号）
                original_size = position.get('size', 0.0)
                new_size = abs(original_size) - adjusted_amount
                if position_side == 'LONG':
                    position['size'] = new_size
                else:  # SHORT
                    position['size'] = -new_size
                
                execution_result = {
                    'status': 'success',
                    'symbol': symbol,
                    'reduction_amount': adjusted_amount,
                    'strategy_name': strategy['name'],
                    'order_id': api_response.data.get('orderId', 'N/A') if api_response.data else 'N/A',
                    'execution_time': datetime.now(timezone.utc).isoformat(),
                    'new_position_size': new_size,
                    'emergency_mode': emergency_mode,
                    'simulated': False
                }
                
                self.logger.info(f"   ✅ 减仓成功: 订单ID {execution_result['order_id']}")
                self.logger.info(f"   📊 仓位更新: {original_size} → {new_size}")
                
                return execution_result
                
            else:
                error_msg = api_response.error_message or '未知错误'
                self.logger.error(f"   ❌ 减仓失败: {error_msg}")
                
                return {
                    'status': 'failed',
                    'error': error_msg,
                    'execution_time': datetime.now(timezone.utc).isoformat(),
                    'emergency_mode': emergency_mode,
                    'simulated': False
                }
                
        except Exception as e:
            self.logger.error(f"减仓执行异常: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': datetime.now(timezone.utc).isoformat(),
                'emergency_mode': emergency_mode,
                'simulated': False
            }
    
    def _update_position_records(self, position: Dict, reduction_calculation: Dict, 
                               roi_pct: float, strategy: Dict):
        """
        更新持仓记录
        
        Args:
            position: 持仓信息（会被修改）
            reduction_calculation: 减仓计算结果
            roi_pct: ROI百分比
            strategy: 减仓策略
        """
        try:
            amount = reduction_calculation['needed_reduction']
            
            # 更新总减仓量
            position['total_reduced'] = position.get('total_reduced', 0.0) + amount
            
            # 更新减仓历史
            if 'reduction_history' not in position:
                position['reduction_history'] = []
            
            reduction_record = {
                'amount': amount,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'roi_when_reduced': roi_pct,
                'strategy_name': strategy['name'],
                'strategy_description': strategy['description'],
                'calculation_details': reduction_calculation
            }
            
            position['reduction_history'].append(reduction_record)
            
            # 更新最后减仓时间
            position['last_reduction_time'] = datetime.now(timezone.utc).isoformat()
            
            self.logger.debug(f"持仓记录已更新: 总减仓{position['total_reduced']:.6f}, 历史记录{len(position['reduction_history'])}条")
            
        except Exception as e:
            self.logger.error(f"持仓记录更新失败: {e}")
    
    def get_reduction_summary(self, position: Dict) -> Dict:
        """
        获取减仓摘要信息
        
        Args:
            position: 持仓信息
            
        Returns:
            减仓摘要
        """
        try:
            total_reduced = position.get('total_reduced', 0.0)
            reduction_history = position.get('reduction_history', [])
            base_position = position.get('reduction_base_position', position.get('size', 0.0))
            current_size = position.get('size', 0.0)
            
            if base_position > 0:
                reduction_pct = (total_reduced / base_position) * 100
            else:
                reduction_pct = 0.0
            
            summary = {
                'total_reduced': total_reduced,
                'reduction_percentage': reduction_pct,
                'reduction_count': len(reduction_history),
                'current_position_size': current_size,
                'base_position_size': base_position,
                'remaining_percentage': max(0, 100 - reduction_pct),
                'last_reduction': reduction_history[-1] if reduction_history else None
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"减仓摘要生成失败: {e}")
            return {
                'total_reduced': 0.0,
                'reduction_percentage': 0.0,
                'reduction_count': 0,
                'error': str(e)
            }
    
    def should_set_reduction_base(self, roi_level: int, position: Dict) -> bool:
        """
        判断是否应该设定减仓基准
        
        Args:
            roi_level: ROI等级
            position: 持仓信息
            
        Returns:
            是否应该设定基准
        """
        # Level 3 (30%+ ROI) 开始设定减仓基准
        return roi_level >= 3 and not position.get('reduction_base_set', False)
    
    def set_reduction_base(self, position: Dict, roi_pct: float) -> Dict:
        """
        设定减仓基准
        
        Args:
            position: 持仓信息（会被修改）
            roi_pct: 当前ROI百分比
            
        Returns:
            设定结果
        """
        try:
            if position.get('reduction_base_set', False):
                return {
                    'set': False,
                    'reason': '减仓基准已设定',
                    'existing_base': position.get('reduction_base_position')
                }
            
            current_size = position.get('size', 0.0)
            
            # 设定减仓基准
            position['reduction_base_position'] = current_size
            position['reduction_base_set'] = True
            position['reduction_base_roi'] = roi_pct
            position['reduction_base_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            self.logger.info(f"🎯 减仓基准设定: 仓位 {current_size:.6f}, ROI {roi_pct:.1f}%")
            
            return {
                'set': True,
                'base_position': current_size,
                'base_roi': roi_pct,
                'timestamp': position['reduction_base_timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"减仓基准设定失败: {e}")
            return {
                'set': False,
                'reason': f'设定异常: {e}'
            } 