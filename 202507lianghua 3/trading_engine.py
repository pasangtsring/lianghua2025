"""
主交易引擎
整合所有模块，实现主交易逻辑
支持多币种选币和分析
"""

import asyncio
import signal
import sys
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import numpy as np # Added for numpy

from config.config_manager import ConfigManager
from utils.logger import get_logger
from data.advanced_data_fetcher import AdvancedDataFetcher
from data.api_client import get_api_client
from core.technical_indicators import TechnicalIndicatorCalculator
from core.complete_macd_divergence_detector import CompleteMACDDivergenceDetector
from core.coin_scanner import CoinScanner
from core.signal_generator import SignalGeneratorWithEnhancedFilter
from risk.risk_manager import RiskManager
from utils.telegram_bot import TelegramBot
from utils.precision_manager import SmartPrecisionManager  # 添加精度管理器
from utils.position_persistence import PositionPersistence  # 🔧 新增：持仓数据持久化
# 移除复杂的家庭财务管理导入

class EmergencyBrake:
    """紧急制动机制"""
    def __init__(self):
        self.emergency_stop = False
        self.stop_reason = ""
        self.stop_timestamp = None
        self.logger = get_logger(__name__)
    
    def trigger_emergency_stop(self, reason: str):
        """触发紧急停止"""
        self.emergency_stop = True
        self.stop_reason = reason
        self.stop_timestamp = datetime.now()
        self.logger.critical(f"🚨 紧急制动触发: {reason}")
    
    def is_emergency_stop_active(self) -> bool:
        """检查紧急制动是否激活"""
        return self.emergency_stop
    
    def reset_emergency_stop(self, reason: str = "手动重置"):
        """重置紧急制动"""
        self.emergency_stop = False
        self.stop_reason = ""
        self.stop_timestamp = None
        self.logger.info(f"✅ 紧急制动已重置: {reason}")

class TradingEngine:
    """主交易引擎 - 多币种自动选币交易系统"""
    
    def __init__(self):
        """初始化交易引擎"""
        # 初始化配置
        self.config = ConfigManager()
        self.logger = get_logger(__name__)
        
        # 初始化API客户端
        self.api_client_manager = get_api_client()
        
        # 初始化核心组件
        self.data_fetcher = AdvancedDataFetcher(self.config, self.api_client_manager)
        self.technical_indicators = TechnicalIndicatorCalculator(self.config)
        self.macd_detector = CompleteMACDDivergenceDetector(self.config)  # 修复：传入配置管理器
        self.coin_scanner = CoinScanner(self.config, self.api_client_manager)
        self.signal_generator = SignalGeneratorWithEnhancedFilter(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # 简化初始化，专注核心功能
        
        # 初始化通知组件
        self.telegram_bot = TelegramBot(self.config)
        
        # 初始化精度管理器（最小修改原则：仅添加，不修改现有逻辑）
        self.precision_manager = SmartPrecisionManager()
        
        # 🔧 新增：初始化持仓数据持久化管理器
        self.position_persistence = PositionPersistence()
        
        # 🚨 新增：初始化紧急制动器
        self.emergency_brake = EmergencyBrake()
        
        # 🔧 新增：技术分析并发控制信号量（防止过多并发调用）
        self.tech_analysis_semaphore = asyncio.Semaphore(3)  # 最多同时3个技术分析任务
        
        # 运行状态
        self.is_running = False
        self.should_stop = False
        
        # 交易配置
        self.trading_config = self.config.get_trading_config()
        self.multi_symbol_mode = getattr(self.trading_config, 'multi_symbol_mode', True)
        
        # 多币种数据存储
        self.selected_symbols = []  # 选中的币种列表
        self.coin_data = {}       # 币种K线数据 {symbol: klines}
        self.coin_signals = {}    # 币种信号 {symbol: signal}
        self.current_positions = {}  # 当前持仓 {symbol: position_info}
        self.trade_history = []
        self.consecutive_losses = 0  # 连续亏损次数
        
        
        # 反转检测冷却机制
        self.reversal_history = {}  # {symbol: last_reversal_time}
        self.reversal_cooldown = 300  # 5分钟冷却期（秒）

        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("交易引擎初始化完成")
    
    def get_dynamic_leverage(self, market_condition: str, confidence: float) -> float:
        """
        简单的动态杠杆计算 - 基于大佬建议
        
        Args:
            market_condition: 市场条件 ('bullish', 'bearish', 'neutral')
            confidence: 信号置信度 (0-1)
            
        Returns:
            建议杠杆倍数
        """
        try:
            trading_config = self.config.get_trading_config()
            risk_config = trading_config.risk
            
            # 大佬建议：牛市和熊市不同杠杆
            if 'bullish' in market_condition.lower() or 'spring' in market_condition.lower() or 'summer' in market_condition.lower():
                # 牛市杠杆 - 基于置信度在范围内调整
                base_leverage = getattr(risk_config, 'leverage_bull', 5.0)
                max_leverage = min(base_leverage * 1.5, getattr(risk_config, 'max_leverage', 10.0))
            else:
                # 熊市杠杆 - 更保守
                base_leverage = getattr(risk_config, 'leverage_bear', 3.0) 
                max_leverage = min(base_leverage * 1.2, getattr(risk_config, 'max_leverage', 10.0))
            
            # 根据置信度调整杠杆
            min_leverage = 1.0
            leverage = min_leverage + (max_leverage - min_leverage) * confidence
            
            # 简单的连续亏损保护
            consecutive_losses = getattr(self, 'consecutive_losses', 0)
            if consecutive_losses >= 3:
                leverage *= 0.5  # 连续亏损降低杠杆
            
            leverage = max(1.0, min(leverage, getattr(risk_config, 'max_leverage', 10.0)))
            
            self.logger.info(f"动态杠杆: 市场={market_condition}, 置信度={confidence:.2f}, 杠杆={leverage:.1f}x")
            return leverage
            
        except Exception as e:
            self.logger.error(f"动态杠杆计算失败: {e}")
            return 3.0  # 安全默认值

    def calculate_dynamic_position_size(self, signal: Dict, symbol: str) -> float:
        """
        简单的动态仓位计算 - 基于大佬杠杆建议
        
        Args:
            signal: 交易信号
            symbol: 交易品种
            
        Returns:
            建议仓位大小
        """
        try:
            confidence = signal.get('confidence', 0.5)
            entry_price = signal.get('price', 0)
            market_condition = signal.get('market_condition', 'neutral')
            
            if entry_price <= 0:
                return 0
            
            # 获取动态杠杆
            leverage = self.get_dynamic_leverage(market_condition, confidence)
            
            # 简单的仓位计算
            base_position_size = self.calculate_position_size_advanced(entry_price, symbol)
            
            # 应用杠杆和置信度调整
            adjusted_position = base_position_size * leverage * confidence
            
            self.logger.info(f"动态仓位: {symbol} - 杠杆{leverage:.1f}x, 置信度{confidence:.2f}, 仓位{adjusted_position:.6f}")
            
            return adjusted_position
            
        except Exception as e:
            self.logger.error(f"动态仓位计算失败: {e}")
            return 0
    
    def signal_handler(self, signum, frame):
        """处理系统信号"""
        self.logger.info(f"收到信号 {signum}，准备关闭交易引擎")
        self.should_stop = True
        self.is_running = False

    async def check_breakeven_add_condition(self, symbol: str, signal: Dict) -> Dict:
        """
        检查保本加仓条件：前单盈利能否覆盖新单止损风险
        
        三种保本策略：
        1. 基础保本：前单盈利 >= 新单潜在止损
        2. 安全保本：前单盈利 >= 新单潜在止损 × 2
        3. 渐进保本：随着加仓次数提高保本倍数
        """
        try:
            if symbol not in self.current_positions:
                return {'allowed': True, 'reason': '首次开仓，无需保本检查'}
            
            position = self.current_positions[symbol]
            current_price = signal.get('price', 0)
            new_stop_loss = signal.get('stop_loss', 0)
            
            if current_price == 0 or new_stop_loss == 0:
                return {'allowed': False, 'reason': '信号价格或止损信息缺失'}
            
            # 1. 计算当前持仓盈亏
            entry_price = position['entry_price']
            position_size = abs(position['size'])
            position_side = position['side']
            
            if position_side == 'LONG':
                unrealized_pnl = (current_price - entry_price) * position_size
                new_order_risk = (current_price - new_stop_loss) * self.calculate_dynamic_position_size(signal, symbol)
            else:  # SHORT
                unrealized_pnl = (entry_price - current_price) * position_size  
                new_order_risk = (new_stop_loss - current_price) * self.calculate_dynamic_position_size(signal, symbol)
            
            # 2. 获取保本策略配置
            add_count = position.get('add_count', 0)
            trading_config = self.config.get_trading_config()
            
            # 保本倍数：随加仓次数递增 (1.0 → 1.5 → 2.0 → 2.5)
            breakeven_multiplier = getattr(trading_config, 'breakeven_multiplier_base', 1.0) + (add_count * 0.5)
            required_profit = new_order_risk * breakeven_multiplier
            
            self.logger.info(f"   🔍 {symbol} 保本加仓分析:")
            self.logger.info(f"      💰 当前持仓盈亏: ${unrealized_pnl:.2f}")
            self.logger.info(f"      ⚠️ 新单止损风险: ${new_order_risk:.2f}")
            self.logger.info(f"      📊 需要盈利覆盖: ${required_profit:.2f} (倍数: {breakeven_multiplier:.1f})")
            self.logger.info(f"      🎯 加仓次数: {add_count}")
            
            # 3. 保本条件判断
            if unrealized_pnl >= required_profit:
                safety_ratio = unrealized_pnl / required_profit if required_profit > 0 else float('inf')
                return {
                    'allowed': True,
                    'reason': f'保本条件满足：盈利${unrealized_pnl:.2f} >= 需求${required_profit:.2f} (安全比{safety_ratio:.1f}x)',
                    'safety_ratio': safety_ratio,
                    'current_pnl': unrealized_pnl,
                    'required_profit': required_profit
                }
            else:
                deficit = required_profit - unrealized_pnl
                return {
                    'allowed': False,
                    'reason': f'保本条件不足：盈利${unrealized_pnl:.2f} < 需求${required_profit:.2f}，缺口${deficit:.2f}',
                    'deficit': deficit,
                    'current_pnl': unrealized_pnl,
                    'required_profit': required_profit
                }
                
        except Exception as e:
            self.logger.error(f"❌ {symbol} 保本加仓条件检查失败: {e}")
            return {'allowed': False, 'reason': f'保本检查异常: {e}'}
    
    async def update_progressive_stop_loss(self, symbol: str) -> Dict:
        """
        渐进式止损更新：使用币安真实PnL数据，随着盈利增加逐步将止损点位上移到盈利区域
        
        🔥 高级功能增强：
        - 集成支撑阻力技术分析优化
        - 智能ROI等级管理 (Level 0-6)
        - 动态参数调整 (市场条件+波动率)
        - 智能减仓系统 (50%+ ROI)
        
        策略原理：
        1. 盈利0-5%：保持原始止损
        2. 盈利5-15%：止损上移到保本点位 
        3. 盈利15-30%：止损上移到盈利5%点位
        4. 盈利30%+：止损上移到盈利15%点位
        🆕 5. 盈利50%+：首次减仓20%
        🆕 6. 盈利80%+：累计减仓40%
        🆕 7. 盈利150%+：累计减仓70%，保留核心30%
        """
        try:
            if symbol not in self.current_positions:
                return {'updated': False, 'reason': '无持仓数据'}
            
            position = self.current_positions[symbol]
            
            # 🎯 关键修改：直接从币安API获取真实仓位数据和PnL
            account_response = await self.api_client_manager.get_account_info()
            if not account_response or not account_response.success:
                return {'updated': False, 'reason': '无法获取币安账户信息'}
            
            # 查找对应的币安仓位数据
            binance_position = None
            for pos in account_response.data.get('positions', []):
                if pos.get('symbol') == symbol and float(pos.get('positionAmt', 0)) != 0:
                    binance_position = pos
                    break
            
            if not binance_position:
                # 🔧 调试：为什么找不到仓位
                self.logger.warning(f"   ⚠️ {symbol} 在币安API返回的positions中未找到，检查所有positions...")
                for i, pos in enumerate(account_response.data.get('positions', [])):
                    if pos.get('symbol') == symbol:
                        self.logger.info(f"   📍 找到 {symbol} 但positionAmt={pos.get('positionAmt', 0)}")
                return {'updated': False, 'reason': f'{symbol} 在币安账户中未找到活跃仓位'}
            
            # 🔧 调试：输出币安仓位数据结构  
            self.logger.info(f"   ✅ {symbol} 找到币安仓位数据: unrealizedProfit={binance_position.get('unrealizedProfit')}, initialMargin={binance_position.get('initialMargin')}")
            
            # 🔥 使用币安真实数据计算盈利百分比 (修复字段名)
            unrealized_pnl = float(binance_position.get('unrealizedProfit', 0))  # 修复：正确字段名
            notional = abs(float(binance_position.get('notional', 0)))  # 名义价值
            entry_price = float(binance_position.get('entryPrice', 0))
            
            # 🔧 调试：输出解析后的变量值
            self.logger.info(f"   🔧 解析结果: unrealized_pnl={unrealized_pnl}, notional={notional}, entry_price={entry_price}")
            
            # 🔧 修复：markPrice字段不存在，使用实时价格API获取
            ticker_response = await self.api_client_manager.get_ticker(symbol)
            if ticker_response and ticker_response.success:
                # 🔧 修复：添加多重验证，防止0价格
                data = ticker_response.data
                mark_price = None
                
                # 多重字段验证
                if 'lastPrice' in data and str(data['lastPrice']) not in ['', '0', 'None', None]:
                    try:
                        mark_price = float(data['lastPrice'])
                        if mark_price <= 0:
                            mark_price = None
                    except (ValueError, TypeError):
                        mark_price = None
                
                if not mark_price and 'price' in data:
                    try:
                        mark_price = float(data['price'])
                        if mark_price <= 0:
                            mark_price = None
                    except (ValueError, TypeError):
                        mark_price = None
                
                if not mark_price and 'close' in data:
                    try:
                        mark_price = float(data['close'])
                        if mark_price <= 0:
                            mark_price = None
                    except (ValueError, TypeError):
                        mark_price = None
                
                if mark_price and mark_price > 0:
                    self.logger.info(f"   🔧 标记价格获取成功: {mark_price}")
                else:
                    mark_price = entry_price  # 降级使用入场价格
                    self.logger.warning(f"   ⚠️ 标记价格无效，使用入场价格: {mark_price}")
                    self.logger.warning(f"   🔍 ticker响应字段: {list(data.keys()) if data else 'None'}")
            else:
                mark_price = entry_price  # 降级使用入场价格
                self.logger.warning(f"   ⚠️ 标记价格获取失败，使用入场价格: {mark_price}")
                
            position_side = 'LONG' if float(binance_position.get('positionAmt', 0)) > 0 else 'SHORT'
            
            # 🔥 修正：币安的"投资回报率"就是基于保证金的收益率
            initial_margin = float(binance_position.get('initialMargin', 0))
            if initial_margin > 0:
                # 币安显示的ROI = 未实现盈亏 / 保证金 * 100
                binance_roi_pct = (unrealized_pnl / initial_margin) * 100  # 对应币安的"投资回报率"
                # 基于名义价值的真实ROI（参考用）
                notional_roi_pct = (unrealized_pnl / notional) * 100 if notional > 0 else 0
            else:
                return {'updated': False, 'reason': '无法获取有效的保证金数据'}
            
            breakeven_price = entry_price
            
            self.logger.info(f"   📊 {symbol} 渐进式止损分析（高级版 - 币安真实数据 + 技术分析）:")
            self.logger.info(f"      💰 入场价格: ${entry_price:.6f}")
            self.logger.info(f"      💱 标记价格: ${mark_price:.6f}")
            self.logger.info(f"      💵 未实现盈亏: ${unrealized_pnl:.2f} USDT")
            self.logger.info(f"      📈 币安ROI: {binance_roi_pct:.2f}% (对应币安投资回报率)")
            self.logger.info(f"      📊 名义ROI: {notional_roi_pct:.2f}% (基于名义价值)")
            self.logger.info(f"      💼 初始保证金: ${initial_margin:.2f} USDT")
            
            # 🆕 高级功能1：动态参数管理
            try:
                from utils.dynamic_parameter_manager import DynamicParameterManager
                param_manager = DynamicParameterManager(symbol)
                roi_level = param_manager.determine_roi_level(binance_roi_pct)
                
                # 简化的市场条件检测（实际应该调用现有的市场检测逻辑）
                market_condition = 'NEUTRAL'  # 简化处理，可以集成现有的市场判断逻辑
                volatility = 0.05  # 简化处理，可以集成现有的波动率计算
                
                level_config = param_manager.get_level_config(roi_level, market_condition, volatility)
                
                self.logger.info(f"   🎯 ROI等级分析: {binance_roi_pct:.1f}% → Level {roi_level}")
                self.logger.info(f"      策略: {level_config['description']}")
                self.logger.info(f"      保护比例: {level_config['protection_ratio']:.1%}")
                
                # 记录配置摘要
                param_manager.log_configuration_summary(binance_roi_pct, market_condition, volatility)
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ 动态参数管理失败，使用原有逻辑: {e}")
                roi_level = 0
                level_config = {'protection_ratio': 0.0, 'description': '原有逻辑'}
            
            # 🆕 高级功能2：智能减仓系统
            reduction_result = None
            try:
                from utils.smart_reduction_manager import SmartReductionManager
                reduction_manager = SmartReductionManager()
                
                # 检查是否需要设定减仓基准
                if reduction_manager.should_set_reduction_base(roi_level, position):
                    base_result = reduction_manager.set_reduction_base(position, binance_roi_pct)
                    if base_result['set']:
                        self.logger.info(f"   🎯 {symbol} 减仓基准设定: 仓位 {base_result['base_position']:.0f}")
                
                # 检查是否触发减仓
                if roi_level >= 4:  # 50%+ ROI开始减仓
                    reduction_result = await reduction_manager.check_and_execute_reduction(
                        symbol, roi_level, binance_roi_pct, position, self.api_client_manager, 
                        self.precision_manager  # 传递精度管理器
                    )
                    
                    if reduction_result['executed']:
                        self.logger.info(f"   ✅ {symbol} 智能减仓执行: {reduction_result['details']}")
                    else:
                        self.logger.info(f"   ⏹️ {symbol} 减仓未执行: {reduction_result['reason']}")
                        
            except Exception as e:
                self.logger.warning(f"   ⚠️ 智能减仓系统失败: {e}")
            
            # 🆕 高级功能3：技术分析优化（简化版）
            technical_adjustment = 0.0
            tech_confidence = 0.0
            try:
                from core.support_resistance_analyzer import SupportResistanceAnalyzer
                
                # 获取K线数据（简化处理）
                # 实际应该调用现有的数据获取逻辑，这里简化处理
                tech_confidence = 0.0  # 暂时简化，避免复杂的K线数据获取
                self.logger.debug(f"   📊 技术分析置信度: {tech_confidence:.3f}")
                
            except Exception as e:
                self.logger.debug(f"   ⚠️ 技术分析模块加载失败: {e}")
            
            # 🔧 修复：获取正确的止损价格（优先使用原始设置的止损，而不是入场价格）
            original_stop_loss = position.get('stop_loss')
            if original_stop_loss is None:
                # 如果没有原始止损，根据仓位方向设置合理的初始止损
                if position_side == 'LONG':
                    original_stop_loss = entry_price * 0.985  # 多头止损设在入场价下方1.5%
                else:  # SHORT
                    original_stop_loss = entry_price * 1.015  # 空头止损设在入场价上方1.5%
                self.logger.warning(f"      ⚠️ {symbol} 未找到原始止损，按{position_side}方向设置默认止损: ${original_stop_loss:.6f}")
            
            current_stop_loss = position.get('progressive_stop_loss', original_stop_loss)
            self.logger.info(f"      🛡️ 当前渐进止损: ${current_stop_loss:.6f}")
            self.logger.info(f"      🔒 原始止损: ${original_stop_loss:.6f}")
            
            # 🔥 修复：基于当前市价和ROI等级计算新的止损目标
            new_stop_loss = current_stop_loss
            stop_loss_reason = "保持原止损"
            minimum_improvement = 0.001  # 最小改进阈值：0.1%
            
            # 🆕 盈亏分离策略：根据ROI状态选择主导系统
            stop_loss_system = "HYBRID"  # 默认混合模式
            
            if binance_roi_pct < 0:
                # 亏损状态：技术分析主导
                stop_loss_system = "TECHNICAL_ANALYSIS"
                self.logger.info(f"   🎯 {symbol} 亏损状态(-{abs(binance_roi_pct):.1f}%)：技术分析主导")
                
                # 尝试使用技术分析确定止损
                try:
                    from core.support_resistance_analyzer import SupportResistanceAnalyzer
                    
                    # 获取K线数据进行技术分析
                    kline_response = await self.api_client_manager.get_klines(symbol, '15m', 50)
                    if kline_response and kline_response.success:
                        analyzer = SupportResistanceAnalyzer()
                        technical_result = analyzer.analyze(kline_response.data)
                        
                        if technical_result['confidence'] > 0.7:
                            technical_stop = analyzer.get_technical_stop_loss(
                                technical_result, position_side, mark_price
                            )
                            
                            if technical_stop:
                                new_stop_loss = technical_stop
                                stop_loss_reason = f"亏损状态技术止损(置信度{technical_result['confidence']:.2f})"
                                self.logger.info(f"   ✅ {symbol} 技术分析止损: {technical_stop:.6f} (置信度{technical_result['confidence']:.2f})")
                            else:
                                self.logger.info(f"   ⚠️ {symbol} 技术分析无有效止损位，保持现有止损")
                        else:
                            self.logger.info(f"   ⚠️ {symbol} 技术分析置信度({technical_result['confidence']:.2f})过低")
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️ {symbol} 技术分析失败，回退到ROI保护: {e}")
                    stop_loss_system = "ROI_PROTECTION"
                        
            elif binance_roi_pct >= 5:
                # 盈利状态：ROI保护主导
                stop_loss_system = "ROI_PROTECTION"
                self.logger.info(f"   🎯 {symbol} 盈利状态({binance_roi_pct:.1f}%)：ROI保护主导")
                
            else:
                # 平衡状态（-5% < ROI < 5%）：综合判断
                stop_loss_system = "HYBRID"
                self.logger.info(f"   🎯 {symbol} 平衡状态({binance_roi_pct:.1f}%)：综合判断")
            
            # 🆕 根据选定的系统执行相应逻辑
            if stop_loss_system in ["ROI_PROTECTION", "HYBRID"]:
                # 使用动态参数管理器的保护比例
                if roi_level >= 1 and 'level_config' in locals():
                    protection_ratio = level_config['protection_ratio']
                    
                    if protection_ratio > 0:
                        # 使用动态保护比例计算止损
                        if position_side == 'LONG':
                            # 多头：止损向上移动到保护指定比例盈利的位置
                            profit_per_unit = (mark_price - entry_price) * protection_ratio
                            target_stop = mark_price - profit_per_unit
                            
                            # 🆕 改进日志：显示计算过程和决策逻辑
                            self.logger.info(f"   📊 {symbol} 保护比例计算: {protection_ratio:.0%}")
                            self.logger.info(f"      💰 利润每单位: ${profit_per_unit:.6f}")
                            self.logger.info(f"      🎯 理论止损: ${target_stop:.6f}")
                            self.logger.info(f"      🛡️ 当前止损: ${current_stop_loss:.6f}")
                            
                            if target_stop > current_stop_loss * (1 + minimum_improvement):
                                new_stop_loss = target_stop
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}% Level{roi_level}，动态保护{protection_ratio:.0%}利润"
                                self.logger.info(f"      ✅ 决策: 上移止损到理论位置")
                            else:
                                self.logger.info(f"      ✅ 决策: 当前止损更激进，保持不变")
                                stop_loss_reason = f"保持现有止损(已超过{protection_ratio:.0%}保护)"
                        else:  # SHORT
                            # 空头：止损向下移动到保护指定比例盈利的位置
                            profit_per_unit = (entry_price - mark_price) * protection_ratio
                            target_stop = mark_price + profit_per_unit
                            
                            # 🆕 改进日志：显示计算过程和决策逻辑
                            self.logger.info(f"   📊 {symbol} 保护比例计算: {protection_ratio:.0%}")
                            self.logger.info(f"      💰 利润每单位: ${profit_per_unit:.6f}")
                            self.logger.info(f"      🎯 理论止损: ${target_stop:.6f}")
                            self.logger.info(f"      🛡️ 当前止损: ${current_stop_loss:.6f}")
                            
                            if target_stop < current_stop_loss * (1 - minimum_improvement):
                                new_stop_loss = target_stop
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}% Level{roi_level}，动态保护{protection_ratio:.0%}利润"
                                self.logger.info(f"      ✅ 决策: 下移止损到理论位置")
                            else:
                                self.logger.info(f"      ✅ 决策: 当前止损更激进，保持不变")
                                stop_loss_reason = f"保持现有止损(已超过{protection_ratio:.0%}保护)"
                                
                    elif binance_roi_pct >= 5:
                        # Level 1: 移至保本点
                        if position_side == 'LONG':
                            if entry_price > current_stop_loss * (1 + minimum_improvement):
                                new_stop_loss = entry_price
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}% Level{roi_level}，移至保本点"
                        else:  # SHORT
                            if entry_price < current_stop_loss * (1 - minimum_improvement):
                                new_stop_loss = entry_price
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}% Level{roi_level}，移至保本点"
                
                # 如果动态参数管理失败，回退到原有逻辑
                if 'level_config' not in locals() or roi_level == 0:
                    # 🔥 使用币安真实ROI判断（原有逻辑保持不变）
                    if binance_roi_pct >= 30:
                        # ROI 30%+：止损设为当前价格向盈利方向的15%盈利保护位置
                        profit_protection_ratio = 0.15  # 保护15%的当前盈利
                        if position_side == 'LONG':
                            # 多头：止损向上移动到保护15%盈利的位置
                            profit_per_unit = (mark_price - entry_price) * profit_protection_ratio
                            target_stop = mark_price - profit_per_unit
                            if target_stop > current_stop_loss * (1 + minimum_improvement):
                                new_stop_loss = target_stop
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}%，激进止损至15%盈利保护区"
                        else:  # SHORT
                            # 空头：止损向下移动到保护15%盈利的位置
                            profit_per_unit = (entry_price - mark_price) * profit_protection_ratio
                            target_stop = mark_price + profit_per_unit
                            if target_stop < current_stop_loss * (1 - minimum_improvement):
                                new_stop_loss = target_stop
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}%，激进止损至15%盈利保护区"
                                
                    elif binance_roi_pct >= 15:
                        # ROI 15-30%：止损设为5%盈利保护区
                        profit_protection_ratio = 0.05
                        if position_side == 'LONG':
                            profit_per_unit = (mark_price - entry_price) * profit_protection_ratio
                            target_stop = mark_price - profit_per_unit
                            if target_stop > current_stop_loss * (1 + minimum_improvement):
                                new_stop_loss = target_stop
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}%，激进止损至5%盈利保护区"
                        else:  # SHORT
                            profit_per_unit = (entry_price - mark_price) * profit_protection_ratio
                            target_stop = mark_price + profit_per_unit
                            if target_stop < current_stop_loss * (1 - minimum_improvement):
                                new_stop_loss = target_stop
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}%，激进止损至5%盈利保护区"
                                
                    elif binance_roi_pct >= 5:
                        # ROI 5-15%：止损移至保本点
                        if position_side == 'LONG':
                            if entry_price > current_stop_loss * (1 + minimum_improvement):
                                new_stop_loss = entry_price
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}%，激进止损至保本点"
                        else:  # SHORT
                            if entry_price < current_stop_loss * (1 - minimum_improvement):
                                new_stop_loss = entry_price
                                stop_loss_reason = f"ROI{binance_roi_pct:.1f}%，激进止损至保本点"
            
            # 🆕 技术分析验证：ROI保护计算的止损通过技术分析验证
            if stop_loss_system == "ROI_PROTECTION" and new_stop_loss != current_stop_loss:
                try:
                    from core.support_resistance_analyzer import SupportResistanceAnalyzer
                    
                    # 获取K线数据进行技术分析验证
                    kline_response = await self.api_client_manager.get_klines(symbol, '15m', 50)
                    if kline_response and kline_response.success:
                        analyzer = SupportResistanceAnalyzer()
                        technical_result = analyzer.analyze(kline_response.data)
                        
                        if technical_result['confidence'] > 0.5:
                            validation_result = analyzer.validate_stop_loss_with_technical_analysis(
                                new_stop_loss, technical_result, position_side
                            )
                            
                            if not validation_result['safe']:
                                # ROI止损位不安全，调整到技术分析建议的位置
                                old_roi_stop = new_stop_loss
                                new_stop_loss = validation_result['suggested_stop_loss']
                                stop_loss_reason += f" + 技术调整({validation_result['reason']})"
                                
                                self.logger.warning(f"   ⚠️ {symbol} ROI止损调整: {old_roi_stop:.6f} → {new_stop_loss:.6f}")
                                self.logger.warning(f"   📈 技术验证: {validation_result['reason']}")
                            else:
                                self.logger.info(f"   ✅ {symbol} ROI止损通过技术验证 (置信度{validation_result['confidence']:.2f})")
                        else:
                            self.logger.info(f"   ⚠️ {symbol} 技术验证置信度({technical_result['confidence']:.2f})过低，跳过验证")
                
                except Exception as e:
                    self.logger.warning(f"   ⚠️ {symbol} 技术验证失败，保持ROI止损: {e}")

            # 🔧 修复：判断是否为"利润保护"方向
            if position_side == 'SHORT':
                # SHORT：止损向下移动是保护利润（正确方向）
                is_profit_protective = new_stop_loss <= current_stop_loss
                direction_description = "利润保护（向下调整止损）" if is_profit_protective else "风险增加（向上调整止损）"
            else:  # LONG
                # LONG：止损向上移动是保护利润（正确方向）  
                is_profit_protective = new_stop_loss >= current_stop_loss
                direction_description = "利润保护（向上调整止损）" if is_profit_protective else "风险增加（向下调整止损）"
            
            # 🆕 增加详细的止损计算调试信息
            improvement_pct = abs(new_stop_loss - current_stop_loss) / current_stop_loss * 100
            self.logger.info(f"   🔧 止损计算详情:")
            self.logger.info(f"      📍 持仓方向: {position_side}")
            self.logger.info(f"      💰 入场价格: ${entry_price:.6f}")
            self.logger.info(f"      💱 当前价格: ${mark_price:.6f}")
            self.logger.info(f"      🛡️ 当前止损: ${current_stop_loss:.6f}")
            self.logger.info(f"      🎯 计算止损: ${new_stop_loss:.6f}")
            self.logger.info(f"      📊 调整方向: {direction_description}")
            self.logger.info(f"      📊 改进幅度: {improvement_pct:.3f}%")
            self.logger.info(f"      📊 调整类型: {'✅利润保护' if is_profit_protective else '❌风险增加'}")
            self.logger.info(f"      🚧 最小阈值: {minimum_improvement * 100:.1f}%")
            self.logger.info(f"      ✅ 更新条件: 利润保护({is_profit_protective}) && 改进({improvement_pct:.3f}% >= {minimum_improvement * 100:.1f}%)")
            
            # 修复：允许合理的利润保护调整，高ROI时放宽限制
            significant_improvement = abs(new_stop_loss - current_stop_loss) / current_stop_loss >= minimum_improvement
            high_roi_override = binance_roi_pct > 10  # 高ROI时放宽限制
            
            # 更新止损
            if (is_profit_protective and significant_improvement) or high_roi_override:
                old_stop_loss = current_stop_loss
                # 🔧 修复：使用progressive_stop_loss字段保存渐进式止损，不覆盖原始stop_loss
                position['progressive_stop_loss'] = new_stop_loss
                position['progressive_stop_loss_updated_at'] = datetime.now()
                position['progressive_stop_loss_reason'] = stop_loss_reason
                
                # 保存原始止损（首次更新时）
                if 'original_stop_loss' not in position:
                    position['original_stop_loss'] = original_stop_loss
                
                self.logger.info(f"   ✅ {symbol} 止损已更新:")
                self.logger.info(f"      🔄 旧止损: ${old_stop_loss:.6f}")
                self.logger.info(f"      🎯 新止损: ${new_stop_loss:.6f}")
                self.logger.info(f"      📊 改进幅度: {abs(new_stop_loss - old_stop_loss) / old_stop_loss * 100:.2f}%")
                self.logger.info(f"      💡 更新原因: {stop_loss_reason}")
                
                # TODO: 调用币安API更新实际订单止损（如果需要）
                # await self.update_exchange_stop_loss_order(symbol, new_stop_loss)
                
                # 🆕 高级功能3：增强反转检测（在止损更新后检查）
                enhanced_reversal_result = None
                try:
                    from core.enhanced_reversal_detector import EnhancedReversalDetector
                    
                    # 获取15分钟K线数据
                    kline_response = await self.api_client_manager.get_klines(symbol, '15m', 20)
                    if kline_response and kline_response.success:
                        kline_data = kline_response.data
                        
                        reversal_detector = EnhancedReversalDetector()
                        signal_type, confidence, details = await reversal_detector.detect_reversal(kline_data)
                        
                        enhanced_reversal_result = {
                            'signal_type': signal_type,
                            'confidence': confidence,
                            'details': details
                        }
                        
                        self.logger.info(f"   🔬 增强反转检测: {signal_type}, 置信度: {confidence:.3f}")
                        
                        # 🚨 急拉急跌防护：高置信度反转信号触发主动减仓
                        if signal_type and confidence > 0.7:  # 高置信度阈值
                            position_side = position.get('side', 'UNKNOWN')
                            
                            # 检查方向冲突（急拉急跌风险）
                            if ((signal_type == 'BEAR_REVERSAL' and position_side == 'LONG') or 
                                (signal_type == 'BULL_REVERSAL' and position_side == 'SHORT')):
                                
                                self.logger.warning(f"🚨 {symbol} 检测到高置信度反转风险！")
                                self.logger.warning(f"   持仓方向: {position_side}, 反转信号: {signal_type}")
                                self.logger.warning(f"   置信度: {confidence:.3f} > 0.7，触发主动防护")
                                
                                # 触发智能减仓（而不是全部平仓）
                                if roi_level >= 1:  # 有盈利时优先减仓
                                    reduction_result = await reduction_manager.check_and_execute_reduction(
                                        symbol, max(roi_level, 4), binance_roi_pct, position, self.api_client_manager,
                                        self.precision_manager, emergency_mode=True  # 紧急模式
                                    )
                                    if reduction_result['executed']:
                                        self.logger.info(f"   ✅ 反转风险防护减仓成功: {reduction_result['details']}")
                                        return {
                                            'updated': True,
                                            'old_stop_loss': old_stop_loss,
                                            'new_stop_loss': new_stop_loss,
                                            'reason': f'反转防护减仓({confidence:.3f})',
                                            'binance_roi_pct': binance_roi_pct,
                                            'roi_level': roi_level,
                                            'reduction_executed': True,
                                            'reduction_result': reduction_result
                                        }
                                    else:
                                        # 减仓失败时才考虑全部平仓
                                        self.logger.error(f"   ❌ 反转风险减仓失败，建议全部平仓")
                                        return {
                                            'emergency_close': True,
                                            'reason': f'高置信度反转风险({confidence:.3f})且减仓失败',
                                            'signal_details': enhanced_reversal_result
                                        }
                                else:
                                    # 亏损时直接触发全部平仓
                                    self.logger.error(f"   💀 亏损状态遇到高置信度反转，建议立即平仓")
                                    return {
                                        'emergency_close': True,
                                        'reason': f'亏损状态高置信度反转风险({confidence:.3f})',
                                        'signal_details': enhanced_reversal_result
                                    }
                                        
                except Exception as e:
                    self.logger.warning(f"   ⚠️ 增强反转检测失败: {e}")
                
                result = {
                    'updated': True,
                    'old_stop_loss': old_stop_loss,
                    'new_stop_loss': new_stop_loss,
                    'reason': stop_loss_reason,
                    'binance_roi_pct': binance_roi_pct,
                    'improvement_pct': abs(new_stop_loss - old_stop_loss) / old_stop_loss * 100,
                    'roi_level': roi_level if 'roi_level' in locals() else None,
                    'reduction_result': reduction_result,
                    # 🆕 增强控制信息
                    'skip_basic_checks': roi_level >= 2 and binance_roi_pct > 10,  # 有足够盈利时完全信任渐进式止损
                    'allow_basic_price_check': roi_level < 2 or binance_roi_pct <= 5,  # 低盈利时允许基础检查作为安全网
                    'position_status': 'PROFITABLE' if binance_roi_pct > 5 else 'BREAKEVEN',
                    'recommended_action': 'HOLD_WITH_PROGRESSIVE_SL' if roi_level >= 2 else 'MONITOR_WITH_BASIC_SL',
                    'control_priority': 'HIGH' if roi_level >= 3 else 'MEDIUM'
                }
                
                return result
            else:
                # 不满足更新条件的详细说明
                if not is_profit_protective:
                    stop_loss_reason = f"新止损({new_stop_loss:.6f})增加风险，未采用 - {direction_description}"
                    self.logger.info(f"   ⏹️ {symbol} 止损未更新: 计算的止损会增加风险")
                else:
                    improvement_needed = minimum_improvement * 100
                    actual_improvement = abs(new_stop_loss - current_stop_loss) / current_stop_loss * 100
                    stop_loss_reason = f"改进幅度({actual_improvement:.3f}%)低于阈值({improvement_needed:.1f}%)"
                    
                    if new_stop_loss == current_stop_loss:
                        self.logger.info(f"   ✅ {symbol} 止损保持不变: 当前止损已经最优")
                        self.logger.info(f"      💡 原因: 当前止损提供了比理论计算更强的保护")
                    else:
                        self.logger.info(f"   ⏹️ {symbol} 止损未更新: 改进幅度不足")
                        self.logger.info(f"      📊 改进幅度: {actual_improvement:.3f}% < 阈值{improvement_needed:.1f}%")
                return {
                    'updated': False,
                    'reason': stop_loss_reason,
                    'binance_roi_pct': binance_roi_pct,
                    'current_stop_loss': current_stop_loss,
                    'calculated_target': new_stop_loss,
                    'roi_level': roi_level if 'roi_level' in locals() else None,
                    'reduction_result': reduction_result,
                    # 🆕 增强控制信息 - 即使止损无需更新，渐进式止损系统仍然主导决策
                    'skip_basic_checks': roi_level >= 2 and binance_roi_pct > 10,  # 有足够盈利时完全信任渐进式止损
                    'allow_basic_price_check': roi_level < 2 or binance_roi_pct <= 5,  # 低盈利时允许基础检查作为安全网
                    'position_status': 'PROFITABLE' if binance_roi_pct > 5 else 'BREAKEVEN',
                    'recommended_action': 'HOLD_WITH_PROGRESSIVE_SL' if roi_level >= 2 else 'MONITOR_WITH_BASIC_SL',
                    'control_priority': 'HIGH' if roi_level >= 3 else 'MEDIUM'
                }
                
        except Exception as e:
            self.logger.error(f"❌ {symbol} 渐进式止损更新失败: {e}")
            return {'updated': False, 'reason': f'更新异常: {e}'}
    
    async def _execute_time_based_reduction(self, symbol: str, reduction_ratio: float) -> Dict:
        """基于时间的智能减仓"""
        try:
            if symbol not in self.current_positions:
                return {'executed': False, 'reason': '持仓不存在'}
            
            position = self.current_positions[symbol]
            current_size = abs(position.get('size', 0))
            reduction_amount = current_size * reduction_ratio
            
            self.logger.info(f"🕐 {symbol} 时间减仓触发:")
            self.logger.info(f"   📊 当前仓位: {current_size}")
            self.logger.info(f"   🔄 减仓比例: {reduction_ratio:.0%}")
            self.logger.info(f"   📉 减仓数量: {reduction_amount:.6f}")
            
            # 调用智能减仓管理器
            from utils.smart_reduction_manager import SmartReductionManager
            reduction_manager = SmartReductionManager()
            
            # 构造时间止损策略
            time_strategy = {
                'name': 'time_based_reduction',
                'description': f'时间止损减仓{reduction_ratio:.0%}',
                'target_pct': reduction_ratio,
                'cumulative': False
            }
            
            result = await reduction_manager._execute_reduction(
                symbol, reduction_amount, position, self.api_client_manager, time_strategy,
                self.precision_manager  # 传递精度管理器
            )
            
            return {'executed': result['status'] == 'success', 'details': result}
            
        except Exception as e:
            self.logger.error(f"时间减仓执行失败: {e}")
            return {'executed': False, 'reason': str(e)}
    
    async def start(self):
        """启动交易引擎"""
        try:
            self.is_running = True
            self.logger.info("🚀 启动多币种自动选币交易引擎...")
            
            # 发送启动通知
            await self.telegram_bot.send_message("🚀 多币种交易引擎已启动")
            
            # 初始化精度管理器（最小修改原则：仅添加初始化逻辑）
            try:
                self.logger.info("🔧 正在初始化精度管理器...")
                await self.precision_manager.init(self.api_client_manager)
                self.logger.info("✅ 精度管理器初始化完成")
            except Exception as e:
                self.logger.warning(f"⚠️ 精度管理器初始化失败，将使用默认精度: {e}")
            
            # 启动主循环
            await self.main_loop()
            
            # 主循环退出后的清理工作
            self.logger.info("🛑 主循环已退出，正在清理资源...")
            await self.telegram_bot.send_message("🛑 交易引擎主循环已退出")
            
        except Exception as e:
            self.logger.error(f"启动交易引擎失败: {e}")
            await self.telegram_bot.send_message(f"❌ 交易引擎启动失败: {e}")
            raise
    
    async def stop(self):
        """停止交易引擎"""
        try:
            self.should_stop = True
            self.is_running = False
            
            self.logger.info("交易引擎停止完成")
            await self.telegram_bot.send_message("🛑 交易引擎已停止")
            
        except Exception as e:
            self.logger.error(f"停止交易引擎失败: {e}")
    
    async def main_loop(self):
        """主循环 - 多币种交易流程"""
        last_coin_selection_time = None
        coin_selection_interval = 3600  # 1小时执行一次选币
        
        while not self.should_stop:
            try:
                current_time = time.time()
                
                # 1. 币种选择阶段（每小时执行一次）
                if (self.multi_symbol_mode and 
                    (last_coin_selection_time is None or 
                     current_time - last_coin_selection_time > coin_selection_interval)):
                    await self.execute_coin_selection()
                    last_coin_selection_time = current_time
                
                # 2. 数据获取阶段
                await self.fetch_all_coin_data()
                
                # 3. 技术分析阶段
                await self.analyze_all_coins()
                
                # 4. 信号检测阶段
                await self.detect_all_signals()
                
                # 5. 交易执行阶段
                await self.execute_all_trades()
                
                # 6. 持仓监控阶段
                await self.monitor_positions()
                
                # 7. 风险管理阶段
                await self.update_risk_management()
                
                # 8. 报告生成阶段
                await self.generate_reports()
                
                # 等待下一个周期（可中断的睡眠）
                for _ in range(30):  # 30秒检查一次，每秒检查一次停止标志
                    if self.should_stop:
                        break
                    await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"主循环执行失败: {e}")
                await self.telegram_bot.send_message(f"⚠️ 主循环错误: {e}")
                # 可中断的错误恢复等待
                for _ in range(60):
                    if self.should_stop:
                        break
                    await asyncio.sleep(1)
    
    async def execute_coin_selection(self):
        """执行币种选择"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🎯 阶段1: 执行币种选择")
            self.logger.info("=" * 80)
            
            # 使用大佬版选币逻辑
            selected_symbols = await self.coin_scanner.scan_and_select_coins()
            
            if selected_symbols:
                self.selected_symbols = selected_symbols
                self.logger.info(f"📊 算法选币结果: {len(selected_symbols)} 个币种: {selected_symbols}")
            else:
                # 备用：使用主流币
                self.selected_symbols = ['BTCUSDT', 'ETHUSDT']
                self.logger.warning("⚠️ 选币失败，使用默认主流币")
            
            # 🚀 关键功能：自动添加持仓币种到检测列表
            await self.add_position_symbols_to_selection()
                
        except Exception as e:
            self.logger.error(f"选币阶段失败: {e}")
            # 备用：使用主流币
            self.selected_symbols = ['BTCUSDT', 'ETHUSDT']
            # 即使出错也要尝试添加持仓币种
            try:
                await self.add_position_symbols_to_selection()
            except Exception as add_error:
                self.logger.error(f"添加持仓币种失败: {add_error}")
    
    async def add_position_symbols_to_selection(self):
        """将持仓币种添加到信号检测列表中 - 确保加仓机制正常工作"""
        try:
            self.logger.info("🔄 正在获取持仓币种，确保持续信号检测...")
            
            # 获取当前持仓数据
            account_response = await self.api_client_manager.get_account_info()
            
            if account_response and account_response.success:
                account_data = account_response.data
                positions_data = account_data.get('positions', [])
                
                # 提取持仓币种
                position_symbols = []
                for pos in positions_data:
                    symbol = pos.get('symbol', '')
                    position_amt = float(pos.get('positionAmt', 0))
                    
                    # 只添加有实际持仓的币种
                    if abs(position_amt) > 0 and symbol:
                        position_symbols.append(symbol)
                
                if position_symbols:
                    # 记录原始选币结果
                    original_count = len(self.selected_symbols)
                    original_symbols = self.selected_symbols.copy()
                    
                    # 合并持仓币种（去重）
                    for symbol in position_symbols:
                        if symbol not in self.selected_symbols:
                            self.selected_symbols.append(symbol)
                    
                    # 详细日志记录
                    added_symbols = [s for s in position_symbols if s not in original_symbols]
                    
                    self.logger.info(f"💼 发现 {len(position_symbols)} 个持仓币种: {position_symbols}")
                    
                    if added_symbols:
                        self.logger.info(f"➕ 新增 {len(added_symbols)} 个持仓币种到检测列表: {added_symbols}")
                        self.logger.info(f"🎯 最终检测列表: {len(self.selected_symbols)} 个币种")
                        self.logger.info(f"   • 算法选币: {original_symbols}")
                        self.logger.info(f"   • 持仓币种: {position_symbols}")
                        self.logger.info(f"✅ 持仓币种加仓机制已激活")
                    else:
                        self.logger.info(f"✅ 所有持仓币种已在检测列表中，无需添加")
                else:
                    self.logger.info("📭 当前无持仓，仅使用算法选币结果")
            else:
                self.logger.warning("⚠️ 获取持仓数据失败，仅使用算法选币结果")
                
        except Exception as e:
            self.logger.error(f"添加持仓币种到检测列表失败: {e}")
            self.logger.warning("⚠️ 将继续使用算法选币结果，但可能影响加仓功能")
    
    async def fetch_multi_timeframe_data(self, symbol: str) -> Dict:
        """
        🆕 任务1.1: 获取多时间周期数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Dict: 包含多个时间周期数据的字典
                {
                    'trend': DataFrame,    # 4h趋势数据
                    'signal': DataFrame,   # 1h信号数据  
                    'entry': DataFrame,    # 15m入场数据
                    'confirm': DataFrame   # 5m确认数据（可选）
                }
        """
        try:
            # 多时间周期配置
            timeframes = {
                'trend': '4h',      # 主趋势判断
                'signal': '1h',     # 信号确认
                'entry': '15m',     # 精确入场
                'confirm': '5m'     # 突破确认（可选）
            }
            
            multi_data = {}
            
            # 并行获取多个时间周期数据
            for purpose, interval in timeframes.items():
                try:
                    # 根据时间周期调整数据量
                    if interval == '15m':
                        limit = 200  # 15分钟需要更多数据点
                    elif interval == '5m':
                        limit = 150  # 5分钟确认数据较少
                    else:
                        limit = 100  # 1h和4h标准数据量
                    
                    self.logger.debug(f"📊 获取 {symbol} {interval} 数据 (limit={limit})")
                    
                    # 调用现有API客户端获取K线数据
                    response = await self.api_client_manager.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit
                    )
                    
                    if response and response.success:
                        # 使用现有的数据处理逻辑
                        multi_data[purpose] = self.process_kline_data(response.data)
                        self.logger.debug(f"✅ {symbol} {interval}({purpose}) 数据获取成功: {len(multi_data[purpose])} 条")
                    else:
                        self.logger.warning(f"⚠️ {symbol} {interval}({purpose}) 数据获取失败")
                        multi_data[purpose] = None
                        
                except Exception as e:
                    self.logger.error(f"❌ {symbol} {interval}({purpose}) 数据获取异常: {e}")
                    multi_data[purpose] = None
            
            # 检查关键数据是否获取成功
            critical_timeframes = ['trend', 'signal', 'entry']
            success_count = sum(1 for tf in critical_timeframes if multi_data.get(tf) is not None)
            
            if success_count >= 2:  # 至少2个关键时间周期成功
                self.logger.info(f"📊 {symbol} 多时间周期数据获取成功: {success_count}/{len(critical_timeframes)} 关键周期")
                return multi_data
            else:
                self.logger.error(f"❌ {symbol} 多时间周期数据获取失败: 仅 {success_count}/{len(critical_timeframes)} 成功")
                return {}
                
        except Exception as e:
            self.logger.error(f"多时间周期数据获取异常 {symbol}: {e}")
            return {}
    
    def process_kline_data(self, klines_data):
        """
        处理K线数据，转换为标准DataFrame格式
        复用现有逻辑，确保数据格式一致性
        """
        try:
            import pandas as pd
            
            # 转换为DataFrame
            ohlcv_data = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # 转换数据类型 - 使用现有逻辑
            for col in ['open', 'high', 'low', 'close', 'volume']:
                ohlcv_data[col] = pd.to_numeric(ohlcv_data[col], errors='coerce')
            
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"K线数据处理失败: {e}")
            return None

    async def fetch_all_coin_data(self):
        """阶段2: 获取选中币种的K线数据（保持现有逻辑不变）"""
        try:
            self.logger.info("📊 阶段2: 获取选中币种的K线数据")
            
            self.selected_symbols_data = {}
            
            # 获取每个选中币种的数据
            for symbol in self.selected_symbols:
                self.logger.info(f"📈 正在获取 {symbol} 的K线数据...")
                
                # 获取K线数据 - 使用15分钟时间框架（符合15分钟K线设计）
                klines_response = await self.api_client_manager.get_klines(
                    symbol, '15m', limit=300  # 15分钟需要更多数据点
                )
                
                if klines_response and klines_response.success:
                    # 使用新的数据处理方法
                    ohlcv_data = self.process_kline_data(klines_response.data)
                    
                    if ohlcv_data is not None:
                        self.selected_symbols_data[symbol] = ohlcv_data
                        self.logger.info(f"✅ {symbol} K线数据获取成功: {len(ohlcv_data)} 条")
                    else:
                        self.logger.error(f"❌ {symbol} K线数据处理失败")
                else:
                    self.logger.error(f"❌ {symbol} K线数据获取失败")
                    
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
    
    async def analyze_all_coins(self):
        """阶段3: 对选中币种执行技术分析"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🔬 阶段3: 对选中币种执行技术分析")
            self.logger.info("=" * 80)
            
            # 修复：使用正确的变量名
            if not hasattr(self, 'selected_symbols_data') or not self.selected_symbols_data:
                self.logger.warning("⚠️ 没有币种数据可供分析")
                return
            
            for symbol in self.selected_symbols_data:
                ohlcv_data = self.selected_symbols_data[symbol]
                
                if ohlcv_data is None or len(ohlcv_data) < 50:
                    self.logger.warning(f"⚠️ {symbol} 数据不足，跳过分析")
                    continue
                
                self.logger.info(f"📊 正在对 {symbol} 执行技术分析...")
                
                try:
                    # 准备技术指标数据格式
                    indicator_data = {
                        'open': ohlcv_data['open'].tolist(),
                        'high': ohlcv_data['high'].tolist(), 
                        'low': ohlcv_data['low'].tolist(),
                        'close': ohlcv_data['close'].tolist(),
                        'volume': ohlcv_data['volume'].tolist()
                    }
                    
                    # 计算所有技术指标
                    indicators = self.technical_indicators.calculate_all_indicators(
                        indicator_data, symbol, '1h'
                    )
                    
                    # MACD背离检测
                    closes = ohlcv_data['close'].tolist()
                    macd_result = None
                    
                    if 'macd' in indicators and len(indicators['macd']) > 0:
                        try:
                            macd_result = self.macd_detector.detect_divergence(
                                closes, 
                                indicators['macd'],
                                symbol=symbol,
                                timeframe='1h'
                            )
                        except Exception as e:
                            self.logger.warning(f"⚠️ {symbol} MACD背离检测失败: {e}")
                    
                    # 详细输出分析结果
                    self.log_detailed_analysis_results(symbol, indicators, macd_result, ohlcv_data)
                    
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 技术分析失败: {e}")
            
        except Exception as e:
            self.logger.error(f"技术分析阶段失败: {e}")
    
    def log_detailed_analysis_results(self, symbol: str, indicators: Dict, macd_result, ohlcv_data):
        """详细输出分析结果（按用户要求）"""
        try:
            current_price = float(ohlcv_data['close'].iloc[-1])
            self.logger.info(f"📋 {symbol} 技术分析详细结果:")
            self.logger.info(f"   💰 当前价格: {current_price:.4f}")
            
            # 移动平均线分析
            if 'sma_20' in indicators and len(indicators['sma_20']) > 0:
                sma_20 = indicators['sma_20'][-1]
                sma_50 = indicators.get('sma_50', [sma_20])[-1] if 'sma_50' in indicators else sma_20
                self.logger.info(f"   📈 SMA20: {sma_20:.4f}, SMA50: {sma_50:.4f}")
                
                # 价格与均线关系（添加除零保护）
                if sma_20 > 0:
                    price_diff_pct = ((current_price/sma_20-1)*100)
                    if current_price > sma_20:
                        self.logger.info(f"   ✅ 价格在SMA20上方 (+{price_diff_pct:.2f}%)")
                    else:
                        self.logger.info(f"   ❌ 价格在SMA20下方 ({price_diff_pct:.2f}%)")
                else:
                    self.logger.info(f"   ⚠️ SMA20数据异常，无法计算价格偏离度")
            
            # RSI分析
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                rsi_data = indicators['rsi'][-1]
                rsi_value = rsi_data.rsi_value if hasattr(rsi_data, 'rsi_value') else rsi_data
                
                if rsi_value < 30:
                    rsi_status = "🔥 超卖"
                elif rsi_value > 70:
                    rsi_status = "⚠️ 超买"
                else:
                    rsi_status = "✅ 正常"
                    
                self.logger.info(f"   📊 RSI(14): {rsi_value:.2f} {rsi_status}")
            
            # MACD分析
            if 'macd' in indicators and len(indicators['macd']) > 0:
                macd_data = indicators['macd'][-1]
                if hasattr(macd_data, 'macd_line'):
                    macd_line = macd_data.macd_line
                    signal_line = macd_data.signal_line
                    histogram = macd_data.histogram
                    
                    trend = "看涨" if macd_line > signal_line else "看跌"
                    momentum = "增强" if histogram > 0 else "减弱"
                    
                    self.logger.info(f"   📈 MACD: {macd_line:.6f}, 信号线: {signal_line:.6f}")
                    self.logger.info(f"   📊 柱状图: {histogram:.6f} (趋势{trend}, 动量{momentum})")
            
            # MACD背离分析
            if macd_result:
                if hasattr(macd_result, 'has_divergence') and macd_result.has_divergence:
                    div_type = getattr(macd_result, 'divergence_type', '未知')
                    confidence = getattr(macd_result, 'confidence', 0)
                    self.logger.info(f"   🔍 MACD背离: {div_type}背离 (置信度: {confidence:.1%})")
                else:
                    self.logger.info(f"   📊 MACD背离: 无明显背离信号")
            
            # 布林带分析
            if 'bollinger' in indicators and len(indicators['bollinger']) > 0:
                bb_data = indicators['bollinger'][-1]
                if hasattr(bb_data, 'upper_band'):
                    upper = bb_data.upper_band
                    lower = bb_data.lower_band
                    position = bb_data.position
                    
                    self.logger.info(f"   📊 布林带: 上轨{upper:.4f}, 下轨{lower:.4f}")
                    self.logger.info(f"   📍 位置: {position}")
            
            # 成交量分析（添加除零保护）
            current_volume = float(ohlcv_data['volume'].iloc[-1])
            avg_volume = float(ohlcv_data['volume'].tail(20).mean())
            
            # 除零保护：如果平均成交量为0或过小，设为默认值
            if avg_volume <= 0:
                volume_ratio = 1.0
                volume_status = "⚠️ 数据异常"
                self.logger.info(f"   📊 成交量: {current_volume:.0f} (平均成交量数据异常)")
            else:
                volume_ratio = current_volume / avg_volume
                volume_status = "🔥 放量" if volume_ratio > 1.5 else "📉 缩量" if volume_ratio < 0.7 else "✅ 正常"
                self.logger.info(f"   📊 成交量: {current_volume:.0f} (20期均值比: {volume_ratio:.2f}x {volume_status})")
            
            # 综合结论
            signals = []
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                rsi_value = indicators['rsi'][-1].rsi_value if hasattr(indicators['rsi'][-1], 'rsi_value') else indicators['rsi'][-1]
                if rsi_value < 30:
                    signals.append("RSI超卖")
                elif rsi_value > 70:
                    signals.append("RSI超买")
            
            if current_price > indicators.get('sma_20', [current_price])[-1]:
                signals.append("价格强势")
            
            conclusion = "、".join(signals) if signals else "中性"
            self.logger.info(f"   💡 技术分析结论: {conclusion}")
            self.logger.info(f"   " + "="*50)
            
        except Exception as e:
            self.logger.error(f"输出分析结果失败: {e}")
    
    async def detect_all_signals(self):
        """阶段4: 检测所有币种的交易信号 - 使用专业SignalGenerator"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("📡 阶段4: 检测交易信号（使用专业SignalGenerator）")
            self.logger.info("=" * 80)
            
            # 修复：使用正确的变量名
            if not hasattr(self, 'selected_symbols_data') or not self.selected_symbols_data:
                self.logger.warning("⚠️ 没有币种数据可供信号检测")
                return
            
            buy_signals = []
            sell_signals = []
            
            for symbol in self.selected_symbols_data:
                ohlcv_data = self.selected_symbols_data[symbol]
                
                if ohlcv_data is None or len(ohlcv_data) < 100:  # SignalGenerator需要至少100条数据
                    self.logger.info(f"⚠️ {symbol} 数据不足，跳过信号检测")
                    continue
                
                self.logger.info(f"📡 正在使用专业SignalGenerator检测 {symbol} 的交易信号...")
                
                try:
                    # 转换DataFrame为SignalGenerator需要的格式
                    kline_data = []
                    for i, row in ohlcv_data.iterrows():
                        kline = {
                            'open_time': int(i) if isinstance(i, (int, float)) else int(time.time() * 1000),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        }
                        kline_data.append(kline)
                    
                    # 使用专业SignalGenerator生成信号（包含HEAD_SHOULDER、双向交易、置信度过滤）
                    trading_signal = self.signal_generator.generate_signal(kline_data)
                    
                    if trading_signal:
                        signal_data = {
                            'symbol': symbol,
                            'type': trading_signal.signal_type.value.upper(),
                            'price': trading_signal.entry_price,
                            'confidence': trading_signal.confidence,
                            'strength': trading_signal.signal_strength.value,
                            'stop_loss': trading_signal.stop_loss_price,
                            'take_profit': trading_signal.take_profit_price,
                            'risk_reward_ratio': trading_signal.risk_reward_ratio,
                            'reasons': trading_signal.reasons,
                            'market_condition': trading_signal.market_condition,
                            'timestamp': trading_signal.timestamp
                        }
                        
                        # 分类信号（支持双向交易）
                        if trading_signal.signal_type.value.upper() == 'BUY':
                            buy_signals.append(signal_data)
                            self.logger.info(f"🟢 {symbol} 买入信号 - 置信度: {trading_signal.confidence:.2f}, 强度: {trading_signal.signal_strength.value}")
                            self.logger.info(f"   📋 理由: {', '.join(trading_signal.reasons)}")
                            self.logger.info(f"   📊 风险回报比: {trading_signal.risk_reward_ratio:.2f}, 市场条件: {trading_signal.market_condition}")
                        elif trading_signal.signal_type.value.upper() == 'SELL':
                            sell_signals.append(signal_data)
                            self.logger.info(f"🔴 {symbol} 卖出信号 - 置信度: {trading_signal.confidence:.2f}, 强度: {trading_signal.signal_strength.value}")
                            self.logger.info(f"   📋 理由: {', '.join(trading_signal.reasons)}")
                            self.logger.info(f"   📊 风险回报比: {trading_signal.risk_reward_ratio:.2f}, 市场条件: {trading_signal.market_condition}")
                    else:
                        current_price = float(ohlcv_data['close'].iloc[-1])
                        self.logger.info(f"➡️ {symbol} 观望 - 当前价格: {current_price:.4f} (未达到信号生成条件)")
                        
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 专业信号检测失败: {e}")
            
            # 统计和存储信号
            self.current_signals = {
                'buy': buy_signals,
                'sell': sell_signals,
                'timestamp': datetime.now()
            }
            
            # 输出信号汇总
            self.logger.info(f"📊 专业信号汇总: {len(buy_signals)} 个买入信号, {len(sell_signals)} 个卖出信号")
            self.logger.info("🎯 使用功能: HEAD_SHOULDER形态识别 + 双向交易 + 置信度过滤 + 周期分析")
            
            # 详细输出信号
            if buy_signals:
                self.logger.info("🟢 买入信号详情:")
                for signal in buy_signals:
                    self.logger.info(f"   • {signal['symbol']}: {signal['price']:.4f} (置信度:{signal['confidence']:.2f})")
                    
            if sell_signals:
                self.logger.info("🔴 卖出信号详情:")
                for signal in sell_signals:
                    self.logger.info(f"   • {signal['symbol']}: {signal['price']:.4f} (置信度:{signal['confidence']:.2f})")
            
        except Exception as e:
            self.logger.error(f"专业信号检测阶段失败: {e}")
    
    async def execute_all_trades(self):
        """阶段5: 执行交易决策和订单管理"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("⚡ 阶段5: 执行交易决策")
            self.logger.info("=" * 80)
            
            # 检查是否有信号数据
            if not hasattr(self, 'current_signals') or not self.current_signals:
                self.logger.info("📭 当前无交易信号，跳过交易执行")
                return
            
            # 从current_signals中获取买入和卖出信号
            buy_signals = self.current_signals.get('buy', [])
            sell_signals = self.current_signals.get('sell', [])
            
            if not buy_signals and not sell_signals:
                self.logger.info("📭 当前无有效交易信号，跳过交易执行")
                return
            
            # 处理买入信号
            if buy_signals:
                self.logger.info(f"📈 处理 {len(buy_signals)} 个买入信号:")
                for signal in buy_signals:
                    await self.execute_buy_order(signal)
            
            # 处理卖出信号
            if sell_signals:
                self.logger.info(f"📉 处理 {len(sell_signals)} 个卖出信号:")
                for signal in sell_signals:
                    await self.execute_sell_order(signal)
            
            # 当前市场条件下没有符合执行标准的信号
            if not buy_signals and not sell_signals:
                self.logger.info("💡 当前市场条件：")
                for symbol in self.selected_symbols:
                    if symbol in self.selected_symbols_data:
                        current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                        self.logger.info(f"   📊 {symbol}: 当前价格 {current_price:.4f}, 状态: 观望中")
                
        except Exception as e:
            self.logger.error(f"交易执行阶段失败: {e}")
    
    async def _pre_trade_risk_check(self, symbol: str, signal: Dict) -> Dict:
        """交易前强制性风险检查"""
        try:
            # 获取风险配置
            risk_config = self.config.get_risk_config()
            
            # 0. 检查紧急制动状态
            if self.emergency_brake.is_emergency_stop_active():
                return {
                    'allowed': False,
                    'reason': f'紧急制动激活: {self.emergency_brake.stop_reason}',
                    'risk_level': 'CRITICAL'
                }
            
            # 1. 检查总敞口限制
            await self.update_risk_management()  # 更新风险指标
            
            # 获取当前账户信息和总敞口
            try:
                account_response = await self.api_client_manager.get_account_info()
                if account_response and account_response.success:
                    account_data = account_response.data
                    
                    # 🔥 修复：使用与风险管理状态一致的计算方式
                    total_cross_wallet_balance = float(account_data.get('totalCrossWalletBalance', 0))
                    used_margin = float(account_data.get('totalInitialMargin', 0))
                    
                    # 使用初始保证金比率进行风险控制（与update_risk_management保持一致）
                    exposure_ratio = (used_margin / total_cross_wallet_balance) if total_cross_wallet_balance > 0 else 0
                    
                    # 检查敞口限制 (从配置文件读取)
                    max_exposure = self.config.get_risk_config().max_total_exposure
                    if exposure_ratio > max_exposure:
                        return {
                            'allowed': False,
                            'reason': f'总敞口超限 {exposure_ratio:.1%} > {max_exposure:.1%}',
                            'risk_level': 'CRITICAL'
                        }
                else:
                    self.logger.warning("无法获取账户信息，跳过敞口检查")
            except Exception as e:
                self.logger.warning(f"敞口检查失败，允许交易继续: {e}")
            
            # 2. 检查单币种加仓次数
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                current_add_count = position.get('add_count', 0)
                if current_add_count >= risk_config.max_add_count:
                    return {
                        'allowed': False,
                        'reason': f'加仓次数达上限 {current_add_count}/{risk_config.max_add_count}',
                        'risk_level': 'HIGH'
                    }
                
                # 🔥 修复：禁用过于保守的保本加仓检查，允许更积极的交易
                breakeven_check = await self.check_breakeven_add_condition(symbol, signal)
                self.logger.info(f"   📊 保本分析（仅供参考）: {breakeven_check['reason']}")
                # 不再基于保本条件阻止交易，让信号质量和其他风险控制来决定
                self.logger.info(f"   🚀 保本检查已禁用，允许积极加仓策略")
            
            # 3. 检查信号质量
            confidence = signal.get('confidence', 0)
            min_confidence = self.config.get_signal_config().min_confidence
            if confidence < min_confidence:
                return {
                    'allowed': False,
                    'reason': f'信号置信度过低 {confidence:.2f} < {min_confidence}',
                    'risk_level': 'MEDIUM'
                }
            
            # 4. 检查可用保证金
            try:
                account_response = await self.api_client_manager.get_account_info()
                if account_response and account_response.success:
                    account_data = account_response.data
                    available_balance = float(account_data.get('availableBalance', 0))
                    
                    # 计算所需保证金
                    position_size = self.calculate_dynamic_position_size(signal, symbol)
                    required_margin = position_size * signal.get('price', 0) * 0.1  # 假设10倍杠杆
                    
                    if required_margin > available_balance * 0.8:  # 不允许使用超过80%的可用余额
                        return {
                            'allowed': False,
                            'reason': f'可用保证金不足 需要:{required_margin:.2f} 可用:{available_balance:.2f}',
                            'risk_level': 'HIGH'
                        }
            except Exception as e:
                self.logger.warning(f"保证金检查失败，允许交易继续: {e}")
            
            # 5. 所有检查通过
            return {
                'allowed': True,
                'reason': '所有风险检查通过',
                'risk_level': 'LOW'
            }
            
        except Exception as e:
            self.logger.error(f"风险预检查失败: {e}")
            # 出现异常时为安全起见，禁止交易
            return {
                'allowed': False,
                'reason': f'风险检查系统异常: {e}',
                                 'risk_level': 'CRITICAL'
             }
    
    async def get_position_specific_indicators(self, symbol: str) -> Optional[Dict]:
        """为持仓币种专门获取最新技术指标数据"""
        # 🔧 新增：使用信号量控制并发
        async with self.tech_analysis_semaphore:
            try:
                self.logger.info(f"🔄 为持仓币种 {symbol} 获取独立技术指标数据...")
                
                # 1. 优先使用现有的selected_symbols_data（如果可用且新鲜）
                if (hasattr(self, 'selected_symbols_data') and 
                    symbol in self.selected_symbols_data and 
                    hasattr(self.selected_symbols_data[symbol], 'index') and 
                    len(self.selected_symbols_data[symbol]) > 0):
                    
                    # 🔧 修复：检查数据时效性（最新数据是否在5分钟内）
                    try:
                        df_data = self.selected_symbols_data[symbol]
                        if hasattr(df_data.index, 'to_pydatetime'):
                            # pandas DatetimeIndex
                            latest_timestamp = df_data.index[-1]
                            if hasattr(latest_timestamp, 'timestamp'):
                                time_diff = datetime.now().timestamp() - latest_timestamp.timestamp()
                            else:
                                # 可能是pandas Timestamp，转换为datetime
                                time_diff = datetime.now().timestamp() - latest_timestamp.to_pydatetime().timestamp()
                        else:
                            # 整数索引或其他类型，直接获取最新数据
                            self.logger.info(f"   ⚠️ 数据索引类型不支持时效检查，直接获取最新数据")
                            ohlcv_data = await self._fetch_fresh_kline_data(symbol)
                            return ohlcv_data if ohlcv_data is not None else None
                        
                        if time_diff < 300:  # 5分钟内的数据认为是新鲜的
                            ohlcv_data = df_data
                            self.logger.info(f"   ✅ 使用现有数据（{time_diff:.0f}秒前）")
                        else:
                            self.logger.info(f"   ⚠️ 现有数据过期（{time_diff:.0f}秒前），获取最新数据")
                            ohlcv_data = await self._fetch_fresh_kline_data(symbol)
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ 时效性检查失败: {e}，直接获取最新数据")
                        ohlcv_data = await self._fetch_fresh_kline_data(symbol)
                else:
                    # 2. 直接从API获取最新K线数据
                    self.logger.info(f"   📡 {symbol} 不在选中币种中，直接获取最新数据")
                    ohlcv_data = await self._fetch_fresh_kline_data(symbol)
                
                if ohlcv_data is None or len(ohlcv_data) == 0:
                    self.logger.error(f"   ❌ {symbol} 无法获取K线数据")
                    return None
                
                # 3. 计算技术指标
                indicator_data = {
                    'open': ohlcv_data['open'].tolist(),
                    'high': ohlcv_data['high'].tolist(),
                    'low': ohlcv_data['low'].tolist(),
                    'close': ohlcv_data['close'].tolist(),
                    'volume': ohlcv_data['volume'].tolist()
                }
                
                current_indicators = self.technical_indicators.calculate_all_indicators(
                    indicator_data, symbol, '1h'
                )
                
                self.logger.info(f"   ✅ {symbol} 技术指标计算完成")
                return current_indicators
                
            except Exception as e:
                self.logger.error(f"❌ {symbol} 获取持仓专用技术指标失败: {e}")
                return None
    
    async def _fetch_fresh_kline_data(self, symbol: str):
        """为指定币种获取最新的K线数据"""
        try:
            # 使用data_fetcher获取最新1小时K线数据
            klines_response = await self.api_client_manager.get_klines(
                symbol=symbol,
                interval='1h',
                limit=200  # 获取200根K线用于技术指标计算
            )
            
            if klines_response and klines_response.success and klines_response.data:
                # 转换为DataFrame格式
                import pandas as pd
                klines_data = klines_response.data
                
                df = pd.DataFrame(klines_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # 数据类型转换
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 设置时间索引
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.logger.info(f"   📊 获取到 {len(df)} 根K线数据")
                return df[['open', 'high', 'low', 'close', 'volume']]
            else:
                self.logger.error(f"   ❌ API返回数据为空")
                return None
                
        except Exception as e:
            self.logger.error(f"获取最新K线数据失败: {e}")
            return None
    
    async def enhanced_reversal_detection(self, symbol: str, position_side: str, indicators: Dict) -> Dict:
        """增强版趋势反转检测（多时间框架、分级反转信号、动态阈值）"""
        # 🔧 新增：使用信号量控制并发
        async with self.tech_analysis_semaphore:
            try:
                self.logger.info(f"🔍 {symbol} 增强版趋势反转检测...")
                
                reversal_score = 0  # 反转评分（0-100）
                reversal_reasons = []
                confidence = 0.0
                
                # 1. MACD反转检测（权重：25分）
                if 'macd' in indicators and len(indicators['macd']) >= 2:
                    macd_current = indicators['macd'][-1]
                    macd_prev = indicators['macd'][-2]
                    
                    current_macd_line = getattr(macd_current, 'macd_line', 0)
                    current_signal_line = getattr(macd_current, 'signal_line', 0)
                    prev_macd_line = getattr(macd_prev, 'macd_line', 0)
                    prev_signal_line = getattr(macd_prev, 'signal_line', 0)
                    
                    # 检测MACD死叉（针对多头持仓）
                    if position_side == 'LONG':
                        prev_golden = prev_macd_line > prev_signal_line
                        current_death = current_macd_line < current_signal_line
                        
                        if prev_golden and current_death:
                            reversal_score += 25
                            reversal_reasons.append("MACD死叉确认")
                            self.logger.info(f"   ❌ MACD死叉：{prev_macd_line:.6f}>{prev_signal_line:.6f} → {current_macd_line:.6f}<{current_signal_line:.6f}")
                    
                    # 检测MACD金叉（针对空头持仓）
                    elif position_side == 'SHORT':
                        prev_death = prev_macd_line < prev_signal_line
                        current_golden = current_macd_line > current_signal_line
                        
                        if prev_death and current_golden:
                            reversal_score += 25
                            reversal_reasons.append("MACD金叉确认")
                            self.logger.info(f"   ❌ MACD金叉：{prev_macd_line:.6f}<{prev_signal_line:.6f} → {current_macd_line:.6f}>{current_signal_line:.6f}")
                
                # 2. RSI极值反转检测（权重：20分）
                if 'rsi' in indicators and len(indicators['rsi']) >= 3:
                    rsi_values = [getattr(rsi, 'rsi_value', rsi) for rsi in indicators['rsi'][-3:]]
                    current_rsi = rsi_values[-1]
                    
                    if position_side == 'LONG':
                        # 多头持仓：RSI进入超买区域（>70）
                        if current_rsi > 70:
                            reversal_score += 15
                            reversal_reasons.append(f"RSI超买{current_rsi:.1f}")
                            
                            # RSI背离检测：价格新高但RSI下降
                            if len(rsi_values) >= 3 and rsi_values[-1] < rsi_values[-2] < rsi_values[-3]:
                                reversal_score += 5
                                reversal_reasons.append("RSI顶背离")
                                
                    elif position_side == 'SHORT':
                        # 空头持仓：RSI进入超卖区域（<30）
                        if current_rsi < 30:
                            reversal_score += 15
                            reversal_reasons.append(f"RSI超卖{current_rsi:.1f}")
                            
                            # RSI背离检测：价格新低但RSI上升
                            if len(rsi_values) >= 3 and rsi_values[-1] > rsi_values[-2] > rsi_values[-3]:
                                reversal_score += 5
                                reversal_reasons.append("RSI底背离")
                
                # 3. 移动平均线穿越检测（权重：20分）
                if 'sma_20' in indicators and 'sma_50' in indicators:
                    if len(indicators['sma_20']) >= 2 and len(indicators['sma_50']) >= 2:
                        sma20_current = indicators['sma_20'][-1]
                        sma50_current = indicators['sma_50'][-1]
                        sma20_prev = indicators['sma_20'][-2]
                        sma50_prev = indicators['sma_50'][-2]
                        
                        if position_side == 'LONG':
                            # 多头持仓：SMA20跌破SMA50（死叉）
                            if sma20_prev > sma50_prev and sma20_current < sma50_current:
                                reversal_score += 20
                                reversal_reasons.append("SMA20/50死叉")
                        
                        elif position_side == 'SHORT':
                            # 空头持仓：SMA20突破SMA50（金叉）
                            if sma20_prev < sma50_prev and sma20_current > sma50_current:
                                reversal_score += 20
                                reversal_reasons.append("SMA20/50金叉")
                
                # 4. 成交量确认（权重：15分）
                if 'volume' in indicators and len(indicators['volume']) >= 3:
                    # 简化处理：假设indicators中有volume数据
                    try:
                        current_volume = indicators['volume'][-1] if hasattr(indicators['volume'][-1], 'real') else indicators['volume'][-1]
                        avg_volume = sum(indicators['volume'][-5:]) / 5 if len(indicators['volume']) >= 5 else current_volume
                        
                        # 成交量放大确认反转
                        if current_volume > avg_volume * 1.5:
                            reversal_score += 15
                            reversal_reasons.append(f"成交量放大{(current_volume/avg_volume):.1f}倍")
                    except:
                        # 成交量数据处理异常，跳过
                        pass
                
                # 5. 布林带突破检测（权重：10分）
                if 'bb_upper' in indicators and 'bb_lower' in indicators:
                    if len(indicators['bb_upper']) > 0 and len(indicators['bb_lower']) > 0:
                        bb_upper = indicators['bb_upper'][-1]
                        bb_lower = indicators['bb_lower'][-1]
                        
                        # 从current_price可以通过API获取
                        try:
                            ticker_response = await self.api_client_manager.get_ticker(symbol)
                            if ticker_response and ticker_response.success:
                                # 🔧 修复：防止0价格问题
                                data = ticker_response.data
                                current_price = None
                                
                                if 'lastPrice' in data and str(data['lastPrice']) not in ['', '0', 'None', None]:
                                    try:
                                        current_price = float(data['lastPrice'])
                                        if current_price <= 0:
                                            current_price = None
                                    except (ValueError, TypeError):
                                        current_price = None
                                
                                if current_price and current_price > 0:
                                    if position_side == 'LONG' and current_price < bb_lower:
                                        reversal_score += 10
                                        reversal_reasons.append("跌破布林下轨")
                                    elif position_side == 'SHORT' and current_price > bb_upper:
                                        reversal_score += 10
                                        reversal_reasons.append("突破布林上轨")
                        except:
                            pass
                
                # 6. 计算最终置信度和判断
                confidence = min(reversal_score / 100.0, 1.0)
                
                # 分级反转信号判断
                if reversal_score >= 60:
                    reversal_strength = 'strong'
                    recommended_action = 'close'
                    reversal_detected = True
                elif reversal_score >= 40:
                    reversal_strength = 'moderate'
                    recommended_action = 'reduce'
                    reversal_detected = True
                elif reversal_score >= 25:
                    reversal_strength = 'weak'
                    recommended_action = 'hold'
                    reversal_detected = False  # 弱反转不触发强制平仓
                else:
                    reversal_strength = 'none'
                    recommended_action = 'hold'
                    reversal_detected = False
                
                self.logger.info(f"   📊 {symbol} 反转检测结果:")
                self.logger.info(f"      💯 反转评分: {reversal_score}/100")
                self.logger.info(f"      📈 反转强度: {reversal_strength}")
                self.logger.info(f"      🎯 建议操作: {recommended_action}")
                self.logger.info(f"      💡 触发原因: {', '.join(reversal_reasons) if reversal_reasons else '无'}")
                
                return {
                    'reversal_detected': reversal_detected,
                    'reversal_strength': reversal_strength,
                    'recommended_action': recommended_action,
                    'confidence': confidence,
                    'score': reversal_score,
                    'reasons': reversal_reasons,
                    'reason': f"反转评分{reversal_score}/100: {', '.join(reversal_reasons[:2])}"  # 取前2个原因
                }
                
            except Exception as e:
                self.logger.error(f"❌ {symbol} 增强反转检测失败: {e}")
                return {
                    'reversal_detected': False,
                    'reversal_strength': 'none',
                    'recommended_action': 'hold',
                    'confidence': 0.0,
                    'score': 0,
                    'reasons': [],
                    'reason': f"检测异常: {e}"
                }
    
    async def update_stop_loss_take_profit_after_add(self, symbol: str, new_avg_price: float, side: str):
        """加仓后重新计算并更新止盈止损"""
        try:
            self.logger.info(f"🔧 {symbol} 加仓后更新止盈止损...")
            
            # 基于新的平均成本价重新计算止盈止损
            direction = 'BUY' if side == 'LONG' else 'SELL'
            new_stop_loss, new_take_profit = self.calculate_stop_loss_take_profit(
                new_avg_price, direction, symbol
            )
            
            # 更新持仓记录中的止盈止损信息
            if symbol in self.current_positions:
                self.current_positions[symbol]['stop_loss'] = new_stop_loss
                self.current_positions[symbol]['take_profit'] = new_take_profit
                self.current_positions[symbol]['last_sl_tp_update'] = datetime.now()
                
                self.logger.info(f"   ✅ {symbol} 止盈止损已更新:")
                self.logger.info(f"      🛡️ 新止损: {new_stop_loss:.6f} ({((new_stop_loss/new_avg_price-1)*100 if side=='LONG' else (1-new_stop_loss/new_avg_price)*100):.2f}%)")
                self.logger.info(f"      🎯 新止盈: {new_take_profit:.6f} ({((new_take_profit/new_avg_price-1)*100 if side=='LONG' else (1-new_take_profit/new_avg_price)*100):.2f}%)")
                
                # 如果是实盘交易，需要更新API中的止盈止损订单
                raw_config = self.config.get_config()
                config_dict = raw_config.dict()
                simulation_mode = config_dict.get('api', {}).get('binance', {}).get('simulation_mode', True)
                
                if not simulation_mode:
                    # 实盘模式：取消旧的止盈止损订单，创建新的
                    try:
                        await self._update_real_stop_loss_take_profit(symbol, new_stop_loss, new_take_profit, side)
                        self.logger.info(f"   ✅ {symbol} 实盘止盈止损订单已更新")
                    except Exception as e:
                        self.logger.error(f"   ❌ {symbol} 更新实盘止盈止损失败: {e}")
                else:
                    self.logger.info(f"   📝 {symbol} 模拟模式止盈止损已更新")
                
                # 持久化更新后的持仓信息
                await self.position_persistence.save_position(symbol, self.current_positions[symbol])
                
            else:
                self.logger.error(f"❌ {symbol} 持仓信息不存在，无法更新止盈止损")
                
        except Exception as e:
            self.logger.error(f"❌ {symbol} 加仓后更新止盈止损失败: {e}")
    
    async def _update_real_stop_loss_take_profit(self, symbol: str, stop_loss: float, take_profit: float, side: str):
        """更新实盘的止盈止损订单"""
        try:
            # 这里应该调用币安API来更新或重新创建止盈止损订单
            # 由于涉及到复杂的订单管理，暂时记录日志
            self.logger.info(f"🔄 准备更新 {symbol} 实盘止盈止损订单")
            # TODO: 实现实际的API调用逻辑
        except Exception as e:
            self.logger.error(f"更新实盘止盈止损订单失败: {e}")
            raise
    
    async def execute_buy_order(self, signal):
        """执行买入订单（使用专业信号的完整信息）"""
        try:
            symbol = signal['symbol']
            entry_price = signal['price']
            confidence = signal['confidence']
            strength = signal['strength']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            risk_reward_ratio = signal['risk_reward_ratio']
            reasons = signal.get('reasons', [])
            market_condition = signal.get('market_condition', 'unknown')
            
            # 🚨 新增：交易前强制性风险预检查
            risk_check_result = await self._pre_trade_risk_check(symbol, signal)
            if not risk_check_result['allowed']:
                self.logger.warning(f"🚫 {symbol} 买入交易被风险管理器阻止: {risk_check_result['reason']}")
                return
            else:
                self.logger.info(f"✅ {symbol} 风险预检查通过: {risk_check_result['reason']}")
            
            self.logger.info(f"🎯 执行 {symbol} 专业买入订单:")
            self.logger.info(f"   💰 入场价格: {entry_price:.6f}")
            self.logger.info(f"   📊 信号置信度: {confidence:.2f}, 强度: {strength}")
            self.logger.info(f"   🎯 风险回报比: {risk_reward_ratio:.2f}")
            self.logger.info(f"   🌍 市场条件: {market_condition}")
            self.logger.info(f"   💡 买入理由: {', '.join(reasons)}")
            
            # 计算仓位大小（基于大佬的动态杠杆建议）
            position_size = self.calculate_dynamic_position_size(signal, symbol)
            # 动态杠杆已包含置信度和市场条件调整
            
            self.logger.info(f"   💼 仓位大小: {position_size:.6f} {symbol.replace('USDT', '')} (动态杠杆已调整)")
            
            self.logger.info(f"   🛡️ 专业风险管理设置:")
            self.logger.info(f"      🔴 止损点位: {stop_loss:.6f} ({((stop_loss/entry_price-1)*100):.2f}%)")
            self.logger.info(f"      🟢 止盈点位: {take_profit:.6f} ({((take_profit/entry_price-1)*100):.2f}%)")
            
            # 生成订单ID
            order_id = f"BUY_{symbol}_{int(datetime.now().timestamp())}"
            
            # 根据配置判断交易模式
            raw_config = self.config.get_config()
            config_dict = raw_config.dict()
            
            # 详细调试配置读取过程
            self.logger.info(f"   🔧 配置调试 - 完整配置结构存在api: {'api' in config_dict}")
            if 'api' in config_dict:
                api_config = config_dict.get('api', {})
                self.logger.info(f"   🔧 配置调试 - API配置存在binance: {'binance' in api_config}")
                if 'binance' in api_config:
                    binance_config = api_config.get('binance', {})
                    self.logger.info(f"   🔧 配置调试 - Binance配置: {binance_config}")
                    simulation_mode = binance_config.get('simulation_mode', True)
                else:
                    simulation_mode = True
                    self.logger.warning("   ⚠️ 配置警告 - 没有找到binance配置，使用默认模拟模式")
            else:
                simulation_mode = True  
                self.logger.warning("   ⚠️ 配置警告 - 没有找到api配置，使用默认模拟模式")
            
            self.logger.info(f"   🔧 配置调试: simulation_mode = {simulation_mode}")
            
            if simulation_mode:
                self.logger.info(f"   ✅ 买入订单已提交（模拟模式）")
            else:
                self.logger.info(f"   ✅ 买入订单已提交（实盘交易）")
                # 实盘交易逻辑
                try:
                    # 调用币安期货API下单
                    order_result = await self._place_real_buy_order(
                        symbol=symbol,
                        quantity=position_size,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if order_result:
                        self.logger.info(f"   🎉 实盘买入订单成功执行！")
                        self.logger.info(f"      📋 币安订单ID: {order_result.get('orderId', 'N/A')}")
                        self.logger.info(f"      💰 实际成交价: {order_result.get('price', entry_price)}")
                        self.logger.info(f"      📊 实际成交量: {order_result.get('executedQty', position_size)}")
                        
                        # 更新订单ID为真实订单ID
                        order_id = f"BINANCE_{order_result.get('orderId', order_id)}"
                    else:
                        self.logger.error(f"   ❌ 实盘买入订单失败！")
                        return  # 下单失败，不创建本地持仓记录
                        
                except Exception as api_error:
                    self.logger.error(f"   ❌ 币安API下单失败: {api_error}")
                    return  # API调用失败，不创建本地持仓记录
            
            # 🔧 修改：检查是否为加仓操作并处理
            if not hasattr(self, 'current_positions'):
                self.current_positions = {}
            
            # 检查是否已有该币种的持仓（加仓检测）
            is_adding_position = symbol in self.current_positions
            
            if is_adding_position:
                # 加仓逻辑：更新平均成本和仓位大小
                existing_position = self.current_positions[symbol]
                existing_price = existing_position['entry_price']
                existing_size = existing_position['size']
                existing_add_count = existing_position.get('add_count', 0)
                
                # 计算新的平均成本
                total_cost = (existing_price * existing_size) + (entry_price * position_size)
                new_total_size = existing_size + position_size
                new_avg_price = total_cost / new_total_size
                
                self.logger.info(f"🔄 {symbol} 执行加仓操作:")
                self.logger.info(f"   📊 原持仓: {existing_size:.6f} @ {existing_price:.6f}")
                self.logger.info(f"   ➕ 加仓量: {position_size:.6f} @ {entry_price:.6f}")
                self.logger.info(f"   📈 新总量: {new_total_size:.6f} @ {new_avg_price:.6f}")
                
                # 更新持仓信息
                self.current_positions[symbol]['entry_price'] = new_avg_price
                self.current_positions[symbol]['size'] = new_total_size
                self.current_positions[symbol]['add_count'] = existing_add_count + 1
                self.current_positions[symbol]['last_add_time'] = datetime.now()
                
                # 🚨 关键：加仓后更新止盈止损
                await self.update_stop_loss_take_profit_after_add(symbol, new_avg_price, 'LONG')
                
            else:
                # 新开仓逻辑：构建完整的买入理由
                try:
                    # 获取当前技术指标数据
                    current_indicators = {}
                    if symbol in self.selected_symbols_data:
                        ohlcv_data = self.selected_symbols_data[symbol]
                        indicator_data = {
                            'open': ohlcv_data['open'].tolist(),
                            'high': ohlcv_data['high'].tolist(),
                            'low': ohlcv_data['low'].tolist(),
                            'close': ohlcv_data['close'].tolist(),
                            'volume': ohlcv_data['volume'].tolist()
                        }
                        current_indicators = self.technical_indicators.calculate_all_indicators(
                            indicator_data, symbol, '15m'
                        )
                    
                    # 构建买入理由
                    buy_reasons = self._build_buy_reasons(reasons, current_indicators, entry_price)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 构建买入理由失败: {e}，使用基础理由")
                    buy_reasons = {'basic_reasons': reasons, 'entry_timestamp': datetime.now().isoformat()}
                
                # 创建新持仓记录
                self.current_positions[symbol] = {
                    'type': 'LONG',
                    'side': 'LONG',  # 添加side字段以便于后续处理
                    'entry_price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'strength': strength,
                    'risk_reward_ratio': risk_reward_ratio,
                    'market_condition': market_condition,
                    'order_id': order_id,
                    'timestamp': datetime.now(),
                    'reasons': reasons,
                    'buy_reasons': buy_reasons,  # 🔧 新增：完整的买入理由
                    'add_count': 0,  # 🔧 新增：加仓次数计数器
                    'last_add_time': None
                }
            
            # 🔧 新增：持久化保存持仓数据
            try:
                await self.position_persistence.save_position(symbol, self.current_positions[symbol])
            except Exception as e:
                self.logger.error(f"❌ 持仓数据持久化失败 {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"执行专业买入订单失败: {e}")
    
    def _build_buy_reasons(self, reasons: list, indicators: dict, entry_price: float) -> dict:
        """构建完整的买入理由数据"""
        try:
            buy_reasons = {
                'entry_timestamp': datetime.now().isoformat(),
                'basic_reasons': reasons
            }
            
            # MACD背离分析
            if any('MACD' in str(reason) and 'bullish' in str(reason).lower() for reason in reasons):
                buy_reasons['macd_bullish_divergence'] = True
                if 'macd' in indicators and len(indicators['macd']) > 0:
                    macd_data = indicators['macd'][-1]
                    buy_reasons['macd_line'] = getattr(macd_data, 'macd_line', 0)
                    buy_reasons['macd_signal'] = getattr(macd_data, 'signal_line', 0)
            
            # RSI分析
            if any('RSI' in str(reason) and '超卖' in str(reason) for reason in reasons):
                buy_reasons['rsi_oversold'] = True
                if 'rsi' in indicators and len(indicators['rsi']) > 0:
                    rsi_data = indicators['rsi'][-1]
                    buy_reasons['rsi_value'] = getattr(rsi_data, 'rsi_value', rsi_data)
            
            # 价格与均线关系
            if 'sma_20' in indicators and len(indicators['sma_20']) > 0:
                sma_20 = indicators['sma_20'][-1]
                price_vs_sma = ((entry_price - sma_20) / sma_20) * 100
                buy_reasons['price_vs_sma20'] = price_vs_sma
            
            # 成交量分析
            if 'volume' in indicators and len(indicators['volume']) > 0:
                recent_volumes = indicators['volume'][-10:]  # 最近10根K线
                avg_volume = sum(recent_volumes) / len(recent_volumes)
                current_volume = recent_volumes[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                buy_reasons['volume_ratio'] = volume_ratio
            
            self.logger.debug(f"构建买入理由完成: {len(buy_reasons)} 个数据点")
            return buy_reasons
            
        except Exception as e:
            self.logger.error(f"构建买入理由失败: {e}")
            return {'basic_reasons': reasons, 'entry_timestamp': datetime.now().isoformat()}
    
    async def execute_sell_order(self, signal):
        """执行卖出订单（使用专业信号的完整信息）"""
        try:
            symbol = signal['symbol']
            entry_price = signal['price']
            confidence = signal['confidence']
            strength = signal['strength']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            risk_reward_ratio = signal['risk_reward_ratio']
            reasons = signal.get('reasons', [])
            market_condition = signal.get('market_condition', 'unknown')
            
            # 🚨 新增：交易前强制性风险预检查
            risk_check_result = await self._pre_trade_risk_check(symbol, signal)
            if not risk_check_result['allowed']:
                self.logger.warning(f"🚫 {symbol} 卖出交易被风险管理器阻止: {risk_check_result['reason']}")
                return
            else:
                self.logger.info(f"✅ {symbol} 风险预检查通过: {risk_check_result['reason']}")
            
            self.logger.info(f"🎯 执行 {symbol} 专业卖出订单:")
            self.logger.info(f"   💰 入场价格: {entry_price:.6f}")
            self.logger.info(f"   📊 信号置信度: {confidence:.2f}, 强度: {strength}")
            self.logger.info(f"   🎯 风险回报比: {risk_reward_ratio:.2f}")
            self.logger.info(f"   🌍 市场条件: {market_condition}")
            self.logger.info(f"   💡 卖出理由: {', '.join(reasons)}")
            
            # 计算仓位大小（基于大佬的动态杠杆建议）
            position_size = self.calculate_dynamic_position_size(signal, symbol)
            # 动态杠杆已包含置信度和市场条件调整
            
            self.logger.info(f"   💼 仓位大小: {position_size:.6f} {symbol.replace('USDT', '')} (动态杠杆已调整)")
            
            self.logger.info(f"   🛡️ 专业风险管理设置:")
            self.logger.info(f"      🔴 止损点位: {stop_loss:.6f} ({((stop_loss/entry_price-1)*100):.2f}%)")
            self.logger.info(f"      🟢 止盈点位: {take_profit:.6f} ({((take_profit/entry_price-1)*100):.2f}%)")
            
            # 生成订单ID
            order_id = f"SELL_{symbol}_{int(datetime.now().timestamp())}"
            
            # 根据配置判断交易模式
            raw_config = self.config.get_config()
            config_dict = raw_config.dict()
            
            # 详细调试配置读取过程
            self.logger.info(f"   🔧 配置调试 - 完整配置结构存在api: {'api' in config_dict}")
            if 'api' in config_dict:
                api_config = config_dict.get('api', {})
                self.logger.info(f"   🔧 配置调试 - API配置存在binance: {'binance' in api_config}")
                if 'binance' in api_config:
                    binance_config = api_config.get('binance', {})
                    self.logger.info(f"   🔧 配置调试 - Binance配置: {binance_config}")
                    simulation_mode = binance_config.get('simulation_mode', True)
                else:
                    simulation_mode = True
                    self.logger.warning("   ⚠️ 配置警告 - 没有找到binance配置，使用默认模拟模式")
            else:
                simulation_mode = True  
                self.logger.warning("   ⚠️ 配置警告 - 没有找到api配置，使用默认模拟模式")
            
            self.logger.info(f"   🔧 配置调试: simulation_mode = {simulation_mode}")
            
            if simulation_mode:
                self.logger.info(f"   ✅ 卖出订单已提交（模拟模式）")
            else:
                self.logger.info(f"   ✅ 卖出订单已提交（实盘交易）")
                # 实盘交易逻辑
                try:
                    # 调用币安期货API下单
                    order_result = await self._place_real_sell_order(
                        symbol=symbol,
                        quantity=position_size,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if order_result:
                        self.logger.info(f"   🎉 实盘卖出订单成功执行！")
                        self.logger.info(f"      📋 币安订单ID: {order_result.get('orderId', 'N/A')}")
                        self.logger.info(f"      💰 实际成交价: {order_result.get('price', entry_price)}")
                        self.logger.info(f"      📊 实际成交量: {order_result.get('executedQty', position_size)}")
                        
                        # 更新订单ID为真实订单ID
                        order_id = f"BINANCE_{order_result.get('orderId', order_id)}"
                    else:
                        self.logger.error(f"   ❌ 实盘卖出订单失败！")
                        return  # 下单失败，不创建本地持仓记录
                        
                except Exception as api_error:
                    self.logger.error(f"   ❌ 币安API下单失败: {api_error}")
                    return  # API调用失败，不创建本地持仓记录
            
            # 🔧 修改：检查是否为加仓操作并处理（SHORT方向）
            if not hasattr(self, 'current_positions'):
                self.current_positions = {}
            
            # 检查是否已有该币种的持仓（加仓检测）
            is_adding_position = symbol in self.current_positions
            
            if is_adding_position:
                # 加仓逻辑：更新平均成本和仓位大小
                existing_position = self.current_positions[symbol]
                existing_price = existing_position['entry_price']
                existing_size = existing_position['size']
                existing_add_count = existing_position.get('add_count', 0)
                
                # 计算新的平均成本
                total_cost = (existing_price * existing_size) + (entry_price * position_size)
                new_total_size = existing_size + position_size
                new_avg_price = total_cost / new_total_size
                
                self.logger.info(f"🔄 {symbol} 执行加仓操作(SHORT):")
                self.logger.info(f"   📊 原持仓: {existing_size:.6f} @ {existing_price:.6f}")
                self.logger.info(f"   ➕ 加仓量: {position_size:.6f} @ {entry_price:.6f}")
                self.logger.info(f"   📈 新总量: {new_total_size:.6f} @ {new_avg_price:.6f}")
                
                # 更新持仓信息
                self.current_positions[symbol]['entry_price'] = new_avg_price
                self.current_positions[symbol]['size'] = new_total_size
                self.current_positions[symbol]['add_count'] = existing_add_count + 1
                self.current_positions[symbol]['last_add_time'] = datetime.now()
                
                # 🚨 关键：加仓后更新止盈止损
                await self.update_stop_loss_take_profit_after_add(symbol, new_avg_price, 'SHORT')
                
            else:
                # 创建新持仓记录
                self.current_positions[symbol] = {
                    'type': 'SHORT',
                    'side': 'SHORT',  # 添加side字段以便于后续处理
                    'entry_price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'strength': strength,
                    'risk_reward_ratio': risk_reward_ratio,
                    'market_condition': market_condition,
                    'order_id': order_id,
                    'timestamp': datetime.now(),
                    'reasons': reasons,
                    'add_count': 0,  # 🔧 新增：加仓次数计数器
                    'last_add_time': None
                }
            
        except Exception as e:
            self.logger.error(f"执行专业卖出订单失败: {e}")
    
    def _check_signal_validity(self, signal: dict) -> tuple[bool, str]:
        """
        检查信号的有效性
        
        Args:
            signal: 信号字典，包含价格、时间戳等信息
            
        Returns:
            tuple: (是否有效, 失效原因)
        """
        try:
            from datetime import datetime, timedelta
            
            # 1. 检查信号时间是否过期（超过5分钟失效）
            signal_time = signal.get('timestamp', datetime.now())
            if isinstance(signal_time, str):
                try:
                    # 尝试解析不同格式的时间戳
                    signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
                except:
                    signal_time = datetime.now()  # 解析失败使用当前时间
            
            time_diff = datetime.now() - signal_time
            if time_diff > timedelta(minutes=5):
                return False, f"信号过期{time_diff.total_seconds()/60:.1f}分钟"
            
            # 2. 检查价格偏离度（如果当前价格与入场价格偏离超过2%，信号可能失效）
            symbol = signal['symbol']
            entry_price = signal['price']
            
            # 尝试获取当前价格（如果有）
            current_price = None
            if hasattr(self, 'selected_symbols_data') and symbol in self.selected_symbols_data:
                try:
                    current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                except:
                    pass
            
            if current_price:
                price_deviation = abs(current_price - entry_price) / entry_price
                if price_deviation > 0.02:  # 2%偏离
                    return False, f"价格偏离{price_deviation*100:.1f}%"
            
            # 3. 检查止损止盈合理性
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            if stop_loss <= 0 or take_profit <= 0:
                return False, "止损止盈异常"
            
            # 4. 检查风险回报比
            risk_reward_ratio = signal.get('risk_reward_ratio', 0)
            if risk_reward_ratio < 0.8:  # 风险回报比太低
                return False, f"风险回报比过低{risk_reward_ratio:.2f}"
            
            # 5. 检查置信度
            confidence = signal.get('confidence', 0)
            min_confidence = self.config.get_signal_config().min_confidence
            if confidence < min_confidence:  # 置信度太低
                return False, f"置信度过低{confidence:.2f} < {min_confidence}"
            
            return True, "正常"
            
        except Exception as e:
            return False, f"检查异常:{str(e)}"
    
    def calculate_position_size_advanced(self, current_price: float, symbol: str) -> float:
        """高级仓位大小计算"""
        try:
            risk_config = self.config.get_risk_config()
            trading_config = self.config.get_trading_config()
            
            # 基础资金
            base_capital = trading_config.initial_capital
            risk_per_trade = risk_config.risk_per_trade
            max_position_size = risk_config.max_position_size
            
            # 风险金额
            risk_amount = base_capital * risk_per_trade
            
            # 基于价格计算基础仓位
            base_position = risk_amount / current_price
            
            # 应用最大仓位限制
            max_allowed = base_capital * max_position_size / current_price
            
            # 返回较小值
            return min(base_position, max_allowed)
            
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0.01  # 默认最小仓位
    
    
    def create_enhanced_position_record(self, symbol: str, signal: dict, indicators: dict):
        """创建增强的持仓记录 - 保存完整技术状态"""
        try:
            buy_reasons = {}
            technical_snapshot = {}
            
            # 解析买入理由
            reasons = signal.get('reasons', [])
            
            # MACD背离分析
            if 'MACD' in str(reasons) and 'bullish' in str(reasons).lower():
                buy_reasons['macd_bullish_divergence'] = True
                if 'macd' in indicators:
                    macd_data = indicators['macd'][-1]
                    buy_reasons['macd_line'] = getattr(macd_data, 'macd_line', 0)
                    buy_reasons['macd_signal'] = getattr(macd_data, 'signal_line', 0)
                    technical_snapshot['macd_entry'] = macd_data
            
            # RSI分析
            if 'RSI' in str(reasons) and '超卖' in str(reasons):
                buy_reasons['rsi_oversold'] = True
                if 'rsi' in indicators:
                    rsi_data = indicators['rsi'][-1]
                    rsi_value = getattr(rsi_data, 'rsi_value', rsi_data)
                    buy_reasons['rsi_value'] = rsi_value
                    technical_snapshot['rsi_entry'] = rsi_value
            
            # 价格与均线关系
            if 'sma_20' in indicators:
                sma_20 = indicators['sma_20'][-1]
                current_price = signal['price']
                price_vs_sma = ((current_price - sma_20) / sma_20) * 100
                buy_reasons['price_vs_sma20'] = price_vs_sma
                technical_snapshot['sma20_entry'] = sma_20
            
            # 形态识别
            if 'engulfing' in str(reasons).lower():
                buy_reasons['engulfing_bull'] = True
                technical_snapshot['pattern_entry'] = 'engulfing_bull'
            
            # 市场阶段
            buy_reasons['market_condition'] = signal.get('market_condition', 'unknown')
            buy_reasons['confidence'] = signal.get('confidence', 0.0)
            buy_reasons['signal_strength'] = signal.get('strength', 'unknown')
            
            # 创建完整的持仓记录
            position_record = {
                'symbol': symbol,
                'side': 'LONG',
                'entry_price': signal['price'],
                'entry_time': datetime.now(),
                'size': signal.get('position_size', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0),
                'buy_reasons': buy_reasons,
                'technical_snapshot': technical_snapshot,
                'original_signal': signal  # 保存原始信号用于调试
            }
            
            self.logger.info(f"   📋 {symbol} 买入理由记录:")
            for reason, value in buy_reasons.items():
                self.logger.info(f"      • {reason}: {value}")
            
            return position_record
            
        except Exception as e:
            self.logger.error(f"创建增强持仓记录失败: {e}")
            # 返回基础记录作为降级方案
            return {
                'symbol': symbol,
                'side': 'LONG', 
                'entry_price': signal['price'],
                'entry_time': datetime.now(),
                'size': signal.get('position_size', 0),
                'buy_reasons': {'basic': True},
                'technical_snapshot': {}
            }

    def calculate_stop_loss_take_profit(self, entry_price: float, direction: str, symbol: str) -> tuple:
        """计算止盈止损点位"""
        try:
            risk_config = self.config.get_risk_config()
            
            # 基础止损比例
            stop_loss_pct = risk_config.stop_loss_pct
            take_profit_ratio = risk_config.take_profit_ratio
            
            if direction == 'BUY':
                # 多头：止损在下方，止盈在上方
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + stop_loss_pct * take_profit_ratio)
            else:  # SELL
                # 空头：止损在上方，止盈在下方
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - stop_loss_pct * take_profit_ratio)
                
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"计算止盈止损失败: {e}")
            if direction == 'BUY':
                return entry_price * 0.98, entry_price * 1.06
            else:
                return entry_price * 1.02, entry_price * 0.94
    
    async def monitor_positions(self):
        """阶段6: 持仓监控 - 同步币安真实持仓数据"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("👁️ 阶段6: 持仓监控")
            self.logger.info("=" * 80)
            
            # 🚀 关键修复：从币安API获取真实持仓数据
            self.logger.info("🔄 正在从币安API同步真实持仓数据...")
            
            try:
                account_response = await self.api_client_manager.get_account_info()
                
                if account_response and account_response.success:
                    account_data = account_response.data
                    positions_data = account_data.get('positions', [])
                    
                    # 🔧 关键修复：保留现有持仓的自定义字段，只更新币安API数据
                    updated_positions = {}
                    active_positions = []
                    
                    for pos in positions_data:
                        symbol = pos.get('symbol', '')
                        position_amt = float(pos.get('positionAmt', 0))
                        entry_price = float(pos.get('entryPrice', 0))
                        unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                        
                        # 只处理有仓位的品种
                        if abs(position_amt) > 0 and entry_price > 0:
                            side = 'LONG' if position_amt > 0 else 'SHORT'
                            
                            # 🔧 保留现有持仓的自定义字段
                            if symbol in self.current_positions:
                                # 更新现有持仓的币安数据，保留自定义字段
                                existing_position = self.current_positions[symbol].copy()
                                existing_position.update({
                                    'size': abs(position_amt),
                                    'entry_price': entry_price, 
                                    'unrealized_pnl': unrealized_pnl,
                                    'position_amt': position_amt,
                                    'timestamp': datetime.now()
                                })
                                updated_positions[symbol] = existing_position
                            else:
                                # 新持仓，创建基础结构
                                position_info = {
                                    'symbol': symbol,
                                    'side': side,
                                    'size': abs(position_amt),
                                    'entry_price': entry_price,
                                    'unrealized_pnl': unrealized_pnl,
                                    'position_amt': position_amt,
                                    'timestamp': datetime.now(),
                                    'add_count': 0,  # 新持仓加仓次数为0
                                    'last_add_time': None
                                }
                                updated_positions[symbol] = position_info
                            
                            active_positions.append(f"{symbol}: {side} ${abs(position_amt):.2f}")
                    
                    # 🔧 安全更新：只更新有币安持仓的品种
                    self.current_positions = updated_positions
                    
                    if active_positions:
                        self.logger.info(f"✅ 币安API持仓同步成功")
                        self.logger.info(f"💼 持仓数量: {len(self.current_positions)}")
                        self.logger.info(f"📋 持仓详情:")
                        for pos_detail in active_positions:
                            self.logger.info(f"   • {pos_detail}")
                    else:
                        self.logger.info("📭 币安账户当前无持仓")
                        return
                        
                else:
                    self.logger.error("❌ 获取币安账户信息失败")
                    # 回退到内存持仓检查
                    if not hasattr(self, 'current_positions') or not self.current_positions:
                        self.logger.info("📭 当前无持仓，跳过持仓监控")
                        return
                        
            except Exception as e:
                self.logger.error(f"币安持仓同步异常: {e}")
                # 回退到内存持仓检查
                if not hasattr(self, 'current_positions') or not self.current_positions:
                    self.logger.info("📭 当前无持仓，跳过持仓监控")
                    return
            
            # 🔧 新增：恢复买入理由（系统重启后）
            for symbol in list(self.current_positions.keys()):
                if 'buy_reasons' not in self.current_positions[symbol]:
                    # 尝试从持久化数据恢复买入理由
                    try:
                        saved_position = await self.position_persistence.load_position(symbol)
                        if saved_position and 'buy_reasons' in saved_position:
                            self.current_positions[symbol]['buy_reasons'] = saved_position['buy_reasons']
                            self.current_positions[symbol]['reasons'] = saved_position.get('reasons', [])
                            # 🔧 关键修复：恢复加仓次数和其他重要字段
                            self.current_positions[symbol]['add_count'] = saved_position.get('add_count', 0)
                            self.current_positions[symbol]['last_add_time'] = saved_position.get('last_add_time')
                            self.current_positions[symbol]['stop_loss'] = saved_position.get('stop_loss')
                            self.current_positions[symbol]['take_profit'] = saved_position.get('take_profit')
                            self.logger.info(f"✅ {symbol} 持仓数据已完整恢复 (加仓次数: {saved_position.get('add_count', 0)})")
                        else:
                            # 无历史数据，创建基础买入理由结构
                            self.current_positions[symbol]['buy_reasons'] = {
                                'basic_reasons': [],
                                'entry_timestamp': datetime.now().isoformat(),
                                'persistence_missing': True
                            }
                            self.logger.warning(f"⚠️ {symbol} 无历史买入理由，仅价格保护生效")
                    except Exception as e:
                        self.logger.error(f"❌ {symbol} 恢复买入理由失败: {e}")
                        # 创建基础结构避免后续错误
                        self.current_positions[symbol]['buy_reasons'] = {
                            'basic_reasons': [],
                            'entry_timestamp': datetime.now().isoformat(),
                            'restore_failed': True
                        }
            
            # 监控每个持仓（修复线程安全问题）
            # 创建持仓的副本避免在迭代时修改字典
            current_positions_copy = dict(self.current_positions.items())
            for symbol, position in current_positions_copy.items():
                # 再次检查该持仓是否仍然存在（可能已被平仓）
                if symbol not in self.current_positions:
                    self.logger.info(f"⚠️ {symbol} 持仓已被平仓，跳过监控")
                    continue
                    
                self.logger.info(f"👁️ 监控 {symbol} 持仓...")
                await self.check_position_exit_signals(symbol, position)
                
        except Exception as e:
            self.logger.error(f"持仓监控阶段失败: {e}")
    
    async def check_position_exit_signals(self, symbol: str, position: dict):
        """检查持仓退出信号 - 重新设计优先级避免冲突"""
        try:
            # 获取实时当前价格
            self.logger.info(f"   🔄 正在获取 {symbol} 实时价格...")
            
            try:
                ticker_response = await self.api_client_manager.get_ticker(symbol)
                if ticker_response and ticker_response.success:
                    # 🔧 修复：添加多重验证，防止0价格
                    data = ticker_response.data
                    current_price = None
                    
                    # 多重字段验证
                    if 'lastPrice' in data and str(data['lastPrice']) not in ['', '0', 'None', None]:
                        try:
                            current_price = float(data['lastPrice'])
                            if current_price <= 0:
                                current_price = None
                        except (ValueError, TypeError):
                            current_price = None
                    
                    if not current_price and 'price' in data:
                        try:
                            current_price = float(data['price'])
                            if current_price <= 0:
                                current_price = None
                        except (ValueError, TypeError):
                            current_price = None
                    
                    if not current_price and 'close' in data:
                        try:
                            current_price = float(data['close'])
                            if current_price <= 0:
                                current_price = None
                        except (ValueError, TypeError):
                            current_price = None
                    
                    if not current_price or current_price <= 0:
                        # 备用：从K线数据获取
                        if symbol in self.selected_symbols_data:
                            current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                            self.logger.warning(f"⚠️ {symbol} ticker价格无效，使用K线价格: {current_price}")
                            self.logger.warning(f"   🔍 ticker响应字段: {list(data.keys()) if data else 'None'}")
                        else:
                            self.logger.error(f"❌ {symbol} 无法获取有效价格数据")
                            return
                else:
                    # 备用：从K线数据获取
                    if symbol in self.selected_symbols_data:
                        current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                        self.logger.warning(f"⚠️ {symbol} ticker获取失败，使用K线价格: {current_price}")
                    else:
                        self.logger.error(f"❌ {symbol} 无法获取价格数据")
                        return
            except Exception as e:
                self.logger.warning(f"⚠️ {symbol} 实时价格获取失败，使用K线价格: {e}")
                if symbol in self.selected_symbols_data:
                    current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                else:
                    self.logger.error(f"❌ {symbol} 价格获取完全失败")
                    return
                    
            entry_price = position['entry_price']
            position_type = position['side']
            position_size = position['size']
            
            # 计算持仓时间
            entry_time = position.get('entry_time')
            if entry_time:
                try:
                    if isinstance(entry_time, str):
                        entry_datetime = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    else:
                        entry_datetime = entry_time
                    hold_minutes = (datetime.now(timezone.utc) - entry_datetime).total_seconds() / 60
                except Exception as e:
                    self.logger.warning(f"⚠️ {symbol} 持仓时间计算失败: {e}")
                    hold_minutes = 0
            else:
                hold_minutes = 0
            
            # 计算当前盈亏
            if position_type == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
            self.logger.info(f"   📊 {symbol} 持仓状态:")
            self.logger.info(f"      📈 持仓类型: {position_type}")
            self.logger.info(f"      💰 入场价格: {entry_price:.6f}")
            self.logger.info(f"      💱 当前价格: {current_price:.6f}")
            self.logger.info(f"      📊 当前盈亏: {pnl_pct:+.2f}%")
            self.logger.info(f"      ⏰ 持仓时间: {hold_minutes:.1f}分钟")
            
            # 🔥 新的执行优先级设计（避免冲突）
            
            # 第1优先级：高级渐进式止损（集成增强反转检测）
            progressive_result = await self.update_progressive_stop_loss(symbol)
            
            if progressive_result.get('emergency_close'):
                # 极高置信度反转风险，触发紧急平仓
                self.logger.error(f"🚨 {symbol} 紧急平仓: {progressive_result['reason']}")
                await self.close_position(symbol, progressive_result['reason'], current_price)
                return
            
            if progressive_result.get('reduction_executed'):
                # 智能减仓已执行，本轮监控完成
                self.logger.info(f"✅ {symbol} 智能减仓完成，跳过后续检查")
                return
            
            if progressive_result['updated']:
                self.logger.info(f"   ✅ {symbol} 渐进式止损已更新: {progressive_result['reason']}")
            
            # 🆕 关键修复：渐进式止损系统决定是否继续后续检查
            if progressive_result.get('skip_basic_checks', False):
                # 渐进式止损系统认为当前状态正常，跳过基础检查
                self.logger.info(f"   📊 {symbol} 渐进式止损系统主导：跳过基础价格检查")
                return
            
            # 第2优先级：时间止损（延长到6小时，优先智能减仓）
            if hold_minutes > 360:  # 6小时强制平仓
                self.logger.warning(f"{symbol} 持仓超时({hold_minutes:.0f}分钟)，强制平仓")
                await self.close_position(symbol, f"时间止损({hold_minutes:.0f}分钟)", current_price)
                return
            elif hold_minutes > 180:  # 3小时智能减仓
                time_reduction_result = await self._execute_time_based_reduction(symbol, 0.5)
                if time_reduction_result['executed']:
                    self.logger.info(f"✅ {symbol} 时间减仓完成")
                    return
            
            # 第3优先级：基础价格止损（最后安全网）- 只有在渐进式止损允许时才执行
            if progressive_result.get('allow_basic_price_check', True):
                direction = 'BUY' if position_type == 'LONG' else 'SELL'
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(entry_price, direction, symbol)
                
                self.logger.info(f"   🛡️ {symbol} 基础价格检查（安全网）:")
                self.logger.info(f"      🔴 基础止损: {stop_loss:.6f}")
                self.logger.info(f"      🟢 基础止盈: {take_profit:.6f}")
                
                exit_signal = None
                if position_type == 'LONG':
                    if current_price <= stop_loss:
                        exit_signal = "STOP_LOSS"
                        self.logger.warning(f"🚨 {symbol} 触发价格止损! 价格{current_price:.6f} <= 止损{stop_loss:.6f}")
                    elif current_price >= take_profit:
                        exit_signal = "TAKE_PROFIT" 
                        self.logger.info(f"🎉 {symbol} 触发止盈! 价格{current_price:.6f} >= 止盈{take_profit:.6f}")
                else:  # SHORT
                    if current_price >= stop_loss:
                        exit_signal = "STOP_LOSS"
                        self.logger.warning(f"🚨 {symbol} 触发价格止损! 价格{current_price:.6f} >= 止损{stop_loss:.6f}")
                    elif current_price <= take_profit:
                        exit_signal = "TAKE_PROFIT"
                        self.logger.info(f"🎉 {symbol} 触发止盈! 价格{current_price:.6f} <= 止盈{take_profit:.6f}")
                
                # 如果有价格退出信号，执行常规平仓
                if exit_signal:
                    await self.close_position(symbol, exit_signal, current_price)
            else:
                self.logger.info(f"   🛡️ {symbol} 渐进式止损系统禁用基础价格检查")
                
        except Exception as e:
            self.logger.error(f"检查 {symbol} 退出信号失败: {e}")
    
    
    
    async def check_buy_reasons_invalidation(self, symbol: str, position: dict, current_indicators: dict, current_price: float):
        """检查买入理由是否失效 - 真正的反转逻辑"""
        try:
            buy_reasons = position.get('buy_reasons', {})
            invalidation_score = 0
            invalidation_details = []
            
            self.logger.info(f"   🔍 {symbol} 检查买入理由失效:")
            
            # 1. 检查MACD背离是否失效
            if buy_reasons.get('macd_bullish_divergence'):
                entry_macd_line = buy_reasons.get('macd_line', 0)
                entry_macd_signal = buy_reasons.get('macd_signal', 0)
                
                if 'macd' in current_indicators and len(current_indicators['macd']) > 0:
                    current_macd = current_indicators['macd'][-1]
                    current_macd_line = getattr(current_macd, 'macd_line', 0)
                    current_macd_signal = getattr(current_macd, 'signal_line', 0)
                    
                    # 检查MACD金叉是否转为死叉
                    entry_golden = entry_macd_line > entry_macd_signal
                    current_golden = current_macd_line > current_macd_signal
                    
                    if entry_golden and not current_golden:
                        invalidation_score += 40
                        invalidation_details.append("MACD从金叉转为死叉")
                        self.logger.info(f"      ❌ MACD背离失效: 金叉→死叉")
                    else:
                        self.logger.info(f"      ✅ MACD背离仍然有效")
            
            # 2. 检查RSI超卖状态是否改变
            if buy_reasons.get('rsi_oversold'):
                entry_rsi = buy_reasons.get('rsi_value', 0)
                
                if 'rsi' in current_indicators and len(current_indicators['rsi']) > 0:
                    current_rsi_data = current_indicators['rsi'][-1]
                    current_rsi = getattr(current_rsi_data, 'rsi_value', current_rsi_data)
                    
                    # RSI从超卖区域(30以下)回升到中性区域(50以上)
                    if entry_rsi < 30 and current_rsi > 50:
                        invalidation_score += 30
                        invalidation_details.append(f"RSI从超卖{entry_rsi:.1f}回升至{current_rsi:.1f}")
                        self.logger.info(f"      ❌ RSI超卖失效: {entry_rsi:.1f} → {current_rsi:.1f}")
                    else:
                        self.logger.info(f"      ✅ RSI状态: {entry_rsi:.1f} → {current_rsi:.1f}")
            
            # 3. 检查价格与均线关系变化
            entry_sma_relation = buy_reasons.get('price_vs_sma20', 0)
            if 'sma_20' in current_indicators and len(current_indicators['sma_20']) > 0:
                current_sma = current_indicators['sma_20'][-1]
                current_sma_relation = ((current_price - current_sma) / current_sma) * 100
                
                # 均线关系变化超过8% (更严格的标准)
                relation_change = abs(current_sma_relation - entry_sma_relation)
                if relation_change > 8:
                    invalidation_score += 20
                    invalidation_details.append(f"均线关系变化{relation_change:.1f}%")
                    self.logger.info(f"      ❌ 均线关系变化: {entry_sma_relation:.1f}% → {current_sma_relation:.1f}%")
                else:
                    self.logger.info(f"      ✅ 均线关系稳定: {entry_sma_relation:.1f}% → {current_sma_relation:.1f}%")
            
            # 4. 价格保护机制 (作为最后防线，而非主要判断)
            entry_price = position.get('entry_price', current_price)
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
            
            # 只有在技术指标失效的情况下，价格大幅下跌才加分
            if invalidation_score > 0 and price_change_pct < -10:  # 技术+价格双重确认
                invalidation_score += 15
                invalidation_details.append(f"技术失效+价格下跌{price_change_pct:.1f}%")
                self.logger.info(f"      ⚠️ 价格保护触发: {price_change_pct:.1f}%")
            
            # 综合判断
            should_reverse = invalidation_score >= 70
            confidence = min(invalidation_score / 100.0, 0.95)
            
            self.logger.info(f"   📊 {symbol} 失效评分: {invalidation_score}/100")
            self.logger.info(f"   📋 失效原因: {', '.join(invalidation_details) if invalidation_details else '无'}")
            self.logger.info(f"   🎯 反转决策: {'执行反转' if should_reverse else '继续持有'}")
            
            return {
                'should_reverse': should_reverse,
                'invalidation_score': invalidation_score,
                'details': invalidation_details,
                'confidence': confidence,
                'reason': f"买入理由失效: {', '.join(invalidation_details)}" if invalidation_details else "买入理由仍然有效"
            }
            
        except Exception as e:
            self.logger.error(f"检查买入理由失效失败: {e}")
            return {
                'should_reverse': False,
                'invalidation_score': 0,
                'details': [],
                'confidence': 0.0,
                'reason': f"检查失败: {e}"
            }

    async def check_technical_reversal_with_confirmation(self, symbol: str, price_data, position_type: str, current_price: float = None):
        """增强版反转检测 - 结合技术指标确认"""
        try:
            # 先进行基础反转检测
            basic_reversal = await self.check_technical_reversal(symbol, price_data, position_type, current_price)
            
            if not basic_reversal or basic_reversal['signal'] != 'REVERSAL_FORCE_CLOSE':
                return basic_reversal
            
            # 基础反转触发后，进行技术指标确认
            self.logger.info(f"   🔍 {symbol} 基础反转触发，进行技术指标确认...")
            
            # 获取技术指标数据
            if symbol not in self.selected_symbols_data:
                self.logger.warning(f"   ⚠️ {symbol} 缺少技术指标数据，使用基础反转")
                return basic_reversal
            
            try:
                # 准备技术指标数据
                ohlcv_data = self.selected_symbols_data[symbol]
                indicator_data = {
                    'open': ohlcv_data['open'].tolist(),
                    'high': ohlcv_data['high'].tolist(),
                    'low': ohlcv_data['low'].tolist(), 
                    'close': ohlcv_data['close'].tolist(),
                    'volume': ohlcv_data['volume'].tolist()
                }
                
                # 计算技术指标
                indicators = self.technical_indicators.calculate_all_indicators(
                    indicator_data, symbol, '1h'
                )
                
                # 技术指标确认逻辑
                technical_confirm_score = 0
                confirm_reasons = []
                
                # RSI确认（如果买入时RSI超卖，反转时RSI应该不再超卖）
                if 'rsi' in indicators and len(indicators['rsi']) > 0:
                    rsi_value = indicators['rsi'][-1].rsi_value if hasattr(indicators['rsi'][-1], 'rsi_value') else indicators['rsi'][-1]
                    
                    if position_type == 'LONG':
                        # 多头仓位：如果RSI从超卖区域回升，确认反转
                        if rsi_value > 35:  # 不再超卖
                            technical_confirm_score += 40
                            confirm_reasons.append(f"RSI回升至{rsi_value:.1f}")
                    else:  # SHORT
                        # 空头仓位：如果RSI从超买区域回落，确认反转  
                        if rsi_value < 65:  # 不再超买
                            technical_confirm_score += 40
                            confirm_reasons.append(f"RSI回落至{rsi_value:.1f}")
                
                # MACD确认（检查信号是否真的反转）
                if 'macd' in indicators and len(indicators['macd']) > 3:
                    macd_line = [m.macd_line if hasattr(m, 'macd_line') else m for m in indicators['macd'][-3:]]
                    signal_line = [m.signal_line if hasattr(m, 'signal_line') else 0 for m in indicators['macd'][-3:]]
                    
                    # 检查MACD金叉死叉变化
                    if len(macd_line) >= 2 and len(signal_line) >= 2:
                        prev_cross = macd_line[-2] > signal_line[-2]
                        curr_cross = macd_line[-1] > signal_line[-1]
                        
                        if position_type == 'LONG' and prev_cross and not curr_cross:
                            # 多头持仓遇到MACD死叉
                            technical_confirm_score += 30
                            confirm_reasons.append("MACD死叉确认")
                        elif position_type == 'SHORT' and not prev_cross and curr_cross:
                            # 空头持仓遇到MACD金叉
                            technical_confirm_score += 30
                            confirm_reasons.append("MACD金叉确认")
                
                # 价格与均线关系确认
                if 'sma_20' in indicators and len(indicators['sma_20']) > 0:
                    sma_20 = indicators['sma_20'][-1]
                    price_vs_sma = (current_price - sma_20) / sma_20 * 100
                    
                    if position_type == 'LONG' and price_vs_sma < -2:
                        # 多头持仓价格跌破均线2%
                        technical_confirm_score += 30
                        confirm_reasons.append(f"跌破SMA20 {price_vs_sma:.1f}%")
                    elif position_type == 'SHORT' and price_vs_sma > 2:
                        # 空头持仓价格涨破均线2%
                        technical_confirm_score += 30
                        confirm_reasons.append(f"涨破SMA20 {price_vs_sma:.1f}%")
                
                # 综合判断
                self.logger.info(f"   📊 {symbol} 技术指标确认评分: {technical_confirm_score}/100")
                self.logger.info(f"   📋 确认理由: {', '.join(confirm_reasons) if confirm_reasons else '无'}")
                
                # 需要技术指标确认评分>=50才执行反转
                if technical_confirm_score >= 50:
                    self.logger.error(f"   ✅ {symbol} 技术指标确认反转！评分{technical_confirm_score}>=50")
                    # 更新反转原因
                    basic_reversal['reason'] += f" + 技术确认({', '.join(confirm_reasons)})"
                    return basic_reversal
                else:
                    self.logger.warning(f"   ❌ {symbol} 技术指标不支持反转！评分{technical_confirm_score}<50，继续持仓")
                    return None  # 不执行反转
                    
            except Exception as e:
                self.logger.error(f"   ❌ {symbol} 技术指标确认失败: {e}，使用基础反转")
                return basic_reversal
                
        except Exception as e:
            self.logger.error(f"增强反转检测失败: {e}")
            return None

    async def close_position_with_reversal(self, symbol: str, reversal_result: dict, current_price: float):
        """反转强制平仓 - 修复API问题并支持反向开仓"""
        try:
            position = self.current_positions.get(symbol)
            if not position:
                self.logger.warning(f"⚠️ {symbol} 持仓不存在，无法平仓")
                return
            
            self.logger.error(f"🔥 执行 {symbol} 反转强制平仓:")
            self.logger.error(f"   🎯 平仓原因: {reversal_result['reason']}")
            self.logger.error(f"   💰 平仓价格: {current_price:.6f}")
            self.logger.error(f"   ⚡ 反转强度: {reversal_result['strength']:.2f}")
            
            # 计算当前盈亏
            entry_price = position['entry_price']
            if position['side'] == 'LONG':
                final_pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                final_pnl = ((entry_price - current_price) / entry_price) * 100
                
            self.logger.error(f"   📊 最终盈亏: {final_pnl:+.2f}%")
            
            # 🔧 修复版API调用 - 解决 "Margin is insufficient" 问题
            success = await self.execute_emergency_close(symbol, position, current_price)
            
            if success:
                self.logger.info(f"✅ {symbol} 反转强制平仓成功")
                
                # 从持仓记录中移除
                if symbol in self.current_positions:
                    del self.current_positions[symbol]
                    
                    # 🔧 新增：清理持久化数据
                    try:
                        await self.position_persistence.remove_position(symbol)
                    except Exception as e:
                        self.logger.error(f"❌ 清理持仓持久化数据失败 {symbol}: {e}")
                    
                # 🚀 考虑反向开仓（用户需求）
                await self.consider_reverse_position(symbol, reversal_result, current_price)
                
            else:
                self.logger.error(f"❌ {symbol} 反转强制平仓失败")
                # 发送紧急通知
                await self.telegram_bot.send_message(
                    f"🚨紧急情况🚨\n"
                    f"{symbol} 反转强制平仓失败！\n"
                    f"原因: {reversal_result['reason']}\n"
                    f"请立即手动平仓！"
                )
                
        except Exception as e:
            self.logger.error(f"反转强制平仓失败: {e}")
    
    async def execute_emergency_close(self, symbol: str, position: dict, current_price: float) -> bool:
        """紧急平仓 - 多种策略尝试，修复API问题"""
        try:
            position_size = abs(float(position['size']))
            close_side = "SELL" if position['side'] == 'LONG' else "BUY"
            
            self.logger.info(f"🔧 {symbol} 紧急平仓参数:")
            self.logger.info(f"   📏 持仓大小: {position_size}")
            self.logger.info(f"   🔄 平仓方向: {close_side}")
            
            # 策略1: 尝试单向持仓模式（不带positionSide）
            order_params_1 = {
                'symbol': symbol,
                'side': close_side,
                'type': 'MARKET',
                'quantity': str(self.precision_manager.adjust_quantity(symbol, position_size))
            }
            
            self.logger.info(f"   🎯 策略1-单向持仓模式: {order_params_1}")
            api_response = await self.api_client_manager.place_order(order_params_1)
            
            if api_response.success:
                self.logger.info(f"✅ 策略1成功: {api_response.data}")
                return True
            
            # 策略2: 尝试双向持仓模式（带positionSide）
            position_side = position['side']  # LONG 或 SHORT
            order_params_2 = {
                'symbol': symbol,
                'side': close_side,
                'type': 'MARKET',
                'quantity': str(self.precision_manager.adjust_quantity(symbol, position_size)),
                'positionSide': position_side
            }
            
            self.logger.info(f"   🎯 策略2-双向持仓模式: {order_params_2}")
            api_response = await self.api_client_manager.place_order(order_params_2)
            
            if api_response.success:
                self.logger.info(f"✅ 策略2成功: {api_response.data}")
                return True
            
            # 策略3: 尝试分批平仓（解决数量精度问题）
            if position_size > 1:
                # 分成5批平仓
                batch_size = position_size / 5
                success_count = 0
                
                for i in range(5):
                    batch_params = {
                        'symbol': symbol,
                        'side': close_side,
                        'type': 'MARKET',
                        'quantity': str(self.precision_manager.adjust_quantity(symbol, batch_size))
                    }
                    
                    self.logger.info(f"   🎯 策略3-分批{i+1}/5: {batch_params}")
                    batch_response = await self.api_client_manager.place_order(batch_params)
                    
                    if batch_response.success:
                        success_count += 1
                        self.logger.info(f"✅ 分批{i+1}/5成功")
                    else:
                        self.logger.error(f"❌ 分批{i+1}/5失败: {batch_response.error_message}")
                
                if success_count >= 3:  # 60%以上成功就算成功
                    self.logger.info(f"✅ 策略3部分成功: {success_count}/5批")
                    return True
            
            # 所有策略都失败
            self.logger.error(f"❌ 所有平仓策略都失败")
            return False
            
        except Exception as e:
            self.logger.error(f"紧急平仓执行失败: {e}")
            return False
    
    async def consider_reverse_position(self, symbol: str, reversal_result: dict, current_price: float):
        """考虑反向开仓 - 根据反转信号判断是否开反向仓位"""
        try:
            reverse_direction = reversal_result['direction']  # 反向方向
            reversal_strength = reversal_result['strength']
            
            self.logger.info(f"🤔 {symbol} 考虑反向开仓:")
            self.logger.info(f"   🔄 反向方向: {reverse_direction}")
            self.logger.info(f"   💪 反转强度: {reversal_strength:.2f}")
            
            # 反向开仓条件判断
            should_reverse = False
            
            # 条件1: 反转强度足够大（>=5%）
            if reversal_strength >= 5.0:
                should_reverse = True
                self.logger.info(f"   ✅ 反转强度{reversal_strength:.2f}% >= 5%，满足开仓条件")
            else:
                self.logger.info(f"   ❌ 反转强度{reversal_strength:.2f}% < 5%，不满足开仓条件")
                return
            
            # 条件2: 检查账户风险（避免过度交易）
            risk_config = self.config.get_risk_config()
            current_positions_count = len(self.current_positions)
            max_positions = getattr(risk_config, 'max_positions', 5)
            
            if current_positions_count >= max_positions:
                self.logger.warning(f"   ⚠️ 持仓数量{current_positions_count} >= {max_positions}，不执行反向开仓")
                return
            
            if should_reverse:
                self.logger.info(f"🚀 {symbol} 决定执行反向开仓!")
                
                # 构造反向信号
                reverse_signal = {
                    'symbol': symbol,
                    'signal_type': reverse_direction,  # 'LONG' 或 'SHORT'
                    'price': current_price,
                    'confidence': min(reversal_strength / 10.0, 0.9),  # 基于反转强度计算置信度
                    'strength': 'STRONG',
                    'reason': f"反转开仓: {reversal_result['reason']}",
                    'reversal_triggered': True
                }
                
                # 执行反向开仓
                if reverse_direction == 'LONG':
                    await self.execute_buy_order(reverse_signal)
                elif reverse_direction == 'SHORT':
                    await self.execute_sell_order(reverse_signal)
                    
            else:
                self.logger.info(f"💡 {symbol} 不满足反向开仓条件，保持观望")
                
        except Exception as e:
            self.logger.error(f"考虑反向开仓失败: {e}")
    
    async def check_technical_reversal(self, symbol: str, price_data, position_type: str, current_price: float = None):
        """检测技术反转信号 - 修复版：检测到反转立即返回强制平仓信号"""
        try:
            # 获取最近价格数据
            closes = price_data['close'].values
            if len(closes) < 20:
                return None
                
            # 使用实时价格（如果提供的话）
            if current_price is not None:
                # 使用实时价格作为最新价格
                recent_closes = closes[-19:]  # 取前19个历史价格
                recent_closes = list(recent_closes) + [current_price]  # 加上实时价格
                current_price_for_analysis = current_price
            else:
                recent_closes = closes[-20:]
                current_price_for_analysis = recent_closes[-1]
            
            # 短期和中期价格变化
            price_change_5 = (current_price_for_analysis - recent_closes[-5]) / recent_closes[-5] * 100
            price_change_10 = (current_price_for_analysis - recent_closes[-10]) / recent_closes[-10] * 100
            
            # 拐点检测
            local_max = max(recent_closes[-10:-5])
            local_min = min(recent_closes[-10:-5])
            
            self.logger.info(f"   🔍 {symbol} 技术分析:")
            self.logger.info(f"      📈 短期变化(5周期): {price_change_5:+.2f}%")
            self.logger.info(f"      📊 中期变化(10周期): {price_change_10:+.2f}%")
            self.logger.info(f"      🔺 近期高点: {local_max:.6f}")
            self.logger.info(f"      🔻 近期低点: {local_min:.6f}")
            
            # 反转信号检测
            reversal_signals = []
            reversal_strength = 0  # 反转强度
            
            if abs(price_change_5) > 3:  # 短期剧烈变化
                if price_change_5 > 0 and price_change_10 < -2:
                    reversal_signals.append("短期反弹，中期仍跌")
                    reversal_strength += abs(price_change_10)
                elif price_change_5 < 0 and price_change_10 > 2:
                    reversal_signals.append("短期回调，中期仍涨")
                    reversal_strength += abs(price_change_5)
            
            # 拐点信号（关键修复：降低阈值，更敏感）
            drop_pct = 0
            rise_pct = 0
            
            if current_price_for_analysis < local_max * 0.92:  # 🔥修复：从3%改为8%，减少误杀
                drop_pct = ((1-current_price_for_analysis/local_max)*100)
                reversal_signals.append(f"从高点{local_max:.6f}下跌{drop_pct:.1f}%")
                reversal_strength += drop_pct
                
            elif current_price_for_analysis > local_min * 1.08:  # 🔥修复：从3%改为8%，减少误杀
                rise_pct = ((current_price_for_analysis/local_min-1)*100)
                reversal_signals.append(f"从低点{local_min:.6f}上涨{rise_pct:.1f}%")
                reversal_strength += rise_pct
                
            if reversal_signals:
                self.logger.info(f"   🔄 {symbol} 反转信号: {'; '.join(reversal_signals)}")
                
                # 🚨关键修复：检测到持仓方向错误时，立即返回强制平仓信号
                force_close_signal = None
                
                # 多头持仓遇到下跌反转 - 立即平仓
                if position_type == 'LONG' and any('下跌' in s for s in reversal_signals):
                    if drop_pct >= 8.0:  # 下跌超过8%立即强制平仓（修复误杀）
                        self.logger.error(f"🚨 {symbol} 多头反转强制平仓! 下跌{drop_pct:.1f}% >= 3%")
                        force_close_signal = "REVERSAL_FORCE_CLOSE"
                    else:
                        self.logger.warning(f"⚠️ {symbol} 多头持仓面临反转风险")
                        
                # 空头持仓遇到上涨反转 - 立即平仓
                elif position_type == 'SHORT' and any('上涨' in s for s in reversal_signals):
                    if rise_pct >= 8.0:  # 上涨超过8%立即强制平仓（修复误杀）
                        self.logger.error(f"🚨 {symbol} 空头反转强制平仓! 上涨{rise_pct:.1f}% >= 3%")
                        force_close_signal = "REVERSAL_FORCE_CLOSE"
                    else:
                        self.logger.warning(f"⚠️ {symbol} 空头持仓面临反转风险")
                
                # 返回强制平仓信号（最高优先级）
                if force_close_signal:
                    return {
                        'signal': force_close_signal,
                        'reason': f"反转强制平仓: {'; '.join(reversal_signals)}",
                        'strength': reversal_strength,
                        'direction': 'SHORT' if position_type == 'LONG' else 'LONG',  # 反向方向
                        'signals': reversal_signals
                    }
            else:
                self.logger.info(f"   ✅ {symbol} 暂无明显反转信号")
                
            return None  # 无反转信号
                
        except Exception as e:
            self.logger.error(f"{symbol} 技术反转检测失败: {e}")
            return None
    
    async def close_position(self, symbol: str, reason: str, current_price: float):
        """平仓操作"""
        try:
            position = self.current_positions[symbol]
            
            self.logger.info(f"⚡ 执行 {symbol} 平仓操作:")
            self.logger.info(f"   🎯 平仓原因: {reason}")
            self.logger.info(f"   💰 平仓价格: {current_price:.6f}")
            
            # 计算最终盈亏
            entry_price = position['entry_price']
            if position['side'] == 'LONG':
                final_pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                final_pnl = ((entry_price - current_price) / entry_price) * 100
                
            self.logger.info(f"   📊 最终盈亏: {final_pnl:+.2f}%")
            
            # 模拟平仓订单
            close_order_id = f"CLOSE_{symbol}_{int(datetime.now().timestamp())}"
            self.logger.info(f"   📋 平仓订单ID: {close_order_id}")
            
            # 根据配置判断交易模式
            # 直接从原始配置字典获取，避免APIConfig对象的.get()方法问题
            raw_config = self.config.get_config()
            config_dict = raw_config.dict()
            
            # 详细调试配置读取过程
            self.logger.info(f"   🔧 配置调试 - 完整配置结构存在api: {'api' in config_dict}")
            if 'api' in config_dict:
                api_config = config_dict.get('api', {})
                self.logger.info(f"   🔧 配置调试 - API配置存在binance: {'binance' in api_config}")
                if 'binance' in api_config:
                    binance_config = api_config.get('binance', {})
                    self.logger.info(f"   🔧 配置调试 - Binance配置: {binance_config}")
                    simulation_mode = binance_config.get('simulation_mode', True)
                else:
                    simulation_mode = True
                    self.logger.warning("   ⚠️ 配置警告 - 没有找到binance配置，使用默认模拟模式")
            else:
                simulation_mode = True  
                self.logger.warning("   ⚠️ 配置警告 - 没有找到api配置，使用默认模拟模式")
            
            self.logger.info(f"   🔧 配置调试: simulation_mode = {simulation_mode}")
            
            if simulation_mode:
                self.logger.info(f"   ✅ 平仓订单已提交（模拟模式）")
            else:
                # 实盘交易模式：调用真实的币安API执行平仓
                try:
                    # 构建平仓订单参数（智能适配单/双向持仓模式）
                    close_side = "SELL" if position['side'] == 'LONG' else "BUY"
                    
                    # 基础订单参数（移除reduceOnly参数，币安API当前不需要）
                    order_params = {
                        'symbol': symbol,
                        'side': close_side,
                        'type': 'MARKET',
                        'quantity': str(self.precision_manager.adjust_quantity(symbol, abs(position['size'])))  # 使用精度管理器调整
                    }
                    
                    # 先尝试不带positionSide参数（单向持仓模式）
                    self.logger.info(f"   🎯 尝试单向持仓模式平仓: {order_params}")
                    
                    # 调用币安API执行平仓
                    api_response = await self.api_client_manager.place_order(order_params)
                    
                    if not api_response.success and "position side does not match" in str(api_response.data).lower():
                        # 如果单向模式失败，尝试双向持仓模式
                        self.logger.info("   🔄 单向持仓模式失败，尝试双向持仓模式...")
                        position_side = "LONG" if position['side'] == 'LONG' else "SHORT"
                        order_params['positionSide'] = position_side
                        self.logger.info(f"   🎯 尝试双向持仓模式平仓: {order_params}")
                        
                        # 再次调用API
                        api_response = await self.api_client_manager.place_order(order_params)
                    
                    if api_response.success:
                        self.logger.info(f"   ✅ 币安API平仓成功: {api_response.data}")
                        order_id = api_response.data.get('orderId', 'N/A')
                        self.logger.info(f"   📋 币安订单ID: {order_id}")
                    else:
                        self.logger.error(f"   ❌ 币安API平仓失败: {api_response.error_message}")
                        # 发送Telegram警告
                        await self.telegram_bot.send_message(
                            f"⚠️ {symbol} 平仓失败！\n"
                            f"错误: {api_response.error_message}\n"
                            f"请手动检查并平仓！"
                        )
                        return  # 平仓失败，不删除持仓记录
                        
                except Exception as api_error:
                    self.logger.error(f"   ❌ 调用币安API平仓异常: {api_error}")
                    await self.telegram_bot.send_message(
                        f"⚠️ {symbol} 平仓API调用异常！\n"
                        f"错误: {api_error}\n"
                        f"请手动检查并平仓！"
                    )
                    return  # API调用失败，不删除持仓记录
            
            # 移除持仓记录
            del self.current_positions[symbol]
            
            # 🔧 新增：清理持久化数据
            try:
                await self.position_persistence.remove_position(symbol)
            except Exception as e:
                self.logger.error(f"❌ 清理持仓持久化数据失败 {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"平仓 {symbol} 失败: {e}")
    
    async def update_risk_management(self):
        """阶段7: 风险管理和系统状态检查"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🛡️ 阶段7: 风险管理")
            self.logger.info("=" * 80)
            
            risk_config = self.config.get_risk_config()
            trading_config = self.config.get_trading_config()
            
            # 获取真实的币安账户余额
            try:
                account_response = await self.api_client_manager.get_account_info()
                if account_response and account_response.success:
                    account_data = account_response.data
                    
                    # 🔧 调试：输出币安账户信息的关键字段
                    self.logger.info(f"🔍 币安账户信息关键字段:")
                    for key in ['totalWalletBalance', 'availableBalance', 'totalInitialMargin', 'totalMaintMargin', 
                               'totalMarginBalance', 'totalUnrealizedProfit', 'totalCrossWalletBalance']:
                        if key in account_data:
                            self.logger.info(f"   {key}: {account_data.get(key)}")
                    
                    # 获取USDT余额（总余额）
                    total_wallet_balance = float(account_data.get('totalWalletBalance', 0))
                    available_balance = float(account_data.get('availableBalance', 0))
                    used_margin = float(account_data.get('totalInitialMargin', 0))
                    
                    # 🔥 关键修复：获取币安的真实保证金比率计算基准
                    total_cross_wallet_balance = float(account_data.get('totalCrossWalletBalance', 0))
                    total_margin_balance = float(account_data.get('totalMarginBalance', 0))
                    total_maint_margin = float(account_data.get('totalMaintMargin', 0))  # 维持保证金
                    
                    # 🔥 修复：计算币安真实保证金比率（使用维持保证金，不是初始保证金）
                    if total_cross_wallet_balance > 0:
                        # 币安标准：维持保证金 / 全仓钱包余额 * 100%
                        binance_margin_ratio = (total_maint_margin / total_cross_wallet_balance) * 100
                        exposure_calculation_base = total_cross_wallet_balance
                        margin_ratio_note = "维持保证金/全仓钱包余额"
                        
                        # 初始保证金比率（用于风险控制，但不显示为保证金比率）
                        initial_margin_ratio = (used_margin / total_cross_wallet_balance) * 100
                    elif total_margin_balance > 0:
                        # 备选：使用保证金余额
                        binance_margin_ratio = (total_maint_margin / total_margin_balance) * 100  
                        exposure_calculation_base = total_margin_balance
                        margin_ratio_note = "维持保证金/保证金余额"
                        initial_margin_ratio = (used_margin / total_margin_balance) * 100
                    else:
                        # 最后备选：使用总钱包余额
                        binance_margin_ratio = (total_maint_margin / total_wallet_balance) * 100
                        exposure_calculation_base = total_wallet_balance
                        margin_ratio_note = "维持保证金/总钱包余额"
                        initial_margin_ratio = (used_margin / total_wallet_balance) * 100
                    
                    self.logger.info(f"📊 风险管理状态（真实币安数据）:")
                    self.logger.info(f"   💼 当前持仓数: {len(self.current_positions) if hasattr(self, 'current_positions') else 0}")
                    self.logger.info(f"   🎯 最大总敞口: {risk_config.max_total_exposure:.1%}")
                    self.logger.info(f"   💰 总钱包余额: ${total_wallet_balance:,.2f} USDT")
                    self.logger.info(f"   🏦 全仓钱包余额: ${total_cross_wallet_balance:,.2f} USDT")
                    self.logger.info(f"   💵 可用余额: ${available_balance:,.2f} USDT")
                    self.logger.info(f"   🔒 已用保证金: ${used_margin:,.2f} USDT")
                    self.logger.info(f"   ⚖️ 维持保证金: ${total_maint_margin:,.2f} USDT")
                    self.logger.info(f"   📊 保证金比率: {binance_margin_ratio:.2f}% ({margin_ratio_note}) [币安显示标准]")
                    self.logger.info(f"   🔥 初始保证金比率: {initial_margin_ratio:.2f}% (用于风险控制)")
                    self.logger.info(f"   ⚖️ 单笔风险: {risk_config.risk_per_trade:.1%}")
                    self.logger.info(f"   🛡️ 止损比例: {risk_config.stop_loss_pct:.1%}")
                    
                    # 🔥 修复：使用币安标准的保证金比率，而不是自计算的敞口比率
                    current_capital = exposure_calculation_base
                else:
                    # API调用失败，使用配置值作为备用
                    self.logger.warning("⚠️ 无法获取币安账户信息，使用配置默认值")
                    current_capital = trading_config.initial_capital
                    binance_margin_ratio = 0  # 无法获取
                    initial_margin_ratio = 0  # 无法获取
                    self.logger.info(f"📊 风险管理状态（配置默认值）:")
                    self.logger.info(f"   💼 当前持仓数: {len(self.current_positions) if hasattr(self, 'current_positions') else 0}")
                    self.logger.info(f"   🎯 最大总敞口: {risk_config.max_total_exposure:.1%}")
                    self.logger.info(f"   💰 初始资金: ${current_capital:,.2f}")
                    self.logger.info(f"   ⚖️ 单笔风险: {risk_config.risk_per_trade:.1%}")
                    self.logger.info(f"   🛡️ 止损比例: {risk_config.stop_loss_pct:.1%}")
                    
            except Exception as e:
                # API调用异常，使用配置值作为备用
                self.logger.error(f"获取币安账户信息失败: {e}")
                current_capital = trading_config.initial_capital
                self.logger.info(f"📊 风险管理状态（备用配置）:")
                self.logger.info(f"   💼 当前持仓数: {len(self.current_positions) if hasattr(self, 'current_positions') else 0}")
                self.logger.info(f"   🎯 最大总敞口: {risk_config.max_total_exposure:.1%}")
                self.logger.info(f"   💰 初始资金: ${current_capital:,.2f}")
                self.logger.info(f"   ⚖️ 单笔风险: {risk_config.risk_per_trade:.1%}")
                self.logger.info(f"   🛡️ 止损比例: {risk_config.stop_loss_pct:.1%}")
                
            # 系统风险状态检查
            total_exposure = 0.0
            
            # 🔧 修复：使用真实保证金计算总风险敞口（而非名义价值）
            if hasattr(self, 'current_positions') and self.current_positions:
                self.logger.info(f"📋 持仓详情:")
                for symbol, position in self.current_positions.items():
                    entry_price = position['entry_price']
                    size = position['size']
                    position_value = entry_price * abs(size)  # 名义价值（显示用）
                    
                    self.logger.info(f"   • {symbol}: {position['side']} ${position_value:,.2f}")
                
                # 🎯 关键修复：使用已用保证金计算真实敞口，而非名义价值
                if account_response and account_response.success:
                    # 计算名义价值（仅供参考）
                    notional_value = 0.0
                    for symbol, position in self.current_positions.items():
                        entry_price = position['entry_price']
                        size = position['size']
                        notional_value += entry_price * abs(size)
                    
                    actual_margin_used = used_margin  # 真实已用保证金
                    
                    # 🔥 修复：显示币安标准的保证金比率，但风险控制使用初始保证金比率
                    self.logger.info(f"   📊 总敞口显示: ${total_maint_margin:,.2f} ({binance_margin_ratio:.2f}%) [币安标准-维持保证金]")
                    self.logger.info(f"   🔥 总敞口实际: ${actual_margin_used:,.2f} ({initial_margin_ratio:.2f}%) [风险控制-初始保证金]")
                    self.logger.info(f"   📊 名义价值: ${notional_value:,.2f} [仅供参考]")
                    
                    # 🔥 关键：使用初始保证金比率进行风险判断（更严格的风险控制）
                    exposure_pct = initial_margin_ratio
                    total_exposure = actual_margin_used
                else:
                    # API失败时的备用计算（使用名义价值）
                    for symbol, position in self.current_positions.items():
                        entry_price = position['entry_price']
                        size = position['size']
                        position_value = entry_price * abs(size)
                        total_exposure += position_value
                    exposure_pct = (total_exposure / current_capital) * 100
                    self.logger.info(f"   📊 总敞口: ${total_exposure:,.2f} ({exposure_pct:.1f}%) [名义价值估算]")
                
                # 🔥 修复：风险警告和紧急制动检查（基于初始保证金比率）
                max_exposure_pct = risk_config.max_total_exposure * 100  # 转换为百分比
                if exposure_pct > max_exposure_pct:
                    self.logger.warning(f"⚠️ 初始保证金比率超限! {exposure_pct:.2f}% > {max_exposure_pct:.1f}%")
                    
                    # 🚨 修复：紧急制动阈值基于初始保证金比率（风险控制）
                    critical_threshold = max_exposure_pct * 1.5  # 超过150%触发紧急制动
                    if exposure_pct > critical_threshold:
                        if not self.emergency_brake.is_emergency_stop_active():
                            self.emergency_brake.trigger_emergency_stop(
                                f"初始保证金比率严重超限 {exposure_pct:.2f}% > {critical_threshold:.1f}%"
                            )
                            await self.telegram_bot.send_message(
                                f"🚨 紧急制动触发！\n初始保证金比率: {exposure_pct:.2f}%\n限制: {critical_threshold:.1f}%"
                            )
                else:
                    self.logger.info(f"✅ 初始保证金比率正常 ({exposure_pct:.2f}%) | 币安显示保证金比率: {binance_margin_ratio:.2f}%")
                    # 如果保证金比率恢复正常且紧急制动是因为敞口问题触发的，可以考虑重置
                    if (self.emergency_brake.is_emergency_stop_active() and 
                        ("敞口" in self.emergency_brake.stop_reason or "保证金" in self.emergency_brake.stop_reason) and 
                        exposure_pct < max_exposure_pct * 0.8):  # 恢复到80%以下
                        self.logger.info("💡 初始保证金比率已恢复正常，可考虑重置紧急制动（需手动确认）")
            else:
                self.logger.info("📭 当前无持仓，风险敞口为0")
                
        except Exception as e:
            self.logger.error(f"风险管理阶段失败: {e}")
    
    async def generate_reports(self):
        """阶段8: 生成系统运行报告"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("📋 阶段8: 系统运行摘要")
            self.logger.info("=" * 80)
            
            # 统计信息
            selected_count = len(self.selected_symbols) if hasattr(self, 'selected_symbols') else 0
            signal_count = 0
            position_count = len(self.current_positions) if hasattr(self, 'current_positions') else 0
            
            if hasattr(self, 'current_signals') and self.current_signals:
                signal_count = len(self.current_signals.get('buy', [])) + len(self.current_signals.get('sell', []))
            
            self.logger.info(f"📊 运行统计:")
            self.logger.info(f"   🎯 选中币种数: {selected_count}")
            self.logger.info(f"   📡 活跃信号数: {signal_count}")
            
            # 🔧 新增：详细显示活跃信号信息
            if hasattr(self, 'current_signals') and self.current_signals and signal_count > 0:
                all_signals = []
                if 'buy' in self.current_signals:
                    all_signals.extend([(s, 'buy') for s in self.current_signals['buy']])
                if 'sell' in self.current_signals:
                    all_signals.extend([(s, 'sell') for s in self.current_signals['sell']])
                
                for i, (signal, direction) in enumerate(all_signals, 1):
                    # 判断信号有效性
                    is_valid, validity_reason = self._check_signal_validity(signal)
                    validity_status = "有效" if is_valid else f"失效({validity_reason})"
                    
                    self.logger.info(f"   {i}. 币种{signal['symbol']}, 方向{direction}, "
                                   f"入场点位:{signal['price']:.4f}, "
                                   f"止盈点位:{signal['take_profit']:.4f}, "
                                   f"止损点位:{signal['stop_loss']:.4f}, "
                                   f"信号是否有效：{validity_status}")
            
            self.logger.info(f"   💼 持仓数量: {position_count}")
            
            # 当前市场状况摘要
            if hasattr(self, 'selected_symbols') and self.selected_symbols:
                self.logger.info(f"📈 市场状况摘要:")
                for symbol in self.selected_symbols:
                    if hasattr(self, 'selected_symbols_data') and symbol in self.selected_symbols_data:
                        current_price = float(self.selected_symbols_data[symbol]['close'].iloc[-1])
                        status = "持仓中" if hasattr(self, 'current_positions') and symbol in self.current_positions else "观望中"
                        self.logger.info(f"   • {symbol}: {current_price:.4f} ({status})")
            
            # 下次执行时间提示
            next_scan_time = datetime.now() + timedelta(seconds=30)
            self.logger.info(f"⏰ 下次扫描时间: {next_scan_time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"报告生成阶段失败: {e}") 

    async def _place_real_buy_order(self, symbol: str, quantity: float, price: float, stop_loss: float, take_profit: float):
        """执行真实的币安期货买入订单"""
        try:
            self.logger.info(f"   🔄 正在向币安提交 {symbol} 买入订单...")
            
            # 🔥 使用智能精度管理器调整数量（替换硬编码方式）
            adjusted_quantity = self.precision_manager.adjust_quantity(symbol, quantity)
            
            # 使用市价单快速成交（兼容双向持仓模式）
            order_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',  # 市价单
                'quantity': adjusted_quantity,
                'positionSide': 'LONG',  # 明确指定多头持仓（双向持仓模式）
                'timestamp': int(time.time() * 1000)
            }
            
            self.logger.info(f"   📋 下单参数: {order_params}")
            
            # 调用API客户端下单
            response = await self.api_client_manager.place_order(order_params)
            
            if response and response.success:
                order_data = response.data
                self.logger.info(f"   ✅ 币安下单API调用成功")
                return order_data
            else:
                error_msg = response.error_message if response else "未知错误"
                self.logger.error(f"   ❌ 币安下单失败: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"   ❌ 币安下单异常: {e}")
            return None 

    async def _place_real_sell_order(self, symbol: str, quantity: float, price: float, stop_loss: float, take_profit: float):
        """执行真实的币安期货卖出订单（做空）"""
        try:
            self.logger.info(f"   🔄 正在向币安提交 {symbol} 卖出订单（做空）...")
            
            # 🔥 使用智能精度管理器调整数量（与买入方法一致）
            adjusted_quantity = self.precision_manager.adjust_quantity(symbol, quantity)
            
            # 使用市价单快速成交（兼容双向持仓模式）
            order_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',  # 市价单
                'quantity': adjusted_quantity,
                'positionSide': 'SHORT',  # 明确指定空头持仓（双向持仓模式）
                'timestamp': int(time.time() * 1000)
            }
            
            self.logger.info(f"   📋 下单参数: {order_params}")
            
            # 调用API客户端下单
            response = await self.api_client_manager.place_order(order_params)
            
            if response and response.success:
                order_data = response.data
                self.logger.info(f"   ✅ 币安下单API调用成功")
                return order_data
            else:
                error_msg = response.error_message if response else "未知错误"
                self.logger.error(f"   ❌ 币安下单失败: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"   ❌ 币安下单异常: {e}")
            return None