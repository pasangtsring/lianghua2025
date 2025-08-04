"""
动态参数管理器 - 根据市场条件动态调整止损参数
基于ROI等级、市场环境、波动率的智能参数优化系统
"""

import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

class DynamicParameterManager:
    """动态参数管理器 - 智能止损参数优化"""
    
    def __init__(self, symbol: str, config=None):
        self.symbol = symbol
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 基础保护比例配置 - 基于改造计划的6级ROI系统
        self.base_protection_ratios = {
            0: 0.0,    # Level 0: 保持原止损 (ROI -10% ~ 5%)
            1: 0.0,    # Level 1: 移至保本点 (ROI 5% ~ 15%)
            2: 0.25,   # Level 2: 保护25%利润 (ROI 15% ~ 30%) ✅提高
            3: 0.40,   # Level 3: 保护40%利润 (ROI 30% ~ 50%) ✅提高
            4: 0.55,   # Level 4: 保护55%利润 (ROI 50% ~ 80%) ✅提高
            5: 0.70,   # Level 5: 保护70%利润 (ROI 80% ~ 150%) ✅提高
            6: 0.85    # Level 6: 保护85%利润 (ROI 150%+) ✅提高
        }
        
        # 减仓配置 - 基于改造计划
        self.reduction_configs = {
            4: {'target_reduction': 0.20, 'cumulative': False},  # Level 4: 减仓20%
            5: {'target_reduction': 0.40, 'cumulative': True},   # Level 5: 累计减仓40%
            6: {'target_reduction': 0.70, 'cumulative': True}    # Level 6: 累计减仓70%
        }
        
        # ROI等级边界
        self.roi_boundaries = {
            0: (-10, 5),
            1: (5, 15),
            2: (15, 30),
            3: (30, 50),
            4: (50, 80),
            5: (80, 150),
            6: (150, 1000)
        }
        
        self.logger.info(f"动态参数管理器初始化完成 - {symbol}")
    
    def determine_roi_level(self, roi_pct: float) -> int:
        """确定ROI等级"""
        try:
            for level, (min_roi, max_roi) in self.roi_boundaries.items():
                if min_roi <= roi_pct < max_roi:
                    return level
            
            # 如果ROI超过最高等级，返回最高等级
            if roi_pct >= 150:
                return 6
            
            # 如果ROI低于最低等级，返回最低等级
            if roi_pct < -10:
                return 0
                
            return 0  # 默认返回0级
            
        except Exception as e:
            self.logger.error(f"ROI等级判断失败: {e}")
            return 0
    
    def get_protection_ratio(self, roi_level: int, market_condition: str = 'NEUTRAL', 
                           volatility: float = 0.05) -> float:
        """
        获取动态保护比例
        
        Args:
            roi_level: ROI等级 (0-6)
            market_condition: 市场条件 ('BULLISH', 'BEARISH', 'NEUTRAL', 'SIDEWAYS')
            volatility: 当前波动率
            
        Returns:
            动态调整后的保护比例
        """
        try:
            # 基础保护比例
            base_ratio = self.base_protection_ratios.get(roi_level, 0.0)
            
            # 市场环境调整 - 基于改造计划
            market_adjustment = self._get_market_adjustment(market_condition)
            
            # 波动率调整 - 基于改造计划
            volatility_adjustment = self._get_volatility_adjustment(volatility)
            
            # 计算最终比例
            final_ratio = base_ratio + market_adjustment + volatility_adjustment
            
            # 限制在合理范围内 (0-60%)
            final_ratio = max(0.0, min(0.6, final_ratio))
            
            self.logger.debug(f"{self.symbol} 保护比例计算: 基础{base_ratio:.2%} + 市场{market_adjustment:+.2%} + 波动{volatility_adjustment:+.2%} = {final_ratio:.2%}")
            
            return final_ratio
            
        except Exception as e:
            self.logger.error(f"保护比例计算失败: {e}")
            return self.base_protection_ratios.get(roi_level, 0.0)
    
    def get_reduction_config(self, roi_level: int) -> Dict:
        """
        获取减仓配置
        
        Args:
            roi_level: ROI等级
            
        Returns:
            减仓配置字典
        """
        try:
            config = self.reduction_configs.get(roi_level, {
                'target_reduction': 0.0, 
                'cumulative': False
            })
            
            self.logger.debug(f"{self.symbol} Level {roi_level} 减仓配置: {config}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"减仓配置获取失败: {e}")
            return {'target_reduction': 0.0, 'cumulative': False}
    
    def get_level_config(self, roi_level: int, market_condition: str = 'NEUTRAL', 
                        volatility: float = 0.05) -> Dict:
        """
        获取指定ROI等级的完整配置
        
        Args:
            roi_level: ROI等级
            market_condition: 市场条件
            volatility: 波动率
            
        Returns:
            完整的等级配置
        """
        try:
            protection_ratio = self.get_protection_ratio(roi_level, market_condition, volatility)
            reduction_config = self.get_reduction_config(roi_level)
            roi_range = self.roi_boundaries.get(roi_level, (0, 0))
            
            config = {
                'level': roi_level,
                'roi_range': roi_range,
                'protection_ratio': protection_ratio,
                'reduction_config': reduction_config,
                'strategy_name': self._get_strategy_name(roi_level),
                'description': self._get_level_description(roi_level)
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"等级配置获取失败: {e}")
            return {
                'level': roi_level,
                'roi_range': (0, 0),
                'protection_ratio': 0.0,
                'reduction_config': {'target_reduction': 0.0, 'cumulative': False},
                'strategy_name': 'unknown',
                'description': 'error'
            }
    
    def _get_market_adjustment(self, market_condition: str) -> float:
        """获取市场环境调整值"""
        market_adjustments = {
            'BULLISH': -0.02,   # 牛市更激进，降低保护比例
            'BEARISH': +0.02,   # 熊市更保守，提高保护比例
            'SIDEWAYS': 0.0,    # 横盘无调整
            'NEUTRAL': 0.0      # 中性无调整
        }
        
        return market_adjustments.get(market_condition.upper(), 0.0)
    
    def _get_volatility_adjustment(self, volatility: float) -> float:
        """获取波动率调整值"""
        try:
            # 假设平均波动率为5%
            avg_volatility = 0.05
            
            if volatility > avg_volatility * 2.0:
                # 高波动(>10%)更保守，提高保护比例
                return +0.05
            elif volatility < avg_volatility * 0.5:
                # 低波动(<2.5%)更激进，降低保护比例
                return -0.02
            else:
                # 正常波动无调整
                return 0.0
                
        except Exception as e:
            self.logger.error(f"波动率调整计算失败: {e}")
            return 0.0
    
    def _get_strategy_name(self, roi_level: int) -> str:
        """获取策略名称"""
        strategy_names = {
            0: "original_stop_loss",
            1: "move_to_breakeven", 
            2: "protect_5pct_profit",
            3: "protect_12pct_profit",
            4: "protect_20pct_profit",
            5: "protect_30pct_profit",
            6: "protect_50pct_profit"
        }
        
        return strategy_names.get(roi_level, "unknown_strategy")
    
    def _get_level_description(self, roi_level: int) -> str:
        """获取等级描述"""
        descriptions = {
            0: "建仓保护期 - 保持原始止损",
            1: "生存确认期 - 移至保本点",
            2: "初步收获期 - 保护5%利润",
            3: "关键决策期 - 保护12%利润，设定减仓基准",
            4: "首次减仓期 - 保护20%利润，减仓20%",
            5: "积极减仓期 - 保护30%利润，累计减仓40%",
            6: "传奇保护期 - 保护50%利润，累计减仓70%"
        }
        
        return descriptions.get(roi_level, "未知等级")
    
    # 为兼容现有代码，提供各个等级的配置方法
    def get_level_0_config(self) -> Dict:
        """Level 0配置"""
        return self.get_level_config(0)
    
    def get_level_1_config(self) -> Dict:
        """Level 1配置"""
        return self.get_level_config(1)
    
    def get_level_2_config(self) -> Dict:
        """Level 2配置"""
        return self.get_level_config(2)
    
    def get_level_3_config(self) -> Dict:
        """Level 3配置"""
        return self.get_level_config(3)
    
    def get_level_4_config(self) -> Dict:
        """Level 4配置"""
        return self.get_level_config(4)
    
    def get_level_5_config(self) -> Dict:
        """Level 5配置"""
        return self.get_level_config(5)
    
    def get_level_6_config(self) -> Dict:
        """Level 6配置"""
        return self.get_level_config(6)
    
    def get_dynamic_reduction_threshold(self, volatility: float) -> int:
        """
        获取动态减仓阈值ROI
        
        Args:
            volatility: 当前波动率
            
        Returns:
            减仓阈值ROI百分比
        """
        try:
            avg_volatility = 0.05  # 5%平均波动率
            
            if volatility > avg_volatility * 1.5:
                # 高波动提前减仓
                return 25
            else:
                # 正常阈值
                return 30
                
        except Exception as e:
            self.logger.error(f"动态减仓阈值计算失败: {e}")
            return 30
    
    def log_configuration_summary(self, roi_pct: float, market_condition: str = 'NEUTRAL', 
                                 volatility: float = 0.05):
        """记录配置摘要信息"""
        try:
            roi_level = self.determine_roi_level(roi_pct)
            config = self.get_level_config(roi_level, market_condition, volatility)
            
            self.logger.info(f"📊 {self.symbol} 动态参数配置摘要:")
            self.logger.info(f"   ROI: {roi_pct:.1f}% → Level {roi_level}")
            self.logger.info(f"   策略: {config['description']}")
            self.logger.info(f"   保护比例: {config['protection_ratio']:.1%}")
            self.logger.info(f"   减仓配置: {config['reduction_config']}")
            self.logger.info(f"   市场条件: {market_condition}")
            self.logger.info(f"   波动率: {volatility:.2%}")
            
        except Exception as e:
            self.logger.error(f"配置摘要记录失败: {e}") 