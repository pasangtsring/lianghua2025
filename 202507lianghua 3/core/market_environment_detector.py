"""
市场环境检测器 - 自动识别牛熊市并调整策略偏向
"""

from typing import Dict, List, Optional
import numpy as np
from utils.logger import get_logger

class MarketEnvironmentDetector:
    """市场环境检测器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def detect_market_environment(self, selected_coins: List) -> Dict:
        """
        检测当前市场环境
        
        Returns:
            Dict: {
                'environment': 'BULL' | 'BEAR' | 'SIDEWAYS',
                'confidence': float,
                'stage_distribution': Dict,
                'recommendation': str
            }
        """
        try:
            if not selected_coins:
                return {
                    'environment': 'SIDEWAYS',
                    'confidence': 0.5,
                    'stage_distribution': {},
                    'recommendation': '无足够数据判断市场环境'
                }
            
            # 统计各阶段币种数量
            stage_count = {1: 0, 2: 0, 3: 0}
            total_coins = len(selected_coins)
            
            for coin in selected_coins:
                stage = getattr(coin, 'stage', 2)
                stage_count[stage] += 1
            
            # 计算阶段比例
            stage_1_ratio = stage_count[1] / total_coins  # 冷启动比例
            stage_2_ratio = stage_count[2] / total_coins  # 高热比例  
            stage_3_ratio = stage_count[3] / total_coins  # 冷却比例
            
            # 判断市场环境
            if stage_1_ratio + stage_2_ratio > 0.6:
                # 超过60%币种处于上涨阶段
                environment = 'BULL'
                confidence = (stage_1_ratio + stage_2_ratio)
                recommendation = '牛市环境，建议优先做多，适度做空对冲'
            elif stage_3_ratio > 0.5:
                # 超过50%币种处于下跌阶段
                environment = 'BEAR'
                confidence = stage_3_ratio
                recommendation = '熊市环境，建议优先做空，谨慎做多'
            else:
                # 震荡市场
                environment = 'SIDEWAYS'
                confidence = max(stage_1_ratio, stage_2_ratio, stage_3_ratio)
                recommendation = '震荡市场，多空并重，灵活应对'
            
            result = {
                'environment': environment,
                'confidence': confidence,
                'stage_distribution': {
                    '冷启动(适合做多)': f"{stage_1_ratio:.1%} ({stage_count[1]}个)",
                    '高热(适合做多)': f"{stage_2_ratio:.1%} ({stage_count[2]}个)",
                    '冷却(适合做空)': f"{stage_3_ratio:.1%} ({stage_count[3]}个)"
                },
                'recommendation': recommendation
            }
            
            self.logger.info(f"📊 市场环境检测结果:")
            self.logger.info(f"   🌍 市场环境: {environment} (置信度: {confidence:.1%})")
            self.logger.info(f"   📈 阶段分布: {result['stage_distribution']}")
            self.logger.info(f"   💡 操作建议: {recommendation}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"市场环境检测失败: {e}")
            return {
                'environment': 'SIDEWAYS',
                'confidence': 0.5,
                'stage_distribution': {},
                'recommendation': '检测失败，建议谨慎操作'
            }
