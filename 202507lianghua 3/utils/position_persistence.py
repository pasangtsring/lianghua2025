#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持仓数据持久化管理器
解决系统重启后买入理由丢失的问题
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from utils.logger import get_logger


class PositionPersistence:
    """持仓数据持久化管理器"""
    
    def __init__(self, data_dir: str = "data/positions"):
        self.data_dir = data_dir
        self.positions_file = os.path.join(data_dir, "active_positions.json")
        self.backup_file = os.path.join(data_dir, "positions_backup.json")
        self.logger = get_logger(__name__)
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        self.logger.info(f"💾 持仓数据持久化管理器初始化完成 - 数据目录: {data_dir}")
    
    async def save_position(self, symbol: str, position_data: Dict[str, Any]):
        """保存单个持仓数据"""
        try:
            # 读取现有数据
            positions = await self.load_all_positions()
            
            # 添加时间戳
            position_data['last_updated'] = datetime.now().isoformat()
            
            # 更新持仓数据
            positions[symbol] = position_data
            
            # 备份现有文件
            if os.path.exists(self.positions_file):
                import shutil
                shutil.copy2(self.positions_file, self.backup_file)
            
            # 保存到文件
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(positions, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 持仓数据已保存: {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存持仓数据失败 {symbol}: {e}")
    
    async def load_all_positions(self) -> Dict[str, Any]:
        """加载所有持仓数据"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"⚠️ 加载持仓数据失败，使用空数据: {e}")
            return {}
    
    async def load_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """加载单个持仓数据"""
        positions = await self.load_all_positions()
        return positions.get(symbol)
    
    async def remove_position(self, symbol: str):
        """移除持仓数据（平仓时调用）"""
        try:
            positions = await self.load_all_positions()
            if symbol in positions:
                del positions[symbol]
                
                # 保存更新后的数据
                with open(self.positions_file, 'w', encoding='utf-8') as f:
                    json.dump(positions, f, indent=2, ensure_ascii=False, default=str)
                    
                self.logger.info(f"🗑️ 持仓数据已移除: {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ 移除持仓数据失败 {symbol}: {e}")
    
    def get_position_summary(self) -> str:
        """获取持仓摘要"""
        try:
            positions = {}
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    positions = json.load(f)
            
            if not positions:
                return "📭 无持久化持仓数据"
            
            summary = f"💾 持久化持仓数据 ({len(positions)}个):\n"
            for symbol, pos in positions.items():
                entry_price = pos.get('entry_price', 0)
                side = pos.get('type', pos.get('side', 'UNKNOWN'))
                last_updated = pos.get('last_updated', 'Unknown')
                has_reasons = 'buy_reasons' in pos
                
                summary += f"   • {symbol}: {side} @ {entry_price:.4f} "
                summary += f"({'有买入理由' if has_reasons else '⚠️无买入理由'}) "
                summary += f"- {last_updated[:19]}\n"
            
            return summary.rstrip()
            
        except Exception as e:
            return f"❌ 获取持仓摘要失败: {e}" 