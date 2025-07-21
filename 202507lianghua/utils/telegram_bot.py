"""
Telegram通知机器人
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.logger import get_logger
from config.config_manager import ConfigManager

class TelegramBot:
    """Telegram通知机器人"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # Telegram配置
        monitoring_config = config_manager.get_monitoring_config()
        self.telegram_config = getattr(monitoring_config, 'telegram', {
            'bot_token': '',
            'chat_id': '',
            'enabled': False
        })
        self.bot_token = self.telegram_config.get('bot_token', '')
        self.chat_id = self.telegram_config.get('chat_id', '')
        self.enabled = self.telegram_config.get('enabled', False)
        
        # 基础URL
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # 消息配置
        self.alerts_config = self.telegram_config.get('alerts', {})
        
        if self.enabled and self.bot_token and self.chat_id:
            self.logger.info("Telegram机器人已启用")
        else:
            self.logger.info("Telegram机器人未启用或配置不完整")
    
    async def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """
        发送消息
        
        Args:
            message: 消息内容
            parse_mode: 解析模式
            
        Returns:
            是否发送成功
        """
        try:
            if not self.enabled:
                return False
            
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.logger.debug("Telegram消息发送成功")
                        return True
                    else:
                        self.logger.error(f"Telegram消息发送失败: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"发送Telegram消息失败: {e}")
            return False
    
    async def send_trade_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        发送交易信号通知
        
        Args:
            signal_data: 信号数据
            
        Returns:
            是否发送成功
        """
        try:
            if not self.alerts_config.get('trade_signals', False):
                return False
            
            message = f"🔔 *交易信号*\n"
            message += f"品种: `{signal_data.get('symbol', 'N/A')}`\n"
            message += f"信号类型: `{signal_data.get('signal_type', 'N/A')}`\n"
            message += f"置信度: `{signal_data.get('confidence', 0):.2f}`\n"
            message += f"当前价格: `{signal_data.get('current_price', 0):.2f}`\n"
            message += f"时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"发送交易信号失败: {e}")
            return False
    
    async def send_error_alert(self, error_msg: str) -> bool:
        """
        发送错误警告
        
        Args:
            error_msg: 错误消息
            
        Returns:
            是否发送成功
        """
        try:
            if not self.alerts_config.get('errors', False):
                return False
            
            message = f"⚠️ *错误警告*\n"
            message += f"错误信息: `{error_msg}`\n"
            message += f"时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"发送错误警告失败: {e}")
            return False
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        发送每日总结
        
        Args:
            summary_data: 总结数据
            
        Returns:
            是否发送成功
        """
        try:
            if not self.alerts_config.get('daily_summary', False):
                return False
            
            message = f"📊 *每日总结*\n"
            message += f"日期: `{datetime.now().strftime('%Y-%m-%d')}`\n"
            message += f"总收益: `{summary_data.get('total_pnl', 0):.2f}`\n"
            message += f"今日收益: `{summary_data.get('daily_pnl', 0):.2f}`\n"
            message += f"胜率: `{summary_data.get('win_rate', 0):.2%}`\n"
            message += f"最大回撤: `{summary_data.get('max_drawdown', 0):.2%}`\n"
            message += f"交易次数: `{summary_data.get('trade_count', 0)}`"
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"发送每日总结失败: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """
        测试连接
        
        Returns:
            是否连接成功
        """
        try:
            if not self.enabled:
                return False
            
            test_message = "🤖 Telegram机器人测试消息"
            return await self.send_message(test_message)
            
        except Exception as e:
            self.logger.error(f"测试Telegram连接失败: {e}")
            return False 