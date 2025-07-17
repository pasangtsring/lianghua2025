#!/usr/bin/env python3
"""
趋势滚仓+周期MACD形态低风险交易策略
主程序入口
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trading_engine import TradingEngine
from utils.logger import Logger
from config.config_manager import ConfigManager

async def main():
    """主函数"""
    logger = Logger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("启动 趋势滚仓+周期MACD形态低风险交易策略")
        logger.info("=" * 80)
        
        # 检查配置
        config = ConfigManager()
        if not config.validate_config():
            logger.error("配置验证失败，程序退出")
            return
        
        # 创建交易引擎
        engine = TradingEngine()
        
        # 启动交易引擎
        await engine.start()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭程序...")
        if 'engine' in locals():
            await engine.stop()
        
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        sys.exit(1)
    
    finally:
        logger.info("程序已退出")

if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误：需要Python 3.8或更高版本")
        sys.exit(1)
    
    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 运行主程序
    asyncio.run(main()) 