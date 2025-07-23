#!/usr/bin/env python3
"""
趋势滚仓+周期MACD形态低风险交易策略
主程序入口
"""

import asyncio
import sys
import os
import schedule
import time
import platform

# Windows兼容性修复：设置正确的事件循环策略
if platform.system() == 'Windows':
    # Windows上强制使用SelectorEventLoop以兼容aiodns
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✅ Windows兼容性修复：已设置SelectorEventLoop策略")

# 可选导入psutil（M芯片兼容性修复）
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil未安装或不兼容，系统资源监控将被禁用")
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trading_engine import TradingEngine
from utils.logger import get_logger
from config.config_manager import ConfigManager

# 条件导入ResourceMonitor（依赖psutil）
try:
    from utils.resource_monitor import ResourceMonitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False
    ResourceMonitor = None

# 全局变量
engine: Optional[TradingEngine] = None
resource_monitor: Optional[ResourceMonitor] = None

async def check_system_resources():
    """检查系统资源状况"""
    try:
        # 如果psutil不可用，跳过资源检查
        if not PSUTIL_AVAILABLE:
            return True
            
        # CPU使用率检查
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # 如果资源使用过高，暂停交易
        # macOS系统90%内存使用率很正常，调整阈值为更合理的95%
        if cpu_percent > 90 or memory_percent > 95:
            logger.warning(f"系统资源使用较高: CPU {cpu_percent}%, 内存 {memory_percent}%")
            if engine:
                await engine.pause_trading("系统资源使用过高")
            await asyncio.sleep(60)  # 等待资源恢复
            return False
        
        return True
    except Exception as e:
        logger.error(f"系统资源检查失败: {e}")
        return True  # 默认允许继续

def schedule_exception_wrapper(func):
    """包装定时任务的异常处理"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"定时任务执行失败: {e}")
    return wrapper

async def main():
    """主函数"""
    global engine, resource_monitor
    logger = get_logger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("启动 趋势滚仓+周期MACD形态低风险交易策略")
        logger.info("=" * 80)
        
        # 系统资源初始检查
        if not await check_system_resources():
            logger.error("系统资源不足，程序退出")
            return
        
        # 检查配置
        config = ConfigManager()
        
        # 初始化资源监控器（仅在配置启用且psutil可用时）
        resource_monitor = None
        system_config = config.get_config().dict().get('system', {})
        resource_config = system_config.get('resource_monitoring', {})
        if resource_config.get('enabled', False) and RESOURCE_MONITOR_AVAILABLE and PSUTIL_AVAILABLE:
            resource_monitor = ResourceMonitor(config)
            await resource_monitor.start_monitoring()
            logger.info("资源监控器已启用")
        else:
            if not RESOURCE_MONITOR_AVAILABLE or not PSUTIL_AVAILABLE:
                logger.info("资源监控器不可用（psutil兼容性问题）")
            else:
                logger.info("资源监控器已禁用")
        
        # 创建交易引擎
        engine = TradingEngine()
        
        # 设置定时任务
        schedule.every(5).minutes.do(schedule_exception_wrapper(lambda: asyncio.create_task(check_system_resources())))
        
        # 启动交易引擎
        await engine.start()
        
        # 主循环
        while True:
            try:
                # 检查交易引擎状态
                if engine and not engine.is_running:
                    logger.info("交易引擎已停止，准备退出主程序...")
                    break
                
                # 运行定时任务
                schedule.run_pending()
                
                # 检查资源状态
                if resource_monitor and not await resource_monitor.check_resources_sync():
                    logger.warning("资源监控器检测到异常，暂停交易")
                    await asyncio.sleep(30)
                    continue
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("收到中断信号，正在关闭程序...")
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}")
                await asyncio.sleep(5)  # 短暂休息后继续
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭程序...")
        
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        sys.exit(1)
    
    finally:
        # 清理资源
        if engine:
            try:
                await engine.stop()
            except Exception as e:
                logger.error(f"停止交易引擎失败: {e}")
        
        if resource_monitor:
            try:
                await resource_monitor.stop_monitoring()
            except Exception as e:
                logger.error(f"停止资源监控失败: {e}")
        
        logger.info("程序已退出")

if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误：需要Python 3.8或更高版本")
        sys.exit(1)
    
    # 检查必要的依赖
    try:
        import schedule
    except ImportError as e:
        print(f"错误：缺少必要的依赖: {e}")
        print("请运行: pip install schedule")
        sys.exit(1)
    
    # 检查可选依赖psutil（资源监控）
    if not PSUTIL_AVAILABLE:
        print("⚠️ psutil不可用，资源监控功能将被禁用")
    
    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 运行主程序
    asyncio.run(main()) 