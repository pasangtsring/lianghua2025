"""
资源监控模块
负责监控系统资源使用情况，在资源紧张时自动暂停交易
实现专家建议的资源管理策略
"""

import asyncio
import psutil
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading

from utils.logger import get_logger
from config.config_manager import ConfigManager

class ResourceStatus(Enum):
    """资源状态枚举"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ActionType(Enum):
    """动作类型枚举"""
    CONTINUE = "continue"
    SLOW_DOWN = "slow_down"
    PAUSE_TRADING = "pause_trading"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ResourceThresholds:
    """资源阈值配置"""
    cpu_warning: float = 70.0
    cpu_critical: float = 80.0
    cpu_emergency: float = 90.0
    
    memory_warning: float = 75.0
    memory_critical: float = 85.0
    memory_emergency: float = 95.0
    
    disk_warning: float = 80.0
    disk_critical: float = 90.0
    disk_emergency: float = 95.0
    
    network_warning: float = 80.0
    network_critical: float = 90.0
    network_emergency: float = 95.0

@dataclass
class ResourceMetrics:
    """资源指标"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_percent: float
    process_count: int
    load_average: List[float]
    timestamp: datetime
    
    def get_overall_status(self, thresholds: ResourceThresholds) -> ResourceStatus:
        """获取整体资源状态"""
        max_usage = max(
            self.cpu_percent,
            self.memory_percent,
            self.disk_percent,
            self.network_percent
        )
        
        if max_usage >= max(thresholds.cpu_emergency, thresholds.memory_emergency, 
                          thresholds.disk_emergency, thresholds.network_emergency):
            return ResourceStatus.EMERGENCY
        elif max_usage >= max(thresholds.cpu_critical, thresholds.memory_critical,
                            thresholds.disk_critical, thresholds.network_critical):
            return ResourceStatus.CRITICAL
        elif max_usage >= max(thresholds.cpu_warning, thresholds.memory_warning,
                            thresholds.disk_warning, thresholds.network_warning):
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.NORMAL

class ResourceMonitor:
    """
    资源监控器 - 实现专家建议的资源管理策略
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 获取系统资源配置
        self.system_config = self.config.get_system_config()
        self.resource_config = self.system_config.resource_monitoring
        
        # 资源阈值
        self.thresholds = ResourceThresholds(
            cpu_warning=self.resource_config.get('cpu_threshold', 70) - 10,
            cpu_critical=self.resource_config.get('cpu_threshold', 80),
            cpu_emergency=self.resource_config.get('cpu_threshold', 80) + 10,
            
            memory_warning=self.resource_config.get('memory_threshold', 75) - 10,
            memory_critical=self.resource_config.get('memory_threshold', 85),
            memory_emergency=self.resource_config.get('memory_threshold', 85) + 10
        )
        
        # 监控配置
        self.check_interval = self.resource_config.get('check_interval', 30)
        self.pause_on_high_usage = self.resource_config.get('pause_on_high_usage', True)
        self.auto_resume = True
        self.resume_delay = 60  # 恢复延迟（秒）
        
        # 监控状态
        self.monitoring_active = False
        self.trading_paused = False
        self.pause_reason = ""
        self.pause_start_time: Optional[datetime] = None
        
        # 监控任务
        self.monitor_task: Optional[asyncio.Task] = None
        
        # 历史数据
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 1440  # 24小时，每分钟一个数据点
        
        # 回调函数
        self.status_callbacks: List[Callable] = []
        self.pause_callbacks: List[Callable] = []
        self.resume_callbacks: List[Callable] = []
        
        # 统计信息
        self.stats = {
            'total_checks': 0,
            'warning_count': 0,
            'critical_count': 0,
            'emergency_count': 0,
            'pause_count': 0,
            'total_pause_time': 0.0,
            'avg_cpu_usage': 0.0,
            'avg_memory_usage': 0.0,
            'max_cpu_usage': 0.0,
            'max_memory_usage': 0.0
        }
        
        # 基线数据
        self.baseline_metrics: Optional[ResourceMetrics] = None
        
        self.logger.info("资源监控器初始化完成")
    
    async def start_monitoring(self):
        """
        启动资源监控
        """
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # 收集基线数据
            await self.collect_baseline_metrics()
            
            # 启动监控任务
            self.monitor_task = asyncio.create_task(self.monitor_loop())
            
            self.logger.info("资源监控已启动")
            
        except Exception as e:
            self.logger.error(f"启动资源监控失败: {e}")
    
    async def stop_monitoring(self):
        """
        停止资源监控
        """
        try:
            self.monitoring_active = False
            
            if self.monitor_task:
                self.monitor_task.cancel()
                await self.monitor_task
            
            self.logger.info("资源监控已停止")
            
        except Exception as e:
            self.logger.error(f"停止资源监控失败: {e}")
    
    async def collect_baseline_metrics(self):
        """
        收集基线资源指标
        """
        try:
            # 收集多个数据点求平均值
            cpu_samples = []
            memory_samples = []
            
            for _ in range(5):
                metrics = await self.collect_current_metrics()
                cpu_samples.append(metrics.cpu_percent)
                memory_samples.append(metrics.memory_percent)
                await asyncio.sleep(1)
            
            # 计算基线
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)
            
            self.baseline_metrics = ResourceMetrics(
                cpu_percent=avg_cpu,
                memory_percent=avg_memory,
                disk_percent=0,
                network_percent=0,
                process_count=0,
                load_average=[0, 0, 0],
                timestamp=datetime.now()
            )
            
            self.logger.info(f"基线指标收集完成: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%")
            
        except Exception as e:
            self.logger.error(f"收集基线指标失败: {e}")
    
    async def monitor_loop(self):
        """
        监控循环
        """
        while self.monitoring_active:
            try:
                # 收集当前指标
                current_metrics = await self.collect_current_metrics()
                
                # 添加到历史记录
                self.add_to_history(current_metrics)
                
                # 更新统计信息
                self.update_stats(current_metrics)
                
                # 检查资源状态
                await self.check_resource_status(current_metrics)
                
                # 等待下次检查
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"资源监控循环错误: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def collect_current_metrics(self) -> ResourceMetrics:
        """
        收集当前资源指标
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 网络使用率（简化计算）
            network_percent = 0  # 实际应该根据网络流量计算
            
            # 进程数量
            process_count = len(psutil.pids())
            
            # 系统负载
            load_average = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_percent=network_percent,
                process_count=process_count,
                load_average=load_average,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"收集资源指标失败: {e}")
            return ResourceMetrics(
                cpu_percent=0, memory_percent=0, disk_percent=0, network_percent=0,
                process_count=0, load_average=[0, 0, 0], timestamp=datetime.now()
            )
    
    def add_to_history(self, metrics: ResourceMetrics):
        """
        添加到历史记录
        """
        try:
            self.metrics_history.append(metrics)
            
            # 限制历史记录大小
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size//2:]
                
        except Exception as e:
            self.logger.error(f"添加历史记录失败: {e}")
    
    def update_stats(self, metrics: ResourceMetrics):
        """
        更新统计信息
        """
        try:
            self.stats['total_checks'] += 1
            
            # 更新平均值
            total_checks = self.stats['total_checks']
            self.stats['avg_cpu_usage'] = (
                (self.stats['avg_cpu_usage'] * (total_checks - 1) + metrics.cpu_percent) / total_checks
            )
            self.stats['avg_memory_usage'] = (
                (self.stats['avg_memory_usage'] * (total_checks - 1) + metrics.memory_percent) / total_checks
            )
            
            # 更新最大值
            self.stats['max_cpu_usage'] = max(self.stats['max_cpu_usage'], metrics.cpu_percent)
            self.stats['max_memory_usage'] = max(self.stats['max_memory_usage'], metrics.memory_percent)
            
            # 更新状态计数
            status = metrics.get_overall_status(self.thresholds)
            if status == ResourceStatus.WARNING:
                self.stats['warning_count'] += 1
            elif status == ResourceStatus.CRITICAL:
                self.stats['critical_count'] += 1
            elif status == ResourceStatus.EMERGENCY:
                self.stats['emergency_count'] += 1
                
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")
    
    async def check_resource_status(self, metrics: ResourceMetrics):
        """
        检查资源状态并执行相应动作
        """
        try:
            status = metrics.get_overall_status(self.thresholds)
            action = self.determine_action(status, metrics)
            
            # 执行动作
            if action == ActionType.PAUSE_TRADING and not self.trading_paused:
                await self.pause_trading(f"资源使用过高: {status.value}")
                
            elif action == ActionType.EMERGENCY_STOP:
                await self.emergency_stop(f"资源紧急状态: {status.value}")
                
            elif action == ActionType.CONTINUE and self.trading_paused and self.auto_resume:
                await self.resume_trading("资源状态恢复正常")
            
            # 调用状态回调
            for callback in self.status_callbacks:
                try:
                    await callback(status, metrics, action)
                except Exception as e:
                    self.logger.error(f"状态回调失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"检查资源状态失败: {e}")
    
    def determine_action(self, status: ResourceStatus, metrics: ResourceMetrics) -> ActionType:
        """
        确定要执行的动作
        """
        try:
            if not self.pause_on_high_usage:
                return ActionType.CONTINUE
            
            if status == ResourceStatus.EMERGENCY:
                return ActionType.EMERGENCY_STOP
            elif status == ResourceStatus.CRITICAL:
                return ActionType.PAUSE_TRADING
            elif status == ResourceStatus.WARNING:
                return ActionType.SLOW_DOWN
            else:
                return ActionType.CONTINUE
                
        except Exception as e:
            self.logger.error(f"确定动作失败: {e}")
            return ActionType.CONTINUE
    
    async def pause_trading(self, reason: str):
        """
        暂停交易
        """
        try:
            if self.trading_paused:
                return
            
            self.trading_paused = True
            self.pause_reason = reason
            self.pause_start_time = datetime.now()
            self.stats['pause_count'] += 1
            
            self.logger.warning(f"交易已暂停: {reason}")
            
            # 调用暂停回调
            for callback in self.pause_callbacks:
                try:
                    await callback(reason)
                except Exception as e:
                    self.logger.error(f"暂停回调失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"暂停交易失败: {e}")
    
    async def resume_trading(self, reason: str):
        """
        恢复交易
        """
        try:
            if not self.trading_paused:
                return
            
            # 检查是否需要延迟恢复
            if self.pause_start_time:
                pause_duration = (datetime.now() - self.pause_start_time).total_seconds()
                if pause_duration < self.resume_delay:
                    # 暂停时间不够，不恢复
                    return
                
                self.stats['total_pause_time'] += pause_duration
            
            self.trading_paused = False
            self.pause_reason = ""
            self.pause_start_time = None
            
            self.logger.info(f"交易已恢复: {reason}")
            
            # 调用恢复回调
            for callback in self.resume_callbacks:
                try:
                    await callback(reason)
                except Exception as e:
                    self.logger.error(f"恢复回调失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"恢复交易失败: {e}")
    
    async def emergency_stop(self, reason: str):
        """
        紧急停止
        """
        try:
            await self.pause_trading(f"紧急停止: {reason}")
            
            # 关闭自动恢复
            self.auto_resume = False
            
            self.logger.critical(f"紧急停止触发: {reason}")
            
        except Exception as e:
            self.logger.error(f"紧急停止失败: {e}")
    
    async def manual_pause(self, reason: str = "手动暂停"):
        """
        手动暂停交易
        """
        await self.pause_trading(reason)
    
    async def manual_resume(self, reason: str = "手动恢复"):
        """
        手动恢复交易
        """
        self.auto_resume = True
        await self.resume_trading(reason)
    
    def add_status_callback(self, callback: Callable):
        """添加状态回调"""
        self.status_callbacks.append(callback)
    
    def add_pause_callback(self, callback: Callable):
        """添加暂停回调"""
        self.pause_callbacks.append(callback)
    
    def add_resume_callback(self, callback: Callable):
        """添加恢复回调"""
        self.resume_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        获取当前状态
        """
        try:
            if not self.metrics_history:
                return {'status': 'no_data'}
            
            latest_metrics = self.metrics_history[-1]
            status = latest_metrics.get_overall_status(self.thresholds)
            
            return {
                'status': status.value,
                'trading_paused': self.trading_paused,
                'pause_reason': self.pause_reason,
                'pause_start_time': self.pause_start_time.isoformat() if self.pause_start_time else None,
                'monitoring_active': self.monitoring_active,
                'auto_resume': self.auto_resume,
                'metrics': {
                    'cpu_percent': latest_metrics.cpu_percent,
                    'memory_percent': latest_metrics.memory_percent,
                    'disk_percent': latest_metrics.disk_percent,
                    'network_percent': latest_metrics.network_percent,
                    'process_count': latest_metrics.process_count,
                    'load_average': latest_metrics.load_average,
                    'timestamp': latest_metrics.timestamp.isoformat()
                },
                'thresholds': {
                    'cpu_warning': self.thresholds.cpu_warning,
                    'cpu_critical': self.thresholds.cpu_critical,
                    'cpu_emergency': self.thresholds.cpu_emergency,
                    'memory_warning': self.thresholds.memory_warning,
                    'memory_critical': self.thresholds.memory_critical,
                    'memory_emergency': self.thresholds.memory_emergency
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取当前状态失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_historical_data(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """
        获取历史数据
        """
        try:
            if not self.metrics_history:
                return []
            
            # 计算时间范围
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # 过滤数据
            filtered_data = [
                {
                    'timestamp': metrics.timestamp.isoformat(),
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'disk_percent': metrics.disk_percent,
                    'network_percent': metrics.network_percent,
                    'process_count': metrics.process_count,
                    'load_average': metrics.load_average
                }
                for metrics in self.metrics_history
                if metrics.timestamp >= cutoff_time
            ]
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        """
        try:
            return {
                'total_checks': self.stats['total_checks'],
                'warning_count': self.stats['warning_count'],
                'critical_count': self.stats['critical_count'],
                'emergency_count': self.stats['emergency_count'],
                'pause_count': self.stats['pause_count'],
                'total_pause_time': self.stats['total_pause_time'],
                'avg_pause_time': (self.stats['total_pause_time'] / self.stats['pause_count'] 
                                 if self.stats['pause_count'] > 0 else 0),
                'avg_cpu_usage': self.stats['avg_cpu_usage'],
                'avg_memory_usage': self.stats['avg_memory_usage'],
                'max_cpu_usage': self.stats['max_cpu_usage'],
                'max_memory_usage': self.stats['max_memory_usage'],
                'uptime_seconds': (datetime.now() - self.baseline_metrics.timestamp).total_seconds()
                                if self.baseline_metrics else 0
            }
            
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def reset_statistics(self):
        """
        重置统计信息
        """
        try:
            self.stats = {
                'total_checks': 0,
                'warning_count': 0,
                'critical_count': 0,
                'emergency_count': 0,
                'pause_count': 0,
                'total_pause_time': 0.0,
                'avg_cpu_usage': 0.0,
                'avg_memory_usage': 0.0,
                'max_cpu_usage': 0.0,
                'max_memory_usage': 0.0
            }
            
            self.logger.info("统计信息已重置")
            
        except Exception as e:
            self.logger.error(f"重置统计信息失败: {e}")
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """
        更新资源阈值
        """
        try:
            for key, value in new_thresholds.items():
                if hasattr(self.thresholds, key):
                    setattr(self.thresholds, key, value)
            
            self.logger.info(f"资源阈值已更新: {new_thresholds}")
            
        except Exception as e:
            self.logger.error(f"更新资源阈值失败: {e}")
    
    async def check_resources_sync(self) -> bool:
        """
        同步检查资源状态（供其他模块调用）
        
        Returns:
            是否允许继续交易
        """
        try:
            if not self.monitoring_active:
                return True
            
            if self.trading_paused:
                return False
            
            # 快速检查当前资源状态
            current_metrics = await self.collect_current_metrics()
            status = current_metrics.get_overall_status(self.thresholds)
            
            if status in [ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY]:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"同步检查资源状态失败: {e}")
            return False
    
    def __del__(self):
        """
        析构函数
        """
        try:
            if self.monitoring_active:
                asyncio.create_task(self.stop_monitoring())
        except:
            pass 