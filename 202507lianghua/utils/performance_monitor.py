"""
性能监控器模块
负责系统性能监控、资源使用追踪、性能分析
"""

import os
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import json
import functools

from utils.logger import get_logger
from config.config_manager import ConfigManager

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"         # 计数器（累计值）
    GAUGE = "gauge"            # 仪表（瞬时值）
    HISTOGRAM = "histogram"     # 直方图
    TIMER = "timer"            # 计时器

@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class PerformanceAlert:
    """性能告警"""
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    load_average: List[float]
    timestamp: datetime

@dataclass
class ProcessMetrics:
    """进程指标"""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    threads_count: int
    connections_count: int
    open_files_count: int
    create_time: datetime
    timestamp: datetime

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 监控配置
        self.monitor_interval = 5  # 监控间隔（秒）
        self.history_size = 1440   # 历史数据保存条数（24小时 * 60分钟 / 5秒）
        self.alert_threshold = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 1000.0
        }
        
        # 数据存储
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        self.system_metrics_history: deque = deque(maxlen=self.history_size)
        self.process_metrics_history: deque = deque(maxlen=self.history_size)
        self.custom_metrics: Dict[str, MetricData] = {}
        self.alerts: List[PerformanceAlert] = []
        
        # 性能计时器
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.timer_stats: Dict[str, Dict[str, float]] = {}
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.thread_lock = threading.RLock()
        
        # 回调函数
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # 当前进程信息
        self.current_process = psutil.Process()
        self.start_time = datetime.now()
        
        # 网络统计基线
        self.network_baseline = self._get_network_stats()
        
        self.logger.info("性能监控器初始化完成")
    
    def start_monitoring(self):
        """启动性能监控"""
        try:
            with self.thread_lock:
                if self.monitoring_active:
                    self.logger.warning("性能监控已经在运行")
                    return
                
                self.monitoring_active = True
                self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitor_thread.start()
                
                self.logger.info("性能监控启动成功")
                
        except Exception as e:
            self.logger.error(f"启动性能监控失败: {e}")
    
    def stop_monitoring(self):
        """停止性能监控"""
        try:
            with self.thread_lock:
                if not self.monitoring_active:
                    self.logger.warning("性能监控未在运行")
                    return
                
                self.monitoring_active = False
                
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=10)
                
                self.logger.info("性能监控停止成功")
                
        except Exception as e:
            self.logger.error(f"停止性能监控失败: {e}")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # 收集进程指标
                process_metrics = self._collect_process_metrics()
                self.process_metrics_history.append(process_metrics)
                
                # 检查告警条件
                self._check_alerts(system_metrics, process_metrics)
                
                # 清理过期数据
                self._cleanup_expired_data()
                
                # 等待下次监控
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"性能监控循环错误: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024
            memory_available_mb = memory.available / 1024 / 1024
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / 1024 / 1024 / 1024
            disk_free_gb = disk.free / 1024 / 1024 / 1024
            
            # 网络使用情况
            network_stats = self._get_network_stats()
            network_sent_mb = (network_stats['bytes_sent'] - self.network_baseline['bytes_sent']) / 1024 / 1024
            network_recv_mb = (network_stats['bytes_recv'] - self.network_baseline['bytes_recv']) / 1024 / 1024
            
            # 进程数量
            process_count = len(psutil.pids())
            
            # 系统负载
            load_average = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                load_average=load_average,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
            return SystemMetrics(
                cpu_percent=0.0, memory_percent=0.0, memory_used_mb=0.0,
                memory_available_mb=0.0, disk_percent=0.0, disk_used_gb=0.0,
                disk_free_gb=0.0, network_sent_mb=0.0, network_recv_mb=0.0,
                process_count=0, load_average=[0.0, 0.0, 0.0], timestamp=datetime.now()
            )
    
    def _collect_process_metrics(self) -> ProcessMetrics:
        """收集当前进程指标"""
        try:
            # 进程信息
            pid = self.current_process.pid
            name = self.current_process.name()
            
            # CPU使用率
            cpu_percent = self.current_process.cpu_percent()
            
            # 内存使用情况
            memory_info = self.current_process.memory_info()
            memory_percent = self.current_process.memory_percent()
            memory_rss_mb = memory_info.rss / 1024 / 1024
            memory_vms_mb = memory_info.vms / 1024 / 1024
            
            # 线程数
            threads_count = self.current_process.num_threads()
            
            # 连接数
            try:
                connections_count = len(self.current_process.connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections_count = 0
            
            # 打开文件数
            try:
                open_files_count = len(self.current_process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files_count = 0
            
            # 创建时间
            create_time = datetime.fromtimestamp(self.current_process.create_time())
            
            return ProcessMetrics(
                pid=pid,
                name=name,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss_mb=memory_rss_mb,
                memory_vms_mb=memory_vms_mb,
                threads_count=threads_count,
                connections_count=connections_count,
                open_files_count=open_files_count,
                create_time=create_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"收集进程指标失败: {e}")
            return ProcessMetrics(
                pid=0, name="unknown", cpu_percent=0.0, memory_percent=0.0,
                memory_rss_mb=0.0, memory_vms_mb=0.0, threads_count=0,
                connections_count=0, open_files_count=0,
                create_time=datetime.now(), timestamp=datetime.now()
            )
    
    def _get_network_stats(self) -> Dict[str, int]:
        """获取网络统计"""
        try:
            stats = psutil.net_io_counters()
            return {
                'bytes_sent': stats.bytes_sent,
                'bytes_recv': stats.bytes_recv,
                'packets_sent': stats.packets_sent,
                'packets_recv': stats.packets_recv
            }
        except Exception as e:
            self.logger.error(f"获取网络统计失败: {e}")
            return {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
    
    def _check_alerts(self, system_metrics: SystemMetrics, process_metrics: ProcessMetrics):
        """检查告警条件"""
        try:
            current_time = datetime.now()
            
            # 检查CPU使用率
            if system_metrics.cpu_percent > self.alert_threshold['cpu_percent']:
                alert = PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric_name="cpu_percent",
                    current_value=system_metrics.cpu_percent,
                    threshold=self.alert_threshold['cpu_percent'],
                    message=f"CPU使用率过高: {system_metrics.cpu_percent:.1f}%",
                    timestamp=current_time
                )
                self._add_alert(alert)
            
            # 检查内存使用率
            if system_metrics.memory_percent > self.alert_threshold['memory_percent']:
                alert = PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric_name="memory_percent",
                    current_value=system_metrics.memory_percent,
                    threshold=self.alert_threshold['memory_percent'],
                    message=f"内存使用率过高: {system_metrics.memory_percent:.1f}%",
                    timestamp=current_time
                )
                self._add_alert(alert)
            
            # 检查磁盘使用率
            if system_metrics.disk_percent > self.alert_threshold['disk_percent']:
                alert = PerformanceAlert(
                    level=AlertLevel.ERROR,
                    metric_name="disk_percent",
                    current_value=system_metrics.disk_percent,
                    threshold=self.alert_threshold['disk_percent'],
                    message=f"磁盘使用率过高: {system_metrics.disk_percent:.1f}%",
                    timestamp=current_time
                )
                self._add_alert(alert)
            
            # 检查进程内存使用
            if process_metrics.memory_percent > 50.0:  # 进程内存超过50%
                alert = PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric_name="process_memory_percent",
                    current_value=process_metrics.memory_percent,
                    threshold=50.0,
                    message=f"进程内存使用率过高: {process_metrics.memory_percent:.1f}%",
                    timestamp=current_time
                )
                self._add_alert(alert)
                
        except Exception as e:
            self.logger.error(f"检查告警条件失败: {e}")
    
    def _add_alert(self, alert: PerformanceAlert):
        """添加告警"""
        try:
            # 检查是否已存在相同的未解决告警
            existing_alert = None
            for existing in self.alerts:
                if (existing.metric_name == alert.metric_name and 
                    existing.level == alert.level and 
                    not existing.resolved):
                    existing_alert = existing
                    break
            
            if existing_alert is None:
                # 添加新告警
                self.alerts.append(alert)
                self.logger.warning(f"新告警: {alert.message}")
                
                # 调用告警回调
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"告警回调执行失败: {e}")
            
            # 限制告警历史数量
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]  # 保留最近500条
                
        except Exception as e:
            self.logger.error(f"添加告警失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def resolve_alert(self, alert_index: int) -> bool:
        """解决告警"""
        try:
            if 0 <= alert_index < len(self.alerts):
                alert = self.alerts[alert_index]
                if not alert.resolved:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    self.logger.info(f"告警已解决: {alert.message}")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"解决告警失败: {e}")
            return False
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None, unit: str = "", description: str = ""):
        """记录自定义指标"""
        try:
            metric = MetricData(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                tags=tags or {},
                unit=unit,
                description=description
            )
            
            self.custom_metrics[name] = metric
            self.metrics_history[name].append(metric)
            
            self.logger.debug(f"记录指标: {name} = {value} {unit}")
            
        except Exception as e:
            self.logger.error(f"记录指标失败: {name} - {e}")
    
    def timer(self, name: str):
        """计时器装饰器"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    self.record_timer(name, duration_ms)
            return wrapper
        return decorator
    
    def record_timer(self, name: str, duration_ms: float):
        """记录计时器"""
        try:
            self.timers[name].append(duration_ms)
            
            # 限制历史记录数量
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-500:]
            
            # 计算统计信息
            durations = self.timers[name]
            self.timer_stats[name] = {
                'count': len(durations),
                'sum': sum(durations),
                'avg': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'latest': duration_ms
            }
            
            # 记录为指标
            self.record_metric(f"timer_{name}_ms", duration_ms, MetricType.TIMER, unit="ms")
            
            # 检查响应时间告警
            if duration_ms > self.alert_threshold['response_time_ms']:
                alert = PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric_name=f"timer_{name}",
                    current_value=duration_ms,
                    threshold=self.alert_threshold['response_time_ms'],
                    message=f"响应时间过长: {name} - {duration_ms:.1f}ms",
                    timestamp=datetime.now()
                )
                self._add_alert(alert)
            
        except Exception as e:
            self.logger.error(f"记录计时器失败: {name} - {e}")
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """递增计数器"""
        try:
            current_metric = self.custom_metrics.get(name)
            if current_metric and current_metric.metric_type == MetricType.COUNTER:
                new_value = current_metric.value + value
            else:
                new_value = value
            
            self.record_metric(name, new_value, MetricType.COUNTER, tags)
            
        except Exception as e:
            self.logger.error(f"递增计数器失败: {name} - {e}")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """设置仪表值"""
        try:
            self.record_metric(name, value, MetricType.GAUGE, tags)
            
        except Exception as e:
            self.logger.error(f"设置仪表值失败: {name} - {e}")
    
    def _cleanup_expired_data(self):
        """清理过期数据"""
        try:
            current_time = datetime.now()
            
            # 清理过期的自定义指标
            expired_metrics = []
            for name, metric in self.custom_metrics.items():
                if current_time - metric.timestamp > timedelta(hours=24):
                    expired_metrics.append(name)
            
            for name in expired_metrics:
                del self.custom_metrics[name]
            
            # 清理过期的计时器数据
            for name in list(self.timers.keys()):
                if name not in self.timer_stats:
                    continue
                
                # 如果计时器长时间没有更新，清理
                if len(self.timers[name]) == 0:
                    del self.timers[name]
                    del self.timer_stats[name]
            
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        try:
            if not self.system_metrics_history:
                return {}
            
            latest = self.system_metrics_history[-1]
            
            # 计算平均值（最近10个数据点）
            recent_metrics = list(self.system_metrics_history)[-10:]
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            
            return {
                'current_cpu_percent': latest.cpu_percent,
                'current_memory_percent': latest.memory_percent,
                'current_disk_percent': latest.disk_percent,
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'memory_used_mb': latest.memory_used_mb,
                'memory_available_mb': latest.memory_available_mb,
                'disk_used_gb': latest.disk_used_gb,
                'disk_free_gb': latest.disk_free_gb,
                'network_sent_mb': latest.network_sent_mb,
                'network_recv_mb': latest.network_recv_mb,
                'process_count': latest.process_count,
                'load_average': latest.load_average,
                'timestamp': latest.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取系统摘要失败: {e}")
            return {}
    
    def get_process_summary(self) -> Dict[str, Any]:
        """获取进程摘要"""
        try:
            if not self.process_metrics_history:
                return {}
            
            latest = self.process_metrics_history[-1]
            uptime = datetime.now() - self.start_time
            
            return {
                'pid': latest.pid,
                'name': latest.name,
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'memory_rss_mb': latest.memory_rss_mb,
                'memory_vms_mb': latest.memory_vms_mb,
                'threads_count': latest.threads_count,
                'connections_count': latest.connections_count,
                'open_files_count': latest.open_files_count,
                'uptime_seconds': int(uptime.total_seconds()),
                'timestamp': latest.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取进程摘要失败: {e}")
            return {}
    
    def get_timer_summary(self) -> Dict[str, Any]:
        """获取计时器摘要"""
        try:
            summary = {}
            for name, stats in self.timer_stats.items():
                summary[name] = {
                    'count': stats['count'],
                    'avg_ms': round(stats['avg'], 2),
                    'min_ms': round(stats['min'], 2),
                    'max_ms': round(stats['max'], 2),
                    'latest_ms': round(stats['latest'], 2),
                    'total_ms': round(stats['sum'], 2)
                }
            return summary
            
        except Exception as e:
            self.logger.error(f"获取计时器摘要失败: {e}")
            return {}
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        try:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            resolved_alerts = [alert for alert in self.alerts if alert.resolved]
            
            # 按级别统计
            level_counts = defaultdict(int)
            for alert in active_alerts:
                level_counts[alert.level.value] += 1
            
            return {
                'total_alerts': len(self.alerts),
                'active_alerts': len(active_alerts),
                'resolved_alerts': len(resolved_alerts),
                'level_counts': dict(level_counts),
                'latest_alerts': [
                    {
                        'level': alert.level.value,
                        'metric_name': alert.metric_name,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'resolved': alert.resolved
                    }
                    for alert in self.alerts[-10:]  # 最近10个告警
                ]
            }
            
        except Exception as e:
            self.logger.error(f"获取告警摘要失败: {e}")
            return {}
    
    def get_custom_metrics_summary(self) -> Dict[str, Any]:
        """获取自定义指标摘要"""
        try:
            summary = {}
            for name, metric in self.custom_metrics.items():
                summary[name] = {
                    'value': metric.value,
                    'type': metric.metric_type.value,
                    'unit': metric.unit,
                    'description': metric.description,
                    'tags': metric.tags,
                    'timestamp': metric.timestamp.isoformat()
                }
            return summary
            
        except Exception as e:
            self.logger.error(f"获取自定义指标摘要失败: {e}")
            return {}
    
    def export_metrics(self, format_type: str = "json") -> str:
        """导出指标数据"""
        try:
            data = {
                'system_summary': self.get_system_summary(),
                'process_summary': self.get_process_summary(),
                'timer_summary': self.get_timer_summary(),
                'alerts_summary': self.get_alerts_summary(),
                'custom_metrics': self.get_custom_metrics_summary(),
                'export_time': datetime.now().isoformat(),
                'monitoring_active': self.monitoring_active
            }
            
            if format_type.lower() == "json":
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return str(data)
                
        except Exception as e:
            self.logger.error(f"导出指标数据失败: {e}")
            return "{}"
    
    def reset_metrics(self):
        """重置所有指标"""
        try:
            with self.thread_lock:
                self.metrics_history.clear()
                self.system_metrics_history.clear()
                self.process_metrics_history.clear()
                self.custom_metrics.clear()
                self.alerts.clear()
                self.timers.clear()
                self.timer_stats.clear()
                
                # 重置网络基线
                self.network_baseline = self._get_network_stats()
                
                self.logger.info("所有指标已重置")
                
        except Exception as e:
            self.logger.error(f"重置指标失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'monitor_interval': self.monitor_interval,
                'history_size': self.history_size,
                'current_system_metrics_count': len(self.system_metrics_history),
                'current_process_metrics_count': len(self.process_metrics_history),
                'custom_metrics_count': len(self.custom_metrics),
                'active_timers_count': len(self.timer_stats),
                'total_alerts_count': len(self.alerts),
                'active_alerts_count': len([a for a in self.alerts if not a.resolved]),
                'uptime_seconds': int((datetime.now() - self.start_time).total_seconds()),
                'alert_thresholds': self.alert_threshold,
                'is_operational': True
            }
            
        except Exception as e:
            self.logger.error(f"获取监控器状态失败: {e}")
            return {'is_operational': False}
    
    def __del__(self):
        """析构函数"""
        try:
            self.stop_monitoring()
        except:
            pass 