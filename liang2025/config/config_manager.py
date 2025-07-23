"""
配置管理模块
功能：配置文件加载、验证、环境变量管理、动态配置更新
作者：Trading System Team
创建时间：2025-01-28
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import pydantic
from pydantic import BaseModel, Field, validator
import threading
import time

# 加载环境变量
load_dotenv()

class APIConfig(BaseModel):
    """API配置模型"""
    base_url: str = Field(..., description="API基础URL")
    timeout: int = Field(30, ge=1, le=120, description="超时时间(秒)")
    max_retries: int = Field(3, ge=1, le=10, description="最大重试次数")
    rate_limit: int = Field(1200, ge=100, le=5000, description="每分钟最大请求数")
    testnet: bool = Field(False, description="是否为测试网")
    weight_limit: int = Field(6000, description="权重限制")
    api_key: Optional[str] = Field(None, description="API密钥")
    api_secret: Optional[str] = Field(None, description="API密钥密码")
    simulation_mode: bool = Field(False, description="是否为模拟模式")  # 添加这个字段！
    
    @validator('base_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL必须以http://或https://开头')
        return v

class MACDConfig(BaseModel):
    """MACD配置模型"""
    fast: int = Field(13, ge=1, le=50, description="快线周期")
    slow: int = Field(34, ge=1, le=100, description="慢线周期")
    signal: int = Field(9, ge=1, le=30, description="信号线周期")
    
    @validator('slow')
    def validate_periods(cls, v, values):
        if 'fast' in values and v <= values['fast']:
            raise ValueError('慢线周期必须大于快线周期')
        return v

class MACDDivergenceConfig(BaseModel):
    """MACD背离配置模型"""
    lookback_period: int = Field(100, ge=20, le=500, description="回看周期")
    min_peak_distance: int = Field(5, ge=1, le=20, description="最小峰值距离")
    prominence_threshold: float = Field(0.1, ge=0.01, le=1.0, description="显著性阈值")
    confirmation_candles: int = Field(3, ge=1, le=10, description="确认K线数")
    min_r_squared: float = Field(0.7, ge=0.1, le=1.0, description="最小R平方值")
    min_trend_points: int = Field(2, ge=2, le=10, description="最小趋势点数")
    min_divergence_strength: float = Field(0.3, ge=0.1, le=1.0, description="最小背离强度")
    min_duration_hours: int = Field(12, ge=1, le=168, description="最小持续时间(小时)")
    max_duration_hours: int = Field(168, ge=1, le=720, description="最大持续时间(小时)")
    dynamic_threshold: bool = Field(True, description="是否使用动态阈值")
    min_significance: float = Field(0.02, ge=0.001, le=0.1, description="最小显著性")
    # 新增连续检测参数
    continuous_detection: bool = Field(True, description="是否启用连续检测")
    consecutive_signals: int = Field(2, ge=1, le=5, description="连续信号数量")
    prominence_mult: float = Field(0.5, ge=0.1, le=2.0, description="显著性乘数")
    strength_filter: float = Field(0.6, ge=0.1, le=1.0, description="强度过滤阈值")

class RiskConfig(BaseModel):
    """风险配置模型"""
    max_position_size: float = Field(0.01, ge=0.001, le=0.1, description="最大单笔仓位比例")
    max_total_exposure: float = Field(0.10, ge=0.01, le=0.5, description="最大总持仓比例")
    max_drawdown: float = Field(0.05, ge=0.01, le=0.2, description="最大回撤比例")
    loss_limit: int = Field(5, ge=1, le=20, description="连续亏损限制")
    emergency_stop_loss: float = Field(0.15, ge=0.05, le=0.3, description="紧急止损比例")
    # 新增风控参数
    risk_per_trade: float = Field(0.005, ge=0.001, le=0.02, description="单笔交易风险比例")
    stop_loss_pct: float = Field(0.01, ge=0.001, le=0.05, description="收益率止损比例")
    take_profit_ratio: float = Field(3.0, ge=1.0, le=10.0, description="盈亏比")
    max_leverage: int = Field(20, ge=1, le=100, description="最大杠杆倍数")
    leverage_bull: float = Field(5.0, ge=1.0, le=20.0, description="牛市杠杆倍数")
    leverage_bear: float = Field(3.0, ge=1.0, le=15.0, description="熊市杠杆倍数")
    stop_loss_mult: float = Field(1.2, ge=1.0, le=3.0, description="止损乘数")
    max_add_positions: int = Field(3, ge=1, le=10, description="最大加仓次数")
    add_profit_thresh: float = Field(0.02, ge=0.001, le=0.1, description="加仓盈利阈值")
    time_stop_min: List[int] = Field(default_factory=lambda: [30, 60], description="时间止损(分钟)")
    funding_thresh: float = Field(0.0001, ge=0.0001, le=0.001, description="资金费率阈值")
    confidence_threshold: float = Field(0.5, ge=0.1, le=1.0, description="信号置信度阈值")
    dynamic_stop_loss: bool = Field(True, description="是否启用动态止损")
    trailing_stop_loss: bool = Field(True, description="是否启用追踪止损")
    partial_close_profit: float = Field(0.03, ge=0.01, le=0.1, description="部分平仓盈利阈值")
    
    @validator('max_total_exposure')
    def validate_exposure(cls, v, values):
        if 'max_position_size' in values and v < values['max_position_size']:
            raise ValueError('总持仓比例必须大于单笔仓位比例')
        return v

class PositionManagementConfig(BaseModel):
    """持仓管理配置模型"""
    portions: int = Field(3, ge=1, le=10, description="持仓分割数")
    add_position_conditions: Dict[str, Union[float, int, List[float]]] = Field(
        default_factory=lambda: {
            "min_profit": 0.02,
            "max_add_count": 3,
            "add_sizes": [0.5, 0.3, 0.2]
        },
        description="加仓条件"
    )
    reduce_position_conditions: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            "profit_levels": [0.03, 0.06, 0.09],
            "reduce_ratios": [0.3, 0.5, 0.7]
        },
        description="减仓条件"
    )
    time_based_management: Dict[str, Union[int, float]] = Field(
        default_factory=lambda: {
            "reduce_after_minutes": 30,
            "close_after_minutes": 60,
            "reduce_ratio": 0.5
        },
        description="基于时间的仓位管理"
    )

class SignalGenerationConfig(BaseModel):
    """信号生成配置模型"""
    volume_mult: float = Field(1.5, ge=0.5, le=5.0, description="成交量乘数")
    atr_mult: float = Field(0.5, ge=0.1, le=2.0, description="ATR乘数")
    vol_factor_mult: float = Field(0.05, ge=0.01, le=0.2, description="波动因子乘数")
    back_div_gap_thresh: float = Field(0.1, ge=0.01, le=0.5, description="背离间隔阈值")
    hang_buffer: float = Field(0.01, ge=0.001, le=0.1, description="挂单缓冲")
    signal_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "macd": 0.4,
            "technical_indicators": 0.3,
            "pattern_recognition": 0.2,
            "cycle_analysis": 0.1
        },
        description="信号权重"
    )
    noise_filter: Dict[str, Union[bool, float]] = Field(
        default_factory=lambda: {
            "enabled": True,
            "min_strength": 0.3,
            "volatility_threshold": 0.02,
            "volume_confirmation": True
        },
        description="噪音过滤"
    )

class MarketConditionsConfig(BaseModel):
    """市场条件配置模型"""
    cycle_detection: Dict[str, Union[bool, List[int], float]] = Field(
        default_factory=lambda: {
            "enabled": True,
            "ma_periods": [50, 200],
            "volume_threshold": 1.2,
            "inflow_detection": True
        },
        description="周期检测"
    )
    pattern_recognition: Dict[str, Union[bool, List[str], float, int]] = Field(
        default_factory=lambda: {
            "enabled": True,
            "patterns": ["ENGULFING", "HEAD_SHOULDER", "CONVERGENCE_TRIANGLE"],
            "min_confidence": 0.6,
            "lookback_period": 50
        },
        description="形态识别"
    )
    liquidity_analysis: Dict[str, Union[bool, int, List[str]]] = Field(
        default_factory=lambda: {
            "enabled": True,
            "min_volume": 1000000,
            "depth_analysis": True,
            "backup_sources": ["coingecko"]
        },
        description="流动性分析"
    )

class ExecutionConfig(BaseModel):
    """执行配置模型"""
    async_enabled: bool = Field(True, description="是否启用异步执行")
    order_types: Dict[str, bool] = Field(
        default_factory=lambda: {
            "market": True,
            "limit": True,
            "stop": True,
            "trailing_stop": True
        },
        description="订单类型"
    )
    slippage_protection: Dict[str, float] = Field(
        default_factory=lambda: {
            "max_slippage": 0.001,
            "price_deviation_threshold": 0.002
        },
        description="滑点保护"
    )
    order_management: Dict[str, Union[int, bool]] = Field(
        default_factory=lambda: {
            "cancel_timeout": 30,
            "retry_attempts": 3,
            "partial_fill_handling": True
        },
        description="订单管理"
    )

class TradingConfig(BaseModel):
    """交易配置模型"""
    symbol: str = Field("BTCUSDT", description="交易对")
    initial_capital: float = Field(10000.0, description="初始资金")
    risk_per_trade: float = Field(0.005, description="每笔交易风险")
    max_positions: int = Field(3, description="最大持仓数")
    max_leverage: int = Field(20, description="最大杠杆")
    base_currency: str = Field("USDT", description="基础货币")
    
    intervals: Dict[str, str] = Field(
        default_factory=lambda: {
            "small": "15m",
            "medium": "1h",
            "large": "4h", 
            "daily": "1d"
        },
        description="K线间隔配置"
    )
    macd: MACDConfig = Field(default_factory=MACDConfig, description="MACD配置")
    ma_periods: List[int] = Field(
        default_factory=lambda: [30, 50, 120, 200, 256],
        description="均线周期列表"
    )
    macd_divergence: MACDDivergenceConfig = Field(
        default_factory=MACDDivergenceConfig, description="MACD背离配置"
    )
    risk: RiskConfig = Field(default_factory=RiskConfig, description="风险管理配置")
    leverage_range: Dict[str, List[int]] = Field(
        default_factory=lambda: {
            "initial": [2, 5],
            "bull": [10, 20],
            "bear": [1, 3]
        },
        description="杠杆范围配置"
    )
    position_management: PositionManagementConfig = Field(
        default_factory=PositionManagementConfig, description="持仓管理配置"
    )
    
    @validator('ma_periods')
    def validate_ma_periods(cls, v):
        if len(v) < 2:
            raise ValueError('至少需要2个均线周期')
        if not all(isinstance(p, int) and p > 0 for p in v):
            raise ValueError('均线周期必须是正整数')
        return sorted(v)

class MonitoringConfig(BaseModel):
    """监控配置模型"""
    metrics_interval: int = Field(60, ge=10, le=300, description="指标收集间隔(秒)")
    alert_thresholds: Dict[str, Union[int, float]] = Field(
        default_factory=lambda: {
            "cpu_usage": 80,
            "memory_usage": 85,
            "error_rate": 0.05,
            "api_latency": 5.0,
            "consecutive_losses": 3,
            "daily_loss_limit": 0.05
        },
        description="告警阈值配置"
    )
    telegram_enabled: bool = Field(True, description="是否启用Telegram通知")
    log_level: str = Field("INFO", description="日志级别")
    performance_tracking: Dict[str, Union[bool, List[str], int]] = Field(
        default_factory=lambda: {
            "enabled": True,
            "metrics": ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown"],
            "reporting_interval": 3600
        },
        description="性能追踪"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'日志级别必须是{valid_levels}之一')
        return v.upper()

class SystemResourceConfig(BaseModel):
    """系统资源配置模型"""
    resource_monitoring: Dict[str, Union[bool, int]] = Field(
        default_factory=lambda: {
            "enabled": True,
            "cpu_threshold": 80,
            "memory_threshold": 85,
            "pause_on_high_usage": True,
            "check_interval": 30
        },
        description="资源监控"
    )
    backup_data_sources: Dict[str, Union[bool, List[str], int]] = Field(
        default_factory=lambda: {
            "enabled": True,
            "fallback_apis": ["coingecko", "coinmarketcap"],
            "failover_delay": 5
        },
        description="备份数据源"
    )

class Config(BaseModel):
    """主配置模型"""
    api: Dict[str, APIConfig] = Field(
        default_factory=lambda: {
            "binance": APIConfig(base_url="https://fapi.binance.com"),
            "coingecko": APIConfig(base_url="https://api.coingecko.com/api/v3")
        },
        description="API配置"
    )
    trading: TradingConfig = Field(default_factory=TradingConfig, description="交易配置")
    signal_generation: SignalGenerationConfig = Field(
        default_factory=SignalGenerationConfig, description="信号生成配置"
    )
    market_conditions: MarketConditionsConfig = Field(
        default_factory=MarketConditionsConfig, description="市场条件配置"
    )
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="执行配置")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="监控配置")
    system: SystemResourceConfig = Field(default_factory=SystemResourceConfig, description="系统资源配置")
    redis_url: str = Field("redis://localhost:6379/0", description="Redis连接URL")
    max_workers: int = Field(5, ge=1, le=20, description="最大工作线程数")
    enable_backtesting: bool = Field(True, description="是否启用回测功能")

class ConfigurationError(Exception):
    """配置错误异常"""
    pass

class ConfigManager:
    """配置管理器"""
    _instance: Optional['ConfigManager'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: str = None):
        """初始化配置管理器"""
        if hasattr(self, '_initialized'):
            return
            
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        
        self.config_file = config_file
        self.config: Optional[Config] = None
        self._last_reload: float = time.time()
        self._watchers: List = []
        self._initialized = True
        
        # 初始化日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            # 加载主配置
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            
            # 验证并创建配置对象
            self.config = Config(**config_data)
            self._last_reload = time.time()
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            # 使用默认配置
            self.config = Config()
            
            # 尝试创建默认配置文件
            try:
                self._create_default_config()
            except Exception as create_error:
                self.logger.error(f"创建默认配置失败: {create_error}")
    
    def _create_default_config(self) -> None:
        """创建默认配置文件"""
        try:
            # 创建配置目录
            config_dir = Path(self.config_file).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建默认配置
            default_config = Config()
            
            # 保存配置文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config.dict(), f, indent=2, ensure_ascii=False)
            
            self.config = default_config
            self.logger.info(f"已创建默认配置文件: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"创建默认配置失败: {e}")
            raise ConfigurationError(f"创建默认配置失败: {e}")
    
    def get_config(self) -> Config:
        """获取配置对象"""
        if self.config is None:
            self.load_config()
        return self.config
    
    def get_api_config(self, api_name: str) -> APIConfig:
        """获取API配置"""
        return self.get_config().api.get(api_name)
    
    def get_trading_config(self) -> TradingConfig:
        """获取交易配置"""
        return self.get_config().trading
    
    def get_risk_config(self) -> RiskConfig:
        """获取风险配置"""
        return self.get_config().trading.risk
    
    def get_signal_config(self) -> SignalGenerationConfig:
        """获取信号生成配置"""
        return self.get_config().signal_generation
    
    def get_execution_config(self) -> ExecutionConfig:
        """获取执行配置"""
        return self.get_config().execution
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        return self.get_config().monitoring
    
    def get_system_config(self) -> SystemResourceConfig:
        """获取系统资源配置"""
        return self.get_config().system
    
    def get_macd_config(self) -> MACDConfig:
        """获取MACD配置"""
        return self.get_config().trading.macd
    
    def get_macd_divergence_config(self) -> MACDDivergenceConfig:
        """获取MACD背离配置"""
        return self.get_config().trading.macd_divergence
    
    def get_position_management_config(self) -> PositionManagementConfig:
        """获取持仓管理配置"""
        return self.get_config().trading.position_management
    
    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            old_config = self.config
            if self.load_config():
                self.logger.info("配置重新加载成功")
                return True
            else:
                self.config = old_config
                return False
        except Exception as e:
            self.logger.error(f"配置重新加载失败: {e}")
            return False
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            # 读取当前配置
            with open(self.config_file, 'r', encoding='utf-8') as f:
                current_config = json.load(f)
            
            # 深度合并配置
            self._deep_merge(current_config, config_updates)
            
            # 验证配置
            Config(**current_config)
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, indent=2, ensure_ascii=False)
            
            # 重新加载
            self.load_config()
            self.logger.info("配置更新成功")
            return True
            
        except Exception as e:
            self.logger.error(f"配置更新失败: {e}")
            return False
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """深度合并字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def get_param(self, param_path: str, default: Any = None) -> Any:
        """获取参数值"""
        try:
            keys = param_path.split('.')
            current = self.get_config().dict()
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except Exception as e:
            self.logger.error(f"获取参数失败: {param_path} - {e}")
            return default
    
    def set_param(self, param_path: str, value: Any) -> bool:
        """设置参数值"""
        try:
            keys = param_path.split('.')
            updates = {}
            current = updates
            
            for key in keys[:-1]:
                current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
            
            return self.update_config(updates)
            
        except Exception as e:
            self.logger.error(f"设置参数失败: {param_path} - {e}")
            return False
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get_param(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set_param(key, value)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"ConfigManager(config_file={self.config_file})"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__() 

    # 简化配置管理，专注核心功能 