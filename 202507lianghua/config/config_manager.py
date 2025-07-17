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

class RiskConfig(BaseModel):
    """风险配置模型"""
    max_position_size: float = Field(0.01, ge=0.001, le=0.1, description="最大单笔仓位比例")
    max_total_exposure: float = Field(0.10, ge=0.01, le=0.5, description="最大总持仓比例")
    max_drawdown: float = Field(0.05, ge=0.01, le=0.2, description="最大回撤比例")
    loss_limit: int = Field(5, ge=1, le=20, description="连续亏损限制")
    emergency_stop_loss: float = Field(0.15, ge=0.05, le=0.3, description="紧急止损比例")
    
    @validator('max_total_exposure')
    def validate_exposure(cls, v, values):
        if 'max_position_size' in values and v < values['max_position_size']:
            raise ValueError('总持仓比例必须大于单笔仓位比例')
        return v

class TradingConfig(BaseModel):
    """交易配置模型"""
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
    risk: RiskConfig = Field(default_factory=RiskConfig, description="风险管理配置")
    leverage_range: Dict[str, List[int]] = Field(
        default_factory=lambda: {
            "initial": [2, 5],
            "bull": [10, 20],
            "bear": [1, 3]
        },
        description="杠杆范围配置"
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
            "api_latency": 5.0
        },
        description="告警阈值配置"
    )
    telegram_enabled: bool = Field(True, description="是否启用Telegram通知")
    log_level: str = Field("INFO", description="日志级别")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'日志级别必须是{valid_levels}之一')
        return v.upper()

class SystemConfig(BaseModel):
    """系统配置模型"""
    api: Dict[str, APIConfig] = Field(
        default_factory=lambda: {
            "binance": APIConfig(base_url="https://fapi.binance.com"),
            "coingecko": APIConfig(base_url="https://api.coingecko.com/api/v3")
        },
        description="API配置"
    )
    trading: TradingConfig = Field(default_factory=TradingConfig, description="交易配置")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="监控配置")
    redis_url: str = Field("redis://localhost:6379/0", description="Redis连接URL")
    max_workers: int = Field(5, ge=1, le=20, description="最大工作线程数")
    enable_backtesting: bool = Field(True, description="是否启用回测功能")

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
    
    def __init__(self, config_file: str = "config/config.json"):
        """初始化配置管理器"""
        if hasattr(self, '_initialized'):
            return
        
        self.config_file = config_file
        self.config: Optional[SystemConfig] = None
        self._last_reload: float = time.time()
        self._watchers: List = []
        self._initialized = True
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_config()
    

    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            # 检查配置文件是否存在
            config_path = Path(self.config_file)
            if not config_path.exists():
                self.logger.warning(f"配置文件不存在: {config_path}")
                self.create_default_config()
                return False
            
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 合并环境变量
            self._merge_env_vars(config_data)
            
            # 验证配置
            self.config = SystemConfig(**config_data)
            
            # 更新最后加载时间
            self._last_reload = time.time()
            
            self.logger.info("配置加载成功")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件JSON格式错误: {e}")
            raise ConfigurationError(f"配置文件格式错误: {e}")
        except pydantic.ValidationError as e:
            self.logger.error(f"配置验证失败: {e}")
            raise ConfigurationError(f"配置验证失败: {e}")
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            raise ConfigurationError(f"配置加载失败: {e}")
    
    def _merge_env_vars(self, config_data: Dict[str, Any]) -> None:
        """合并环境变量到配置中"""
        env_mappings = {
            'BINANCE_API_KEY': ['api', 'binance', 'api_key'],
            'BINANCE_API_SECRET': ['api', 'binance', 'api_secret'],
            'BINANCE_BASE_URL': ['api', 'binance', 'base_url'],
            'COINGECKO_API_KEY': ['api', 'coingecko', 'api_key'],
            'REDIS_URL': ['redis_url'],
            'LOG_LEVEL': ['monitoring', 'log_level'],
            'MAX_WORKERS': ['max_workers'],
            'TELEGRAM_BOT_TOKEN': ['monitoring', 'telegram_bot_token'],
            'TELEGRAM_CHAT_ID': ['monitoring', 'telegram_chat_id'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 设置嵌套配置值
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # 类型转换
                if env_var == 'MAX_WORKERS':
                    value = int(value)
                elif env_var == 'TELEGRAM_CHAT_ID':
                    value = int(value)
                
                current[config_path[-1]] = value
    
    def create_default_config(self) -> None:
        """创建默认配置文件"""
        try:
            # 创建配置目录
            config_dir = Path(self.config_file).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建默认配置
            default_config = SystemConfig()
            
            # 保存配置文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config.dict(), f, indent=2, ensure_ascii=False)
            
            self.config = default_config
            self.logger.info(f"已创建默认配置文件: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"创建默认配置失败: {e}")
            raise ConfigurationError(f"创建默认配置失败: {e}")
    
    def get_config(self) -> SystemConfig:
        """获取配置对象"""
        if self.config is None:
            raise ConfigurationError("配置未加载")
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if self.config is None:
            return default
        
        try:
            keys = key.split('.')
            value = self.config.dict()
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        try:
            if self.config is None:
                return False
            
            # 更新配置对象
            config_dict = self.config.dict()
            keys = key.split('.')
            current = config_dict
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            
            # 重新验证配置
            self.config = SystemConfig(**config_dict)
            
            # 保存到文件
            self.save_config()
            
            self.logger.info(f"配置已更新: {key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"设置配置失败: {e}")
            return False
    
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            if self.config is None:
                return False
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config.dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info("配置已保存")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            return False
    
    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            old_config = self.config
            success = self.load_config()
            
            if success:
                # 通知观察者配置已更新
                self._notify_watchers(old_config, self.config)
            
            return success
            
        except Exception as e:
            self.logger.error(f"重新加载配置失败: {e}")
            return False
    
    def add_watcher(self, callback) -> None:
        """添加配置变化监听器"""
        if callback not in self._watchers:
            self._watchers.append(callback)
    
    def remove_watcher(self, callback) -> None:
        """移除配置变化监听器"""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def _notify_watchers(self, old_config: SystemConfig, new_config: SystemConfig) -> None:
        """通知配置变化监听器"""
        for callback in self._watchers:
            try:
                callback(old_config, new_config)
            except Exception as e:
                self.logger.error(f"通知配置监听器失败: {e}")
    
    def validate_api_credentials(self) -> Dict[str, bool]:
        """验证API凭据"""
        results = {}
        
        # 验证Binance API
        binance_key = os.getenv('BINANCE_API_KEY')
        binance_secret = os.getenv('BINANCE_API_SECRET')
        results['binance'] = bool(binance_key and binance_secret)
        
        # 验证CoinGecko API
        coingecko_key = os.getenv('COINGECKO_API_KEY')
        results['coingecko'] = bool(coingecko_key)
        
        # 验证Telegram配置
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat = os.getenv('TELEGRAM_CHAT_ID')
        results['telegram'] = bool(telegram_token and telegram_chat)
        
        return results
    
    def get_api_config(self, provider: str) -> Optional[APIConfig]:
        """获取API配置"""
        if self.config is None:
            return None
        
        api_configs = self.config.api
        return api_configs.get(provider)
    
    def is_config_expired(self, max_age: int = 3600) -> bool:
        """检查配置是否过期"""
        return time.time() - self._last_reload > max_age

class ConfigurationError(Exception):
    """配置错误异常"""
    pass

# 全局配置管理器实例（延迟创建）
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# 便捷函数
def get_config() -> SystemConfig:
    """获取配置对象"""
    return get_config_manager().get_config()

def get_setting(key: str, default: Any = None) -> Any:
    """获取配置值"""
    return get_config_manager().get(key, default)

def set_setting(key: str, value: Any) -> bool:
    """设置配置值"""
    return get_config_manager().set(key, value)

def reload_config() -> bool:
    """重新加载配置"""
    return get_config_manager().reload_config()

def validate_credentials() -> Dict[str, bool]:
    """验证凭据"""
    return get_config_manager().validate_api_credentials()

if __name__ == "__main__":
    # 测试配置管理器
    try:
        config = get_config()
        print("配置加载成功")
        print(f"MACD配置: {config.trading.macd}")
        print(f"风险配置: {config.trading.risk}")
        print(f"API凭据验证: {validate_credentials()}")
    except ConfigurationError as e:
        print(f"配置错误: {e}") 