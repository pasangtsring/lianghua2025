"""
基础模块测试文件
测试配置管理模块和日志系统模块的功能
作者：Trading System Team
创建时间：2025-01-28
"""

import pytest
import tempfile
import os
import json
import time
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import (
    ConfigManager, 
    ConfigurationError, 
    get_config, 
    get_setting, 
    set_setting,
    validate_credentials,
    SystemConfig,
    MACDConfig,
    RiskConfig
)

from utils.logger import (
    LoggerManager,
    init_logger,
    get_logger,
    log_trade_signal,
    log_trade_order,
    log_position_change,
    log_performance_metrics,
    log_system_performance,
    performance_monitor,
    LogLevel,
    TelegramLogHandler,
    JsonFormatter,
    PerformanceLogHandler
)

class TestConfigManager:
    """配置管理器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        
        # 重置全局实例
        ConfigManager._instance = None
        
        # 创建测试配置
        self.test_config = {
            "api": {
                "binance": {
                    "base_url": "https://fapi.binance.com",
                    "timeout": 30,
                    "max_retries": 3,
                    "rate_limit": 1200
                }
            },
            "trading": {
                "macd": {
                    "fast": 13,
                    "slow": 34,
                    "signal": 9
                },
                "risk": {
                    "max_position_size": 0.01,
                    "max_total_exposure": 0.10,
                    "max_drawdown": 0.05,
                    "loss_limit": 5,
                    "emergency_stop_loss": 0.15
                }
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_interval": 60,
                "telegram_enabled": True
            }
        }
    
    def teardown_method(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # 重置全局实例
        ConfigManager._instance = None
    
    def test_config_creation(self):
        """测试配置文件创建"""
        # 创建配置文件
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        # 创建配置管理器
        config_manager = ConfigManager(self.config_file)
        
        # 验证配置加载
        assert config_manager.config is not None
        assert config_manager.config.api['binance'].base_url == "https://fapi.binance.com"
        assert config_manager.config.trading.macd.fast == 13
        assert config_manager.config.trading.risk.max_position_size == 0.01
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        # 使用不存在的配置文件
        config_manager = ConfigManager(self.config_file)
        
        # 验证默认配置创建
        assert os.path.exists(self.config_file)
        assert config_manager.config is not None
        
        # 验证默认值
        assert config_manager.config.trading.macd.fast == 13
        assert config_manager.config.trading.risk.max_position_size == 0.01
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效配置
        invalid_config = {
            "trading": {
                "macd": {
                    "fast": 50,  # 快线周期大于慢线周期
                    "slow": 30
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        # 验证配置验证失败
        with pytest.raises(ConfigurationError):
            ConfigManager(self.config_file)
    
    def test_env_var_merging(self):
        """测试环境变量合并"""
        # 设置环境变量
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_api_key',
            'BINANCE_API_SECRET': 'test_api_secret',
            'LOG_LEVEL': 'DEBUG'
        }):
            # 创建配置文件
            with open(self.config_file, 'w') as f:
                json.dump(self.test_config, f)
            
            config_manager = ConfigManager(self.config_file)
            
            # 验证环境变量被合并
            assert config_manager.config.api['binance'].api_key == 'test_api_key'
            assert config_manager.config.api['binance'].api_secret == 'test_api_secret'
            assert config_manager.config.monitoring.log_level == 'DEBUG'
    
    def test_config_get_set(self):
        """测试配置获取和设置"""
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        config_manager = ConfigManager(self.config_file)
        
        # 测试获取配置
        assert config_manager.get('trading.macd.fast') == 13
        assert config_manager.get('trading.risk.max_position_size') == 0.01
        assert config_manager.get('nonexistent.key', 'default') == 'default'
        
        # 测试设置配置
        assert config_manager.set('trading.macd.fast', 15) == True
        assert config_manager.get('trading.macd.fast') == 15
    
    def test_config_reload(self):
        """测试配置重新加载"""
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        config_manager = ConfigManager(self.config_file)
        original_value = config_manager.get('trading.macd.fast')
        
        # 修改配置文件
        modified_config = self.test_config.copy()
        modified_config['trading']['macd']['fast'] = 15
        
        with open(self.config_file, 'w') as f:
            json.dump(modified_config, f)
        
        # 重新加载配置
        assert config_manager.reload_config() == True
        assert config_manager.get('trading.macd.fast') == 15
    
    def test_api_credentials_validation(self):
        """测试API凭据验证"""
        config_manager = ConfigManager(self.config_file)
        
        # 测试无凭据情况
        credentials = config_manager.validate_api_credentials()
        assert credentials['binance'] == False
        assert credentials['coingecko'] == False
        assert credentials['telegram'] == False
        
        # 测试有凭据情况
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_key',
            'BINANCE_API_SECRET': 'test_secret',
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': '123456'
        }):
            credentials = config_manager.validate_api_credentials()
            assert credentials['binance'] == True
            assert credentials['telegram'] == True
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        config_manager1 = ConfigManager(self.config_file)
        config_manager2 = ConfigManager(self.config_file)
        
        assert config_manager1 is config_manager2

class TestLoggerManager:
    """日志管理器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        
        # 清理全局日志管理器
        import utils.logger
        utils.logger.logger_manager = None
        
        # 创建测试配置
        self.test_config = {
            'log_level': 'DEBUG',
            'log_dir': self.log_dir,
            'telegram_bot_token': 'test_token',
            'telegram_chat_id': '123456'
        }
    
    def teardown_method(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # 清理全局日志管理器
        import utils.logger
        if utils.logger.logger_manager:
            utils.logger.logger_manager.close()
            utils.logger.logger_manager = None
    
    def test_logger_initialization(self):
        """测试日志系统初始化"""
        logger_manager = LoggerManager(self.test_config)
        
        # 验证日志目录创建
        assert os.path.exists(self.log_dir)
        
        # 验证处理器创建
        assert logger_manager.handlers['console'] is not None
        assert logger_manager.handlers['file'] is not None
        assert logger_manager.handlers['error'] is not None
        assert logger_manager.handlers['performance'] is not None
        
        # 验证交易日志记录器
        assert logger_manager.trading_logger is not None
    
    def test_logger_creation(self):
        """测试日志记录器创建"""
        logger_manager = LoggerManager(self.test_config)
        
        # 创建日志记录器
        logger = logger_manager.get_logger('test_module')
        
        # 验证日志记录器
        assert logger is not None
        assert logger.name == 'test_module'
    
    def test_custom_log_levels(self):
        """测试自定义日志级别"""
        logger_manager = LoggerManager(self.test_config)
        logger = logger_manager.get_logger('test_custom_levels')
        
        # 测试TRADE级别
        assert hasattr(logger, 'trade')
        assert logging.getLevelName(LogLevel.TRADE.value) == 'TRADE'
        
        # 测试PERFORMANCE级别
        assert hasattr(logger, 'performance')
        assert logging.getLevelName(LogLevel.PERFORMANCE.value) == 'PERFORMANCE'
    
    def test_trade_logging(self):
        """测试交易日志记录"""
        logger_manager = LoggerManager(self.test_config)
        
        # 测试交易信号日志
        test_signal_data = {
            'symbol': 'BTCUSDT',
            'price': 45000,
            'confidence': 0.85,
            'macd_divergence': True
        }
        
        logger_manager.log_trade_signal('BTCUSDT', 'LONG', test_signal_data)
        
        # 验证交易日志文件创建
        trade_log_file = os.path.join(self.log_dir, "trades.log")
        assert os.path.exists(trade_log_file)
        
        # 验证日志内容
        with open(trade_log_file, 'r') as f:
            log_content = f.read()
            assert 'BTCUSDT' in log_content
            assert 'LONG' in log_content
            assert 'signal' in log_content
    
    def test_performance_logging(self):
        """测试性能日志记录"""
        logger_manager = LoggerManager(self.test_config)
        
        # 记录性能指标
        logger_manager.log_system_performance('api_latency', 0.15, {
            'endpoint': '/api/v1/klines',
            'method': 'GET'
        })
        
        # 验证性能日志文件创建
        performance_log_file = os.path.join(self.log_dir, "performance.json")
        
        # 等待性能日志写入
        time.sleep(0.1)
        
        # 手动刷新性能处理器
        if logger_manager.performance_handler:
            logger_manager.performance_handler._flush_metrics()
        
        # 验证性能日志内容
        if os.path.exists(performance_log_file):
            with open(performance_log_file, 'r') as f:
                content = f.read()
                if content.strip():
                    log_data = json.loads(content.strip().split('\n')[0])
                    assert 'api_latency' in log_data['metric']
                    assert log_data['data']['value'] == 0.15
    
    def test_json_formatter(self):
        """测试JSON格式化器"""
        formatter = JsonFormatter()
        
        # 创建测试日志记录
        logger = logging.getLogger('test_json')
        record = logging.LogRecord(
            name='test_json',
            level=logging.INFO,
            pathname='test.py',
            lineno=100,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # 添加额外数据
        record.extra_data = {'test_key': 'test_value'}
        
        # 格式化日志
        formatted = formatter.format(record)
        
        # 验证JSON格式
        log_data = json.loads(formatted)
        assert log_data['message'] == 'Test message'
        assert log_data['level'] == 'INFO'
        assert log_data['extra']['test_key'] == 'test_value'
    
    def test_telegram_handler(self):
        """测试Telegram处理器"""
        # 创建Telegram处理器
        handler = TelegramLogHandler('test_token', 123456)
        
        # 测试速率限制
        assert handler._check_rate_limit() == True
        
        # 测试消息格式化
        record = logging.LogRecord(
            name='test_telegram',
            level=logging.ERROR,
            pathname='test.py',
            lineno=100,
            msg='Test error message',
            args=(),
            exc_info=None
        )
        
        formatted = handler._format_message(record)
        
        # 验证格式化消息
        assert '❌' in formatted  # 错误级别表情
        assert 'ERROR' in formatted
        assert 'Test error message' in formatted
    
    def test_performance_monitor_decorator(self):
        """测试性能监控装饰器"""
        logger_manager = LoggerManager(self.test_config)
        
        @performance_monitor
        def test_function(x, y):
            time.sleep(0.01)  # 模拟处理时间
            return x + y
        
        # 执行被装饰的函数
        result = test_function(1, 2)
        
        # 验证函数执行结果
        assert result == 3
        
        # 验证性能日志记录
        performance_log_file = os.path.join(self.log_dir, "performance.json")
        
        # 等待性能日志写入
        time.sleep(0.1)
        
        # 手动刷新性能处理器
        if logger_manager.performance_handler:
            logger_manager.performance_handler._flush_metrics()
    
    def test_log_levels_integration(self):
        """测试日志级别集成"""
        logger_manager = LoggerManager(self.test_config)
        logger = logger_manager.get_logger('test_levels')
        
        # 测试不同级别的日志
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')
        
        # 验证系统日志文件
        system_log_file = os.path.join(self.log_dir, "system.log")
        assert os.path.exists(system_log_file)
        
        # 验证错误日志文件
        error_log_file = os.path.join(self.log_dir, "errors.log")
        assert os.path.exists(error_log_file)

class TestIntegration:
    """集成测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        
        # 重置全局实例
        ConfigManager._instance = None
        import utils.logger
        utils.logger.logger_manager = None
    
    def teardown_method(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # 清理全局实例
        ConfigManager._instance = None
        import utils.logger
        if utils.logger.logger_manager:
            utils.logger.logger_manager.close()
            utils.logger.logger_manager = None
    
    def test_config_logger_integration(self):
        """测试配置管理器和日志系统集成"""
        # 创建配置文件
        test_config = {
            "monitoring": {
                "log_level": "DEBUG",
                "log_dir": self.log_dir,
                "telegram_enabled": True
            },
            "trading": {
                "macd": {
                    "fast": 13,
                    "slow": 34,
                    "signal": 9
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        # 初始化配置管理器
        config_manager = ConfigManager(self.config_file)
        
        # 使用配置初始化日志系统
        logger_config = {
            'log_level': config_manager.get('monitoring.log_level'),
            'log_dir': config_manager.get('monitoring.log_dir'),
            'telegram_enabled': config_manager.get('monitoring.telegram_enabled')
        }
        
        logger_manager = LoggerManager(logger_config)
        
        # 验证集成
        assert logger_manager is not None
        assert os.path.exists(self.log_dir)
        
        # 测试日志记录
        logger = logger_manager.get_logger('integration_test')
        logger.info('Integration test message')
        
        # 验证日志文件
        system_log_file = os.path.join(self.log_dir, "system.log")
        assert os.path.exists(system_log_file)
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 配置管理器初始化
        config_manager = ConfigManager(self.config_file)
        
        # 2. 日志系统初始化
        logger_config = {
            'log_level': 'INFO',
            'log_dir': self.log_dir
        }
        logger_manager = LoggerManager(logger_config)
        
        # 3. 获取配置
        macd_config = config_manager.get('trading.macd.fast', 13)
        
        # 4. 记录交易信号
        signal_data = {
            'macd_fast': macd_config,
            'price': 45000,
            'confidence': 0.85
        }
        
        logger_manager.log_trade_signal('BTCUSDT', 'LONG', signal_data)
        
        # 5. 记录性能指标
        logger_manager.log_system_performance('strategy_execution', 0.05, {
            'symbol': 'BTCUSDT',
            'strategy': 'macd_divergence'
        })
        
        # 6. 验证结果
        trade_log_file = os.path.join(self.log_dir, "trades.log")
        assert os.path.exists(trade_log_file)
        
        # 验证交易日志内容
        with open(trade_log_file, 'r') as f:
            log_content = f.read()
            assert 'BTCUSDT' in log_content
            assert 'LONG' in log_content

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"]) 