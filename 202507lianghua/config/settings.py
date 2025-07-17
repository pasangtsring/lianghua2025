"""
默认配置文件
"""

# 交易配置
TRADING_CONFIG = {
    'symbol': 'BTCUSDT',
    'interval': '1h',
    'initial_capital': 10000,
    'max_position_size': 0.5,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'risk_per_trade': 0.02
}

# MACD配置
MACD_CONFIG = {
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9,
    'lookback_period': 50,
    'min_distance': 5,
    'prominence_multiplier': 0.5,
    'min_gap': 0.1,
    'time_tolerance': 2,
    'min_consecutive_divergences': 2,
    'max_consecutive_divergences': 3
}

# 技术指标配置
TECHNICAL_CONFIG = {
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std': 2,
    'atr_period': 14,
    'ema_periods': [5, 10, 20, 50, 200],
    'sma_periods': [20, 50, 200]
}

# 数据获取配置
DATA_CONFIG = {
    'binance_api_key': '',
    'binance_secret_key': '',
    'data_limit': 1000,
    'cache_ttl': 300,
    'retry_attempts': 3,
    'retry_delay': 1
}

# 风险管理配置
RISK_CONFIG = {
    'max_daily_loss': 0.05,
    'max_drawdown': 0.15,
    'max_positions': 5,
    'var_confidence': 0.95,
    'var_period': 252,
    'emergency_stop_loss': 0.1
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/trading_system.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Telegram配置
TELEGRAM_CONFIG = {
    'bot_token': '',
    'chat_id': '',
    'enabled': False,
    'alerts': {
        'trade_signals': True,
        'errors': True,
        'daily_summary': True
    }
}

# Redis配置
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': '',
    'socket_timeout': 5,
    'connection_pool_size': 10
}

# 性能监控配置
PERFORMANCE_CONFIG = {
    'metrics_interval': 60,
    'memory_threshold': 80,
    'cpu_threshold': 80,
    'disk_threshold': 90,
    'network_timeout': 30
}

# WebSocket配置
WEBSOCKET_CONFIG = {
    'reconnect_attempts': 5,
    'reconnect_delay': 5,
    'ping_interval': 20,
    'ping_timeout': 10,
    'max_message_size': 1024 * 1024  # 1MB
}

# 回测配置
BACKTEST_CONFIG = {
    'commission': 0.001,
    'slippage': 0.0005,
    'start_date': '2023-01-01',
    'end_date': '2024-01-01',
    'benchmark': 'BTC',
    'output_format': 'html'
}

# 监控配置
MONITORING_CONFIG = {
    'dashboard_port': 8080,
    'metrics_port': 8081,
    'update_interval': 1,
    'history_retention': 7 * 24 * 60  # 7 days in minutes
}

# 默认配置合并
DEFAULT_CONFIG = {
    'trading': TRADING_CONFIG,
    'macd': MACD_CONFIG,
    'technical': TECHNICAL_CONFIG,
    'data': DATA_CONFIG,
    'risk': RISK_CONFIG,
    'log': LOG_CONFIG,
    'telegram': TELEGRAM_CONFIG,
    'redis': REDIS_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'websocket': WEBSOCKET_CONFIG,
    'backtest': BACKTEST_CONFIG,
    'monitoring': MONITORING_CONFIG
} 