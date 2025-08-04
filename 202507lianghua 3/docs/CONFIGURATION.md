# ⚙️ 配置参数说明

## 📋 目录

- [概述](#概述)
- [配置文件结构](#配置文件结构)
- [核心交易配置](#核心交易配置)
- [MACD背离配置](#macd背离配置)
- [风险管理配置](#风险管理配置)
- [系统资源配置](#系统资源配置)
- [监控与告警配置](#监控与告警配置)
- [数据源配置](#数据源配置)
- [环境变量配置](#环境变量配置)
- [高级配置](#高级配置)
- [配置模板](#配置模板)

## 📖 概述

本文档详细说明了交易系统的所有配置参数，包括参数含义、取值范围、默认值和配置建议。

### 🎯 配置文件位置

- **主配置文件**: `config/config.json`
- **环境变量**: `.env`
- **用户配置**: `config/user_config.json`（可选）
- **回测配置**: `config/backtest_config.json`（可选）

### 📝 配置原则

1. **安全第一**: 敏感信息使用环境变量
2. **模块化**: 按功能模块组织配置
3. **可扩展**: 支持自定义配置项
4. **容错性**: 提供合理的默认值
5. **文档化**: 每个参数都有详细说明

## 🏗️ 配置文件结构

```json
{
  "api": {
    "binance": { ... },
    "coingecko": { ... }
  },
  "trading": {
    "intervals": { ... },
    "macd": { ... },
    "macd_divergence": { ... },
    "risk": { ... },
    "leverage_range": { ... }
  },
  "signal_generation": { ... },
  "market_conditions": { ... },
  "execution": { ... },
  "monitoring": { ... },
  "system": { ... }
}
```

## 💹 核心交易配置

### 基本交易参数

```json
{
  "trading": {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "initial_capital": 10000,
    "max_positions": 3,
    "base_currency": "USDT"
  }
}
```

#### 详细参数说明

| 参数 | 类型 | 默认值 | 描述 | 取值范围 |
|------|------|--------|------|----------|
| `symbol` | string | "BTCUSDT" | 主要交易品种 | 任何有效的交易对 |
| `interval` | string | "1h" | 主时间周期 | "1m", "5m", "15m", "1h", "4h", "1d" |
| `initial_capital` | number | 10000 | 初始资金 | > 0 |
| `max_positions` | number | 3 | 最大持仓数 | 1-10 |
| `base_currency` | string | "USDT" | 基础货币 | "USDT", "BTC", "ETH" |

### 多时间周期配置

```json
{
  "trading": {
    "intervals": {
      "small": "15m",
      "medium": "1h", 
      "large": "4h",
      "daily": "1d"
    }
  }
}
```

**配置说明**:
- `small`: 短周期，用于精确入场
- `medium`: 中周期，主要分析周期
- `large`: 长周期，确认趋势方向
- `daily`: 日周期，判断整体市场状态

### MACD基础配置

```json
{
  "trading": {
    "macd": {
      "fast": 12,
      "slow": 26,
      "signal": 9
    }
  }
}
```

**参数优化建议**:
- **快速市场**: fast=8, slow=21, signal=5
- **标准市场**: fast=12, slow=26, signal=9（默认）
- **慢速市场**: fast=19, slow=39, signal=9

### 移动平均线配置

```json
{
  "trading": {
    "ma_periods": [20, 50, 120, 200, 256]
  }
}
```

**应用场景**:
- `20`: 短期趋势判断
- `50`: 中期趋势支撑/阻力
- `120`: 长期趋势确认
- `200`: 牛熊分界线
- `256`: 超长期趋势

## 🔍 MACD背离配置

### 核心背离参数

```json
{
  "trading": {
    "macd_divergence": {
      "lookback_period": 100,
      "min_peak_distance": 5,
      "prominence_threshold": 0.1,
      "confirmation_candles": 3,
      "min_r_squared": 0.7,
      "min_trend_points": 2,
      "min_divergence_strength": 0.3,
      "min_duration_hours": 12,
      "max_duration_hours": 168,
      "dynamic_threshold": true,
      "min_significance": 0.02
    }
  }
}
```

#### 详细参数说明

| 参数 | 类型 | 默认值 | 描述 | 优化建议 |
|------|------|--------|------|----------|
| `lookback_period` | number | 100 | 回看周期数 | 波动市场: 50-80, 趋势市场: 100-150 |
| `min_peak_distance` | number | 5 | 最小峰值距离 | 快速市场: 3-4, 慢速市场: 6-8 |
| `prominence_threshold` | number | 0.1 | 显著性阈值 | 噪音多: 0.15-0.2, 信号少: 0.05-0.1 |
| `confirmation_candles` | number | 3 | 确认K线数 | 保守: 5-7, 激进: 1-2 |
| `min_r_squared` | number | 0.7 | 最小R²值 | 严格: 0.8-0.9, 宽松: 0.5-0.6 |
| `min_divergence_strength` | number | 0.3 | 最小背离强度 | 质量优先: 0.5-0.7, 数量优先: 0.2-0.3 |
| `min_duration_hours` | number | 12 | 最小持续时间 | 短线: 6-12, 中线: 24-48 |
| `max_duration_hours` | number | 168 | 最大持续时间 | 1周: 168, 1月: 720 |

### 连续背离配置

```json
{
  "trading": {
    "macd_divergence": {
      "continuous_detection": true,
      "consecutive_signals": 2,
      "prominence_mult": 0.5,
      "strength_filter": 0.6
    }
  }
}
```

**参数说明**:
- `continuous_detection`: 是否启用连续背离检测
- `consecutive_signals`: 连续信号数量要求
- `prominence_mult`: 显著性倍数
- `strength_filter`: 强度过滤阈值

## 🛡️ 风险管理配置

### 基础风控参数

```json
{
  "trading": {
    "risk": {
      "max_position_size": 0.1,
      "max_total_exposure": 0.3,
      "max_drawdown": 0.05,
      "loss_limit": 5,
      "emergency_stop_loss": 0.15,
      "risk_per_trade": 0.005,
      "stop_loss_pct": 0.02,
      "take_profit_ratio": 3.0,
      "max_leverage": 10,
      "confidence_threshold": 0.5
    }
  }
}
```

#### 详细风控参数

| 参数 | 类型 | 默认值 | 描述 | 推荐设置 |
|------|------|--------|------|----------|
| `max_position_size` | number | 0.1 | 最大单仓比例 | 保守: 0.05, 激进: 0.15 |
| `max_total_exposure` | number | 0.3 | 最大总持仓比例 | 保守: 0.2, 激进: 0.5 |
| `max_drawdown` | number | 0.05 | 最大回撤比例 | 新手: 0.03, 专业: 0.08 |
| `loss_limit` | number | 5 | 连续亏损限制 | 保守: 3, 激进: 8 |
| `emergency_stop_loss` | number | 0.15 | 紧急止损比例 | 固定: 0.1-0.2 |
| `risk_per_trade` | number | 0.005 | 单笔交易风险 | 保守: 0.002, 激进: 0.01 |
| `stop_loss_pct` | number | 0.02 | 止损百分比 | 波动小: 0.01, 波动大: 0.03 |
| `take_profit_ratio` | number | 3.0 | 盈亏比 | 最低: 2.0, 理想: 3.0-5.0 |
| `max_leverage` | number | 10 | 最大杠杆 | 新手: 3-5, 专业: 10-20 |
| `confidence_threshold` | number | 0.5 | 信号置信度阈值 | 保守: 0.7, 激进: 0.3 |

### 高级风控配置

```json
{
  "trading": {
    "risk": {
      "dynamic_stop_loss": true,
      "trailing_stop_loss": true,
      "partial_close_profit": 0.03,
      "add_profit_thresh": 0.02,
      "max_add_positions": 3,
      "time_stop_min": [30, 60],
      "funding_thresh": 0.00005,
      "emergency_atr_mult": 2.0
    }
  }
}
```

**高级参数说明**:
- `dynamic_stop_loss`: 动态止损，根据市场波动调整
- `trailing_stop_loss`: 移动止损，锁定盈利
- `partial_close_profit`: 部分平仓盈利阈值
- `add_profit_thresh`: 加仓盈利阈值
- `max_add_positions`: 最大加仓次数
- `time_stop_min`: 时间止损（分钟）
- `funding_thresh`: 资金费率阈值
- `emergency_atr_mult`: 紧急ATR倍数

### 杠杆管理配置

```json
{
  "trading": {
    "leverage_range": {
      "initial": [2, 5],
      "bull": [5, 15],
      "bear": [1, 3]
    }
  }
}
```

**市场条件杠杆策略**:
- `initial`: 初始杠杆范围
- `bull`: 牛市杠杆范围
- `bear`: 熊市杠杆范围

## 💻 系统资源配置

### 资源监控配置

```json
{
  "system": {
    "resource_monitoring": {
      "enabled": true,
      "cpu_threshold": 80,
      "memory_threshold": 85,
      "disk_threshold": 90,
      "pause_on_high_usage": true,
      "check_interval": 30,
      "alert_threshold": 75
    }
  }
}
```

#### 资源监控参数

| 参数 | 类型 | 默认值 | 描述 | 调整建议 |
|------|------|--------|------|----------|
| `enabled` | boolean | true | 是否启用资源监控 | 生产环境: true |
| `cpu_threshold` | number | 80 | CPU使用率阈值(%) | 低配机器: 70 |
| `memory_threshold` | number | 85 | 内存使用率阈值(%) | 内存少: 75 |
| `disk_threshold` | number | 90 | 磁盘使用率阈值(%) | 固定: 90 |
| `pause_on_high_usage` | boolean | true | 高使用率时暂停交易 | 安全: true |
| `check_interval` | number | 30 | 检查间隔（秒） | 频繁: 15, 节省: 60 |
| `alert_threshold` | number | 75 | 告警阈值(%) | 提前预警: 65 |

### 数据源备份配置

```json
{
  "system": {
    "backup_data_sources": {
      "enabled": true,
      "fallback_apis": ["coingecko", "coinmarketcap"],
      "failover_delay": 5,
      "max_retries": 3,
      "retry_delay": 2
    }
  }
}
```

## 📊 监控与告警配置

### 监控指标配置

```json
{
  "monitoring": {
    "metrics_interval": 60,
    "alert_thresholds": {
      "cpu_usage": 80,
      "memory_usage": 85,
      "error_rate": 0.05,
      "api_latency": 5.0,
      "consecutive_losses": 3,
      "daily_loss_limit": 0.05
    },
    "telegram_enabled": true,
    "log_level": "INFO"
  }
}
```

#### 告警阈值说明

| 指标 | 默认值 | 描述 | 建议设置 |
|------|--------|------|----------|
| `cpu_usage` | 80 | CPU使用率告警阈值 | 服务器: 85, 个人PC: 75 |
| `memory_usage` | 85 | 内存使用率告警阈值 | 内存充足: 90, 紧张: 80 |
| `error_rate` | 0.05 | 错误率告警阈值 | 严格: 0.02, 宽松: 0.1 |
| `api_latency` | 5.0 | API延迟告警阈值（秒） | 快速: 3.0, 容忍: 10.0 |
| `consecutive_losses` | 3 | 连续亏损告警阈值 | 保守: 2, 激进: 5 |
| `daily_loss_limit` | 0.05 | 每日亏损限制 | 严格: 0.03, 宽松: 0.08 |

### 性能监控配置

```json
{
  "monitoring": {
    "performance_tracking": {
      "enabled": true,
      "metrics": ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown"],
      "reporting_interval": 3600,
      "history_length": 10000
    }
  }
}
```

## 🔌 数据源配置

### API配置

```json
{
  "api": {
    "binance": {
      "base_url": "https://fapi.binance.com",
      "timeout": 30,
      "max_retries": 3,
      "rate_limit": 1200,
      "weight_limit": 6000
    },
    "coingecko": {
      "base_url": "https://api.coingecko.com/api/v3",
      "timeout": 30,
      "max_retries": 3,
      "rate_limit": 100
    }
  }
}
```

#### API参数说明

| 参数 | 描述 | 推荐值 |
|------|------|--------|
| `base_url` | API基础URL | 官方URL |
| `timeout` | 请求超时时间（秒） | 30 |
| `max_retries` | 最大重试次数 | 3 |
| `rate_limit` | 每分钟请求限制 | 按API限制 |
| `weight_limit` | 权重限制 | 按API限制 |

## 🔐 环境变量配置

### .env文件配置

```bash
# API密钥配置
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_TESTNET=true

# 数据库配置
REDIS_URL=redis://localhost:6379/0
MONGODB_URL=mongodb://localhost:27017/trading

# 通知配置
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/trading_system.log
LOG_MAX_SIZE=10485760
LOG_BACKUP_COUNT=5

# 安全配置
ENCRYPTION_KEY=your_encryption_key
JWT_SECRET=your_jwt_secret
```

### 环境变量说明

#### API配置
- `BINANCE_API_KEY`: 币安API密钥
- `BINANCE_API_SECRET`: 币安API密钥
- `BINANCE_TESTNET`: 是否使用测试网（建议新手使用）

#### 安全配置
- `ENCRYPTION_KEY`: 数据加密密钥
- `JWT_SECRET`: JWT令牌密钥

## 🎯 信号生成配置

### 信号过滤配置

```json
{
  "signal_generation": {
    "min_confidence": 0.6,
    "max_signals_per_hour": 5,
    "signal_expiry_minutes": 30,
    "duplicate_filter": true,
    "vol_factor_mult": 0.05,
    "market_hours_only": false
  }
}
```

### 市场条件配置

```json
{
  "market_conditions": {
    "volatility_filter": {
      "enabled": true,
      "min_volatility": 0.01,
      "max_volatility": 0.15,
      "calculation_period": 24
    },
    "volume_filter": {
      "enabled": true,
      "min_volume_ratio": 0.5,
      "volume_sma_period": 20
    },
    "trend_filter": {
      "enabled": true,
      "min_trend_strength": 0.3,
      "trend_lookback": 50
    }
  }
}
```

## 📈 执行配置

### 订单执行配置

```json
{
  "execution": {
    "order_type": "LIMIT",
    "price_offset": 0.001,
    "max_slippage": 0.005,
    "order_timeout": 300,
    "partial_fill_enabled": true,
    "post_only": false,
    "reduce_only": false
  }
}
```

## 🔧 高级配置

### 机器学习配置

```json
{
  "ml": {
    "enabled": false,
    "model_type": "lightgbm",
    "feature_engineering": {
      "enabled": true,
      "lookback_periods": [5, 10, 20, 50],
      "indicators": ["rsi", "macd", "bollinger", "atr"]
    },
    "training": {
      "retrain_interval": 168,
      "min_samples": 1000,
      "test_size": 0.2,
      "validation_size": 0.2
    }
  }
}
```

### 插件配置

```json
{
  "plugins": {
    "enabled": true,
    "plugin_dir": "plugins/",
    "load_plugins": ["trading_stats", "risk_monitor", "performance_tracker"],
    "plugin_config": {
      "trading_stats": {
        "enabled": true,
        "report_interval": 3600
      }
    }
  }
}
```

## 📋 配置模板

### 保守型配置

```json
{
  "trading": {
    "risk": {
      "max_position_size": 0.05,
      "max_total_exposure": 0.15,
      "max_drawdown": 0.03,
      "risk_per_trade": 0.002,
      "stop_loss_pct": 0.015,
      "take_profit_ratio": 4.0,
      "max_leverage": 3,
      "confidence_threshold": 0.7
    },
    "macd_divergence": {
      "min_divergence_strength": 0.5,
      "confirmation_candles": 5,
      "min_r_squared": 0.8
    }
  }
}
```

### 激进型配置

```json
{
  "trading": {
    "risk": {
      "max_position_size": 0.15,
      "max_total_exposure": 0.5,
      "max_drawdown": 0.08,
      "risk_per_trade": 0.01,
      "stop_loss_pct": 0.025,
      "take_profit_ratio": 2.5,
      "max_leverage": 20,
      "confidence_threshold": 0.4
    },
    "macd_divergence": {
      "min_divergence_strength": 0.2,
      "confirmation_candles": 1,
      "min_r_squared": 0.5
    }
  }
}
```

### 回测专用配置

```json
{
  "backtest": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000,
    "commission": 0.001,
    "slippage": 0.0005,
    "benchmark": "BTCUSDT",
    "rebalance_frequency": "daily",
    "output_format": "json"
  }
}
```

## 🔍 配置验证

### 验证脚本

```bash
# 验证配置文件
python scripts/validate_config.py

# 验证特定配置项
python scripts/validate_config.py --section trading

# 生成配置报告
python scripts/config_report.py --output config_report.html
```

### 配置检查清单

- [ ] API密钥配置正确
- [ ] 风险参数设置合理
- [ ] 监控阈值适当
- [ ] 数据库连接正常
- [ ] 通知服务配置
- [ ] 日志设置正确
- [ ] 资源监控启用

## 🔗 相关文档

- [用户使用指南](USER_GUIDE.md)
- [API参考文档](API_REFERENCE.md)
- [架构设计文档](ARCHITECTURE.md)
- [故障排除指南](TROUBLESHOOTING.md)

---

💡 **配置建议**: 建议新用户从保守型配置开始，随着经验积累逐步调整为更适合的参数。 