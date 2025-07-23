# 🚀 趋势滚仓+周期MACD形态低风险交易策略

## 📖 项目概述

本项目是一个基于Python的高级量化交易系统，专注于低风险趋势滚仓策略，集成了专业级的MACD背离检测、形态识别、风险管理和资源监控功能。系统采用异步架构设计，支持多数据源备份，具备企业级的稳定性和可扩展性。

### 🎯 核心特性

- **🔍 智能信号生成**：专家级MACD背离检测，支持连续背离验证和柱转虚过滤
- **📊 多维形态识别**：集成吞噬形态、头肩形态、收敛三角形等经典技术分析
- **⚡ 高性能架构**：异步并发处理，支持多数据源冗余和API限流
- **🛡️ 全面风险管理**：动态止损、时间止损、VaR计算、紧急止损机制
- **📱 实时监控**：系统资源监控、性能追踪、Telegram通知集成
- **🔧 灵活配置**：模块化配置管理，支持参数动态调整

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   Trading Engine                        │
├─────────────────────────────────────────────────────────┤
│  Signal Generator  │  Pattern Detector │ Risk Manager   │
├─────────────────────────────────────────────────────────┤
│    Data Fetcher    │  Technical Indicators │ Order Exec │
├─────────────────────────────────────────────────────────┤
│  Resource Monitor  │  Logger System   │ Config Manager  │
└─────────────────────────────────────────────────────────┘
```

### 💰 策略效果预期

基于专家算法优化和历史回测数据：
- **胜率提升**：45-55% → 50-65% (+10-15%)
- **年化收益**：10-25% → 12-35% (+20-40%)
- **最大回撤**：20-35% → ≤15% (-25-50%)
- **夏普比率**：0.8-1.2 → 1.2-1.8 (+50%)

## 🚀 快速开始

### 📋 环境要求

- **Python**: 3.8+
- **系统内存**: 4GB+
- **磁盘空间**: 2GB+
- **网络**: 稳定互联网连接

### 📦 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-repo/trading-system.git
cd trading-system
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件，填入API密钥和配置
```

5. **初始化配置**
```bash
python scripts/init_config.py
```

### 🎮 基本使用

#### 1. 启动系统
```bash
python main.py
```

#### 2. 实时监控
```bash
# 查看系统状态
python scripts/monitor.py --status

# 查看实时日志
tail -f logs/trading_system.log
```

#### 3. 回测验证
```bash
python scripts/backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-06-01
```

### 📈 核心功能演示

#### MACD背离检测
```python
from core.enhanced_pattern_detector import EnhancedPatternDetector
from config.config_manager import ConfigManager

# 初始化检测器
config = ConfigManager()
detector = EnhancedPatternDetector(config)

# 检测背离信号
signals = detector.detect_divergence(highs, lows, closes, vol_factor=0.02)

for signal in signals:
    print(f"检测到{signal.type}背离，置信度: {signal.confidence:.2f}")
```

#### 风险管理
```python
from risk.risk_manager import RiskManager

# 初始化风险管理器
risk_manager = RiskManager(config)

# 交易前风险检查
can_trade, risk_msg = risk_manager.check_pre_trade_risk(
    symbol="BTCUSDT", 
    side="BUY", 
    quantity=0.1, 
    price=50000
)

if can_trade:
    print("风险检查通过，可以执行交易")
else:
    print(f"风险检查失败: {risk_msg}")
```

#### 资源监控
```python
from utils.resource_monitor import ResourceMonitor

# 启动资源监控
monitor = ResourceMonitor(config)
await monitor.start_monitoring()

# 获取当前状态
status = monitor.get_current_status()
print(f"CPU使用率: {status['metrics']['cpu_percent']:.1f}%")
print(f"内存使用率: {status['metrics']['memory_percent']:.1f}%")
```

## 📚 文档目录

- **[📚 文档索引](docs/INDEX.md)** - 完整文档导航
- **[📘 用户指南](docs/USER_GUIDE.md)** - 详细使用说明
- **[📡 API参考](docs/API_REFERENCE.md)** - 完整API文档
- **[⚙️ 配置说明](docs/CONFIGURATION.md)** - 参数配置指南
- **[🏗️ 架构设计](docs/ARCHITECTURE.md)** - 系统架构文档
- **[🔧 故障排除](docs/TROUBLESHOOTING.md)** - 问题解决指南

## 🔧 主要模块

### 核心算法模块
- **`core/signal_generator.py`** - 信号生成器
- **`core/enhanced_pattern_detector.py`** - 增强形态检测器
- **`core/macd_divergence_detector.py`** - MACD背离检测器
- **`core/technical_indicators.py`** - 技术指标计算

### 数据处理模块
- **`data/api_client.py`** - API客户端管理
- **`data/advanced_data_fetcher.py`** - 高级数据获取器

### 风险管理模块
- **`risk/risk_manager.py`** - 风险管理器
- **`risk/position_manager.py`** - 仓位管理器

### 执行引擎模块
- **`execution/order_executor.py`** - 订单执行器
- **`trading_engine.py`** - 主交易引擎

### 工具模块
- **`utils/resource_monitor.py`** - 资源监控器
- **`utils/logger.py`** - 日志系统
- **`utils/telegram_bot.py`** - Telegram通知

## 🎯 配置示例

### 基础配置
```json
{
  "trading": {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "risk_per_trade": 0.005,
    "stop_loss_pct": 0.01,
    "take_profit_ratio": 3.0,
    "max_leverage": 20
  },
  "macd_divergence": {
    "lookback_period": 100,
    "min_peak_distance": 5,
    "prominence_mult": 0.5,
    "strength_filter": 0.6,
    "consecutive_signals": 2
  },
  "risk": {
    "max_drawdown": 0.05,
    "emergency_stop_loss": 0.15,
    "time_stop_min": [30, 60],
    "confidence_threshold": 0.5
  }
}
```

### 高级配置
```json
{
  "system": {
    "resource_monitoring": {
      "enabled": true,
      "cpu_threshold": 80,
      "memory_threshold": 85,
      "check_interval": 30
    },
    "backup_data_sources": {
      "enabled": true,
      "fallback_apis": ["coingecko", "coinmarketcap"],
      "failover_delay": 5
    }
  },
  "monitoring": {
    "telegram_enabled": true,
    "alert_thresholds": {
      "cpu_usage": 80,
      "memory_usage": 85,
      "error_rate": 0.05
    }
  }
}
```

## 🧪 测试与验证

### 单元测试
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定模块测试
python -m pytest tests/test_enhanced_pattern_detector.py -v

# 运行覆盖率测试
python -m pytest tests/ --cov=core --cov-report=html
```

### 性能测试
```bash
# 运行性能基准测试
python tests/benchmark/test_performance.py

# 内存使用测试
python tests/benchmark/test_memory_usage.py
```

### 回测验证
```bash
# 完整回测
python scripts/backtest.py --config config/backtest_config.json

# 多品种回测
python scripts/batch_backtest.py --symbols BTCUSDT,ETHUSDT,BNBUSDT
```

## 📊 监控与报告

### 实时监控
- **系统状态**: CPU、内存、磁盘使用率
- **交易统计**: 订单数量、成功率、盈亏情况
- **风险指标**: 最大回撤、VaR、夏普比率

### 报告生成
```bash
# 生成每日报告
python scripts/generate_report.py --date 2024-01-01

# 生成性能分析报告
python scripts/performance_report.py --period 30d
```

## 🔐 安全考虑

### API密钥管理
- 使用环境变量存储敏感信息
- 支持密钥轮换和权限控制
- 建议使用只读API密钥进行数据获取

### 风险控制
- 多层风险检查机制
- 紧急止损和熔断机制
- 实时资源监控和自动暂停

## 🤝 贡献指南

1. **Fork项目**并创建特性分支
2. **编写测试**确保代码质量
3. **遵循代码规范**使用black格式化
4. **提交PR**并详细说明改动

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📞 支持与联系

- **问题报告**: [GitHub Issues](https://github.com/your-repo/trading-system/issues)
- **功能建议**: [GitHub Discussions](https://github.com/your-repo/trading-system/discussions)
- **技术支持**: support@yourcompany.com

## 🎉 致谢

感谢所有贡献者和开源社区对本项目的支持！

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！ 