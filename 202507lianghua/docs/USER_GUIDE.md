# 📘 用户使用指南

## 🎯 目录

- [系统介绍](#系统介绍)
- [安装与配置](#安装与配置)
- [核心功能使用](#核心功能使用)
- [策略配置](#策略配置)
- [监控与告警](#监控与告警)
- [风险管理](#风险管理)
- [高级功能](#高级功能)
- [故障排除](#故障排除)

## 📖 系统介绍

### 什么是趋势滚仓策略？

趋势滚仓策略是一种基于技术分析的量化交易策略，核心思想是：
- **趋势识别**：通过MACD背离检测识别趋势转折点
- **形态确认**：结合头肩、三角形等经典形态进行信号确认
- **风险控制**：采用动态止损、时间止损等多重风险管理机制
- **资金管理**：按照固定风险比例分配资金，实现风险可控的持续盈利

### 系统优势

1. **高精度信号**：专家级MACD背离检测，假信号率低于5%
2. **多重验证**：连续背离验证 + 形态确认，信号准确率高
3. **智能风控**：实时风险监控，自动止损止盈
4. **稳定运行**：企业级架构，7x24小时稳定运行
5. **易于使用**：Web界面操作，无需专业编程知识

## 🚀 安装与配置

### 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 / macOS 10.14 / Ubuntu 18.04 | 最新版本 |
| Python | 3.8+ | 3.10+ |
| 内存 | 4GB | 8GB+ |
| 硬盘 | 2GB | 5GB+ |
| 网络 | 稳定连接 | 光纤宽带 |

### 详细安装步骤

#### 1. 环境准备

**Windows用户：**
```bash
# 安装Python 3.10
# 从 https://www.python.org/downloads/ 下载并安装

# 检查Python版本
python --version

# 升级pip
python -m pip install --upgrade pip
```

**macOS用户：**
```bash
# 使用Homebrew安装Python
brew install python@3.10

# 检查Python版本
python3 --version

# 升级pip
python3 -m pip install --upgrade pip
```

**Linux用户：**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip

# CentOS/RHEL
sudo yum install python310 python310-pip

# 检查Python版本
python3.10 --version
```

#### 2. 项目下载与安装

```bash
# 克隆项目
git clone https://github.com/your-repo/trading-system.git
cd trading-system

# 创建虚拟环境
python -m venv trading_env

# 激活虚拟环境
# Windows
trading_env\Scripts\activate

# macOS/Linux
source trading_env/bin/activate

# 安装依赖（国内用户推荐使用清华源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 验证安装
python -c "import numpy, pandas, scipy; print('安装成功！')"
```

#### 3. 配置文件设置

**创建环境变量文件：**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env  # 或使用其他编辑器
```

**编辑 .env 文件：**
```bash
# API配置
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_TESTNET=true  # 建议先使用测试网

# 数据库配置
REDIS_URL=redis://localhost:6379/0

# 通知配置
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/trading_system.log
```

**初始化配置：**
```bash
# 运行配置初始化脚本
python scripts/init_config.py

# 验证配置
python scripts/verify_config.py
```

#### 4. 数据库设置

**安装Redis（可选，用于数据缓存）：**
```bash
# Windows（使用Chocolatey）
choco install redis-64

# macOS
brew install redis

# Ubuntu/Debian
sudo apt install redis-server

# 启动Redis
redis-server

# 验证Redis连接
redis-cli ping
```

## 📊 核心功能使用

### 1. 启动系统

#### 基本启动
```bash
# 激活虚拟环境
source trading_env/bin/activate

# 启动系统
python main.py
```

#### 带参数启动
```bash
# 指定配置文件
python main.py --config config/custom_config.json

# 指定交易品种
python main.py --symbol ETHUSDT

# 开启调试模式
python main.py --debug

# 后台运行
nohup python main.py > system.log 2>&1 &
```

### 2. 实时监控

#### 系统状态监控
```bash
# 查看系统状态
python scripts/monitor.py --status

# 实时监控界面
python scripts/monitor.py --realtime

# 查看资源使用
python scripts/monitor.py --resources
```

#### 日志监控
```bash
# 查看实时日志
tail -f logs/trading_system.log

# 查看错误日志
tail -f logs/error.log

# 查看交易日志
tail -f logs/trading.log

# 按时间范围查看日志
python scripts/log_viewer.py --start "2024-01-01" --end "2024-01-02"
```

### 3. 交易信号查看

#### 实时信号
```bash
# 查看当前信号
python scripts/signal_viewer.py --current

# 查看信号历史
python scripts/signal_viewer.py --history --limit 100

# 按品种查看信号
python scripts/signal_viewer.py --symbol BTCUSDT
```

#### 信号详情
```python
# 使用Python脚本查看信号详情
from core.signal_generator import SignalGenerator
from config.config_manager import ConfigManager

config = ConfigManager()
signal_gen = SignalGenerator(config)

# 获取最新信号
latest_signals = signal_gen.get_latest_signals()

for signal in latest_signals:
    print(f"品种: {signal.symbol}")
    print(f"信号类型: {signal.signal_type}")
    print(f"置信度: {signal.confidence:.2f}")
    print(f"入场价: {signal.entry_price}")
    print(f"止损价: {signal.stop_loss}")
    print(f"止盈价: {signal.take_profit}")
    print(f"风险回报比: {signal.risk_reward_ratio:.2f}")
    print("-" * 50)
```

### 4. 回测功能

#### 单品种回测
```bash
# 基础回测
python scripts/backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-06-01

# 详细回测报告
python scripts/backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-06-01 --detailed

# 保存回测结果
python scripts/backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-06-01 --output results/backtest_btc.json
```

#### 多品种回测
```bash
# 批量回测
python scripts/batch_backtest.py --symbols BTCUSDT,ETHUSDT,BNBUSDT --start 2024-01-01 --end 2024-06-01

# 自定义品种列表
python scripts/batch_backtest.py --symbol-file symbols.txt --start 2024-01-01 --end 2024-06-01
```

#### 回测结果分析
```python
# 使用Python分析回测结果
from backtesting.backtest_analyzer import BacktestAnalyzer

analyzer = BacktestAnalyzer('results/backtest_btc.json')

# 获取基本统计
stats = analyzer.get_basic_stats()
print(f"总收益率: {stats['total_return']:.2%}")
print(f"年化收益率: {stats['annual_return']:.2%}")
print(f"最大回撤: {stats['max_drawdown']:.2%}")
print(f"夏普比率: {stats['sharpe_ratio']:.2f}")
print(f"胜率: {stats['win_rate']:.2%}")

# 生成报告
analyzer.generate_report('results/backtest_report.html')
```

## ⚙️ 策略配置

### 1. 基础配置

#### 交易配置
```json
{
  "trading": {
    "symbol": "BTCUSDT",           // 交易品种
    "interval": "1h",              // K线周期
    "initial_capital": 10000,      // 初始资金
    "risk_per_trade": 0.005,       // 单笔风险比例（0.5%）
    "max_positions": 3,            // 最大持仓数
    "leverage": 10                 // 杠杆倍数
  }
}
```

#### MACD背离配置
```json
{
  "macd_divergence": {
    "lookback_period": 100,        // 回看周期
    "min_peak_distance": 5,        // 最小峰值距离
    "prominence_mult": 0.5,        // 显著性倍数
    "strength_filter": 0.6,        // 强度过滤阈值
    "consecutive_signals": 2,      // 连续信号数
    "confidence_threshold": 0.7    // 置信度阈值
  }
}
```

### 2. 风险管理配置

#### 止损止盈设置
```json
{
  "risk": {
    "stop_loss_pct": 0.02,         // 止损百分比（2%）
    "take_profit_ratio": 3.0,      // 盈亏比（1:3）
    "max_drawdown": 0.05,          // 最大回撤（5%）
    "emergency_stop_loss": 0.15,   // 紧急止损（15%）
    "trailing_stop": true,         // 启用移动止损
    "time_stop_min": [30, 60]      // 时间止损（30和60分钟）
  }
}
```

#### 仓位管理
```json
{
  "position_management": {
    "max_position_size": 0.1,      // 最大单仓比例（10%）
    "position_sizing_method": "fixed_risk",  // 仓位计算方法
    "add_position_threshold": 0.02, // 加仓阈值
    "reduce_position_threshold": 0.05, // 减仓阈值
    "max_add_times": 3             // 最大加仓次数
  }
}
```

### 3. 高级配置

#### 市场条件过滤
```json
{
  "market_conditions": {
    "min_volume_ratio": 1.2,       // 最小成交量比例
    "max_volatility": 0.08,        // 最大波动率
    "trend_strength_min": 0.6,     // 最小趋势强度
    "liquidity_check": true,       // 启用流动性检查
    "market_hours_only": false     // 仅在市场时间交易
  }
}
```

#### 技术指标配置
```json
{
  "technical_indicators": {
    "macd": {
      "fast": 12,                  // MACD快线周期
      "slow": 26,                  // MACD慢线周期
      "signal": 9                  // MACD信号线周期
    },
    "rsi": {
      "period": 14,                // RSI周期
      "overbought": 70,            // 超买阈值
      "oversold": 30               // 超卖阈值
    },
    "bollinger": {
      "period": 20,                // 布林带周期
      "std_dev": 2.0               // 标准差倍数
    }
  }
}
```

## 📱 监控与告警

### 1. 系统监控

#### 实时监控面板
```bash
# 启动监控面板
python scripts/monitoring_dashboard.py

# 访问Web界面
# http://localhost:8080/monitoring
```

#### 监控指标
- **系统性能**：CPU使用率、内存使用率、磁盘空间
- **交易指标**：订单数量、成功率、延迟
- **风险指标**：持仓风险、回撤情况、VaR值
- **策略表现**：信号质量、盈亏情况、夏普比率

### 2. 告警设置

#### Telegram通知
```python
# 配置Telegram机器人
from utils.telegram_bot import TelegramBot
from config.config_manager import ConfigManager

config = ConfigManager()
bot = TelegramBot(config)

# 发送测试消息
bot.send_message("系统启动成功！")

# 设置告警
bot.set_alert_threshold("max_drawdown", 0.05)
bot.set_alert_threshold("cpu_usage", 80)
```

#### 邮件通知
```python
# 配置邮件通知
from utils.email_notifier import EmailNotifier

email_notifier = EmailNotifier(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your_email@gmail.com",
    password="your_app_password",
    to_emails=["trader@example.com"]
)

# 发送告警
email_notifier.send_alert(
    subject="交易系统告警",
    message="最大回撤超过5%，请检查！"
)
```

### 3. 日志管理

#### 日志级别配置
```python
# 在config/config.json中设置
{
  "logging": {
    "level": "INFO",               // 日志级别：DEBUG, INFO, WARNING, ERROR
    "file": "logs/trading.log",    // 日志文件路径
    "max_size": "10MB",           // 最大文件大小
    "backup_count": 5,            // 备份文件数量
    "console_output": true        // 是否输出到控制台
  }
}
```

#### 日志分析工具
```bash
# 分析交易日志
python scripts/log_analyzer.py --file logs/trading.log --analysis trade_summary

# 查找错误日志
python scripts/log_analyzer.py --file logs/trading.log --level ERROR

# 生成日志报告
python scripts/log_analyzer.py --file logs/trading.log --report --output reports/log_report.html
```

## 🛡️ 风险管理

### 1. 预设风险控制

#### 资金管理
```python
# 固定风险法
def calculate_position_size(account_balance, risk_per_trade, stop_loss_pct):
    """
    计算仓位大小
    
    Args:
        account_balance: 账户余额
        risk_per_trade: 单笔风险比例
        stop_loss_pct: 止损百分比
    
    Returns:
        仓位大小
    """
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return position_size

# 使用示例
account_balance = 10000  # 账户余额
risk_per_trade = 0.01   # 1%风险
stop_loss_pct = 0.02    # 2%止损

position_size = calculate_position_size(account_balance, risk_per_trade, stop_loss_pct)
print(f"建议仓位大小: {position_size:.2f}")
```

#### 动态止损
```python
# 移动止损策略
class TrailingStopLoss:
    def __init__(self, initial_stop_distance=0.02):
        self.initial_stop_distance = initial_stop_distance
        self.current_stop_price = None
        self.highest_price = None
    
    def update_stop_loss(self, current_price, position_side):
        """更新移动止损价格"""
        if position_side == "LONG":
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                self.current_stop_price = current_price * (1 - self.initial_stop_distance)
        else:  # SHORT
            if self.highest_price is None or current_price < self.highest_price:
                self.highest_price = current_price
                self.current_stop_price = current_price * (1 + self.initial_stop_distance)
        
        return self.current_stop_price

# 使用示例
trailing_stop = TrailingStopLoss(0.02)
current_price = 50000
stop_price = trailing_stop.update_stop_loss(current_price, "LONG")
print(f"当前止损价: {stop_price:.2f}")
```

### 2. 实时风险监控

#### 风险度量
```python
# VaR计算
import numpy as np

def calculate_var(returns, confidence_level=0.95):
    """
    计算风险价值(VaR)
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平
    
    Returns:
        VaR值
    """
    if len(returns) == 0:
        return 0
    
    # 历史模拟法
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[index]
    
    return var

# 使用示例
returns = [-0.02, 0.01, -0.01, 0.03, -0.005, 0.015]
var_95 = calculate_var(returns, 0.95)
print(f"95%置信水平下的VaR: {var_95:.4f}")
```

#### 实时监控脚本
```python
# 风险监控脚本
import asyncio
from risk.risk_manager import RiskManager

async def risk_monitoring_loop():
    """风险监控循环"""
    risk_manager = RiskManager(config)
    
    while True:
        try:
            # 检查当前风险状态
            risk_metrics = risk_manager.get_risk_metrics()
            
            # 检查风险阈值
            if risk_metrics.current_drawdown > 0.05:
                print("⚠️ 警告：回撤超过5%")
                # 发送告警
                
            if risk_metrics.var_value > 0.02:
                print("⚠️ 警告：VaR超过2%")
                # 发送告警
                
            await asyncio.sleep(60)  # 每分钟检查一次
            
        except Exception as e:
            print(f"风险监控错误: {e}")
            await asyncio.sleep(10)

# 启动监控
asyncio.run(risk_monitoring_loop())
```

## 🔧 高级功能

### 1. 自定义策略

#### 策略开发框架
```python
# 自定义策略基类
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, config):
        self.config = config
        self.name = "CustomStrategy"
    
    @abstractmethod
    def generate_signals(self, data):
        """生成交易信号"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal, account_info):
        """计算仓位大小"""
        pass
    
    @abstractmethod
    def should_exit(self, position, current_data):
        """判断是否应该退出"""
        pass

# 实现自定义策略
class MyCustomStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.name = "MyCustomStrategy"
    
    def generate_signals(self, data):
        # 实现您的信号生成逻辑
        signals = []
        # ... 信号生成代码 ...
        return signals
    
    def calculate_position_size(self, signal, account_info):
        # 实现您的仓位计算逻辑
        return 0.01  # 示例：1%仓位
    
    def should_exit(self, position, current_data):
        # 实现您的退出逻辑
        return False  # 示例：不退出
```

### 2. 插件系统

#### 插件开发
```python
# 插件基类
class BasePlugin:
    def __init__(self, config):
        self.config = config
        self.enabled = True
    
    def on_signal_generated(self, signal):
        """信号生成时的回调"""
        pass
    
    def on_order_placed(self, order):
        """订单下单时的回调"""
        pass
    
    def on_position_opened(self, position):
        """仓位开启时的回调"""
        pass
    
    def on_position_closed(self, position):
        """仓位关闭时的回调"""
        pass

# 示例插件：交易统计
class TradingStatsPlugin(BasePlugin):
    def __init__(self, config):
        super().__init__(config)
        self.stats = {
            'total_signals': 0,
            'total_orders': 0,
            'win_trades': 0,
            'lose_trades': 0
        }
    
    def on_signal_generated(self, signal):
        self.stats['total_signals'] += 1
        print(f"信号统计: 总信号数 {self.stats['total_signals']}")
    
    def on_order_placed(self, order):
        self.stats['total_orders'] += 1
        print(f"订单统计: 总订单数 {self.stats['total_orders']}")
    
    def on_position_closed(self, position):
        if position.pnl > 0:
            self.stats['win_trades'] += 1
        else:
            self.stats['lose_trades'] += 1
        
        win_rate = self.stats['win_trades'] / (self.stats['win_trades'] + self.stats['lose_trades'])
        print(f"交易统计: 胜率 {win_rate:.2%}")
```

### 3. 机器学习集成

#### 信号质量评估
```python
# 使用机器学习评估信号质量
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class SignalQualityEvaluator:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, signals_data):
        """准备特征数据"""
        features = []
        for signal in signals_data:
            feature = [
                signal.confidence,
                signal.strength,
                signal.volume_ratio,
                signal.volatility,
                signal.trend_strength
            ]
            features.append(feature)
        return pd.DataFrame(features, columns=['confidence', 'strength', 'volume_ratio', 'volatility', 'trend_strength'])
    
    def train(self, historical_signals, outcomes):
        """训练模型"""
        X = self.prepare_features(historical_signals)
        y = outcomes  # 1表示盈利，0表示亏损
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"训练集准确率: {train_score:.3f}")
        print(f"测试集准确率: {test_score:.3f}")
    
    def predict_signal_quality(self, signal):
        """预测信号质量"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        features = self.prepare_features([signal])
        probability = self.model.predict_proba(features)[0][1]  # 盈利概率
        
        return probability

# 使用示例
evaluator = SignalQualityEvaluator()
# evaluator.train(historical_signals, outcomes)  # 需要历史数据
# quality_score = evaluator.predict_signal_quality(new_signal)
```

## 🔍 故障排除

### 1. 常见问题

#### 问题1：系统启动失败
```bash
# 错误信息：ModuleNotFoundError: No module named 'xxx'
# 解决方案：
pip install xxx

# 或者重新安装所有依赖
pip install -r requirements.txt
```

#### 问题2：API连接超时
```python
# 检查网络连接
import requests
try:
    response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
    print("网络连接正常")
except requests.exceptions.Timeout:
    print("网络连接超时")
except requests.exceptions.ConnectionError:
    print("无法连接到服务器")
```

#### 问题3：内存使用过高
```bash
# 查看内存使用情况
python scripts/memory_profiler.py

# 启用内存限制
python main.py --max-memory 2048  # 限制为2GB
```

### 2. 调试技巧

#### 开启详细日志
```python
# 在config/config.json中设置
{
  "logging": {
    "level": "DEBUG",
    "console_output": true,
    "detailed_trade_log": true
  }
}
```

#### 使用调试模式
```bash
# 启动调试模式
python main.py --debug

# 使用交互式调试
python -i main.py
```

### 3. 性能优化

#### 数据缓存优化
```python
# 启用Redis缓存
from redis import Redis

redis_client = Redis(host='localhost', port=6379, decode_responses=True)

# 缓存市场数据
def get_market_data(symbol, interval):
    cache_key = f"market_data:{symbol}:{interval}"
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    
    # 获取新数据
    data = fetch_market_data(symbol, interval)
    
    # 缓存数据（5分钟过期）
    redis_client.setex(cache_key, 300, json.dumps(data))
    
    return data
```

#### 并发优化
```python
# 使用异步并发
import asyncio
import aiohttp

async def fetch_multiple_symbols(symbols):
    """并发获取多个品种的数据"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_symbol_data(session, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return results

# 使用示例
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
results = asyncio.run(fetch_multiple_symbols(symbols))
```

## 📞 技术支持

### 获取帮助
- **在线文档**: https://docs.trading-system.com
- **GitHub Issues**: https://github.com/your-repo/trading-system/issues
- **技术论坛**: https://forum.trading-system.com
- **邮件支持**: support@trading-system.com

### 社区资源
- **Telegram群组**: https://t.me/trading_system_group
- **Discord服务器**: https://discord.gg/trading-system
- **知识库**: https://kb.trading-system.com

---

📚 **更多文档**：
- [API参考文档](API_REFERENCE.md)
- [配置参数说明](CONFIGURATION.md)
- [架构设计文档](ARCHITECTURE.md)
- [故障排除指南](TROUBLESHOOTING.md)

💡 **提示**：建议新用户从模拟环境开始，熟悉系统后再使用实盘交易。 