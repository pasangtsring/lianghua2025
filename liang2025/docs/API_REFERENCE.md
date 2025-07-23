# 📡 API参考文档

## 🎯 目录

- [概述](#概述)
- [核心模块API](#核心模块api)
- [数据获取API](#数据获取api)
- [风险管理API](#风险管理api)
- [监控系统API](#监控系统api)
- [配置管理API](#配置管理api)
- [工具类API](#工具类api)
- [错误码参考](#错误码参考)

## 📖 概述

本文档提供了交易系统所有主要模块的API参考，包括类、方法、参数和返回值的详细说明。

### 🔧 导入说明

```python
# 核心模块
from core.signal_generator import SignalGenerator
from core.enhanced_pattern_detector import EnhancedPatternDetector
from core.macd_divergence_detector import MACDDivergenceDetector

# 数据模块
from data.api_client import APIClientManager
from data.advanced_data_fetcher import AdvancedDataFetcher

# 风险管理
from risk.risk_manager import RiskManager
from risk.position_manager import PositionManager

# 工具模块
from utils.resource_monitor import ResourceMonitor
from utils.logger import Logger
from config.config_manager import ConfigManager
```

### 🎯 通用约定

- **异步方法**: 使用 `async/await` 模式
- **返回值**: 大多数方法返回数据类或字典
- **错误处理**: 使用标准Python异常
- **类型提示**: 使用Python类型注解

## 🧠 核心模块API

### SignalGenerator

信号生成器，负责生成交易信号。

#### 类定义
```python
class SignalGenerator:
    def __init__(self, config: ConfigManager)
```

#### 主要方法

##### `generate_signal()`
```python
async def generate_signal(self, symbol: str, timeframe: str = "1h") -> Optional[TradingSignal]:
    """
    生成交易信号
    
    Args:
        symbol: 交易品种符号，如 "BTCUSDT"
        timeframe: 时间周期，如 "1h", "4h", "1d"
    
    Returns:
        TradingSignal: 交易信号对象，包含以下属性：
            - signal_type: 信号类型 ("BUY", "SELL", "HOLD")
            - confidence: 置信度 (0.0-1.0)
            - entry_price: 入场价格
            - stop_loss: 止损价格
            - take_profit: 止盈价格
            - timestamp: 信号时间戳
            - metadata: 额外信息
    
    Raises:
        ValueError: 当symbol格式不正确时
        ConnectionError: 当无法获取数据时
    """
```

**使用示例**:
```python
config = ConfigManager()
signal_gen = SignalGenerator(config)

# 生成信号
signal = await signal_gen.generate_signal("BTCUSDT", "1h")

if signal:
    print(f"信号类型: {signal.signal_type}")
    print(f"置信度: {signal.confidence:.2f}")
    print(f"入场价: {signal.entry_price}")
    print(f"止损价: {signal.stop_loss}")
    print(f"止盈价: {signal.take_profit}")
```

##### `analyze_macd_divergence()`
```python
def analyze_macd_divergence(self, kline_data: List[Dict]) -> Dict:
    """
    分析MACD背离
    
    Args:
        kline_data: K线数据列表，每个元素包含：
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - timestamp: 时间戳
    
    Returns:
        Dict: 背离分析结果，包含：
            - has_divergence: 是否存在背离
            - divergence_type: 背离类型 ("bullish", "bearish")
            - confidence: 置信度
            - strength: 背离强度
            - score: 综合评分
    """
```

##### `get_signal_history()`
```python
def get_signal_history(self, symbol: str, limit: int = 100) -> List[TradingSignal]:
    """
    获取历史信号
    
    Args:
        symbol: 交易品种
        limit: 返回数量限制
    
    Returns:
        List[TradingSignal]: 历史信号列表
    """
```

### EnhancedPatternDetector

增强形态检测器，专门用于检测技术分析形态。

#### 类定义
```python
class EnhancedPatternDetector:
    def __init__(self, config_manager: ConfigManager)
```

#### 主要方法

##### `detect_divergence()`
```python
def detect_divergence(self, highs: np.ndarray, lows: np.ndarray, 
                     closes: np.ndarray, vol_factor: float = 0.0) -> List[DivergenceSignal]:
    """
    检测MACD背离
    
    Args:
        highs: 最高价数组
        lows: 最低价数组
        closes: 收盘价数组
        vol_factor: 波动性因子，用于动态调整阈值
    
    Returns:
        List[DivergenceSignal]: 背离信号列表，每个信号包含：
            - type: 背离类型 (DivergenceType.BULLISH/BEARISH)
            - confidence: 置信度
            - strength: 背离强度
            - indices: 相关K线索引
            - macd_values: MACD值
            - price_values: 价格值
    """
```

##### `detect_pattern()`
```python
async def detect_pattern(self, df: pd.DataFrame, pattern_type: str) -> List[PatternSignal]:
    """
    检测特定形态
    
    Args:
        df: 包含OHLCV数据的DataFrame
        pattern_type: 形态类型 ("ENGULFING", "HEAD_SHOULDER", "CONVERGENCE_TRIANGLE")
    
    Returns:
        List[PatternSignal]: 形态信号列表
    """
```

### MACDDivergenceDetector

MACD背离检测器，专门用于检测和验证MACD背离。

#### 类定义
```python
class MACDDivergenceDetector:
    def __init__(self, config_manager: ConfigManager)
```

#### 主要方法

##### `detect_divergence_enhanced()`
```python
def detect_divergence_enhanced(self, df: pd.DataFrame, symbol: str) -> Optional[DivergencePattern]:
    """
    增强型背离检测
    
    Args:
        df: 包含OHLCV数据的DataFrame
        symbol: 交易品种
    
    Returns:
        Optional[DivergencePattern]: 背离模式对象，包含：
            - divergence_type: 背离类型
            - confidence: 置信度
            - price_trend: 价格趋势线
            - macd_trend: MACD趋势线
            - duration: 持续时间
            - risk_reward_ratio: 风险回报比
    """
```

## 📊 数据获取API

### APIClientManager

API客户端管理器，负责管理多个数据源的API连接。

#### 类定义
```python
class APIClientManager:
    def __init__(self, config: ConfigManager)
```

#### 主要方法

##### `get_klines()`
```python
async def get_klines(self, symbol: str, interval: str, 
                    limit: int = 1000, start_time: Optional[int] = None,
                    end_time: Optional[int] = None) -> APIResponse:
    """
    获取K线数据
    
    Args:
        symbol: 交易品种符号
        interval: 时间间隔 ("1m", "5m", "15m", "1h", "4h", "1d")
        limit: 数据条数限制
        start_time: 开始时间（毫秒时间戳）
        end_time: 结束时间（毫秒时间戳）
    
    Returns:
        APIResponse: API响应对象，包含：
            - success: 是否成功
            - data: K线数据列表
            - error_message: 错误信息（如果失败）
            - source: 数据源名称
            - cached: 是否来自缓存
    """
```

##### `get_ticker()`
```python
async def get_ticker(self, symbol: str) -> APIResponse:
    """
    获取24小时价格变动统计
    
    Args:
        symbol: 交易品种符号
    
    Returns:
        APIResponse: 包含ticker信息的响应
    """
```

##### `get_account_info()`
```python
async def get_account_info() -> APIResponse:
    """
    获取账户信息
    
    Returns:
        APIResponse: 包含账户余额、持仓等信息
    """
```

### AdvancedDataFetcher

高级数据获取器，提供增强的数据获取功能。

#### 类定义
```python
class AdvancedDataFetcher:
    def __init__(self, config: ConfigManager, api_client: APIClient)
```

#### 主要方法

##### `fetch_all_advanced_data()`
```python
async def fetch_all_advanced_data(self, symbol: str) -> Dict[str, Any]:
    """
    获取所有高级数据
    
    Args:
        symbol: 交易品种符号
    
    Returns:
        Dict[str, Any]: 包含以下数据的字典：
            - x_heat: 社交媒体热度数据
            - liquidity: 流动性数据
            - coingecko: CoinGecko增强数据
            - validation: 数据验证结果
    """
```

## 🛡️ 风险管理API

### RiskManager

风险管理器，负责交易前风险检查和持仓风险监控。

#### 类定义
```python
class RiskManager:
    def __init__(self, config_manager: ConfigManager)
```

#### 主要方法

##### `check_pre_trade_risk()`
```python
def check_pre_trade_risk(self, symbol: str, side: str, 
                        quantity: float, price: float) -> Tuple[bool, str]:
    """
    交易前风险检查
    
    Args:
        symbol: 交易品种
        side: 交易方向 ("BUY", "SELL")
        quantity: 交易数量
        price: 交易价格
    
    Returns:
        Tuple[bool, str]: (是否允许交易, 风险信息)
    """
```

##### `update_position_risk()`
```python
def update_position_risk(self, positions: Dict) -> None:
    """
    更新仓位风险
    
    Args:
        positions: 当前持仓信息字典
    """
```

##### `get_risk_metrics()`
```python
def get_risk_metrics(self) -> RiskMetrics:
    """
    获取风险指标
    
    Returns:
        RiskMetrics: 风险指标对象，包含：
            - current_drawdown: 当前回撤
            - max_drawdown: 最大回撤
            - daily_pnl: 每日盈亏
            - total_pnl: 总盈亏
            - var_value: VaR值
            - leverage: 杠杆比率
    """
```

### PositionManager

仓位管理器，负责仓位计算和管理。

#### 类定义
```python
class PositionManager:
    def __init__(self, config_manager: ConfigManager)
```

#### 主要方法

##### `calculate_position_size()`
```python
def calculate_position_size(self, symbol: str, entry_price: float, 
                          stop_loss_price: float, current_equity: float) -> float:
    """
    计算仓位大小
    
    Args:
        symbol: 交易品种
        entry_price: 入场价格
        stop_loss_price: 止损价格
        current_equity: 当前权益
    
    Returns:
        float: 建议的仓位大小
    """
```

## 📱 监控系统API

### ResourceMonitor

资源监控器，负责监控系统资源使用情况。

#### 类定义
```python
class ResourceMonitor:
    def __init__(self, config_manager: ConfigManager)
```

#### 主要方法

##### `start_monitoring()`
```python
async def start_monitoring(self) -> None:
    """
    启动资源监控
    
    开始监控系统资源使用情况，包括CPU、内存、磁盘等
    """
```

##### `stop_monitoring()`
```python
async def stop_monitoring(self) -> None:
    """
    停止资源监控
    """
```

##### `get_current_status()`
```python
def get_current_status(self) -> Dict[str, Any]:
    """
    获取当前系统状态
    
    Returns:
        Dict[str, Any]: 包含以下信息的字典：
            - status: 系统状态 ("normal", "warning", "critical")
            - trading_paused: 是否暂停交易
            - metrics: 资源使用指标
                - cpu_percent: CPU使用率
                - memory_percent: 内存使用率
                - disk_percent: 磁盘使用率
            - thresholds: 阈值设置
    """
```

##### `check_resources_sync()`
```python
async def check_resources_sync(self) -> bool:
    """
    同步检查资源状态
    
    Returns:
        bool: 是否允许继续交易
    """
```

## ⚙️ 配置管理API

### ConfigManager

配置管理器，负责管理系统配置。

#### 类定义
```python
class ConfigManager:
    def __init__(self, config_file: str = "config/config.json")
```

#### 主要方法

##### `get_trading_config()`
```python
def get_trading_config(self) -> Dict[str, Any]:
    """
    获取交易配置
    
    Returns:
        Dict[str, Any]: 交易配置字典，包含：
            - symbol: 交易品种
            - interval: 时间周期
            - risk_per_trade: 单笔风险比例
            - max_positions: 最大持仓数
            - leverage: 杠杆倍数
    """
```

##### `get_risk_config()`
```python
def get_risk_config(self) -> Dict[str, Any]:
    """
    获取风险配置
    
    Returns:
        Dict[str, Any]: 风险配置字典
    """
```

##### `get_macd_divergence_config()`
```python
def get_macd_divergence_config(self) -> Dict[str, Any]:
    """
    获取MACD背离配置
    
    Returns:
        Dict[str, Any]: MACD背离检测配置
    """
```

##### `update_config()`
```python
def update_config(self, config_path: str, new_value: Any) -> bool:
    """
    更新配置项
    
    Args:
        config_path: 配置路径，如 "trading.risk_per_trade"
        new_value: 新值
    
    Returns:
        bool: 是否更新成功
    """
```

##### `validate_config()`
```python
def validate_config(self) -> bool:
    """
    验证配置
    
    Returns:
        bool: 配置是否有效
    """
```

## 🔧 工具类API

### Logger

日志系统，提供结构化日志记录。

#### 使用方法
```python
from utils.logger import Logger

logger = Logger(__name__)

# 记录不同级别的日志
logger.info("系统启动")
logger.warning("检测到异常情况")
logger.error("发生错误", exc_info=True)
logger.debug("调试信息")

# 记录交易相关日志
logger.log_trade("BTCUSDT", "BUY", 0.1, 50000, "市价单")
logger.log_signal("BTCUSDT", "看涨背离", 0.75)
```

### TelegramBot

Telegram通知机器人。

#### 类定义
```python
class TelegramBot:
    def __init__(self, config: ConfigManager)
```

#### 主要方法

##### `send_message()`
```python
async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
    """
    发送消息
    
    Args:
        message: 消息内容
        parse_mode: 解析模式 ("HTML", "Markdown")
    
    Returns:
        bool: 是否发送成功
    """
```

##### `send_alert()`
```python
async def send_alert(self, alert_type: str, message: str, 
                    level: str = "INFO") -> bool:
    """
    发送告警
    
    Args:
        alert_type: 告警类型
        message: 告警消息
        level: 告警级别 ("INFO", "WARNING", "ERROR", "CRITICAL")
    
    Returns:
        bool: 是否发送成功
    """
```

## 📊 数据模型

### TradingSignal

交易信号数据模型。

```python
@dataclass
class TradingSignal:
    signal_type: str              # 信号类型 ("BUY", "SELL", "HOLD")
    symbol: str                   # 交易品种
    confidence: float             # 置信度 (0.0-1.0)
    entry_price: float            # 入场价格
    stop_loss: float              # 止损价格
    take_profit: float            # 止盈价格
    position_size: float          # 建议仓位大小
    timestamp: datetime           # 信号时间
    risk_reward_ratio: float      # 风险回报比
    metadata: Dict[str, Any]      # 额外信息
```

### DivergenceSignal

背离信号数据模型。

```python
@dataclass
class DivergenceSignal:
    type: DivergenceType          # 背离类型
    confidence: float             # 置信度
    strength: float               # 背离强度
    indices: List[int]            # 相关K线索引
    macd_values: List[float]      # MACD值
    price_values: List[float]     # 价格值
    timestamp: datetime           # 检测时间
```

### RiskMetrics

风险指标数据模型。

```python
@dataclass
class RiskMetrics:
    current_drawdown: float       # 当前回撤
    max_drawdown: float           # 最大回撤
    daily_pnl: float              # 每日盈亏
    total_pnl: float              # 总盈亏
    var_value: float              # VaR值
    position_risk: float          # 仓位风险
    leverage: float               # 杠杆比率
    margin_usage: float           # 保证金使用率
```

## 🚨 错误码参考

### 系统错误码

| 错误码 | 错误类型 | 描述 | 解决方案 |
|--------|----------|------|----------|
| 1001 | ConfigError | 配置文件错误 | 检查config.json格式 |
| 1002 | APIError | API连接错误 | 检查网络连接和API密钥 |
| 1003 | DataError | 数据获取错误 | 检查数据源可用性 |
| 1004 | RiskError | 风险检查失败 | 检查风险参数设置 |
| 1005 | ResourceError | 资源不足 | 释放系统资源 |

### 交易错误码

| 错误码 | 错误类型 | 描述 | 解决方案 |
|--------|----------|------|----------|
| 2001 | OrderError | 订单错误 | 检查订单参数 |
| 2002 | PositionError | 仓位错误 | 检查仓位状态 |
| 2003 | BalanceError | 余额不足 | 检查账户余额 |
| 2004 | LeverageError | 杠杆错误 | 调整杠杆设置 |
| 2005 | SymbolError | 品种错误 | 检查交易品种 |

### 使用示例

```python
from utils.exceptions import TradingSystemError

try:
    signal = await signal_gen.generate_signal("BTCUSDT")
except TradingSystemError as e:
    if e.error_code == 1002:
        logger.error("API连接错误，请检查网络")
    elif e.error_code == 2001:
        logger.error("订单错误，请检查参数")
    else:
        logger.error(f"未知错误: {e}")
```

## 📈 性能优化

### 异步最佳实践

```python
# 推荐：并行处理多个请求
async def fetch_multiple_data():
    tasks = [
        api_client.get_klines("BTCUSDT", "1h"),
        api_client.get_klines("ETHUSDT", "1h"),
        api_client.get_klines("BNBUSDT", "1h")
    ]
    results = await asyncio.gather(*tasks)
    return results

# 避免：串行处理
async def fetch_multiple_data_slow():
    result1 = await api_client.get_klines("BTCUSDT", "1h")
    result2 = await api_client.get_klines("ETHUSDT", "1h")
    result3 = await api_client.get_klines("BNBUSDT", "1h")
    return [result1, result2, result3]
```

### 缓存使用

```python
# 启用缓存以提高性能
api_client = APIClientManager(config)
api_client.enable_cache(ttl=300)  # 5分钟缓存

# 批量获取数据
data = await api_client.get_klines("BTCUSDT", "1h", limit=1000)
```

## 🔗 相关链接

- [用户使用指南](USER_GUIDE.md)
- [配置参数说明](CONFIGURATION.md)
- [架构设计文档](ARCHITECTURE.md)
- [故障排除指南](TROUBLESHOOTING.md)

---

📝 **注意**: 本文档持续更新中，如有疑问请参考源代码或联系技术支持。 