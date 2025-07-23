# 项目基础设施搭建指南

## 1. 环境准备

### 1.1 Python环境设置
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 升级pip
pip install --upgrade pip
```

### 1.2 项目目录创建
```bash
# 创建项目根目录
mkdir trading_system
cd trading_system

# 创建子目录结构
mkdir -p config core risk execution data utils backtesting monitoring tests/{unit,integration,performance}

# 创建__init__.py文件
touch config/__init__.py core/__init__.py risk/__init__.py execution/__init__.py
touch data/__init__.py utils/__init__.py backtesting/__init__.py monitoring/__init__.py
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/performance/__init__.py
```

### 1.3 依赖管理
```bash
# 安装核心依赖
pip install binance-futures-connector ccxt pandas numpy scipy ta-lib
pip install asyncio aiohttp websockets redis psutil
pip install pytest pytest-asyncio pytest-cov
pip install python-telegram-bot schedule
pip install pydantic python-dotenv
pip install matplotlib seaborn plotly

# 导出依赖列表
pip freeze > requirements.txt
```

## 2. 基础配置文件

### 2.1 .env文件模板
```env
# API配置
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_BASE_URL=https://fapi.binance.com

# 备份API
COINGECKO_API_KEY=your_coingecko_api_key

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Telegram配置
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# 系统配置
LOG_LEVEL=INFO
MAX_WORKERS=5
```

### 2.2 config.json模板
```json
{
  "api": {
    "binance": {
      "base_url": "https://fapi.binance.com",
      "timeout": 30,
      "max_retries": 3
    },
    "coingecko": {
      "base_url": "https://api.coingecko.com/api/v3",
      "timeout": 30
    }
  },
  "trading": {
    "intervals": {
      "small": "15m",
      "medium": "1h", 
      "large": "4h",
      "daily": "1d"
    },
    "macd": {
      "fast": 13,
      "slow": 34,
      "signal": 9
    },
    "ma_periods": [30, 50, 120, 200, 256],
    "risk": {
      "max_position_size": 0.01,
      "max_total_exposure": 0.10,
      "max_drawdown": 0.05,
      "loss_limit": 5
    }
  },
  "monitoring": {
    "metrics_interval": 60,
    "alert_thresholds": {
      "cpu_usage": 80,
      "memory_usage": 85,
      "error_rate": 0.05
    }
  }
}
```

## 3. Git设置

### 3.1 .gitignore文件
```gitignore
# 环境变量
.env
.env.local
.env.production

# Python缓存
__pycache__/
*.py[cod]
*$py.class
*.so

# 虚拟环境
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 日志文件
logs/
*.log

# 数据文件
data/
cache/
backups/

# 配置文件（包含敏感信息）
config/production.json
config/local.json
```

### 3.2 Git初始化
```bash
git init
git add .
git commit -m "Initial project structure"
```

## 4. 验证清单

- [ ] 虚拟环境成功创建并激活
- [ ] 所有依赖包正确安装
- [ ] 项目目录结构完整
- [ ] 配置文件模板已创建
- [ ] 环境变量文件已设置
- [ ] Git仓库初始化完成
- [ ] 基础测试环境可用

## 5. 下一步

完成基础设施搭建后，请更新任务状态并开始"配置管理模块"的开发。 