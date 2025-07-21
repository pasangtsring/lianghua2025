#!/bin/bash
# 系统启动脚本

echo "🚀 启动量化交易系统..."
echo "=" * 50

# 设置Python环境
export PYTHONPATH=/Users/xiaoyang/PycharmProjects/pythonProject1/202507lianghua:$PYTHONPATH
PYTHON_CMD="/Users/xiaoyang/miniconda3/bin/python"

# 检查Python环境
echo "🔍 检查Python环境..."
$PYTHON_CMD -c "import sys; print(f'Python版本: {sys.version}')"
$PYTHON_CMD -c "import pydantic; print(f'Pydantic版本: {pydantic.__version__}')"

# 检查核心模块
echo "🧠 检查核心模块..."
$PYTHON_CMD -c "from config.config_manager import ConfigManager; print('✅ 配置管理器正常')"

# 提供选项
echo ""
echo "请选择运行模式："
echo "1. 🧪 测试网模式 (推荐)"
echo "2. 🏭 生产环境模式 (谨慎使用)"
echo "3. 🔍 系统验证"
echo "4. ⚙️ 配置测试网"
echo "5. 🏭 配置生产环境"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "🧪 启动测试网模式..."
        $PYTHON_CMD scripts/setup_testnet.py
        ;;
    2)
        echo "🏭 启动生产环境模式..."
        $PYTHON_CMD scripts/setup_production.py
        ;;
    3)
        echo "🔍 运行系统验证..."
        $PYTHON_CMD scripts/verify_system.py
        ;;
    4)
        echo "⚙️ 配置测试网..."
        $PYTHON_CMD scripts/setup_testnet.py
        ;;
    5)
        echo "🏭 配置生产环境..."
        $PYTHON_CMD scripts/setup_production.py
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo "✅ 操作完成" 