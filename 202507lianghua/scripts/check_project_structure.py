#!/usr/bin/env python3
"""
项目结构检查脚本
验证项目架构的完整性
"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """检查项目结构"""
    
    print("🔍 项目结构检查报告")
    print("=" * 80)
    
    # 定义预期的项目结构
    expected_structure = {
        'config/': [
            '__init__.py',
            'config_manager.py',
            'settings.py',
            'config.json'
        ],
        'core/': [
            '__init__.py',
            'data_fetcher.py',
            'technical_indicators.py',
            'macd_divergence_detector.py',
            'complete_macd_divergence_detector.py',
            'pattern_recognizer.py',
            'cycle_analyzer.py',
            'signal_generator.py'
        ],
        'risk/': [
            '__init__.py',
            'risk_manager.py',
            'position_manager.py',
            'var_calculator.py'
        ],
        'execution/': [
            '__init__.py',
            'order_executor.py',
            'position_tracker.py',
            'emergency_handler.py'
        ],
        'data/': [
            '__init__.py',
            'advanced_data_fetcher.py',
            'api_client.py',
            'websocket_handler.py',
            'data_validator.py'
        ],
        'utils/': [
            '__init__.py',
            'logger.py',
            'telegram_bot.py',
            'redis_client.py',
            'performance_monitor.py'
        ],
        'backtesting/': [
            '__init__.py',
            'backtest_engine.py',
            'performance_analyzer.py',
            'report_generator.py'
        ],
        'monitoring/': [
            '__init__.py',
            'dashboard.py',
            'metrics_collector.py',
            'alert_system.py'
        ],
        'tests/': [
            '__init__.py',
            'unit/',
            'integration/',
            'performance/'
        ],
        'root': [
            'trading_engine.py',
            'main.py',
            'requirements.txt',
            'config.json'
        ]
    }
    
    # 检查结果
    total_files = 0
    existing_files = 0
    missing_files = []
    
    for directory, files in expected_structure.items():
        if directory == 'root':
            base_path = Path('.')
            print(f"\n📁 根目录:")
        else:
            base_path = Path(directory)
            print(f"\n📁 {directory}")
        
        for file in files:
            total_files += 1
            file_path = base_path / file
            
            if file_path.exists():
                existing_files += 1
                print(f"  ✅ {file}")
            else:
                missing_files.append(str(file_path))
                print(f"  ❌ {file} (缺失)")
    
    # 总结
    print(f"\n📊 总结:")
    print(f"  预期文件: {total_files}")
    print(f"  已存在: {existing_files}")
    print(f"  缺失: {len(missing_files)}")
    print(f"  完成度: {existing_files/total_files*100:.1f}%")
    
    if missing_files:
        print(f"\n❌ 缺失文件列表:")
        for file in missing_files:
            print(f"  - {file}")
    
    # 检查关键文件
    print(f"\n🔑 关键文件检查:")
    key_files = [
        'main.py',
        'trading_engine.py',
        'config/config_manager.py',
        'core/technical_indicators.py',
        'core/complete_macd_divergence_detector.py',
        'data/advanced_data_fetcher.py',
        'risk/risk_manager.py',
        'utils/logger.py'
    ]
    
    for file in key_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    return existing_files, total_files

if __name__ == "__main__":
    existing, total = check_project_structure()
    
    if existing == total:
        print("\n🎉 项目架构完整！")
        sys.exit(0)
    else:
        print(f"\n⚠️  项目架构不完整，完成度: {existing/total*100:.1f}%")
        sys.exit(1) 