#!/usr/bin/env python3
"""
数据验证器测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from config.config_manager import ConfigManager
from data.data_validator import DataValidator, DataType, ValidationLevel, ValidationRule, ValidationResult

def test_data_validator():
    """测试数据验证器"""
    print("🧪 测试数据验证器模块")
    print("=" * 60)
    
    try:
        # 1. 初始化配置
        config = ConfigManager()
        print("✅ 配置管理器初始化成功")
        
        # 2. 初始化数据验证器
        validator = DataValidator(config)
        print("✅ 数据验证器初始化成功")
        
        # 3. 测试K线数据验证
        print("\n📊 测试K线数据验证：")
        
        # 有效的K线数据
        valid_kline = {
            "timestamp": 1642723200000,
            "open": "50000.00",
            "high": "51000.00",
            "low": "49000.00",
            "close": "50500.00",
            "volume": "100.5"
        }
        
        # 验证有效数据
        report = validator.validate_data(valid_kline, DataType.KLINE)
        print(f"✅ 有效K线数据验证完成")
        print(f"   成功率: {report.success_rate:.2%}")
        print(f"   问题数量: {len(report.issues)}")
        
        # 无效的K线数据
        invalid_kline = {
            "timestamp": "invalid_time",  # 无效时间戳
            "open": "-100.00",           # 负价格
            "high": "49000.00",          # 最高价小于其他价格
            "low": "52000.00",           # 最低价大于其他价格
            "close": "50500.00",
            "volume": "-10"              # 负成交量
        }
        
        # 验证无效数据
        report = validator.validate_data(invalid_kline, DataType.KLINE)
        print(f"✅ 无效K线数据验证完成")
        print(f"   成功率: {report.success_rate:.2%}")
        print(f"   问题数量: {len(report.issues)}")
        
        if report.issues:
            print("   发现的问题:")
            for issue in report.issues[:5]:  # 显示前5个问题
                print(f"      {issue.field}: {issue.message}")
        
        # 4. 测试Ticker数据验证
        print("\n📊 测试Ticker数据验证：")
        
        # 有效的Ticker数据
        valid_ticker = {
            "symbol": "BTCUSDT",
            "price": "50000.00",
            "change_percent": "2.5"
        }
        
        report = validator.validate_data(valid_ticker, DataType.TICKER)
        print(f"✅ 有效Ticker数据验证: 成功率 {report.success_rate:.2%}")
        
        # 无效的Ticker数据
        invalid_ticker = {
            "symbol": "invalid_symbol",   # 无效品种格式
            "price": "-100",             # 负价格
            "change_percent": "999"      # 异常涨跌幅
        }
        
        report = validator.validate_data(invalid_ticker, DataType.TICKER)
        print(f"✅ 无效Ticker数据验证: 成功率 {report.success_rate:.2%}, 问题数: {len(report.issues)}")
        
        # 5. 测试深度数据验证
        print("\n📊 测试深度数据验证：")
        
        # 有效的深度数据
        valid_depth = {
            "symbol": "BTCUSDT",
            "bids": [
                ["49900.00", "1.5"],
                ["49800.00", "2.0"],
                ["49700.00", "0.8"]
            ],
            "asks": [
                ["50100.00", "1.2"],
                ["50200.00", "1.8"],
                ["50300.00", "0.9"]
            ]
        }
        
        report = validator.validate_data(valid_depth, DataType.DEPTH)
        print(f"✅ 有效深度数据验证: 成功率 {report.success_rate:.2%}")
        
        # 无效的深度数据（买卖价格交叉）
        invalid_depth = {
            "symbol": "BTCUSDT",
            "bids": [
                ["50200.00", "1.5"],  # 买价高于卖价
            ],
            "asks": [
                ["50100.00", "1.2"],  # 卖价低于买价
            ]
        }
        
        report = validator.validate_data(invalid_depth, DataType.DEPTH)
        print(f"✅ 无效深度数据验证: 成功率 {report.success_rate:.2%}, 问题数: {len(report.issues)}")
        
        # 6. 测试交易数据验证
        print("\n📊 测试交易数据验证：")
        
        # 有效的交易数据
        valid_trade = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": "0.1",
            "price": "50000.00",
            "timestamp": 1642723200000
        }
        
        report = validator.validate_data(valid_trade, DataType.TRADE)
        print(f"✅ 有效交易数据验证: 成功率 {report.success_rate:.2%}")
        
        # 无效的交易数据
        invalid_trade = {
            "symbol": "BTCUSDT",
            "side": "invalid_side",      # 无效交易方向
            "quantity": "0",             # 零数量
            "price": "",                 # 空价格
            # 缺少timestamp
        }
        
        report = validator.validate_data(invalid_trade, DataType.TRADE)
        print(f"✅ 无效交易数据验证: 成功率 {report.success_rate:.2%}, 问题数: {len(report.issues)}")
        
        # 7. 测试账户数据验证
        print("\n📊 测试账户数据验证：")
        
        # 有效的账户数据
        valid_account = {
            "total_balance": "10000.00",
            "available_balance": "8000.00"
        }
        
        report = validator.validate_data(valid_account, DataType.ACCOUNT)
        print(f"✅ 有效账户数据验证: 成功率 {report.success_rate:.2%}")
        
        # 无效的账户数据（可用余额大于总余额）
        invalid_account = {
            "total_balance": "5000.00",
            "available_balance": "8000.00"  # 可用余额大于总余额
        }
        
        report = validator.validate_data(invalid_account, DataType.ACCOUNT)
        print(f"✅ 无效账户数据验证: 成功率 {report.success_rate:.2%}, 问题数: {len(report.issues)}")
        
        # 8. 测试批量数据验证
        print("\n📊 测试批量数据验证：")
        
        # 批量K线数据
        batch_klines = [
            {
                "timestamp": 1642723200000,
                "open": "50000.00",
                "high": "51000.00",
                "low": "49000.00",
                "close": "50500.00",
                "volume": "100.5"
            },
            {
                "timestamp": 1642726800000,  # 1小时后
                "open": "50500.00",
                "high": "52000.00",
                "low": "50000.00",
                "close": "51500.00",
                "volume": "150.2"
            },
            {
                "timestamp": 1642730400000,  # 再1小时后
                "open": "51500.00",
                "high": "52500.00",
                "low": "51000.00",
                "close": "52000.00",
                "volume": "200.8"
            }
        ]
        
        report = validator.validate_data(batch_klines, DataType.KLINE)
        print(f"✅ 批量K线数据验证: 处理 {report.total_records} 条记录")
        print(f"   成功率: {report.success_rate:.2%}")
        print(f"   执行时间: {report.execution_time:.4f}秒")
        
        # 9. 测试自定义验证规则
        print("\n📊 测试自定义验证规则：")
        
        # 添加自定义规则
        custom_rule = ValidationRule(
            field="custom_field",
            rule_type="numeric",
            parameters={"min_value": 100, "max_value": 1000},
            error_message="自定义字段必须在100-1000范围内"
        )
        
        validator.add_custom_rule(DataType.KLINE, custom_rule)
        print("✅ 自定义验证规则添加成功")
        
        # 测试包含自定义字段的数据
        custom_data = {
            "timestamp": 1642723200000,
            "open": "50000.00",
            "high": "51000.00",
            "low": "49000.00",
            "close": "50500.00",
            "volume": "100.5",
            "custom_field": "50"  # 小于最小值
        }
        
        report = validator.validate_data(custom_data, DataType.KLINE)
        print(f"✅ 自定义规则验证: 成功率 {report.success_rate:.2%}, 问题数: {len(report.issues)}")
        
        # 10. 测试验证规则查询
        print("\n📊 测试验证规则查询：")
        
        kline_rules = validator.get_validation_rules(DataType.KLINE)
        ticker_rules = validator.get_validation_rules(DataType.TICKER)
        
        print(f"✅ 验证规则查询成功")
        print(f"   K线验证规则数: {len(kline_rules)}")
        print(f"   Ticker验证规则数: {len(ticker_rules)}")
        
        # 11. 测试统计信息
        print("\n📊 测试统计信息：")
        
        stats = validator.get_stats()
        print("✅ 统计信息获取成功")
        print(f"   总验证次数: {stats['total_validations']}")
        print(f"   成功验证次数: {stats['successful_validations']}")
        print(f"   失败验证次数: {stats['failed_validations']}")
        print(f"   成功率: {stats['success_rate']:.2%}")
        print(f"   总处理记录数: {stats['total_records_processed']}")
        print(f"   总发现问题数: {stats['total_issues_found']}")
        print(f"   平均验证时间: {stats['average_validation_time']:.4f}秒")
        
        if stats['validation_rules_count']:
            print("   验证规则统计:")
            for data_type, count in stats['validation_rules_count'].items():
                print(f"      {data_type}: {count} 条规则")
        
        # 12. 测试状态查询
        print("\n📊 测试状态查询：")
        
        status = validator.get_status()
        print("✅ 状态查询成功")
        print(f"   总验证次数: {status['total_validations']}")
        print(f"   总处理记录数: {status['total_records_processed']}")
        print(f"   支持的数据类型: {', '.join(status['supported_data_types'])}")
        print(f"   缓存大小: {status['cache_size']}")
        print(f"   运行状态: {status['is_operational']}")
        
        # 13. 测试错误处理
        print("\n📊 测试错误处理：")
        
        # 测试空数据
        empty_report = validator.validate_data([], DataType.KLINE)
        print(f"✅ 空数据验证: 处理 {empty_report.total_records} 条记录")
        
        # 测试无效数据类型
        try:
            invalid_data = "这不是字典或列表"
            report = validator.validate_data(invalid_data, DataType.KLINE)
            print("✅ 无效数据类型处理正常")
        except Exception as e:
            print(f"❌ 无效数据类型处理失败: {e}")
        
        # 14. 测试性能验证
        print("\n📊 测试性能验证：")
        
        # 创建大量测试数据
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                "timestamp": 1642723200000 + i * 3600000,
                "open": f"{50000 + i}.00",
                "high": f"{51000 + i}.00",
                "low": f"{49000 + i}.00",
                "close": f"{50500 + i}.00",
                "volume": f"{100 + i}.5"
            })
        
        start_time = datetime.now()
        report = validator.validate_data(large_dataset, DataType.KLINE)
        end_time = datetime.now()
        
        performance_time = (end_time - start_time).total_seconds()
        records_per_second = report.total_records / performance_time if performance_time > 0 else 0
        
        print(f"✅ 性能验证完成")
        print(f"   处理记录数: {report.total_records}")
        print(f"   总耗时: {performance_time:.4f}秒")
        print(f"   处理速度: {records_per_second:.1f} 记录/秒")
        print(f"   成功率: {report.success_rate:.2%}")
        
        print("\n✅ 数据验证器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_validator()
    sys.exit(0 if success else 1) 