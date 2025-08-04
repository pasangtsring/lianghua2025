"""
数据验证器模块
负责数据完整性检查、格式验证、异常值检测
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from decimal import Decimal, InvalidOperation

from utils.logger import get_logger
from config.config_manager import ConfigManager

class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CUSTOM = "custom"

class DataType(Enum):
    """数据类型"""
    KLINE = "kline"
    TICKER = "ticker"
    DEPTH = "depth"
    TRADE = "trade"
    ACCOUNT = "account"
    ORDER = "order"
    POSITION = "position"

class ValidationResult(Enum):
    """验证结果"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationRule:
    """验证规则"""
    field: str
    rule_type: str
    parameters: Dict[str, Any]
    error_message: str
    level: ValidationLevel = ValidationLevel.STANDARD

@dataclass
class ValidationIssue:
    """验证问题"""
    field: str
    issue_type: str
    severity: ValidationResult
    message: str
    value: Any
    expected: Any = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ValidationReport:
    """验证报告"""
    data_type: DataType
    total_records: int
    valid_records: int
    issues: List[ValidationIssue]
    validation_time: datetime
    execution_time: float
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.valid_records / self.total_records if self.total_records > 0 else 0.0
    
    @property
    def has_errors(self) -> bool:
        """是否有错误"""
        return any(issue.severity in [ValidationResult.ERROR, ValidationResult.CRITICAL] for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """是否有警告"""
        return any(issue.severity == ValidationResult.WARNING for issue in self.issues)

class DataValidator:
    """数据验证器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # 验证规则
        self.validation_rules: Dict[DataType, List[ValidationRule]] = {}
        
        # 统计信息
        self.stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_records_processed': 0,
            'total_issues_found': 0,
            'validation_time_total': 0.0
        }
        
        # 缓存
        self.validation_cache: Dict[str, ValidationReport] = {}
        self.cache_ttl = 300  # 5分钟
        
        # 初始化默认规则
        self.setup_default_rules()
        
        self.logger.info("数据验证器初始化完成")
    
    def setup_default_rules(self):
        """设置默认验证规则"""
        try:
            # K线数据验证规则
            kline_rules = [
                ValidationRule("timestamp", "required", {}, "时间戳不能为空"),
                ValidationRule("timestamp", "type", {"expected_type": "int"}, "时间戳必须是整数"),
                ValidationRule("timestamp", "range", {"min_value": 1000000000000, "max_value": 9999999999999}, "时间戳格式错误"),
                ValidationRule("open", "required", {}, "开盘价不能为空"),
                ValidationRule("open", "numeric", {"min_value": 0}, "开盘价必须为正数"),
                ValidationRule("high", "required", {}, "最高价不能为空"),
                ValidationRule("high", "numeric", {"min_value": 0}, "最高价必须为正数"),
                ValidationRule("low", "required", {}, "最低价不能为空"),
                ValidationRule("low", "numeric", {"min_value": 0}, "最低价必须为正数"),
                ValidationRule("close", "required", {}, "收盘价不能为空"),
                ValidationRule("close", "numeric", {"min_value": 0}, "收盘价必须为正数"),
                ValidationRule("volume", "required", {}, "成交量不能为空"),
                ValidationRule("volume", "numeric", {"min_value": 0}, "成交量必须为非负数"),
                ValidationRule("", "price_logic", {}, "价格逻辑检查")
            ]
            self.validation_rules[DataType.KLINE] = kline_rules
            
            # Ticker数据验证规则
            ticker_rules = [
                ValidationRule("symbol", "required", {}, "交易品种不能为空"),
                ValidationRule("symbol", "format", {"pattern": r"^[A-Z]+USDT?$"}, "交易品种格式错误"),
                ValidationRule("price", "required", {}, "价格不能为空"),
                ValidationRule("price", "numeric", {"min_value": 0}, "价格必须为正数"),
                ValidationRule("change_percent", "numeric", {"min_value": -100, "max_value": 1000}, "涨跌幅范围异常")
            ]
            self.validation_rules[DataType.TICKER] = ticker_rules
            
            # 深度数据验证规则
            depth_rules = [
                ValidationRule("symbol", "required", {}, "交易品种不能为空"),
                ValidationRule("bids", "required", {}, "买单深度不能为空"),
                ValidationRule("asks", "required", {}, "卖单深度不能为空"),
                ValidationRule("bids", "array", {"min_length": 1}, "买单深度至少包含一条记录"),
                ValidationRule("asks", "array", {"min_length": 1}, "卖单深度至少包含一条记录"),
                ValidationRule("", "depth_logic", {}, "深度数据逻辑检查")
            ]
            self.validation_rules[DataType.DEPTH] = depth_rules
            
            # 交易数据验证规则
            trade_rules = [
                ValidationRule("symbol", "required", {}, "交易品种不能为空"),
                ValidationRule("side", "required", {}, "交易方向不能为空"),
                ValidationRule("side", "enum", {"allowed_values": ["buy", "sell"]}, "交易方向只能是buy或sell"),
                ValidationRule("quantity", "required", {}, "交易数量不能为空"),
                ValidationRule("quantity", "numeric", {"min_value": 0}, "交易数量必须为正数"),
                ValidationRule("price", "required", {}, "交易价格不能为空"),
                ValidationRule("price", "numeric", {"min_value": 0}, "交易价格必须为正数"),
                ValidationRule("timestamp", "required", {}, "交易时间不能为空")
            ]
            self.validation_rules[DataType.TRADE] = trade_rules
            
            # 账户数据验证规则
            account_rules = [
                ValidationRule("total_balance", "required", {}, "总余额不能为空"),
                ValidationRule("total_balance", "numeric", {"min_value": 0}, "总余额不能为负数"),
                ValidationRule("available_balance", "required", {}, "可用余额不能为空"),
                ValidationRule("available_balance", "numeric", {"min_value": 0}, "可用余额不能为负数"),
                ValidationRule("", "balance_logic", {}, "余额逻辑检查")
            ]
            self.validation_rules[DataType.ACCOUNT] = account_rules
            
            self.logger.info("默认验证规则设置完成")
            
        except Exception as e:
            self.logger.error(f"设置默认验证规则失败: {e}")
    
    def validate_data(self, data: Union[Dict, List[Dict]], data_type: DataType, 
                     validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        """
        验证数据
        
        Args:
            data: 待验证的数据
            data_type: 数据类型
            validation_level: 验证级别
            
        Returns:
            验证报告
        """
        start_time = datetime.now()
        execution_start = start_time.timestamp()
        
        try:
            # 标准化数据格式
            if isinstance(data, dict):
                data_list = [data]
            else:
                data_list = data
            
            total_records = len(data_list)
            valid_records = 0
            all_issues = []
            
            # 获取验证规则
            rules = self.validation_rules.get(data_type, [])
            
            # 验证每条记录
            for i, record in enumerate(data_list):
                record_issues = self.validate_single_record(record, rules, data_type, i)
                
                if not any(issue.severity in [ValidationResult.ERROR, ValidationResult.CRITICAL] for issue in record_issues):
                    valid_records += 1
                
                all_issues.extend(record_issues)
            
            # 计算执行时间
            execution_time = datetime.now().timestamp() - execution_start
            
            # 创建验证报告
            report = ValidationReport(
                data_type=data_type,
                total_records=total_records,
                valid_records=valid_records,
                issues=all_issues,
                validation_time=start_time,
                execution_time=execution_time
            )
            
            # 更新统计
            self.update_stats(report)
            
            # 记录日志
            if report.has_errors:
                self.logger.warning(f"数据验证发现错误: {data_type.value}, 成功率: {report.success_rate:.2%}")
            else:
                self.logger.info(f"数据验证完成: {data_type.value}, 成功率: {report.success_rate:.2%}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            execution_time = datetime.now().timestamp() - execution_start
            
            return ValidationReport(
                data_type=data_type,
                total_records=0,
                valid_records=0,
                issues=[ValidationIssue("system", "exception", ValidationResult.CRITICAL, str(e), None)],
                validation_time=start_time,
                execution_time=execution_time
            )
    
    def validate_single_record(self, record: Dict, rules: List[ValidationRule], 
                              data_type: DataType, record_index: int) -> List[ValidationIssue]:
        """
        验证单条记录
        
        Args:
            record: 记录数据
            rules: 验证规则
            data_type: 数据类型
            record_index: 记录索引
            
        Returns:
            验证问题列表
        """
        issues = []
        
        try:
            for rule in rules:
                if rule.field == "":
                    # 自定义逻辑验证
                    custom_issues = self.apply_custom_validation(record, rule, data_type, record_index)
                    issues.extend(custom_issues)
                else:
                    # 字段验证
                    field_issues = self.validate_field(record, rule, record_index)
                    issues.extend(field_issues)
            
            return issues
            
        except Exception as e:
            self.logger.error(f"验证单条记录失败: {e}")
            return [ValidationIssue("record", "validation_error", ValidationResult.ERROR, str(e), record)]
    
    def validate_field(self, record: Dict, rule: ValidationRule, record_index: int) -> List[ValidationIssue]:
        """
        验证字段
        
        Args:
            record: 记录数据
            rule: 验证规则
            record_index: 记录索引
            
        Returns:
            验证问题列表
        """
        issues = []
        field = rule.field
        
        try:
            # 必填检查
            if rule.rule_type == "required":
                if field not in record or record[field] is None or record[field] == "":
                    issues.append(ValidationIssue(
                        field, "required", ValidationResult.ERROR, 
                        f"记录{record_index}: {rule.error_message}", record.get(field)
                    ))
                    return issues
            
            # 如果字段不存在，跳过其他验证
            if field not in record:
                return issues
            
            value = record[field]
            
            # 类型检查
            if rule.rule_type == "type":
                expected_type = rule.parameters.get("expected_type")
                if expected_type == "int" and not isinstance(value, int):
                    try:
                        int(value)
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            field, "type", ValidationResult.ERROR,
                            f"记录{record_index}: {rule.error_message}", value, expected_type
                        ))
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            field, "type", ValidationResult.ERROR,
                            f"记录{record_index}: {rule.error_message}", value, expected_type
                        ))
            
            # 数值检查
            elif rule.rule_type == "numeric":
                try:
                    numeric_value = float(value)
                    
                    min_value = rule.parameters.get("min_value")
                    max_value = rule.parameters.get("max_value")
                    
                    if min_value is not None and numeric_value < min_value:
                        issues.append(ValidationIssue(
                            field, "min_value", ValidationResult.ERROR,
                            f"记录{record_index}: {field}值{numeric_value}小于最小值{min_value}", 
                            value, min_value
                        ))
                    
                    if max_value is not None and numeric_value > max_value:
                        issues.append(ValidationIssue(
                            field, "max_value", ValidationResult.WARNING,
                            f"记录{record_index}: {field}值{numeric_value}大于最大值{max_value}", 
                            value, max_value
                        ))
                
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        field, "numeric", ValidationResult.ERROR,
                        f"记录{record_index}: {field}不是有效数值", value
                    ))
            
            # 范围检查
            elif rule.rule_type == "range":
                try:
                    numeric_value = float(value)
                    min_value = rule.parameters.get("min_value")
                    max_value = rule.parameters.get("max_value")
                    
                    if min_value is not None and numeric_value < min_value:
                        issues.append(ValidationIssue(
                            field, "range", ValidationResult.ERROR,
                            f"记录{record_index}: {rule.error_message}", value, f"{min_value}-{max_value}"
                        ))
                    
                    if max_value is not None and numeric_value > max_value:
                        issues.append(ValidationIssue(
                            field, "range", ValidationResult.ERROR,
                            f"记录{record_index}: {rule.error_message}", value, f"{min_value}-{max_value}"
                        ))
                
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        field, "range", ValidationResult.ERROR,
                        f"记录{record_index}: 无法进行范围检查，值不是数值", value
                    ))
            
            # 格式检查
            elif rule.rule_type == "format":
                pattern = rule.parameters.get("pattern")
                if pattern and not re.match(pattern, str(value)):
                    issues.append(ValidationIssue(
                        field, "format", ValidationResult.ERROR,
                        f"记录{record_index}: {rule.error_message}", value, pattern
                    ))
            
            # 枚举检查
            elif rule.rule_type == "enum":
                allowed_values = rule.parameters.get("allowed_values", [])
                if value not in allowed_values:
                    issues.append(ValidationIssue(
                        field, "enum", ValidationResult.ERROR,
                        f"记录{record_index}: {field}值不在允许范围内", value, allowed_values
                    ))
            
            # 数组检查
            elif rule.rule_type == "array":
                if not isinstance(value, list):
                    issues.append(ValidationIssue(
                        field, "array", ValidationResult.ERROR,
                        f"记录{record_index}: {field}必须是数组", value
                    ))
                else:
                    min_length = rule.parameters.get("min_length")
                    max_length = rule.parameters.get("max_length")
                    
                    if min_length is not None and len(value) < min_length:
                        issues.append(ValidationIssue(
                            field, "array_length", ValidationResult.ERROR,
                            f"记录{record_index}: {rule.error_message}", len(value), min_length
                        ))
                    
                    if max_length is not None and len(value) > max_length:
                        issues.append(ValidationIssue(
                            field, "array_length", ValidationResult.WARNING,
                            f"记录{record_index}: {field}数组长度超过限制", len(value), max_length
                        ))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"验证字段失败: {e}")
            return [ValidationIssue(field, "validation_error", ValidationResult.ERROR, str(e), record.get(field))]
    
    def apply_custom_validation(self, record: Dict, rule: ValidationRule, 
                               data_type: DataType, record_index: int) -> List[ValidationIssue]:
        """
        应用自定义验证
        
        Args:
            record: 记录数据
            rule: 验证规则
            data_type: 数据类型
            record_index: 记录索引
            
        Returns:
            验证问题列表
        """
        issues = []
        
        try:
            # K线价格逻辑检查
            if rule.rule_type == "price_logic" and data_type == DataType.KLINE:
                issues.extend(self.validate_kline_price_logic(record, record_index))
            
            # 深度数据逻辑检查
            elif rule.rule_type == "depth_logic" and data_type == DataType.DEPTH:
                issues.extend(self.validate_depth_logic(record, record_index))
            
            # 余额逻辑检查
            elif rule.rule_type == "balance_logic" and data_type == DataType.ACCOUNT:
                issues.extend(self.validate_balance_logic(record, record_index))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"自定义验证失败: {e}")
            return [ValidationIssue("custom", "validation_error", ValidationResult.ERROR, str(e), None)]
    
    def validate_kline_price_logic(self, record: Dict, record_index: int) -> List[ValidationIssue]:
        """验证K线价格逻辑"""
        issues = []
        
        try:
            open_price = float(record.get('open', 0))
            high_price = float(record.get('high', 0))
            low_price = float(record.get('low', 0))
            close_price = float(record.get('close', 0))
            
            # 最高价应该是最高的
            if high_price < max(open_price, close_price, low_price):
                issues.append(ValidationIssue(
                    "high", "price_logic", ValidationResult.ERROR,
                    f"记录{record_index}: 最高价应该是所有价格中的最高值",
                    high_price, max(open_price, close_price, low_price)
                ))
            
            # 最低价应该是最低的
            if low_price > min(open_price, close_price, high_price):
                issues.append(ValidationIssue(
                    "low", "price_logic", ValidationResult.ERROR,
                    f"记录{record_index}: 最低价应该是所有价格中的最低值",
                    low_price, min(open_price, close_price, high_price)
                ))
            
            # 价格异常波动检查
            max_change = max(abs(high_price - open_price), abs(low_price - open_price), 
                           abs(close_price - open_price)) / open_price if open_price > 0 else 0
            
            if max_change > 0.2:  # 20%异常波动
                issues.append(ValidationIssue(
                    "price", "abnormal_change", ValidationResult.WARNING,
                    f"记录{record_index}: 价格波动异常大: {max_change:.2%}",
                    max_change, 0.2
                ))
                
        except (ValueError, TypeError, ZeroDivisionError) as e:
            issues.append(ValidationIssue(
                "price", "logic_error", ValidationResult.ERROR,
                f"记录{record_index}: 价格逻辑验证失败: {e}", None
            ))
        
        return issues
    
    def validate_depth_logic(self, record: Dict, record_index: int) -> List[ValidationIssue]:
        """验证深度数据逻辑"""
        issues = []
        
        try:
            bids = record.get('bids', [])
            asks = record.get('asks', [])
            
            # 检查买单价格递减
            for i in range(1, len(bids)):
                if len(bids[i]) >= 1 and len(bids[i-1]) >= 1:
                    current_price = float(bids[i][0])
                    prev_price = float(bids[i-1][0])
                    
                    if current_price >= prev_price:
                        issues.append(ValidationIssue(
                            "bids", "order_logic", ValidationResult.WARNING,
                            f"记录{record_index}: 买单价格应该递减", current_price, prev_price
                        ))
            
            # 检查卖单价格递增
            for i in range(1, len(asks)):
                if len(asks[i]) >= 1 and len(asks[i-1]) >= 1:
                    current_price = float(asks[i][0])
                    prev_price = float(asks[i-1][0])
                    
                    if current_price <= prev_price:
                        issues.append(ValidationIssue(
                            "asks", "order_logic", ValidationResult.WARNING,
                            f"记录{record_index}: 卖单价格应该递增", current_price, prev_price
                        ))
            
            # 检查买卖价差
            if bids and asks and len(bids[0]) >= 1 and len(asks[0]) >= 1:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                
                if best_bid >= best_ask:
                    issues.append(ValidationIssue(
                        "spread", "logic_error", ValidationResult.ERROR,
                        f"记录{record_index}: 最佳买价不能大于等于最佳卖价", best_bid, best_ask
                    ))
                
        except (ValueError, TypeError, IndexError) as e:
            issues.append(ValidationIssue(
                "depth", "logic_error", ValidationResult.ERROR,
                f"记录{record_index}: 深度逻辑验证失败: {e}", None
            ))
        
        return issues
    
    def validate_balance_logic(self, record: Dict, record_index: int) -> List[ValidationIssue]:
        """验证余额逻辑"""
        issues = []
        
        try:
            total_balance = float(record.get('total_balance', 0))
            available_balance = float(record.get('available_balance', 0))
            
            # 可用余额不能大于总余额
            if available_balance > total_balance:
                issues.append(ValidationIssue(
                    "available_balance", "balance_logic", ValidationResult.ERROR,
                    f"记录{record_index}: 可用余额不能大于总余额", available_balance, total_balance
                ))
                
        except (ValueError, TypeError) as e:
            issues.append(ValidationIssue(
                "balance", "logic_error", ValidationResult.ERROR,
                f"记录{record_index}: 余额逻辑验证失败: {e}", None
            ))
        
        return issues
    
    def validate_batch_consistency(self, data_list: List[Dict], data_type: DataType) -> List[ValidationIssue]:
        """验证批量数据一致性"""
        issues = []
        
        try:
            if data_type == DataType.KLINE and len(data_list) > 1:
                # 检查时间序列连续性
                for i in range(1, len(data_list)):
                    current_time = int(data_list[i].get('timestamp', 0))
                    prev_time = int(data_list[i-1].get('timestamp', 0))
                    
                    if current_time <= prev_time:
                        issues.append(ValidationIssue(
                            "timestamp", "sequence", ValidationResult.WARNING,
                            f"K线时间序列不连续: 位置{i}", current_time, prev_time
                        ))
                
                # 检查价格连续性
                for i in range(1, len(data_list)):
                    current_open = float(data_list[i].get('open', 0))
                    prev_close = float(data_list[i-1].get('close', 0))
                    
                    price_gap = abs(current_open - prev_close) / prev_close if prev_close > 0 else 0
                    
                    if price_gap > 0.05:  # 5%价格跳空
                        issues.append(ValidationIssue(
                            "price", "gap", ValidationResult.WARNING,
                            f"价格跳空: 位置{i}, 跳空{price_gap:.2%}", current_open, prev_close
                        ))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"批量一致性验证失败: {e}")
            return [ValidationIssue("batch", "consistency_error", ValidationResult.ERROR, str(e), None)]
    
    def add_custom_rule(self, data_type: DataType, rule: ValidationRule):
        """添加自定义验证规则"""
        try:
            if data_type not in self.validation_rules:
                self.validation_rules[data_type] = []
            
            self.validation_rules[data_type].append(rule)
            self.logger.info(f"添加自定义验证规则: {data_type.value}.{rule.field}")
            
        except Exception as e:
            self.logger.error(f"添加自定义验证规则失败: {e}")
    
    def update_stats(self, report: ValidationReport):
        """更新统计信息"""
        try:
            self.stats['total_validations'] += 1
            self.stats['total_records_processed'] += report.total_records
            self.stats['total_issues_found'] += len(report.issues)
            self.stats['validation_time_total'] += report.execution_time
            
            if report.has_errors:
                self.stats['failed_validations'] += 1
            else:
                self.stats['successful_validations'] += 1
                
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")
    
    def get_validation_rules(self, data_type: DataType) -> List[ValidationRule]:
        """获取验证规则"""
        return self.validation_rules.get(data_type, [])
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_validation_time = (self.stats['validation_time_total'] / self.stats['total_validations'] 
                              if self.stats['total_validations'] > 0 else 0)
        
        success_rate = (self.stats['successful_validations'] / self.stats['total_validations'] 
                       if self.stats['total_validations'] > 0 else 0)
        
        return {
            'total_validations': self.stats['total_validations'],
            'successful_validations': self.stats['successful_validations'],
            'failed_validations': self.stats['failed_validations'],
            'success_rate': success_rate,
            'total_records_processed': self.stats['total_records_processed'],
            'total_issues_found': self.stats['total_issues_found'],
            'average_validation_time': avg_validation_time,
            'validation_rules_count': {dt.value: len(rules) for dt, rules in self.validation_rules.items()}
        }
    
    def get_status(self) -> Dict:
        """获取验证器状态"""
        return {
            'total_validations': self.stats['total_validations'],
            'total_records_processed': self.stats['total_records_processed'],
            'supported_data_types': [dt.value for dt in self.validation_rules.keys()],
            'cache_size': len(self.validation_cache),
            'is_operational': True
        } 