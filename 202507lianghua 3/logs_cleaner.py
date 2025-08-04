#!/usr/bin/env python3
"""
日志清理脚本
定期清理旧的日志文件，避免磁盘空间不足
"""

import os
import glob
import time
from datetime import datetime, timedelta
from pathlib import Path

class LogCleaner:
    """日志清理器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.retention_days = {
            'system': 30,      # 系统日志保留30天
            'errors': 30,      # 错误日志保留30天 
            'trades': 90,      # 交易日志保留90天（重要）
            'performance': 14   # 性能日志保留14天
        }
    
    def clean_logs(self):
        """清理过期日志"""
        print(f"🧹 开始清理日志文件 ({self.log_dir})")
        
        current_time = datetime.now()
        total_deleted = 0
        total_size_freed = 0
        
        for log_type, retention_days in self.retention_days.items():
            cutoff_date = current_time - timedelta(days=retention_days)
            
            # 查找匹配的日志文件
            patterns = [
                f"????????_{log_type}.log*",
                f"????????_{log_type}.json*",
            ]
            
            for pattern in patterns:
                files = glob.glob(str(self.log_dir / pattern))
                
                for file_path in files:
                    try:
                        # 从文件名提取日期
                        filename = os.path.basename(file_path)
                        date_str = filename[:8]
                        
                        if len(date_str) == 8 and date_str.isdigit():
                            file_date = datetime.strptime(date_str, "%Y%m%d")
                            
                            if file_date < cutoff_date:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                
                                total_deleted += 1
                                total_size_freed += file_size
                                
                                print(f"   🗑️ 删除: {filename} ({file_size/1024/1024:.1f}MB)")
                                
                    except Exception as e:
                        print(f"   ❌ 删除失败 {file_path}: {e}")
        
        print(f"✅ 清理完成: 删除{total_deleted}个文件，释放{total_size_freed/1024/1024:.1f}MB空间")
    
    def get_log_statistics(self):
        """获取日志统计信息"""
        print("📊 日志文件统计:")
        
        for log_type in self.retention_days.keys():
            patterns = [
                f"????????_{log_type}.log*",
                f"????????_{log_type}.json*",
            ]
            
            total_files = 0
            total_size = 0
            
            for pattern in patterns:
                files = glob.glob(str(self.log_dir / pattern))
                total_files += len(files)
                
                for file_path in files:
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
            
            if total_files > 0:
                print(f"   📁 {log_type}: {total_files}个文件, {total_size/1024/1024:.1f}MB")

if __name__ == "__main__":
    cleaner = LogCleaner()
    cleaner.get_log_statistics()
    cleaner.clean_logs()
