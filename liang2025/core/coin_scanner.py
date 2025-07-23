#!/usr/bin/env python3
"""
币种扫描器 - 基于大佬建议的优秀选币逻辑
实现3阶段市场判断（冷启动、高热、冷却）+ 资金流入分析
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from utils.logger import get_logger
from config.config_manager import ConfigManager

@dataclass
class CoinAnalysis:
    """币种分析结果"""
    symbol: str
    stage: int  # 1=冷启动, 2=高热, 3=冷却
    stage_name: str
    price: float
    ma50: float
    ma200: float
    volume_ratio: float  # 当前量/平均量
    money_flow_ratio: float  # 资金流入比例
    reasons: List[str]
    score: float
    
class CoinScanner:
    """币种扫描器 - 大佬版选币逻辑"""
    
    def __init__(self, config_manager: ConfigManager, api_client_manager):
        self.config = config_manager
        self.api_client_manager = api_client_manager
        self.logger = get_logger(__name__)
        
        # 选币参数（大佬建议）
        self.mainstream_coins = ['BTCUSDT', 'ETHUSDT']  # 主流币保底
        self.top_volume_count = 10  # 成交量Top10
        self.max_symbols = 10  # 最终选择限制
        
        # 阶段判断参数
        self.stage_params = {
            # 阶段1: 冷启动（最佳多头满仓）
            1: {'price_vs_ma200': 1.1, 'volume_vs_avg': 0.8},
            # 阶段2: 高热（滚仓机会） 
            2: {'volume_vs_avg': 1.5},
            # 阶段3: 冷却（试空轻仓）
            3: {'volume_vs_avg': 1.0}
        }
        
        # 资金流入参数（适当放宽）
        self.money_flow_multiplier = 1.0  # 降低到1.0倍平均流入
        self.money_flow_period = 20  # 20期平均流入
        
        self.logger.info("🎯 币种扫描器初始化完成（大佬版选币逻辑）")
    
    async def scan_and_select_coins(self) -> List[str]:
        """
        大佬版选币逻辑主流程
        """
        try:
            self.logger.info("🔍 正在扫描所有币种，执行选币逻辑...")
            
            # 步骤1: 获取候选币种池（主流+成交量Top10）
            candidates = await self.get_candidate_pool()
            self.logger.info(f"📊 候选币种池: {len(candidates)} 个，包括主流/成交量top")
            
            # 步骤2: 阶段过滤 + 流动性过滤
            selected_coins = []
            for symbol in candidates:
                self.logger.info(f"📈 正在分析 {symbol}...")
                
                analysis = await self.analyze_coin_stage_and_flow(symbol)
                if analysis:
                    # 显示详细分析结果
                    self.logger.info(f"   📊 {symbol}: 阶段{analysis.stage}({analysis.stage_name}), "
                                   f"价格:{analysis.price:.4f}, MA50:{analysis.ma50:.4f}, MA200:{analysis.ma200:.4f}")
                    self.logger.info(f"   📈 成交量比例:{analysis.volume_ratio:.2f}x, "
                                   f"资金流入比例:{analysis.money_flow_ratio:.2f}x, 评分:{analysis.score:.1f}")
                    
                    # 选择阶段1（冷启动）和阶段2（高热）
                    if analysis.stage in [1, 2] and analysis.money_flow_ratio >= self.money_flow_multiplier:
                        selected_coins.append(analysis)
                        self.logger.info(f"✅ 选择 {symbol}: {analysis.stage_name}，流动性流入OK")
                    else:
                        reason = "流动性不足" if analysis.money_flow_ratio < self.money_flow_multiplier else "阶段不符"
                        self.logger.info(f"❌ 排除 {symbol}: {analysis.stage_name}，{reason}")
                else:
                    self.logger.warning(f"❌ {symbol} 分析失败，跳过")
            
            # 步骤3: 排序和限制数量
            selected_coins.sort(key=lambda x: x.score, reverse=True)
            final_coins = selected_coins[:self.max_symbols]
            
            # 如果没有选出任何币种，使用备用宽松标准
            if not final_coins:
                self.logger.warning("🔄 严格标准未选出币种，使用宽松标准重新筛选...")
                backup_selected = []
                
                for symbol in candidates:
                    analysis = await self.analyze_coin_stage_and_flow(symbol)
                    if analysis:
                        # 放宽条件：所有阶段都可以，资金流入比例>0.5
                        if analysis.money_flow_ratio >= 0.5:
                            backup_selected.append(analysis)
                            self.logger.info(f"🔄 宽松标准选择 {symbol}: {analysis.stage_name}")
                            if len(backup_selected) >= 3:  # 至少选3个
                                break
                
                if backup_selected:
                    selected_coins.sort(key=lambda x: x.score, reverse=True)
                    final_coins = backup_selected[:self.max_symbols]
                    self.logger.info(f"✅ 宽松标准成功选出 {len(final_coins)} 个币种")
                else:
                    # 最后的备用方案：从候选池中选择成交量最大的币种
                    self.logger.warning("⚠️ 所有选币标准都失败，使用最终备用方案...")
                    backup_final = []
                    
                    try:
                        # 获取候选池的24小时成交量数据
                        volume_data = []
                        for symbol in candidates[:5]:  # 只检查前5个
                            try:
                                ticker_response = await self.api_client_manager.get_24hr_ticker_stats(symbol)
                                if ticker_response and ticker_response.success:
                                    volume = float(ticker_response.data.get('volume', 0))
                                    volume_data.append((symbol, volume))
                            except Exception as e:
                                self.logger.debug(f"获取{symbol}成交量失败: {e}")
                        
                        # 按成交量排序，选择前2个
                        if volume_data:
                            volume_data.sort(key=lambda x: x[1], reverse=True)
                            for symbol, volume in volume_data[:2]:
                                # 创建基础分析结果
                                backup_analysis = CoinAnalysis(
                                    symbol=symbol,
                                    stage=2,
                                    stage_name="备用选择",
                                    price=0.0,  # 临时值
                                    ma50=0.0,
                                    ma200=0.0,
                                    volume_ratio=1.0,
                                    money_flow_ratio=1.0,
                                    reasons=["备用选币机制"],
                                    score=60.0
                                )
                                backup_final.append(backup_analysis)
                                self.logger.info(f"🔄 最终备用选择 {symbol} (成交量: {volume:.0f})")
                        
                        final_coins = backup_final
                        
                    except Exception as e:
                        self.logger.error(f"备用方案也失败: {e}")
                        # 绝对最后的备用：返回主流币
                        final_coins = [
                            CoinAnalysis(
                                symbol="BTCUSDT",
                                stage=2, stage_name="默认选择", price=0.0, ma50=0.0, ma200=0.0,
                                volume_ratio=1.0, money_flow_ratio=1.0, reasons=["默认主流币"], score=50.0
                            ),
                            CoinAnalysis(
                                symbol="ETHUSDT", 
                                stage=2, stage_name="默认选择", price=0.0, ma50=0.0, ma200=0.0,
                                volume_ratio=1.0, money_flow_ratio=1.0, reasons=["默认主流币"], score=50.0
                            )
                        ]
                        self.logger.warning("🔄 使用绝对备用方案：BTCUSDT + ETHUSDT")
            
            # 输出最终结果
            final_symbols = [coin.symbol for coin in final_coins]
            self.logger.info(f"🎯 最终选择的币种: {final_symbols}")
            
            for i, coin in enumerate(final_coins):
                if coin.price > 0:  # 只有真实分析的币种才输出详细信息
                    self.logger.info(f"   {i+1}. {coin.symbol}: {coin.stage_name}, 价格:{coin.price:.4f}, "
                                   f"资金流入比例:{coin.money_flow_ratio:.2f}x")
                else:
                    self.logger.info(f"   {i+1}. {coin.symbol}: {coin.stage_name}")
            
            return final_symbols
            
        except Exception as e:
            self.logger.error(f"选币逻辑执行失败: {e}")
            # 返回默认主流币
            return self.mainstream_coins
    
    async def get_candidate_pool(self) -> List[str]:
        """
        获取候选币种池：主流币 + 成交量Top10
        """
        candidates = self.mainstream_coins.copy()
        
        try:
            # 获取成交量Top10
            top_volume_coins = await self.get_top_volume_coins(self.top_volume_count)
            
            # 去重合并
            for coin in top_volume_coins:
                if coin not in candidates:
                    candidates.append(coin)
                    
            self.logger.info(f"候选池构建完成: 主流币{len(self.mainstream_coins)}个 + Top成交量{len(top_volume_coins)}个")
            
        except Exception as e:
            self.logger.warning(f"获取Top成交量失败: {e}，仅使用主流币")
            
        return candidates
    
    async def get_top_volume_coins(self, limit: int) -> List[str]:
        """
        获取成交量Top N的USDT交易对
        """
        try:
            # 获取24小时统计数据
            stats_response = await self.api_client_manager.get_24hr_ticker_stats()
            if not stats_response or not stats_response.success:
                return []
            
            stats_data = stats_response.data
            if not isinstance(stats_data, list):
                return []
            
            # 筛选USDT交易对并按成交量排序
            usdt_pairs = []
            for stat in stats_data:
                symbol = stat.get('symbol', '')
                if symbol.endswith('USDT') and symbol not in ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT']:
                    volume = float(stat.get('quoteVolume', 0))
                    usdt_pairs.append((symbol, volume))
            
            # 按成交量排序，取前N个
            usdt_pairs.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [pair[0] for pair in usdt_pairs[:limit]]
            
            self.logger.info(f"获取成交量Top{limit}: {top_symbols}")
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"获取Top成交量币种失败: {e}")
            return []
    
    async def analyze_coin_stage_and_flow(self, symbol: str) -> Optional[CoinAnalysis]:
        """
        分析币种的市场阶段和资金流入情况（大佬核心逻辑）
        增强版：添加多重容错和降级策略
        """
        try:
            # 获取日线K线数据（用于阶段判断）
            klines_response = await self.api_client_manager.get_klines(symbol, '1d', limit=250)
            
            # 第一重检查：API响应
            if not klines_response or not klines_response.success:
                self.logger.warning(f"❌ {symbol} API请求失败，尝试降级到小时数据")
                # 降级策略：使用小时数据
                klines_response = await self.api_client_manager.get_klines(symbol, '1h', limit=200)
                
            if not klines_response or not klines_response.success:
                self.logger.error(f"❌ {symbol} 所有数据源都失败")
                return None
                
            klines = klines_response.data
            
            # 第二重检查：数据长度
            if len(klines) < 50:
                self.logger.warning(f"❌ {symbol} 数据不足({len(klines)}条)，需要至少50条")
                return None
            elif len(klines) < 200:
                self.logger.warning(f"⚠️ {symbol} 数据不足({len(klines)}条)，使用降级计算")
                
            # 提取OHLCV数据，添加数据验证
            try:
                opens = np.array([float(k[1]) for k in klines])
                highs = np.array([float(k[2]) for k in klines]) 
                lows = np.array([float(k[3]) for k in klines])
                closes = np.array([float(k[4]) for k in klines])
                volumes = np.array([float(k[5]) for k in klines])
                
                # 数据合理性检查
                if np.any(closes <= 0) or np.any(volumes < 0):
                    self.logger.error(f"❌ {symbol} 数据异常：包含非正价格或负成交量")
                    return None
                    
            except (ValueError, IndexError) as e:
                self.logger.error(f"❌ {symbol} 数据解析失败: {e}")
                return None
            
            current_price = closes[-1]
            
            # 计算移动平均线（大佬标准），使用自适应长度
            ma50_period = min(50, len(closes))
            ma200_period = min(200, len(closes))
            
            ma50 = np.mean(closes[-ma50_period:])
            ma200 = np.mean(closes[-ma200_period:])
            
            # 计算成交量比例，使用自适应长度
            volume_period = min(20, len(volumes))
            avg_volume = np.mean(volumes[-volume_period:])  # 20期平均成交量
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # 计算资金流入比例（大佬核心算法）
            money_flow_ratio = self.calculate_money_flow_ratio(opens, closes, volumes)
            
            # 判断市场阶段（大佬3阶段理论）
            stage, stage_name, reasons = self.determine_market_stage(
                current_price, ma50, ma200, volume_ratio
            )
            
            # 计算综合评分
            score = self.calculate_comprehensive_score(stage, volume_ratio, money_flow_ratio)
            
            # 输出详细分析结果（与用户要求一致）
            self.logger.info(f"   📊 {symbol}: 阶段{stage}({stage_name}), "
                           f"价格:{current_price:.4f}, MA50:{ma50:.4f}, MA200:{ma200:.4f}")
            self.logger.info(f"   📈 成交量比例:{volume_ratio:.2f}x, "
                           f"资金流入比例:{money_flow_ratio:.2f}x, 评分:{score:.1f}")
            
            return CoinAnalysis(
                symbol=symbol,
                stage=stage,
                stage_name=stage_name, 
                price=current_price,
                ma50=ma50,
                ma200=ma200,
                volume_ratio=volume_ratio,
                money_flow_ratio=money_flow_ratio,
                reasons=reasons,
                score=score
            )
            
        except Exception as e:
            self.logger.error(f"❌ 分析币种 {symbol} 出现异常: {e}")
            import traceback
            self.logger.debug(f"详细错误堆栈: {traceback.format_exc()}")
            return None
    
    def calculate_money_flow_ratio(self, opens: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
        """
        计算资金流入比例（大佬算法）
        资金流入 = 成交量 * (收盘价 - 开盘价)
        """
        try:
            # 计算每日资金流入
            daily_flows = volumes * (closes - opens)
            
            # 最近流入
            recent_flow = daily_flows[-1]
            
            # 20期平均流入
            avg_flow_period = min(self.money_flow_period, len(daily_flows))
            avg_flow = np.mean(daily_flows[-avg_flow_period:])
            
            # 资金流入比例
            flow_ratio = abs(recent_flow) / abs(avg_flow) if avg_flow != 0 else 1
            
            # 考虑流入方向（正流入 vs 负流入）
            if recent_flow > 0 and avg_flow > 0:
                return flow_ratio
            elif recent_flow > 0 > avg_flow:
                return flow_ratio * 1.5  # 转正奖励
            else:
                return flow_ratio * 0.5  # 负流入惩罚
                
        except Exception as e:
            self.logger.warning(f"计算资金流入失败: {e}")
            return 1.0
    
    def determine_market_stage(self, price: float, ma50: float, ma200: float, volume_ratio: float) -> Tuple[int, str, List[str]]:
        """
        判断市场阶段（大佬3阶段理论）
        """
        reasons = []
        
        # 阶段1: 冷启动（最佳多头满仓）
        if (price < ma200 * self.stage_params[1]['price_vs_ma200'] and 
            volume_ratio < self.stage_params[1]['volume_vs_avg']):
            reasons.append("低位盘整，蓄势待发")
            reasons.append("成交量温和，资金等待")
            return 1, "冷启动", reasons
        
        # 阶段2: 高热（滚仓机会）
        if (price > ma50 and volume_ratio > self.stage_params[2]['volume_vs_avg']):
            reasons.append("突破主升浪")
            reasons.append("成交量放大")
            if price > ma200:
                reasons.append("长期趋势向上")
            return 2, "高热", reasons
        
        # 阶段3: 冷却（试空轻仓）
        if (price < ma50 and volume_ratio < self.stage_params[3]['volume_vs_avg']):
            reasons.append("下跌末期")
            reasons.append("成交量萎缩")
            return 3, "冷却", reasons
        
        # 其他情况（中性）
        reasons.append("市场中性")
        return 2, "中性", reasons  # 归类为阶段2以便进入候选
    
    def calculate_comprehensive_score(self, stage: int, volume_ratio: float, money_flow_ratio: float) -> float:
        """
        计算综合评分
        """
        base_score = 50
        
        # 阶段评分
        stage_scores = {1: 85, 2: 80, 3: 30}  # 冷启动和高热得高分
        base_score += stage_scores.get(stage, 50)
        
        # 成交量评分
        if volume_ratio > 1.5:
            base_score += 20
        elif volume_ratio > 1.2:
            base_score += 15
        elif volume_ratio > 1.0:
            base_score += 10
        
        # 资金流入评分
        if money_flow_ratio > 1.5:
            base_score += 20
        elif money_flow_ratio > 1.2:
            base_score += 15
        elif money_flow_ratio > 1.0:
            base_score += 10
        
        return min(base_score, 100) 