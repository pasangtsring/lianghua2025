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
    preferred_direction: str = 'LONG'  # 推荐交易方向：LONG/SHORT
    
class CoinScanner:
    """币种扫描器 - 大佬版选币逻辑"""
    
    def __init__(self, config_manager: ConfigManager, api_client_manager):
        self.config = config_manager
        self.api_client_manager = api_client_manager
        self.logger = get_logger(__name__)
        
        # 选币参数（大佬建议）
        self.mainstream_coins = ['BTCUSDT', 'ETHUSDT']  # 主流币保底
        self.top_volume_count = self.config.get_coin_selection_config().top_volume_count  # 从配置读取Top N
        self.max_symbols = self.config.get_coin_selection_config().max_symbols  # 最终选择限制
        
        # 阶段判断参数
        self.stage_params = {
            # 阶段1: 冷启动（最佳多头满仓）
            1: {'price_vs_ma200': 1.1, 'volume_vs_avg': 0.8},
            # 阶段2: 高热（滚仓机会） 
            2: {'volume_vs_avg': 1.5},
            # 阶段3: 冷却（试空轻仓）
            3: {'volume_vs_avg': 1.0}
        }
        
        # 资金流入参数（大幅放宽以适应当前市场）
        self.money_flow_multiplier = 0.5  # 大幅降低到0.5倍，增加选币数量
        self.money_flow_period = 20  # 20期平均流入
        
        self.logger.info("🎯 币种扫描器初始化完成（大佬版选币逻辑）")
    
    def validate_coin_data(self, klines_data: List, symbol: str) -> tuple[bool, str]:
        """验证币种数据是否正常，过滤异常币种（如BNXUSDT、ALPACAUSDT等）"""
        if not klines_data or len(klines_data) < 10:
            return False, "数据不足"
        
        try:
            # 检查最近20根K线的价格变化和成交量
            recent_klines = klines_data[-20:] if len(klines_data) >= 20 else klines_data
            closes = [float(k[4]) for k in recent_klines]
            volumes = [float(k[5]) for k in recent_klines]
            
            price_range = max(closes) - min(closes)
            total_volume = sum(volumes)
            min_price = min(closes)
            
            # 检查价格是否完全无变化且成交量为0（疑似暂停交易）
            if price_range == 0 and total_volume == 0:
                return False, f"价格无变化且成交量为0，疑似暂停交易"
            
            # 检查价格变化是否过小（流动性极差）
            if min_price > 0 and price_range / min_price < 0.001:  # 变化小于0.1%
                return False, f"价格变化过小({price_range/min_price:.4%})，流动性不足"
            
            # 检查成交量是否全为0（无交易活动）
            if total_volume == 0:
                return False, f"成交量为0，无交易活动"
            
            return True, "数据正常"
            
        except Exception as e:
            return False, f"数据验证失败: {str(e)}"
    
    async def scan_and_select_coins(self) -> List[str]:
        """
        大佬版选币逻辑主流程
        """
        try:
            self.logger.info("🔍 正在扫描所有币种，执行选币逻辑...")
            
            # 市场环境感知：降低选币门槛以适应下跌行情
            market_sentiment = await self.detect_market_sentiment()
            if market_sentiment == "BEARISH":
                # 🔥 修复：熊市中平衡多空机会
                long_multiplier = 0.8   # 熊市做多门槛适度提高
                short_multiplier = 0.4  # 熊市做空门槛降低
                self.logger.info("📉 熊市环境：平衡做多做空机会")
            elif market_sentiment == "BULLISH":
                # 🔥 修复：牛市中增加做多机会
                long_multiplier = 0.3   # 牛市做多门槛大幅降低
                short_multiplier = 0.8  # 牛市做空门槛提高
                self.logger.info("📈 牛市环境：增加做多机会")
            else:
                # 🔥 新增：中性市场平衡处理
                long_multiplier = 0.5   # 中性市场平衡处理
                short_multiplier = 0.5
                self.logger.info("⚖️ 中性市场：多空平衡选币")
            
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
                    
                    # 🔥 修复：多空平衡选币逻辑
                    
                    # 做多机会判断 - 条件大幅放宽
                    if (analysis.stage in [1, 2] and analysis.money_flow_ratio >= long_multiplier) or \
                       (analysis.stage == 1 and analysis.price > analysis.ma50) or \
                       (analysis.stage == 2 and analysis.volume_ratio >= 1.2) or \
                       (analysis.price > analysis.ma200 and analysis.volume_ratio >= 1.0):
                        # 多头机会：大幅放宽条件
                        analysis.preferred_direction = 'LONG'
                        selected_coins.append(analysis)
                        self.logger.info(f"✅ 做多选择: {symbol} - {analysis.stage_name}，技术面支持")
                    
                    # 做空机会判断 - 条件适度收紧
                    elif (analysis.stage == 3 and analysis.money_flow_ratio <= short_multiplier) or \
                         (analysis.stage in [2, 3] and analysis.price < analysis.ma200 and analysis.volume_ratio >= 1.0):
                        # 空头机会：适度收紧条件
                        analysis.preferred_direction = 'SHORT'
                        selected_coins.append(analysis)
                        self.logger.info(f"✅ 做空选择: {symbol} - {analysis.stage_name}，趋势转弱")
                    else:
                        if analysis.stage in [1, 2]:
                            reason = "流动性不足"
                        else:
                            reason = "资金流入过多，不适合做空"
                        self.logger.info(f"❌ 排除 {symbol}: {analysis.stage_name}，{reason}")
                else:
                    self.logger.warning(f"❌ {symbol} 分析失败，跳过")
            
            # 🔥 修复：步骤3: 多空平衡排序和数量控制
            selected_coins.sort(key=lambda x: x.score, reverse=True)
            
            # 分离多空候选
            long_coins = [coin for coin in selected_coins if coin.preferred_direction == 'LONG']
            short_coins = [coin for coin in selected_coins if coin.preferred_direction == 'SHORT']
            
            # 平衡选择：尽量保持多空比例
            max_total = min(self.max_symbols, 20)  # 支持最多20个币种
            target_long = max_total // 2
            target_short = max_total - target_long
            
            final_coins = []
            final_coins.extend(long_coins[:target_long])    # 选择评分最高的做多币种
            final_coins.extend(short_coins[:target_short])  # 选择评分最高的做空币种
            
            # 如果某一方不足，用另一方补充
            if len(long_coins) < target_long and len(short_coins) > target_short:
                remaining = target_long - len(long_coins)
                final_coins.extend(short_coins[target_short:target_short + remaining])
            elif len(short_coins) < target_short and len(long_coins) > target_long:
                remaining = target_short - len(short_coins)
                final_coins.extend(long_coins[target_long:target_long + remaining])
            
            self.logger.info(f"📊 多空分配: 做多{len([c for c in final_coins if c.preferred_direction == 'LONG'])}个, "
                           f"做空{len([c for c in final_coins if c.preferred_direction == 'SHORT'])}个, 总计{len(final_coins)}个")
            
            # 如果没有选出任何币种，使用备用宽松标准
            if not final_coins:
                self.logger.warning("🔄 严格标准未选出币种，使用宽松标准重新筛选...")
                
                # 宽松标准重新筛选
                for symbol in candidates:
                    analysis = await self.analyze_coin_stage_and_flow(symbol)
                    if analysis:
                        # 宽松的做多条件
                        if analysis.stage in [1, 2] and analysis.money_flow_ratio >= 0.3:
                            analysis.preferred_direction = 'LONG'
                            selected_coins.append(analysis)
                            self.logger.info(f"🔄 宽松选择 {symbol}: 做多机会")
                        # 宽松的做空条件
                        elif analysis.price < analysis.ma50 or analysis.money_flow_ratio <= 1.0:
                            analysis.preferred_direction = 'SHORT'
                            selected_coins.append(analysis)
                            self.logger.info(f"🔄 宽松选择 {symbol}: 做空机会")
                
                # 重新排序和限制
                selected_coins.sort(key=lambda x: x.score, reverse=True)
                final_coins = selected_coins[:self.max_symbols]
                self.logger.info(f"🔄 宽松标准选出 {len(final_coins)} 个币种")
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
                    direction = getattr(coin, 'preferred_direction', 'LONG')
                    direction_msg = "适合做多" if direction == 'LONG' else "适合做空"
                    self.logger.info(f"   {i+1}. {coin.symbol}: {coin.stage_name}, {direction_msg}, 价格:{coin.price:.4f}, "
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
        🔥 修复：获取扩大的候选币种池：主流币 + 成交量Top50
        """
        candidates = self.mainstream_coins.copy()
        
        try:
            # 🔥 扩大：获取成交量Top50 (从配置读取，默认50)
            expanded_count = min(self.top_volume_count, 80)  # 最多80个
            top_volume_coins = await self.get_top_volume_coins(expanded_count)
            
            # 去重合并
            for coin in top_volume_coins:
                if coin not in candidates:
                    candidates.append(coin)
            
            # 🔥 新增：确保候选池足够大
            if len(candidates) < 30:  # 如果候选池太小，进一步放宽条件
                self.logger.warning("候选池偏小，启用宽松筛选...")
                additional_coins = await self.get_additional_candidates()
                for coin in additional_coins:
                    if coin not in candidates:
                        candidates.append(coin)
                        
            self.logger.info(f"🎯 扩大候选池完成: 主流币{len(self.mainstream_coins)}个 + "
                           f"Top成交量{len(top_volume_coins)}个, 总计{len(candidates)}个")
            
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
    
    async def get_additional_candidates(self) -> List[str]:
        """
        🔥 新增：获取额外候选币种（宽松条件）
        """
        try:
            # 获取交易所信息
            exchange_response = await self.api_client_manager.get_exchange_info()
            if not exchange_response or not exchange_response.success:
                return []
            
            # 获取所有USDT交易对
            all_usdt_symbols = []
            for symbol_info in exchange_response.data.get('symbols', []):
                symbol = symbol_info.get('symbol', '')
                if (symbol.endswith('USDT') and 
                    symbol not in self.config.get_coin_selection_config().excluded_symbols and
                    symbol_info.get('status') == 'TRADING'):
                    all_usdt_symbols.append(symbol)
            
            # 获取24小时ticker
            ticker_response = await self.api_client_manager.get_24hr_tickers()
            if not ticker_response or not ticker_response.success:
                return all_usdt_symbols[:20]  # 降级返回前20个
            
            # 按成交量过滤（降低门槛）
            volume_candidates = []
            for ticker in ticker_response.data:
                symbol = ticker.get('symbol')
                if symbol in all_usdt_symbols:
                    try:
                        volume = float(ticker.get('quoteVolume', 0))
                        if volume >= 20000000:  # 进一步降低到2千万USDT
                            volume_candidates.append((symbol, volume))
                    except (ValueError, TypeError):
                        continue
            
            # 排序并返回前30个
            volume_candidates.sort(key=lambda x: x[1], reverse=True)
            additional_symbols = [symbol for symbol, _ in volume_candidates[:30]]
            
            self.logger.info(f"   🎯 额外候选: {len(additional_symbols)}个 (门槛2千万USDT)")
            return additional_symbols
            
        except Exception as e:
            self.logger.error(f"获取额外候选失败: {e}")
            return []
    
    async def analyze_coin_stage_and_flow(self, symbol: str) -> Optional[CoinAnalysis]:
        """
        分析币种的市场阶段和资金流入情况（大佬核心逻辑）
        增强版：添加多重容错和降级策略
        """
        try:
            # 🔧 修复：使用1小时K线数据（用于阶段判断），兼容新币种
            # 改为1h数据，既能支持新币种，又符合15分钟交易策略
            klines_response = await self.api_client_manager.get_klines(symbol, '1h', limit=500)
            
            # 第一重检查：API响应
            if not klines_response or not klines_response.success:
                self.logger.warning(f"❌ {symbol} 1h数据请求失败，尝试降级到15分钟数据")
                # 降级策略：使用15分钟数据  
                klines_response = await self.api_client_manager.get_klines(symbol, '15m', limit=800)
                
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
            
            # 🔧 新增：数据有效性验证（过滤异常币种）
            is_valid, reason = self.validate_coin_data(klines, symbol)
            if not is_valid:
                self.logger.warning(f"❌ 排除 {symbol}: {reason}")
                return None
                
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
    
    async def detect_market_sentiment(self) -> str:
        """
        检测市场整体情绪
        通过分析主流币种的价格趋势判断市场环境
        """
        try:
            # 分析主流币种的短期趋势
            mainstream_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
            bearish_count = 0
            bullish_count = 0
            
            for symbol in mainstream_symbols:
                try:
                    # 获取24小时价格变化
                    ticker_data = await self.api_client_manager.binance_client.get_ticker_24hr(symbol)
                    price_change_pct = float(ticker_data.get('priceChangePercent', 0))
                    
                    if price_change_pct < -2:  # 跌超2%
                        bearish_count += 1
                    elif price_change_pct > 2:  # 涨超2%
                        bullish_count += 1
                        
                except Exception:
                    continue
            
            # 判断市场情绪
            if bearish_count > bullish_count:
                return "BEARISH"
            elif bullish_count > bearish_count:
                return "BULLISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.warning(f"市场情绪检测失败: {e}")
            return "NEUTRAL"
    
    def calculate_comprehensive_score(self, stage: int, volume_ratio: float, money_flow_ratio: float) -> float:
        """
        计算综合评分
        """
        base_score = 50
        
        # 阶段评分
        stage_scores = {1: 85, 2: 80, 3: 85}  # 多空平衡：所有阶段都有机会
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