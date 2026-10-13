"""
scanner.py - 每日買賣訊號掃描器

功能：
1. 掃描自選股清單
2. 支援單一策略或多策略（自動綜合判斷）
3. 輸出報告（螢幕 + CSV + 文字檔）
4. 支援批次掃描
5. 共識判斷（多策略投票）

使用方式：
    from scanner import DailyScanner
    from strategy import BollingerBandsStrategy, GoldenCrossStrategy
    
    # 單一策略
    strategy_map = {
        "0700.HK": BollingerBandsStrategy(20, 2.0),
    }
    
    # 多策略
    strategy_map = {
        "0700.HK": [
            BollingerBandsStrategy(20, 2.0),
            TrendFilterStrategy(RSIStrategy(14, 30, 70), 200),
        ],
    }
    
    scanner = DailyScanner(db=db, watchlist=["0700.HK"], strategy_map=strategy_map)
    results = scanner.scan_and_export()

作者: HK-Quant-Radar
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 一、每日掃描器
# ============================================================

class DailyScanner:
    """
    每日買賣訊號掃描器
    
    支援：
    - 單一策略
    - 多策略（自動綜合判斷）
    """
    
    def __init__(
        self,
        db,
        watchlist: List[str],
        strategy_map: Dict[str, Union[Any, List[Any]]],
        default_strategy: Any = None,
    ):
        """
        Args:
            db: MarketDatabase 實例
            watchlist: 自選股清單
            strategy_map: {ticker: strategy 或 [strategy1, strategy2, ...]}
            default_strategy: 預設策略
        """
        self.db = db
        self.watchlist = watchlist
        self.strategy_map = strategy_map
        self.default_strategy = default_strategy
    
    # ==================== 主掃描方法 ====================
    
    def scan(
        self,
        date: Optional[str] = None,
        lookback_days: int = 250,
    ) -> pd.DataFrame:
        """掃描所有自選股"""
        date = date or datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n🔄 開始掃描 {len(self.watchlist)} 支股票...")
        print(f"   日期: {date}")
        print(f"   回看: {lookback_days} 天")
        
        results = []
        
        for ticker in self.watchlist:
            try:
                result = self._scan_single(ticker, date, lookback_days)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"⚠️ {ticker} 掃描失敗: {e}")
                continue
        
        if not results:
            print("⚠️ 無掃描結果")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # 統計
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        hold_count = (df['signal'] == 0).sum()
        
        print(f"\n✅ 掃描完成:")
        print(f"   🟢 買入: {buy_count} 支")
        print(f"   🔴 賣出: {sell_count} 支")
        print(f"   ⚪ 持有: {hold_count} 支")
        
        return df
    
    def _scan_single(
        self,
        ticker: str,
        date: str,
        lookback_days: int,
    ) -> Optional[Dict]:
        """掃描單一股票（支援單一策略或多策略）"""
        
        # ============================================
        # 1. 取得策略
        # ============================================
        strategies = self.strategy_map.get(ticker, self.default_strategy)
        if strategies is None:
            return None
        
        # 統一轉為列表
        if not isinstance(strategies, list):
            strategies = [strategies]
        
        if not strategies:
            return None
        
        # ============================================
        # 2. 取得數據
        # ============================================
        df = self.db.get_ohlcv(ticker)
        
        if df.empty:
            return None
        
        df = df.tail(lookback_days)
        
        if len(df) < 60:
            return None
        
        # ============================================
        # 3. 對每個策略掃描（並快取結果）
        # ============================================
        strategy_results = []
        signals_cache = {}   # ✅ 快取，避免重複計算
        
        for strategy in strategies:
            try:
                signals = strategy.generate_signals(df)
                
                if 'signal' not in signals.columns:
                    continue
                
                latest_signal = int(signals['signal'].iloc[-1])
                
                strategy_results.append({
                    'name': strategy.name,
                    'signal': latest_signal,
                })
                
                # 快取
                signals_cache[strategy.name] = signals
            except Exception as e:
                logger.debug(f"   {ticker}: 策略 {strategy.name} 失敗: {e}")
                continue
        
        if not strategy_results:
            return None
        
        # ============================================
        # 4. 綜合判斷（多數決）
        # ============================================
        total = len(strategy_results)
        buy_count = sum(1 for s in strategy_results if s['signal'] == 1)
        sell_count = sum(1 for s in strategy_results if s['signal'] == -1)
        hold_count = total - buy_count - sell_count
        
        # 主訊號（多數決）
        if buy_count > sell_count and buy_count > total / 2:
            final_signal = 1
        elif sell_count > buy_count and sell_count > total / 2:
            final_signal = -1
        else:
            final_signal = 0
        
        # 共識標籤
        consensus = self._get_consensus(buy_count, sell_count, hold_count, total)
        
        # ============================================
        # 5. 組裝結果
        # ============================================
        latest_price = float(df['close'].iloc[-1])
        prev_price = float(df['close'].iloc[-2]) if len(df) > 1 else latest_price
        change_pct = (latest_price / prev_price - 1) * 100
        
        action = self._signal_to_action(final_signal)
        
        # 策略名稱
        if len(strategies) == 1:
            strategy_display = strategies[0].name
        else:
            strategy_display = f"{total} 個策略"
        
        result = {
            'ticker': ticker,
            'date': date,
            'signal': final_signal,
            'action': action,
            'consensus': consensus,
            'price': round(latest_price, 2),
            'change_pct': round(change_pct, 2),
            'strategy': strategy_display,
            'total_strategies': total,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'strategies': strategy_results,
        }
        
        # ============================================
        # 6. 額外指標（從快取取）
        # ============================================
        first_name = strategies[0].name
        if first_name in signals_cache:
            first_signals = signals_cache[first_name]
            if 'rsi' in first_signals.columns:
                result['rsi'] = round(float(first_signals['rsi'].iloc[-1]), 2)
            if 'macd' in first_signals.columns:
                result['macd'] = round(float(first_signals['macd'].iloc[-1]), 4)
        
        return result
    
    @staticmethod
    def _get_consensus(buy: int, sell: int, hold: int, total: int) -> str:
        """計算共識標籤"""
        if total == 1:
            return "單一策略"
        
        buy_pct = buy / total
        sell_pct = sell / total
        
        if buy_pct >= 0.8:
            return f"🟢🟢🟢 極強買入 ({buy}/{total})"
        elif buy_pct >= 0.6:
            return f"🟢🟢 強買入 ({buy}/{total})"
        elif buy > total / 2:
            return f"🟢 買入 ({buy}/{total})"
        elif sell_pct >= 0.8:
            return f"🔴🔴🔴 極強賣出 ({sell}/{total})"
        elif sell_pct >= 0.6:
            return f"🔴🔴 強賣出 ({sell}/{total})"
        elif sell > total / 2:
            return f"🔴 賣出 ({sell}/{total})"
        else:
            return f"⚪ 觀望 (買{buy}/賣{sell}/持{hold})"
    
    @staticmethod
    def _signal_to_action(signal: int) -> str:
        """訊號轉動作"""
        if signal == 1:
            return "🟢 買入"
        elif signal == -1:
            return "🔴 賣出"
        else:
            return "⚪ 持有"
    
    # ==================== 報告方法 ====================
    
    def print_report(self, results_df: pd.DataFrame):
        """列印報告"""
        if results_df.empty:
            print("⚠️ 無結果")
            return
        
        print("\n" + "="*110)
        print(f"📊 每日買賣訊號報告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*110)
        
        # ---------- 買入訊號 ----------
        buy_df = results_df[results_df['signal'] == 1]
        if not buy_df.empty:
            # ✅ 排序：先按買入數，再按漲跌
            sort_cols = []
            ascending = []
            if 'buy_count' in buy_df.columns:
                sort_cols.append('buy_count')
                ascending.append(False)
            if 'change_pct' in buy_df.columns:
                sort_cols.append('change_pct')
                ascending.append(False)
            
            if sort_cols:
                buy_df = buy_df.sort_values(sort_cols, ascending=ascending)
            
            print(f"\n🟢 買入訊號 ({len(buy_df)} 支):")
            print("-" * 110)
            print(f"{'代碼':<12} {'價格':>10} {'漲跌':>10} {'共識':<30} {'策略':<20}")
            print("-" * 110)
            for _, row in buy_df.iterrows():
                print(f"{row['ticker']:<12} {row['price']:>10.2f} "
                      f"{row['change_pct']:>9.2f}% "
                      f"{row.get('consensus', ''):<30} "
                      f"{row['strategy']:<20}")
        else:
            print(f"\n🟢 買入訊號 (0 支):")
            print("-" * 110)
            print("   （無）")
        
        # ---------- 賣出訊號 ----------
        sell_df = results_df[results_df['signal'] == -1]
        if not sell_df.empty:
            sort_cols = []
            ascending = []
            if 'sell_count' in sell_df.columns:
                sort_cols.append('sell_count')
                ascending.append(False)
            
            if sort_cols:
                sell_df = sell_df.sort_values(sort_cols, ascending=ascending)
            
            print(f"\n🔴 賣出訊號 ({len(sell_df)} 支):")
            print("-" * 110)
            print(f"{'代碼':<12} {'價格':>10} {'漲跌':>10} {'共識':<30} {'策略':<20}")
            print("-" * 110)
            for _, row in sell_df.iterrows():
                print(f"{row['ticker']:<12} {row['price']:>10.2f} "
                      f"{row['change_pct']:>9.2f}% "
                      f"{row.get('consensus', ''):<30} "
                      f"{row['strategy']:<20}")
        else:
            print(f"\n🔴 賣出訊號 (0 支):")
            print("-" * 110)
            print("   （無）")
        
        # ---------- 持有 ----------
        hold_df = results_df[results_df['signal'] == 0]
        print(f"\n⚪ 持有/觀望 ({len(hold_df)} 支):")
        if not hold_df.empty:
            tickers = hold_df['ticker'].tolist()
            for i in range(0, len(tickers), 5):
                print(f"   {', '.join(tickers[i:i+5])}")
        
        print("\n" + "="*110)
    
    # ==================== 導出方法 ====================
    
    def export(self, results_df: pd.DataFrame, output_dir="outputs"):
        """導出結果"""
        if results_df.empty:
            print("⚠️ 無結果可導出")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        today = datetime.now().strftime("%Y%m%d")
        
        # ---------- 導出 CSV ----------
        export_df = results_df.copy()
        export_df['strategies'] = export_df['strategies'].apply(
            lambda x: '; '.join([f"{s['name']}({s['signal']})" for s in x]) if isinstance(x, list) else ''
        )
        
        file_csv = output_dir / f"daily_signals_{today}.csv"
        export_df.to_csv(file_csv, index=False, encoding='utf-8-sig')
        print(f"✅ CSV: {file_csv}")
        
        # ---------- 導出文字報告 ----------
        file_txt = output_dir / f"daily_signals_{today}.txt"
        with open(file_txt, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"每日買賣訊號報告\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # 買入
            buy_df = results_df[results_df['signal'] == 1]
            f.write(f"🟢 買入訊號 ({len(buy_df)} 支):\n")
            f.write("-"*70 + "\n")
            for _, row in buy_df.iterrows():
                f.write(f"  {row['ticker']:<12} 價格={row['price']:>8.2f} "
                        f"({row['change_pct']:+.2f}%) "
                        f"共識={row.get('consensus', 'N/A')}\n")
            
            # 賣出
            sell_df = results_df[results_df['signal'] == -1]
            f.write(f"\n🔴 賣出訊號 ({len(sell_df)} 支):\n")
            f.write("-"*70 + "\n")
            for _, row in sell_df.iterrows():
                f.write(f"  {row['ticker']:<12} 價格={row['price']:>8.2f} "
                        f"({row['change_pct']:+.2f}%) "
                        f"共識={row.get('consensus', 'N/A')}\n")
            
            # 持有
            hold_df = results_df[results_df['signal'] == 0]
            f.write(f"\n⚪ 持有 ({len(hold_df)} 支):\n")
            f.write("-"*70 + "\n")
            for _, row in hold_df.iterrows():
                f.write(f"  {row['ticker']:<12} 價格={row['price']:>8.2f}\n")
        
        print(f"✅ 報告: {file_txt}")
    
    # ==================== 便捷方法 ====================
    
    def scan_and_export(self, **kwargs) -> pd.DataFrame:
        """掃描 + 列印 + 導出"""
        results = self.scan(**kwargs)
        if not results.empty:
            self.print_report(results)
            self.export(results)
        return results


# ============================================================
# 二、輔助函數
# ============================================================

def build_strategy_map_from_backtest(
    backtest_results: pd.DataFrame,
    strategy_pool: Dict[str, Any],
    top_n: int = 1,
    strategy_map_lists: bool = False,
) -> Dict[str, Union[Any, List[Any]]]:
    """
    從回測結果建立「每支股票最佳策略」對應
    
    Args:
        backtest_results: 來自 Cell 6 的多股票測試結果
        strategy_pool: {strategy_name: strategy_instance} 策略池
        top_n: 每支股票選前 N 個策略
        strategy_map_lists: 是否回傳「策略列表」
        
    Returns:
        {ticker: strategy} 或 {ticker: [strategy1, ...]}
    """
    strategy_map = {}
    
    for ticker in backtest_results['ticker'].unique():
        ticker_df = backtest_results[backtest_results['ticker'] == ticker]
        top_n_df = ticker_df.nlargest(top_n, 'sharpe')
        
        strategies = []
        for _, row in top_n_df.iterrows():
            strategy_name = row['strategy']
            if strategy_name in strategy_pool:
                strategies.append(strategy_pool[strategy_name])
        
        if not strategies:
            continue
        
        if strategy_map_lists or top_n > 1:
            strategy_map[ticker] = strategies
        else:
            strategy_map[ticker] = strategies[0]
    
    return strategy_map


# ============================================================
# 三、測試區塊
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("🧪 scanner.py 測試")
    print("="*70)
    
    # 檢查依賴
    try:
        from data_engine import MarketDatabase
        from strategy import (
            BollingerBandsStrategy,
            GoldenCrossStrategy,
            RSIStrategy,
            TrendFilterStrategy,
        )
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        exit(1)
    
    # 建立測試資料庫
    print("\n📊 建立測試資料庫...")
    db = MarketDatabase("test_scanner.db")
    
    TEST_WATCHLIST = ["0700.HK", "9988.HK", "0005.HK"]
    
    stats = db.sync_data(
        tickers=TEST_WATCHLIST,
        max_workers=3,
        full_download_years=1,
    )
    print(f"   同步結果: {stats}")
    
    # 測試 1：單一策略
    print("\n" + "="*70)
    print("📡 測試 1: 單一策略")
    print("="*70)
    
    single_strategy_map = {
        "0700.HK": BollingerBandsStrategy(20, 2.0),
        "9988.HK": BollingerBandsStrategy(20, 2.0),
        "0005.HK": TrendFilterStrategy(GoldenCrossStrategy(20, 60), 200),
    }
    
    scanner1 = DailyScanner(
        db=db,
        watchlist=TEST_WATCHLIST,
        strategy_map=single_strategy_map,
    )
    
    results1 = scanner1.scan(lookback_days=250)
    scanner1.print_report(results1)
    
    # 測試 2：多策略
    print("\n" + "="*70)
    print("📡 測試 2: 多策略")
    print("="*70)
    
    multi_strategy_map = {
        "0700.HK": [
            BollingerBandsStrategy(20, 2.0),
            TrendFilterStrategy(RSIStrategy(14, 30, 70), 200),
        ],
        "9988.HK": [
            BollingerBandsStrategy(20, 2.0),
            RSIStrategy(14, 30, 70),
        ],
        "0005.HK": [
            TrendFilterStrategy(GoldenCrossStrategy(20, 60), 200),
            BollingerBandsStrategy(20, 2.0),
            RSIStrategy(14, 30, 70),
        ],
    }
    
    scanner2 = DailyScanner(
        db=db,
        watchlist=TEST_WATCHLIST,
        strategy_map=multi_strategy_map,
    )
    
    results2 = scanner2.scan(lookback_days=250)
    scanner2.print_report(results2)
    scanner2.export(results2, output_dir="test_outputs")
    
    # 顯示完整結果
    print("\n📊 多策略完整結果:")
    display_cols = ['ticker', 'price', 'change_pct', 'consensus', 
                   'total_strategies', 'buy_count', 'sell_count', 'hold_count']
    display_cols = [c for c in display_cols if c in results2.columns]
    print(results2[display_cols].to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ 測試完成！")
    print("="*70)