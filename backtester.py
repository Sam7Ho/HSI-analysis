"""
backtester.py - 回測引擎

功能：
1. 接受策略和價格數據
2. 模擬真實交易（手續費、整股）
3. 計算績效指標（報酬、風險、交易統計）
4. 產生完整的回測報告

架構：
- Backtester：單股票回測
- PortfolioBacktester：多股票組合回測
- Trade：交易記錄
- BacktestResult：回測結果

使用方式：
    from backtester import Backtester
    from strategy import GoldenCrossStrategy
    
    strategy = GoldenCrossStrategy(20, 60)
    bt = Backtester(initial_capital=1_000_000)
    result = bt.run(df, strategy)
    
    print(result.summary())

作者: HK-Quant-Radar
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np


# ============================================================
# 一、資料結構
# ============================================================

@dataclass
class Trade:
    """單筆交易記錄"""
    date: pd.Timestamp
    action: str          # 'BUY' 或 'SELL'
    price: float
    shares: int
    amount: float        # 交易金額
    commission: float    # 手續費
    cash_after: float    # 交易後現金
    pnl: Optional[float] = None  # 損益（僅賣出時）

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'action': self.action,
            'price': round(self.price, 2),
            'shares': self.shares,
            'amount': round(self.amount, 2),
            'commission': round(self.commission, 2),
            'cash_after': round(self.cash_after, 2),
            'pnl': round(self.pnl, 2) if self.pnl is not None else None,
        }


@dataclass
class BacktestResult:
    """回測結果"""
    strategy_name: str
    initial_capital: float
    final_capital: float
    equity: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    signals: pd.DataFrame

    def summary(self) -> str:
        """列印回測摘要"""
        m = self.metrics
        lines = [
            "="*70,
            f"📊 回測報告: {self.strategy_name}",
            "="*70,
            "",
            "💰 資金表現:",
            f"   初始資金: {self.initial_capital:>15,.2f}",
            f"   最終資金: {self.final_capital:>15,.2f}",
            f"   總報酬:   {m['total_return']*100:>14.2f}%",
            f"   年化報酬: {m['annual_return']*100:>14.2f}%",
            "",
            "📈 風險指標:",
            f"   年化波動: {m['annual_volatility']*100:>14.2f}%",
            f"   Sharpe:   {m['sharpe_ratio']:>15.2f}",
            f"   Sortino:  {m['sortino_ratio']:>15.2f}",
            f"   最大回撤: {m['max_drawdown']*100:>14.2f}%",
            f"   Calmar:   {m['calmar_ratio']:>15.2f}",
            "",
            "📉 交易統計:",
            f"   總交易數: {m['total_trades']:>15.0f}",
            f"   勝率:     {m['win_rate']*100:>14.2f}%",
            f"   平均獲利: {m['avg_win']*100:>14.2f}%",
            f"   平均虧損: {m['avg_loss']*100:>14.2f}%",
            f"   盈虧比:   {m['profit_factor']:>15.2f}",
            "",
            "="*70,
        ]
        return "\n".join(lines)


# ============================================================
# 二、回測引擎
# ============================================================

class Backtester:
    """
    單股票回測引擎
    
    模擬真實交易：
    - 手續費（預設 0.1%）
    - 整股交易（不能買 0.5 股）
    - 買入用現金、賣出換現金
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,      # 0.1%
        slippage: float = 0.0,          # 滑價（0.0 = 無）
        allow_short: bool = False,      # 是否允許做空
    ):
        """
        Args:
            initial_capital: 初始資金
            commission: 手續費率（0.001 = 0.1%）
            slippage: 滑價（0.001 = 0.1%）
            allow_short: 是否允許做空
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.allow_short = allow_short
    
    def run(self, data: pd.DataFrame, strategy) -> BacktestResult:
        """
        執行回測
        
        Args:
            data: 價格數據（含 open, high, low, close, volume）
            strategy: 策略實例
            
        Returns:
            BacktestResult
        """
        # ---------- 驗證數據 ----------
        if data.empty:
            raise ValueError("數據為空")
        
        required_cols = ['close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少欄位: {col}")
        
        # ---------- 產生信號 ----------
        signals = strategy.generate_signals(data)
        
        if 'signal' not in signals.columns:
            raise ValueError("策略必須產生 'signal' 欄位")
        
        # ---------- 模擬交易 ----------
        equity, trades = self._simulate_trading(data, signals['signal'])
        
        # ---------- 計算績效 ----------
        metrics = self._calculate_metrics(equity, trades)
        
        # ---------- 打包結果 ----------
        return BacktestResult(
            strategy_name=strategy.name,
            initial_capital=self.initial_capital,
            final_capital=equity.iloc[-1],
            equity=equity,
            trades=trades,
            metrics=metrics,
            signals=signals,
        )
    
    def _simulate_trading(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> Tuple[pd.Series, List[Trade]]:
        """
        模擬交易
        
        狀態機：
        - 空手（position=0）+ 買入信號 → 買入
        - 持倉（position=1）+ 賣出信號 → 賣出
        - 其他 → 持有
        """
        # ---------- 初始化 ----------
        cash = self.initial_capital
        shares = 0
        position = 0  # 0=空手, 1=持倉
        buy_price = 0.0  # 記錄買入價（計算損益）
        
        equity = pd.Series(index=data.index, dtype=float)
        trades: List[Trade] = []
        
        # ---------- 逐日模擬 ----------
        for i, (date, row) in enumerate(data.iterrows()):
            price = float(row['close'])
            signal = int(signals.iloc[i]) if pd.notna(signals.iloc[i]) else 0
            
            # ----- 買入 -----
            if signal == 1 and position == 0:
                # 考慮滑價
                exec_price = price * (1 + self.slippage)
                
                # 計算可買股數（整股）
                max_shares = int(cash / (exec_price * (1 + self.commission)))
                
                if max_shares > 0:
                    cost = max_shares * exec_price
                    comm = cost * self.commission
                    total_cost = cost + comm
                    
                    cash -= total_cost
                    shares = max_shares
                    position = 1
                    buy_price = exec_price
                    
                    trades.append(Trade(
                        date=date,
                        action='BUY',
                        price=exec_price,
                        shares=max_shares,
                        amount=cost,
                        commission=comm,
                        cash_after=cash,
                    ))
            
            # ----- 賣出 -----
            elif signal == -1 and position == 1:
                # 考慮滑價
                exec_price = price * (1 - self.slippage)
                
                # 賣出所有持倉
                revenue = shares * exec_price
                comm = revenue * self.commission
                net_revenue = revenue - comm
                
                # 計算損益
                pnl = (exec_price - buy_price) * shares - comm
                
                cash += net_revenue
                shares = 0
                position = 0
                buy_price = 0.0
                
                trades.append(Trade(
                    date=date,
                    action='SELL',
                    price=exec_price,
                    shares=trades[-1].shares if trades else 0,
                    amount=revenue,
                    commission=comm,
                    cash_after=cash,
                    pnl=pnl,
                ))
            
            # ----- 記錄淨值 -----
            equity.iloc[i] = cash + shares * price
        
        return equity, trades
    
    def _calculate_metrics(
        self,
        equity: pd.Series,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """計算績效指標"""
        # ---------- 基本報酬 ----------
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        # ---------- 日報酬 ----------
        returns = equity.pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # ---------- 年化報酬 ----------
        n_days = len(equity)
        if n_days > 0:
            annual_return = (1 + total_return) ** (252 / n_days) - 1
        else:
            annual_return = 0.0
        
        # ---------- 年化波動 ----------
        annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # ---------- Sharpe Ratio ----------
        if annual_volatility > 0:
            sharpe_ratio = annual_return / annual_volatility
        else:
            sharpe_ratio = 0.0
        
        # ---------- Sortino Ratio（只考慮下行波動）----------
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0.0
        if downside_std > 0:
            sortino_ratio = annual_return / downside_std
        else:
            sortino_ratio = 0.0
        
        # ---------- 最大回撤 ----------
        rolling_max = equity.cummax()
        drawdown = (equity / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # ---------- Calmar Ratio ----------
        if abs(max_drawdown) > 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        # ---------- 交易統計 ----------
        sell_trades = [t for t in trades if t.action == 'SELL' and t.pnl is not None]
        
        if sell_trades:
            pnls = np.array([t.pnl for t in sell_trades])
            # 用「收益率」計算勝率
            returns_per_trade = []
            for i, t in enumerate(sell_trades):
                # 找到對應的買入交易
                buy_trade = None
                for j in range(len(trades) - 1, -1, -1):
                    if trades[j].action == 'BUY' and trades[j].date < t.date:
                        buy_trade = trades[j]
                        break
                
                if buy_trade:
                    trade_return = (t.price / buy_trade.price) - 1
                    returns_per_trade.append(trade_return)
            
            returns_per_trade = np.array(returns_per_trade) if returns_per_trade else np.array([])
            
            # 勝率
            if len(returns_per_trade) > 0:
                wins = returns_per_trade[returns_per_trade > 0]
                losses = returns_per_trade[returns_per_trade < 0]
                
                win_rate = len(wins) / len(returns_per_trade)
                avg_win = wins.mean() if len(wins) > 0 else 0.0
                avg_loss = losses.mean() if len(losses) > 0 else 0.0
                
                # 盈虧比
                total_wins = wins.sum() if len(wins) > 0 else 0.0
                total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            else:
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0
                profit_factor = 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
        }


# ============================================================
# 三、多股票組合回測
# ============================================================

class PortfolioBacktester:
    """
    多股票組合回測
    
    將資金平均分配到多支股票，分別回測後匯總。
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy
    ) -> Dict[str, BacktestResult]:
        """
        執行多股票回測
        
        Args:
            data_dict: Dict[ticker, DataFrame]
            strategy: 策略實例
            
        Returns:
            Dict[ticker, BacktestResult]
        """
        n_stocks = len(data_dict)
        if n_stocks == 0:
            return {}
        
        # 每支股票分配相同資金
        capital_per_stock = self.initial_capital / n_stocks
        
        results = {}
        for ticker, df in data_dict.items():
            bt = Backtester(
                initial_capital=capital_per_stock,
                commission=self.commission,
                slippage=self.slippage,
            )
            results[ticker] = bt.run(df, strategy)
        
        return results
    
    @staticmethod
    def aggregate(results: Dict[str, BacktestResult]) -> Dict[str, float]:
        """匯總多股票結果"""
        if not results:
            return {}
        
        total_return = np.mean([r.metrics['total_return'] for r in results.values()])
        avg_sharpe = np.mean([r.metrics['sharpe_ratio'] for r in results.values()])
        avg_drawdown = np.mean([r.metrics['max_drawdown'] for r in results.values()])
        avg_win_rate = np.mean([r.metrics['win_rate'] for r in results.values()])
        total_trades = sum(r.metrics['total_trades'] for r in results.values())
        
        return {
            'n_stocks': len(results),
            'avg_total_return': total_return,
            'avg_sharpe': avg_sharpe,
            'avg_max_drawdown': avg_drawdown,
            'avg_win_rate': avg_win_rate,
            'total_trades': total_trades,
        }


# ============================================================
# 四、測試區塊
# ============================================================

if __name__ == "__main__":
    """
    獨立測試 backtester.py
    執行: python backtester.py
    """
    print("="*70)
    print("🧪 backtester.py 測試")
    print("="*70)
    
    # ========================================
    # 1. 建立測試數據
    # ========================================
    print("\n📊 建立測試數據...")
    
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    
    # 模擬有趨勢的價格
    trend = np.linspace(0, 30, 500)
    noise = np.cumsum(np.random.randn(500) * 1.5)
    price = pd.Series(100 + trend + noise, index=dates)
    
    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(500) * 0.5),
        'low': price - np.abs(np.random.randn(500) * 0.5),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, 500),
    })
    
    print(f"   {len(df)} 筆數據")
    print(f"   價格範圍: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    print(f"   期間: {dates[0].date()} ~ {dates[-1].date()}")
    
    # ========================================
    # 2. 測試單一策略
    # ========================================
    print("\n" + "="*70)
    print("📌 測試: 黃金交叉策略")
    print("="*70)
    
    from strategy import GoldenCrossStrategy
    
    strategy = GoldenCrossStrategy(fast_period=20, slow_period=60)
    bt = Backtester(
        initial_capital=1_000_000,
        commission=0.001,
        slippage=0.0,
    )
    
    result = bt.run(df, strategy)
    print(result.summary())
    
    # ========================================
    # 3. 顯示交易記錄
    # ========================================
    print("\n📋 交易記錄（前 10 筆）:")
    print("-"*70)
    
    if result.trades:
        trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
        print(trades_df.head(10).to_string(index=False))
    else:
        print("   無交易")
    
    # ========================================
    # 4. 測試多個策略
    # ========================================
    print("\n" + "="*70)
    print("📌 測試: 多策略比較")
    print("="*70)
    
    from strategy import (
        GoldenCrossStrategy,
        MACDStrategy,
        RSIStrategy,
        BollingerBandsStrategy,
    )
    
    strategies = [
        GoldenCrossStrategy(20, 60),
        MACDStrategy(),
        RSIStrategy(14, 30, 70),
        BollingerBandsStrategy(20, 2.0),
    ]
    
    print(f"\n{'策略':<30} {'總報酬':>10} {'年化':>10} {'Sharpe':>8} {'回撤':>10} {'交易':>6}")
    print("-" * 80)
    
    for strategy in strategies:
        result = bt.run(df, strategy)
        m = result.metrics
        print(f"{strategy.name:<30} "
              f"{m['total_return']*100:>9.2f}% "
              f"{m['annual_return']*100:>9.2f}% "
              f"{m['sharpe_ratio']:>8.2f} "
              f"{m['max_drawdown']*100:>9.2f}% "
              f"{m['total_trades']:>6}")
    
    # ========================================
    # 5. 測試多股票組合
    # ========================================
    print("\n" + "="*70)
    print("📌 測試: 多股票組合回測")
    print("="*70)
    
    # 模擬 3 支股票的數據
    data_dict = {}
    for i, ticker in enumerate(["STOCK_A", "STOCK_B", "STOCK_C"]):
        np.random.seed(42 + i)
        trend = np.linspace(0, 30 - i*5, 500)
        noise = np.cumsum(np.random.randn(500) * 1.5)
        price = pd.Series(100 + trend + noise, index=dates)
        
        data_dict[ticker] = pd.DataFrame({
            'open': price,
            'high': price + np.abs(np.random.randn(500) * 0.5),
            'low': price - np.abs(np.random.randn(500) * 0.5),
            'close': price,
            'volume': np.random.randint(1000000, 5000000, 500),
        })
    
    portfolio_bt = PortfolioBacktester(
        initial_capital=3_000_000,
        commission=0.001,
    )
    
    portfolio_results = portfolio_bt.run(data_dict, strategy=GoldenCrossStrategy(20, 60))
    
    print(f"\n{'股票':<12} {'總報酬':>12} {'Sharpe':>10} {'交易數':>8}")
    print("-" * 50)
    
    for ticker, result in portfolio_results.items():
        m = result.metrics
        print(f"{ticker:<12} "
              f"{m['total_return']*100:>11.2f}% "
              f"{m['sharpe_ratio']:>10.2f} "
              f"{m['total_trades']:>8}")
    
    # 匯總
    aggregate = portfolio_bt.aggregate(portfolio_results)
    print(f"\n📊 組合匯總:")
    print(f"   股票數: {aggregate['n_stocks']}")
    print(f"   平均報酬: {aggregate['avg_total_return']*100:.2f}%")
    print(f"   平均 Sharpe: {aggregate['avg_sharpe']:.2f}")
    print(f"   平均回撤: {aggregate['avg_max_drawdown']*100:.2f}%")
    print(f"   總交易數: {aggregate['total_trades']}")
    
    print("\n" + "="*70)
    print("✅ 所有測試完成！")
    print("="*70)