"""
strategy.py - 策略庫

架構：
1. BaseStrategy（抽象基底類別）
2. 單一指標策略（GoldenCross, RSI, MACD, Bollinger）
3. 組合策略（MultiIndicator）
4. 策略註冊表（方便呼叫）

使用方式：
    from strategy import GoldenCrossStrategy
    
    strategy = GoldenCrossStrategy(fast_period=20, slow_period=60)
    signals = strategy.generate_signals(df)
    
    # signals 是一個 DataFrame，含 'signal' 欄位：
    #   1  = 買入
    #   -1 = 賣出
    #   0  = 持有/觀望

作者: HK-Quant-Radar
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from alpha_library import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
)


# ============================================================
# 一、策略基底類別
# ============================================================

class BaseStrategy(ABC):
    """
    策略基底類別
    
    所有策略必須繼承此類別並實作 `generate_signals`。
    子類別會自動獲得 `__repr__` 和 `get_params` 方法。
    """
    
    def __init__(self, name: str = "Base"):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        產生買賣信號
        
        Args:
            data: 價格數據（必須含 open, high, low, close, volume 欄位）
        
        Returns:
            DataFrame，至少含 'signal' 欄位：
                1  = 買入
                -1 = 賣出
                0  = 持有/觀望
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """取得策略參數"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'name'
        }
    
    def __repr__(self) -> str:
        params = self.get_params()
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_str})"


# ============================================================
# 二、趨勢策略
# ============================================================

class GoldenCrossStrategy(BaseStrategy):
    """
    黃金交叉策略
    
    邏輯：
    - 快線 > 慢線 → 買入（黃金交叉）
    - 快線 < 慢線 → 賣出（死亡交叉）
    
    適用：趨勢明顯的市場
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 60):
        super().__init__(name=f"GoldenCross({fast_period},{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ma_fast = calculate_sma(data['close'], self.fast_period)
        ma_slow = calculate_sma(data['close'], self.slow_period)
        
        # 狀態：1=快線在上，-1=快線在下
        state = pd.Series(0, index=data.index)
        state[ma_fast > ma_slow] = 1
        state[ma_fast < ma_slow] = -1
        
        # 信號：只在「狀態改變」時發出
        signal = pd.Series(0, index=data.index)
        signal[(state == 1) & (state.shift(1) == -1)] = 1    # 黃金交叉
        signal[(state == -1) & (state.shift(1) == 1)] = -1   # 死亡交叉
        
        return pd.DataFrame({
            'signal': signal,
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
        })


class MACDStrategy(BaseStrategy):
    """
    MACD 策略
    
    邏輯：
    - MACD 線 > 信號線 → 買入
    - MACD 線 < 信號線 → 賣出
    
    適用：中長期趨勢
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name=f"MACD({fast},{slow},{signal})")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        macd_line, signal_line, histogram = calculate_macd(
            data['close'], self.fast, self.slow, self.signal
        )
        
        # 狀態
        state = pd.Series(0, index=data.index)
        state[macd_line > signal_line] = 1
        state[macd_line < signal_line] = -1
        
        # 信號
        signal = pd.Series(0, index=data.index)
        signal[(state == 1) & (state.shift(1) == -1)] = 1
        signal[(state == -1) & (state.shift(1) == 1)] = -1
        
        return pd.DataFrame({
            'signal': signal,
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram,
        })


# ============================================================
# 三、動量策略
# ============================================================

class RSIStrategy(BaseStrategy):
    """
    RSI 超買超賣策略
    
    邏輯：
    - RSI < 超賣線（預設 30）→ 買入
    - RSI > 超買線（預設 70）→ 賣出
    
    適用：震盪市場
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name=f"RSI({period},{oversold},{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi = calculate_rsi(data['close'], self.period)
        
        # 信號
        signal = pd.Series(0, index=data.index)
        signal[rsi < self.oversold] = 1
        signal[rsi > self.overbought] = -1
        
        return pd.DataFrame({
            'signal': signal,
            'rsi': rsi,
        })


# ============================================================
# 四、波動策略
# ============================================================

class BollingerBandsStrategy(BaseStrategy):
    """
    布林通道策略
    
    邏輯：
    - 價格觸及下軌 → 買入（超賣反彈）
    - 價格觸及上軌 → 賣出（超買回落）
    
    適用：震盪市場
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(name=f"Bollinger({period},{std_dev})")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        upper, middle, lower = calculate_bollinger_bands(
            data['close'], self.period, self.std_dev
        )
        
        # 信號
        signal = pd.Series(0, index=data.index)
        signal[data['close'] <= lower] = 1   # 觸及下軌
        signal[data['close'] >= upper] = -1  # 觸及上軌
        
        return pd.DataFrame({
            'signal': signal,
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
        })


class BreakoutStrategy(BaseStrategy):
    """
    突破策略（唐奇安通道）
    
    邏輯：
    - 價格突破 N 日新高 → 買入
    - 價格跌破 N 日新低 → 賣出
    
    適用：趨勢市場
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__(name=f"Breakout({lookback})")
        self.lookback = lookback
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        high_roll = data['high'].rolling(self.lookback).max()
        low_roll = data['low'].rolling(self.lookback).min()
        
        # 信號
        signal = pd.Series(0, index=data.index)
        signal[data['close'] > high_roll.shift(1)] = 1   # 突破新高
        signal[data['close'] < low_roll.shift(1)] = -1   # 跌破新低
        
        return pd.DataFrame({
            'signal': signal,
            'high_channel': high_roll,
            'low_channel': low_roll,
        })


# ============================================================
# 五、組合策略
# ============================================================

class MultiIndicatorStrategy(BaseStrategy):
    """
    多指標組合策略
    
    邏輯（AND 條件）：
    - 買入：所有子策略都發出買入信號
    - 賣出：所有子策略都發出賣出信號
    
    優點：信號更嚴格，減少假信號
    缺點：可能錯過機會
    """
    
    def __init__(self, strategies: List[BaseStrategy], mode: str = "and"):
        """
        Args:
            strategies: 子策略清單
            mode: "and"（全部同意）或 "or"（任一同意）
        """
        names = " + ".join(s.name for s in strategies)
        super().__init__(name=f"Multi({mode}): {names}")
        self.strategies = strategies
        self.mode = mode
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 收集所有策略的信號
        all_signals = []
        for strategy in self.strategies:
            sig = strategy.generate_signals(data)
            all_signals.append(sig['signal'])
        
        signals_df = pd.concat(all_signals, axis=1)
        
        # 組合
        if self.mode == "and":
            # 全部同意買入 → 1
            buy_signal = (signals_df == 1).all(axis=1)
            sell_signal = (signals_df == -1).all(axis=1)
        else:  # "or"
            # 任一同意買入 → 1
            buy_signal = (signals_df == 1).any(axis=1)
            sell_signal = (signals_df == -1).any(axis=1)
        
        signal = pd.Series(0, index=data.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        
        return pd.DataFrame({
            'signal': signal,
            'agreement': (signals_df == 1).sum(axis=1),  # 同意買入的策略數
        })


class TrendFilterStrategy(BaseStrategy):
    """
    趨勢過濾策略
    
    邏輯：
    - 只在長期趨勢向上時才買入（避免逆勢）
    - 買入：短期策略發出買入信號 + 長期趨勢向上
    - 賣出：短期策略發出賣出信號
    
    優點：過濾逆勢信號
    """
    
    def __init__(self, entry_strategy: BaseStrategy, trend_period: int = 200):
        super().__init__(name=f"TrendFilter({entry_strategy.name},{trend_period})")
        self.entry_strategy = entry_strategy
        self.trend_period = trend_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 長期趨勢
        ma_trend = calculate_sma(data['close'], self.trend_period)
        trend_up = data['close'] > ma_trend
        
        # 入場信號
        entry_signals = self.entry_strategy.generate_signals(data)
        
        # 過濾
        signal = pd.Series(0, index=data.index)
        signal[(entry_signals['signal'] == 1) & trend_up] = 1   # 買入（趨勢向上）
        signal[entry_signals['signal'] == -1] = -1               # 賣出（不受趨勢限制）
        
        return pd.DataFrame({
            'signal': signal,
            'trend_up': trend_up,
            'entry_signal': entry_signals['signal'],
        })


# ============================================================
# 六、策略註冊表
# ============================================================

STRATEGIES = {
    "golden_cross": GoldenCrossStrategy,
    "macd": MACDStrategy,
    "rsi": RSIStrategy,
    "bollinger": BollingerBandsStrategy,
    "breakout": BreakoutStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    根據名稱取得策略實例
    
    Args:
        name: 策略名稱（見 STRATEGIES）
        **kwargs: 策略參數
    
    Returns:
        策略實例
    
    Example:
        >>> strategy = get_strategy("golden_cross", fast_period=20, slow_period=60)
    """
    if name not in STRATEGIES:
        raise ValueError(f"未知策略: {name}，可選: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


# ============================================================
# 七、測試區塊
# ============================================================

if __name__ == "__main__":
    """
    獨立測試 strategy.py
    執行: python strategy.py
    """
    print("="*70)
    print("🧪 strategy.py 測試")
    print("="*70)
    
    # ========================================
    # 1. 建立測試數據
    # ========================================
    print("\n📊 建立測試數據...")
    
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    
    # 模擬有趨勢的價格
    trend = np.linspace(0, 20, 300)
    noise = np.cumsum(np.random.randn(300) * 1.5)
    price = pd.Series(100 + trend + noise, index=dates)
    
    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(300) * 0.5),
        'low': price - np.abs(np.random.randn(300) * 0.5),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, 300),
    })
    
    print(f"   {len(df)} 筆數據")
    print(f"   價格範圍: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    
    # ========================================
    # 2. 測試單一策略
    # ========================================
    print("\n" + "="*70)
    print("📌 測試單一策略")
    print("="*70)
    
    strategies = [
        GoldenCrossStrategy(fast_period=20, slow_period=60),
        MACDStrategy(),
        RSIStrategy(period=14, oversold=30, overbought=70),
        BollingerBandsStrategy(),
        BreakoutStrategy(lookback=20),
    ]
    
    for strategy in strategies:
        print(f"\n🔹 {strategy.name}")
        
        signals = strategy.generate_signals(df)
        
        buy_count = (signals['signal'] == 1).sum()
        sell_count = (signals['signal'] == -1).sum()
        hold_count = (signals['signal'] == 0).sum()
        
        print(f"   買入信號: {buy_count}")
        print(f"   賣出信號: {sell_count}")
        print(f"   持有信號: {hold_count}")
        
        # 顯示最近 3 個信號
        recent_signals = signals[signals['signal'] != 0].tail(3)
        if not recent_signals.empty:
            print(f"   最近信號:")
            for idx, row in recent_signals.iterrows():
                action = "買入" if row['signal'] == 1 else "賣出"
                print(f"      {idx.strftime('%Y-%m-%d')}: {action}")
    
    # ========================================
    # 3. 測試組合策略
    # ========================================
    print("\n" + "="*70)
    print("📌 測試組合策略")
    print("="*70)
    
    # 組合策略
    combo = MultiIndicatorStrategy(
        strategies=[
            GoldenCrossStrategy(20, 60),
            RSIStrategy(14, 30, 70),
        ],
        mode="and"
    )
    
    print(f"\n🔹 {combo.name}")
    signals = combo.generate_signals(df)
    print(f"   買入信號: {(signals['signal'] == 1).sum()}")
    print(f"   賣出信號: {(signals['signal'] == -1).sum()}")
    
    # 趨勢過濾
    filtered = TrendFilterStrategy(
        entry_strategy=RSIStrategy(14, 30, 70),
        trend_period=200
    )
    
    print(f"\n🔹 {filtered.name}")
    signals = filtered.generate_signals(df)
    print(f"   買入信號: {(signals['signal'] == 1).sum()}")
    print(f"   賣出信號: {(signals['signal'] == -1).sum()}")
    
    # ========================================
    # 4. 測試策略註冊表
    # ========================================
    print("\n" + "="*70)
    print("📌 測試策略註冊表")
    print("="*70)
    
    print(f"\n可用策略: {list(STRATEGIES.keys())}")
    
    strategy = get_strategy("golden_cross", fast_period=10, slow_period=30)
    print(f"\n取得策略: {strategy}")
    
    # ========================================
    # 5. 顯示信號分佈
    # ========================================
    print("\n" + "="*70)
    print("📌 信號分佈比較")
    print("="*70)
    
    print(f"\n{'策略':<35} {'買入':>6} {'賣出':>6} {'持有':>6}")
    print("-" * 70)
    
    for strategy in strategies:
        signals = strategy.generate_signals(df)
        buy = (signals['signal'] == 1).sum()
        sell = (signals['signal'] == -1).sum()
        hold = (signals['signal'] == 0).sum()
        print(f"{strategy.name:<35} {buy:>6} {sell:>6} {hold:>6}")
    
    print("\n" + "="*70)
    print("✅ 所有測試完成！")
    print("="*70)