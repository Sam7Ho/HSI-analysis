"""
alpha_library.py - 技術指標庫

分類：
1. 趨勢指標：SMA, EMA, MACD, ADX
2. 動量指標：RSI, Stochastic, Williams %R
3. 波動指標：Bollinger Bands, ATR, Keltner Channel
4. 量價指標：OBV, Volume MA, VWAP
5. 綜合指標：自訂組合

使用方式：
    from alpha_library import calculate_rsi, calculate_macd
    
    rsi = calculate_rsi(price, period=14)
    macd_line, signal_line, histogram = calculate_macd(price)

作者: HK-Quant-Radar
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union

# 型別別名
PriceData = Union[pd.Series, pd.DataFrame]


# ============================================================
# 一、趨勢指標（Trend Indicators）
# ============================================================

def calculate_sma(price: pd.Series, period: int = 20) -> pd.Series:
    """
    簡單移動平均（Simple Moving Average）
    
    Args:
        price: 價格序列
        period: 週期（預設 20）
    
    Returns:
        SMA 序列
    
    Example:
        >>> sma20 = calculate_sma(df['close'], 20)
    """
    return price.rolling(window=period, min_periods=1).mean()


def calculate_ema(price: pd.Series, period: int = 20) -> pd.Series:
    """
    指數移動平均（Exponential Moving Average）
    
    比 SMA 更靈敏，對近期價格賦予更高權重。
    
    Args:
        price: 價格序列
        period: 週期
    
    Returns:
        EMA 序列
    
    Example:
        >>> ema12 = calculate_ema(df['close'], 12)
    """
    return price.ewm(span=period, adjust=False, min_periods=1).mean()


def calculate_wma(price: pd.Series, period: int = 20) -> pd.Series:
    """
    加權移動平均（Weighted Moving Average）
    
    Args:
        price: 價格序列
        period: 週期
    
    Returns:
        WMA 序列
    """
    weights = np.arange(1, period + 1)
    return price.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def calculate_macd(
    price: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 指標（Moving Average Convergence Divergence）
    
    Args:
        price: 價格序列
        fast: 快線週期（預設 12）
        slow: 慢線週期（預設 26）
        signal: 信號線週期（預設 9）
    
    Returns:
        (macd_line, signal_line, histogram)
        - macd_line: MACD 線（快線 - 慢線）
        - signal_line: 信號線（MACD 線的 EMA）
        - histogram: 柱狀圖（MACD 線 - 信號線）
    
    Example:
        >>> macd, signal, hist = calculate_macd(df['close'])
    """
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=1).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    ADX 指標（Average Directional Index）
    
    衡量趨勢強度（不看方向）。
    - ADX > 25：強趨勢
    - ADX < 20：弱趨勢（盤整）
    
    Args:
        high, low, close: 價格序列
        period: 週期（預設 14）
    
    Returns:
        ADX 序列
    """
    # 真實波幅（True Range）
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    
    # 上升/下降動向
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index
    )
    
    # ATR
    atr = tr.ewm(span=period, adjust=False, min_periods=1).mean()
    
    # +DI, -DI
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False, min_periods=1).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False, min_periods=1).mean() / atr
    
    # DX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    
    # ADX（DX 的 EMA）
    adx = dx.ewm(span=period, adjust=False, min_periods=1).mean()
    
    return adx


# ============================================================
# 二、動量指標（Momentum Indicators）
# ============================================================

def calculate_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """
    相對強弱指標（Relative Strength Index）
    
    數值範圍：0-100
    - RSI > 70：超買
    - RSI < 30：超賣
    - RSI 40-60：中性
    
    Args:
        price: 價格序列
        period: 週期（預設 14）
    
    Returns:
        RSI 序列（0-100）
    
    Example:
        >>> rsi = calculate_rsi(df['close'], 14)
    """
    delta = price.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # 處理除以 0 的情況
    rsi = rsi.fillna(50)  # 資料不足時給中性值
    
    return rsi


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    隨機指標（Stochastic Oscillator）
    
    %K 與 %D 都在 0-100 之間
    - > 80：超買
    - < 20：超賣
    
    Args:
        high, low, close: 價格序列
        k_period: %K 週期（預設 14）
        d_period: %D 週期（預設 3）
        smooth_k: %K 平滑週期（預設 3）
    
    Returns:
        (k_line, d_line)
    """
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    
    # 未平滑的 %K
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    
    # 平滑 %K
    k_line = raw_k.rolling(window=smooth_k, min_periods=1).mean()
    
    # %D = %K 的移動平均
    d_line = k_line.rolling(window=d_period, min_periods=1).mean()
    
    return k_line, d_line


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Williams %R
    
    數值範圍：-100 到 0
    - > -20：超買
    - < -80：超賣
    
    Args:
        high, low, close: 價格序列
        period: 週期
    
    Returns:
        Williams %R 序列
    """
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    
    return wr


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    順勢指標（Commodity Channel Index）
    
    - CCI > 100：超買
    - CCI < -100：超賣
    
    Args:
        high, low, close: 價格序列
        period: 週期
    
    Returns:
        CCI 序列
    """
    tp = (high + low + close) / 3  # 典型價格
    sma_tp = tp.rolling(window=period, min_periods=1).mean()
    
    # 平均絕對偏差
    mad = tp.rolling(window=period, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    
    cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    
    return cci


def calculate_momentum(price: pd.Series, period: int = 10) -> pd.Series:
    """
    動量指標（Momentum）
    
    Momentum = 今日價格 - N 日前價格
    
    Args:
        price: 價格序列
        period: 週期
    
    Returns:
        Momentum 序列
    """
    return price - price.shift(period)


def calculate_roc(price: pd.Series, period: int = 10) -> pd.Series:
    """
    變動率（Rate of Change）
    
    ROC = (今日價格 - N 日前價格) / N 日前價格 × 100
    
    Args:
        price: 價格序列
        period: 週期
    
    Returns:
        ROC 序列（%）
    """
    return price.pct_change(periods=period) * 100


# ============================================================
# 三、波動指標（Volatility Indicators）
# ============================================================

def calculate_bollinger_bands(
    price: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林通道（Bollinger Bands）
    
    - 中軌：SMA
    - 上軌：SMA + N × 標準差
    - 下軌：SMA - N × 標準差
    
    價格觸及上下軌代表超買/超賣。
    
    Args:
        price: 價格序列
        period: 週期（預設 20）
        std_dev: 標準差倍數（預設 2.0）
    
    Returns:
        (upper, middle, lower)
    
    Example:
        >>> upper, middle, lower = calculate_bollinger_bands(df['close'])
    """
    middle = price.rolling(window=period, min_periods=1).mean()
    std = price.rolling(window=period, min_periods=1).std()
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return upper, middle, lower


def calculate_bb_width(
    price: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.Series:
    """
    布林通道寬度（Bollinger Band Width）
    
    衡量波動性：
    - 寬度擴大：波動增加
    - 寬度縮小：波動減少（可能突破）
    
    Args:
        price: 價格序列
        period: 週期
        std_dev: 標準差倍數
    
    Returns:
        Band Width 序列
    """
    upper, middle, lower = calculate_bollinger_bands(price, period, std_dev)
    return (upper - lower) / middle.replace(0, np.nan) * 100


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    平均真實波幅（Average True Range）
    
    衡量波動性，常用於設定停損。
    
    Args:
        high, low, close: 價格序列
        period: 週期
    
    Returns:
        ATR 序列
    """
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return atr


def calculate_keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_multiplier: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    肯特納通道（Keltner Channel）
    
    類似布林通道，但用 ATR 而非標準差。
    
    Args:
        high, low, close: 價格序列
        period: 週期
        atr_multiplier: ATR 倍數
    
    Returns:
        (upper, middle, lower)
    """
    middle = close.ewm(span=period, adjust=False, min_periods=1).mean()
    atr = calculate_atr(high, low, close, period)
    
    upper = middle + atr_multiplier * atr
    lower = middle - atr_multiplier * atr
    
    return upper, middle, lower


def calculate_historical_volatility(
    price: pd.Series,
    period: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    歷史波動率（Historical Volatility）
    
    Args:
        price: 價格序列
        period: 週期
        annualize: 是否年化（×√252）
    
    Returns:
        波動率序列（小數，例如 0.25 = 25%）
    """
    log_returns = np.log(price / price.shift())
    vol = log_returns.rolling(window=period, min_periods=1).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


# ============================================================
# 四、量價指標（Volume Indicators）
# ============================================================

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    能量潮（On-Balance Volume）
    
    累積成交量：
    - 價格上漲：+ 成交量
    - 價格下跌：- 成交量
    
    Args:
        close: 收盤價序列
        volume: 成交量序列
    
    Returns:
        OBV 序列
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    
    return obv


def calculate_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    成交量移動平均
    
    Args:
        volume: 成交量序列
        period: 週期
    
    Returns:
        Volume MA 序列
    """
    return volume.rolling(window=period, min_periods=1).mean()


def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    量比（Volume Ratio）
    
    量比 = 今日成交量 / N 日平均成交量
    - > 1.5：放量
    - < 0.5：縮量
    
    Args:
        volume: 成交量序列
        period: 週期
    
    Returns:
        量比序列
    """
    vol_ma = calculate_volume_ma(volume, period)
    return volume / vol_ma.replace(0, np.nan)


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    資金流量指標（Money Flow Index）
    
    類似 RSI，但考慮成交量。
    - MFI > 80：超買
    - MFI < 20：超賣
    
    Args:
        high, low, close, volume: 價格與成交量序列
        period: 週期
    
    Returns:
        MFI 序列（0-100）
    """
    tp = (high + low + close) / 3
    money_flow = tp * volume
    
    positive_flow = money_flow.where(tp > tp.shift(), 0.0)
    negative_flow = money_flow.where(tp < tp.shift(), 0.0)
    
    positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
    
    mfr = positive_mf / negative_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi


# ============================================================
# 五、綜合工具函數
# ============================================================

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    一次計算所有常用指標
    
    Args:
        df: 含 open, high, low, close, volume 的 DataFrame
    
    Returns:
        加上所有指標的 DataFrame
    
    Example:
        >>> df_with_indicators = calculate_all_indicators(df)
    """
    result = df.copy()
    
    # 趨勢
    result['sma_20'] = calculate_sma(df['close'], 20)
    result['sma_60'] = calculate_sma(df['close'], 60)
    result['ema_12'] = calculate_ema(df['close'], 12)
    result['ema_26'] = calculate_ema(df['close'], 26)
    
    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_hist'] = hist
    
    # RSI
    result['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # 布林通道
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    result['bb_upper'] = upper
    result['bb_middle'] = middle
    result['bb_lower'] = lower
    
    # ATR
    result['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    # KD
    k, d = calculate_stochastic(df['high'], df['low'], df['close'])
    result['kd_k'] = k
    result['kd_d'] = d
    
    # 量價
    result['obv'] = calculate_obv(df['close'], df['volume'])
    result['vol_ma_20'] = calculate_volume_ma(df['volume'], 20)
    
    return result


# ============================================================
# 六、測試區塊
# ============================================================

if __name__ == "__main__":
    """
    獨立測試 alpha_library.py
    執行: python alpha_library.py
    """
    print("="*70)
    print("🧪 alpha_library 測試")
    print("="*70)
    
    # 建立測試數據
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    price = pd.Series(
        100 + np.cumsum(np.random.randn(200) * 2),
        index=dates
    )
    high = price + np.abs(np.random.randn(200))
    low = price - np.abs(np.random.randn(200))
    volume = pd.Series(np.random.randint(1000000, 5000000, 200), index=dates)
    
    df = pd.DataFrame({
        'open': price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume,
    })
    
    print(f"\n📊 測試數據: {len(df)} 筆")
    print(df.head())
    
    # 測試各指標
    print("\n" + "-"*70)
    print("1. 趨勢指標")
    print("-"*70)
    sma = calculate_sma(df['close'], 20)
    ema = calculate_ema(df['close'], 12)
    macd, signal, hist = calculate_macd(df['close'])
    adx = calculate_adx(df['high'], df['low'], df['close'])
    print(f"   SMA(20) 最後值: {sma.iloc[-1]:.2f}")
    print(f"   EMA(12) 最後值: {ema.iloc[-1]:.2f}")
    print(f"   MACD 最後值: {macd.iloc[-1]:.4f}")
    print(f"   ADX 最後值: {adx.iloc[-1]:.2f}")
    
    print("\n" + "-"*70)
    print("2. 動量指標")
    print("-"*70)
    rsi = calculate_rsi(df['close'], 14)
    k, d = calculate_stochastic(df['high'], df['low'], df['close'])
    wr = calculate_williams_r(df['high'], df['low'], df['close'])
    cci = calculate_cci(df['high'], df['low'], df['close'])
    print(f"   RSI(14) 最後值: {rsi.iloc[-1]:.2f}")
    print(f"   KD 最後值: K={k.iloc[-1]:.2f}, D={d.iloc[-1]:.2f}")
    print(f"   Williams %R 最後值: {wr.iloc[-1]:.2f}")
    print(f"   CCI 最後值: {cci.iloc[-1]:.2f}")
    
    print("\n" + "-"*70)
    print("3. 波動指標")
    print("-"*70)
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    atr = calculate_atr(df['high'], df['low'], df['close'])
    hv = calculate_historical_volatility(df['close'])
    print(f"   布林上軌: {upper.iloc[-1]:.2f}")
    print(f"   布林中軌: {middle.iloc[-1]:.2f}")
    print(f"   布林下軌: {lower.iloc[-1]:.2f}")
    print(f"   ATR(14) 最後值: {atr.iloc[-1]:.2f}")
    print(f"   歷史波動率: {hv.iloc[-1]*100:.2f}%")
    
    print("\n" + "-"*70)
    print("4. 量價指標")
    print("-"*70)
    obv = calculate_obv(df['close'], df['volume'])
    vol_ma = calculate_volume_ma(df['volume'], 20)
    mfi = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
    print(f"   OBV 最後值: {obv.iloc[-1]:,.0f}")
    print(f"   Volume MA(20): {vol_ma.iloc[-1]:,.0f}")
    print(f"   MFI 最後值: {mfi.iloc[-1]:.2f}")
    
    print("\n" + "-"*70)
    print("5. 綜合指標（一次計算）")
    print("-"*70)
    df_with_indicators = calculate_all_indicators(df)
    print(f"   原始欄位: {df.shape[1]}")
    print(f"   加入指標後: {df_with_indicators.shape[1]} 個欄位")
    print(f"   新增欄位: {[c for c in df_with_indicators.columns if c not in df.columns]}")
    
    print("\n" + "="*70)
    print("✅ 所有測試完成！")
    print("="*70)