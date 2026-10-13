# 📊 HSI Analysis - 港股量化分析系統

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

一個專為香港恆生指數成份股設計的量化分析系統。

## ✨ 功能

- 📥 自動同步港股數據（yfinance）
- 📈 20+ 技術指標（SMA、EMA、RSI、MACD、Bollinger）
- 🎯 8 種交易策略
- 🔬 多策略回測引擎
- 📡 每日買賣訊號掃描
- 📊 視覺化報告
- 🔍 個股基本面分析

## 🚀 快速開始

### 1. 安裝

```bash
git clone https://github.com/Sam7Ho/HSI_analysis.git
cd HSI_analysis

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. 執行

```bash
jupyter notebook main.ipynb
```

按順序執行 Cell 0 → 7。

## 📁 專案結構

```
HSI_analysis/
├── main.ipynb              # 主程式
├── data_engine.py          # 資料庫引擎
├── alpha_library.py        # 技術指標庫
├── strategy.py             # 策略庫
├── backtester.py           # 回測引擎
├── scanner.py              # 訊號掃描器
├── fundamental_analyzer.py # 基本面分析
├── requirements.txt        # 依賴清單
└── README.md               # 本文件
```

## 📊 使用範例

### 範例 1：同步數據

```python
from data_engine import MarketDatabase

db = MarketDatabase("hk_trading.db")
stats = db.sync_data(["0700.HK", "9988.HK"], full_download_years=5)
print(stats)
```

### 範例 2：計算指標

```python
from alpha_library import calculate_rsi, calculate_macd

rsi = calculate_rsi(df['close'], 14)
macd, signal, hist = calculate_macd(df['close'])
```

### 範例 3：單股回測

```python
from strategy import BollingerBandsStrategy
from backtester import Backtester

strategy = BollingerBandsStrategy(20, 2.0)
bt = Backtester(initial_capital=1_000_000)
result = bt.run(df, strategy)
print(result.summary())
```

### 範例 4：每日掃描

```python
from scanner import DailyScanner

scanner = DailyScanner(db=db, watchlist=WATCHLIST, strategy_map=strategy_map)
results = scanner.scan_and_export()
```

### 範例 5：個股基本面分析

```python
# 獨立工具，不影響其他檔案
%run fundamental_analyzer.py

# 單股分析
analyze_stock("0700.HK")

# 多股比較
compare_stocks(["0700.HK", "9988.HK", "0005.HK"])


## 📚 策略說明

| 策略 | 買入 | 賣出 | 適用 |
|------|------|------|------|
| **Bollinger** | 觸及下軌 | 觸及上軌 | 震盪 |
| **GoldenCross** | 快線上穿慢線 | 快線下穿慢線 | 趨勢 |
| **MACD** | MACD 上穿信號 | MACD 下穿信號 | 趨勢 |
| **RSI** | RSI < 30 | RSI > 70 | 震盪 |
| **Breakout** | 突破新高 | 跌破新低 | 趨勢 |


## 🛠️ 技術棧

- Python 3.10+
- Pandas / NumPy
- yfinance（數據源）
- SQLite（資料庫）
- Matplotlib / Seaborn（視覺化）
- Jupyter Notebook

## ⚠️ 免責聲明

**本專案僅供學習和研究使用，不構成任何投資建議。**

- 所有數據來自公開來源（yfinance）
- 歷史表現不代表未來收益
- 使用本工具的投資風險由使用者自行承擔
- 作者不對任何投資損失負責

## 📄 授權

MIT License - 詳見 [LICENSE](LICENSE)

## 🙏 致謝

- [yfinance](https://github.com/ranaroussi/yfinance)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

## 📬 聯絡

- GitHub: [@Sam7Ho](https://github.com/Sam7Ho)