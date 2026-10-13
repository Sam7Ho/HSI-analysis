"""
fundamental_analyzer.py - 個股基本面分析工具

獨立工具：
- 不被其他檔案引用
- 純看個股基本面
- 不作篩選
- 不影響其他模組

使用方式：
    # 方法 1：直接執行
    python fundamental_analyzer.py
    
    # 方法 2：在 Jupyter 中
    %run fundamental_analyzer.py
    
    # 方法 3：導入使用
    from fundamental_analyzer import analyze_stock
    analyze_stock("0700.HK")

作者: HK-Quant-Radar
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


# ============================================================
# 一、核心分析函數
# ============================================================

def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    取得股票基本面數據
    
    Args:
        ticker: 股票代碼（如 "0700.HK"）
    
    Returns:
        基本面數據字典
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return {}
        
        return info
    
    except Exception as e:
        print(f"❌ 取得 {ticker} 數據失敗: {e}")
        return {}


def safe_get(info: Dict, key: str, default: Any = None) -> Any:
    """安全取得字典值"""
    value = info.get(key, default)
    
    if value is None or value == "N/A" or value == "":
        return default
    
    return value


def analyze_stock(ticker: str, show_raw: bool = False) -> Optional[Dict[str, Any]]:
    """
    分析個股基本面
    
    Args:
        ticker: 股票代碼
        show_raw: 是否顯示原始數據
    
    Returns:
        分析結果字典
    """
    print("=" * 70)
    print(f"📊 個股基本面分析: {ticker}")
    print("=" * 70)
    
    # 取得數據
    info = get_stock_info(ticker)
    
    if not info:
        print(f"❌ 無法取得 {ticker} 的數據")
        return None
    
    # ============================================
    # 1. 基本資訊
    # ============================================
    print("\n📌 基本資訊")
    print("-" * 70)
    
    company_name = safe_get(info, 'longName', safe_get(info, 'shortName', ticker))
    sector = safe_get(info, 'sector', 'N/A')
    industry = safe_get(info, 'industry', 'N/A')
    country = safe_get(info, 'country', 'N/A')
    website = safe_get(info, 'website', 'N/A')
    
    print(f"   公司名稱: {company_name}")
    print(f"   行業: {sector}")
    print(f"   子行業: {industry}")
    print(f"   國家: {country}")
    print(f"   網站: {website}")
    
    # ============================================
    # 2. 股價資訊
    # ============================================
    print("\n💰 股價資訊")
    print("-" * 70)
    
    current_price = safe_get(info, 'currentPrice', safe_get(info, 'regularMarketPrice', 0))
    previous_close = safe_get(info, 'previousClose', 0)
    open_price = safe_get(info, 'open', 0)
    day_high = safe_get(info, 'dayHigh', 0)
    day_low = safe_get(info, 'dayLow', 0)
    fifty_two_high = safe_get(info, 'fiftyTwoWeekHigh', 0)
    fifty_two_low = safe_get(info, 'fiftyTwoWeekLow', 0)
    
    if current_price and previous_close:
        change = current_price - previous_close
        change_pct = (change / previous_close) * 100
        change_str = f"{change:+.2f} ({change_pct:+.2f}%)"
    else:
        change_str = "N/A"
    
    print(f"   現價: {current_price}")
    print(f"   漲跌: {change_str}")
    print(f"   開盤: {open_price}")
    print(f"   最高: {day_high}")
    print(f"   最低: {day_low}")
    print(f"   52 週高: {fifty_two_high}")
    print(f"   52 週低: {fifty_two_low}")
    
    # ============================================
    # 3. 估值指標
    # ============================================
    print("\n📈 估值指標")
    print("-" * 70)
    
    market_cap = safe_get(info, 'marketCap', 0)
    pe = safe_get(info, 'trailingPE', safe_get(info, 'forwardPE', 0))
    pb = safe_get(info, 'priceToBook', 0)
    ps = safe_get(info, 'priceToSalesTrailing12Months', 0)
    peg = safe_get(info, 'pegRatio', 0)
    ev_ebitda = safe_get(info, 'enterpriseToEbitda', 0)
    
    if market_cap:
        market_cap_b = market_cap / 1e9
        print(f"   市值: {market_cap_b:.2f}B")
    
    print(f"   PE（本益比）: {pe}")
    print(f"   PB（股價淨值比）: {pb}")
    print(f"   PS（股價營收比）: {ps}")
    print(f"   PEG: {peg}")
    print(f"   EV/EBITDA: {ev_ebitda}")
    
    # ============================================
    # 4. 獲利能力
    # ============================================
    print("\n💵 獲利能力")
    print("-" * 70)
    
    roe = safe_get(info, 'returnOnEquity', 0)
    roa = safe_get(info, 'returnOnAssets', 0)
    profit_margin = safe_get(info, 'profitMargins', 0)
    operating_margin = safe_get(info, 'operatingMargins', 0)
    gross_margin = safe_get(info, 'grossMargins', 0)
    
    if roe:
        print(f"   ROE（股東權益報酬率）: {roe*100:.2f}%")
    if roa:
        print(f"   ROA（資產報酬率）: {roa*100:.2f}%")
    if profit_margin:
        print(f"   淨利率: {profit_margin*100:.2f}%")
    if operating_margin:
        print(f"   營業利益率: {operating_margin*100:.2f}%")
    if gross_margin:
        print(f"   毛利率: {gross_margin*100:.2f}%")
    
    # ============================================
    # 5. 成長指標
    # ============================================
    print("\n📊 成長指標")
    print("-" * 70)
    
    revenue_growth = safe_get(info, 'revenueGrowth', 0)
    earnings_growth = safe_get(info, 'earningsGrowth', 0)
    earnings_quarterly_growth = safe_get(info, 'earningsQuarterlyGrowth', 0)
    
    if revenue_growth:
        print(f"   營收成長（年）: {revenue_growth*100:.2f}%")
    if earnings_growth:
        print(f"   獲利成長（年）: {earnings_growth*100:.2f}%")
    if earnings_quarterly_growth:
        print(f"   獲利成長（季）: {earnings_quarterly_growth*100:.2f}%")
    
    # ============================================
    # 6. 財務健康
    # ============================================
    print("\n🏦 財務健康")
    print("-" * 70)
    
    total_cash = safe_get(info, 'totalCash', 0)
    total_debt = safe_get(info, 'totalDebt', 0)
    debt_to_equity = safe_get(info, 'debtToEquity', 0)
    current_ratio = safe_get(info, 'currentRatio', 0)
    quick_ratio = safe_get(info, 'quickRatio', 0)
    free_cash_flow = safe_get(info, 'freeCashflow', 0)
    operating_cash_flow = safe_get(info, 'operatingCashflow', 0)
    
    if total_cash:
        print(f"   現金: {total_cash/1e9:.2f}B")
    if total_debt:
        print(f"   負債: {total_debt/1e9:.2f}B")
    if debt_to_equity:
        print(f"   負債權益比: {debt_to_equity:.2f}")
    if current_ratio:
        print(f"   流動比率: {current_ratio:.2f}")
    if quick_ratio:
        print(f"   速動比率: {quick_ratio:.2f}")
    if free_cash_flow:
        print(f"   自由現金流: {free_cash_flow/1e9:.2f}B")
    if operating_cash_flow:
        print(f"   營運現金流: {operating_cash_flow/1e9:.2f}B")
    
    # ============================================
    # 7. 股息資訊
    # ============================================
    print("\n💸 股息資訊")
    print("-" * 70)
    
    dividend_yield = safe_get(info, 'dividendYield', 0)
    dividend_rate = safe_get(info, 'dividendRate', 0)
    payout_ratio = safe_get(info, 'payoutRatio', 0)
    ex_dividend_date = safe_get(info, 'exDividendDate', 0)
    
    if dividend_yield:
        print(f"   股息率: {dividend_yield:.2f}%")
    if dividend_rate:
        print(f"   每股股息: {dividend_rate}")
    if payout_ratio:
        print(f"   派息比率: {payout_ratio*100:.2f}%")
    if ex_dividend_date:
        try:
            date = datetime.fromtimestamp(ex_dividend_date).strftime('%Y-%m-%d')
            print(f"   除息日: {date}")
        except:
            pass
    
    # ============================================
    # 8. 風險指標
    # ============================================
    print("\n⚠️ 風險指標")
    print("-" * 70)
    
    beta = safe_get(info, 'beta', 0)
    
    if beta:
        print(f"   Beta: {beta:.2f}")
        if beta < 0.8:
            print("   → 波動小於市場（防禦型）")
        elif beta < 1.2:
            print("   → 波動接近市場（穩健型）")
        else:
            print("   → 波動大於市場（進取型）")
    
    # ============================================
    # 9. 分析師評級
    # ============================================
    print("\n🎯 分析師評級")
    print("-" * 70)
    
    recommendation = safe_get(info, 'recommendationKey', 'N/A')
    target_mean = safe_get(info, 'targetMeanPrice', 0)
    target_high = safe_get(info, 'targetHighPrice', 0)
    target_low = safe_get(info, 'targetLowPrice', 0)
    num_analysts = safe_get(info, 'numberOfAnalystOpinions', 0)
    
    print(f"   評級: {recommendation}")
    
    if target_mean and current_price:
        upside = ((target_mean - current_price) / current_price) * 100
        print(f"   目標價（平均）: {target_mean:.2f} ({upside:+.2f}%)")
    if target_high:
        print(f"   目標價（最高）: {target_high:.2f}")
    if target_low:
        print(f"   目標價（最低）: {target_low:.2f}")
    if num_analysts:
        print(f"   分析師數量: {num_analysts}")
    
    # ============================================
    # 10. 綜合評分
    # ============================================
    print("\n🏆 綜合評分")
    print("-" * 70)
    
    score = 0
    max_score = 0
    
    # ROE（30 分）
    if roe:
        max_score += 30
        score += min(roe / 0.20, 1.0) * 30
    
    # 淨利率（20 分）
    if profit_margin:
        max_score += 20
        score += min(profit_margin / 0.20, 1.0) * 20
    
    # 負債（20 分）
    if debt_to_equity:
        max_score += 20
        score += max(1 - debt_to_equity / 2.0, 0) * 20
    
    # 成長（30 分）
    if revenue_growth:
        max_score += 15
        score += min(revenue_growth / 0.30, 1.0) * 15
    if earnings_growth:
        max_score += 15
        score += min(earnings_growth / 0.30, 1.0) * 15
    
    if max_score > 0:
        final_score = (score / max_score) * 100
    else:
        final_score = 0
    
    print(f"   評分: {final_score:.1f} / 100")
    
    if final_score >= 80:
        print("   → 🟢 優秀")
    elif final_score >= 60:
        print("   → 🟢 良好")
    elif final_score >= 40:
        print("   → 🟡 一般")
    elif final_score >= 20:
        print("   → 🟠 較弱")
    else:
        print("   → 🔴 差")
    
    # ============================================
    # 顯示原始數據（可選）
    # ============================================
    if show_raw:
        print("\n📋 原始數據")
        print("-" * 70)
        for key, value in sorted(info.items()):
            if value is not None and value != "N/A":
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    
    # 返回結果
    return {
        'ticker': ticker,
        'company_name': company_name,
        'sector': sector,
        'industry': industry,
        'current_price': current_price,
        'market_cap': market_cap,
        'pe': pe,
        'pb': pb,
        'roe': roe,
        'profit_margin': profit_margin,
        'debt_to_equity': debt_to_equity,
        'current_ratio': current_ratio,
        'revenue_growth': revenue_growth,
        'earnings_growth': earnings_growth,
        'dividend_yield': dividend_yield,
        'beta': beta,
        'recommendation': recommendation,
        'target_mean': target_mean,
        'score': final_score,
    }


def compare_stocks(tickers: list):
    """
    比較多支股票的基本面
    
    Args:
        tickers: 股票代碼列表
    """
    print("=" * 70)
    print(f"📊 多股基本面比較（{len(tickers)} 支）")
    print("=" * 70)
    
    results = []
    
    for ticker in tickers:
        info = get_stock_info(ticker)
        if not info:
            continue
        
        # 計算評分
        roe = safe_get(info, 'returnOnEquity', 0)
        profit_margin = safe_get(info, 'profitMargins', 0)
        debt_to_equity = safe_get(info, 'debtToEquity', 0)
        revenue_growth = safe_get(info, 'revenueGrowth', 0)
        earnings_growth = safe_get(info, 'earningsGrowth', 0)
        
        score = 0
        max_score = 0
        
        if roe:
            max_score += 30
            score += min(roe / 0.20, 1.0) * 30
        if profit_margin:
            max_score += 20
            score += min(profit_margin / 0.20, 1.0) * 20
        if debt_to_equity:
            max_score += 20
            score += max(1 - debt_to_equity / 2.0, 0) * 20
        if revenue_growth:
            max_score += 15
            score += min(revenue_growth / 0.30, 1.0) * 15
        if earnings_growth:
            max_score += 15
            score += min(earnings_growth / 0.30, 1.0) * 15
        
        final_score = (score / max_score) * 100 if max_score > 0 else 0
        
        results.append({
            'Ticker': ticker,
            'Name': safe_get(info, 'longName', ticker)[:20],
            'Price': safe_get(info, 'currentPrice', 0),
            'PE': safe_get(info, 'trailingPE', 0),
            'ROE': f"{roe*100:.1f}%" if roe else "N/A",
            'Margin': f"{profit_margin*100:.1f}%" if profit_margin else "N/A",
            'D/E': f"{debt_to_equity:.2f}" if debt_to_equity else "N/A",
            'Growth': f"{revenue_growth*100:.1f}%" if revenue_growth else "N/A",
            'Yield': f"{safe_get(info, 'dividendYield', 0):.2f}%",
            'Score': f"{final_score:.1f}",
        })
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('Score', ascending=False)
        print("\n" + df.to_string(index=False))
    else:
        print("❌ 無數據")
    
    print("\n" + "=" * 70)
    
    return pd.DataFrame(results)


# ============================================================
# 二、測試區塊
# ============================================================

if __name__ == "__main__":
    """
    獨立測試
    執行: python fundamental_analyzer.py
    """
    
    # ========================================
    # 測試 1：單股分析
    # ========================================
    analyze_stock("0700.HK")
    
    # ========================================
    # 測試 2：多股比較
    # ========================================
    print("\n\n")
    compare_stocks([
        "0700.HK",  # 騰訊
        "9988.HK",  # 阿里
        "0005.HK",  # 滙豐
        "0939.HK",  # 建行
        "1810.HK",  # 小米
    ])