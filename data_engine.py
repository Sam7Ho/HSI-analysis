"""
data_engine.py - 港股量化資料庫引擎

功能：
1. OHLCV 數據同步（增量/全量）
2. 篩選結果儲存與查詢
3. Panel Data 查詢（向量化計算用）
4. 交易日感知（跳過週末/假期）
5. 多執行緒安全（Threading Lock）
6. 時區修正（避免日期偏差）

作者: HK-Quant-Radar
"""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import yfinance as yf
import logging

# ==================== Logging 設定 ====================

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)


# ==================== 主類別 ====================

class MarketDatabase:
    """
    SQLite 港股/美股行情資料庫管理器
    
    特性：
    - WAL 模式（多執行緒讀寫安全）
    - Threading Lock（避免 database is locked）
    - 保留 Adj Close（除權息復權）
    - Chunking 查詢（突破 SQL 變數上限）
    - 交易日感知（跳過週末/假期）
    """
    
    # 類別常數
    VALID_FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]
    NUMERIC_COLS = ["open", "high", "low", "close", "adj_close"]
    
    def __init__(self, db_path: str = "hk_trading.db"):
        """
        初始化資料庫
        
        Args:
            db_path: SQLite 資料庫檔案路徑
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"✅ MarketDatabase 初始化: {db_path}")
    
    # ==================== 連線與初始化 ====================
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        建立資料庫連線（啟用 WAL 模式）
        
        WAL 模式優點：
        - 讀寫並行（一個寫入時仍可讀取）
        - 更少的鎖衝突
        - 更好的多執行緒支援
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-10000;")   # 10MB cache
        return conn
    
    def _init_db(self):
        """初始化資料庫表架構"""
        with self._get_connection() as conn:
            # ---------- OHLCV 表格 ----------
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            # 雙索引（支援不同查詢模式）
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_date 
                ON ohlcv (ticker, date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_date_ticker 
                ON ohlcv (date, ticker)
            """)
            
            # ---------- 篩選結果表格 ----------
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screening_results (
                    screening_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    passed INTEGER,
                    score REAL,
                    roe REAL,
                    pe REAL,
                    pb REAL,
                    profit_margin REAL,
                    debt_to_equity REAL,
                    current_ratio REAL,
                    free_cash_flow REAL,
                    revenue_growth REAL,
                    market_cap REAL,
                    dividend_yield REAL,
                    beta REAL,
                    sector TEXT,
                    industry TEXT,
                    currency TEXT,
                    reason TEXT,
                    failed_criteria TEXT,
                    PRIMARY KEY (screening_date, ticker)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_screening_date 
                ON screening_results (screening_date)
            """)
            
            conn.commit()
    
    # ==================== 交易日工具 ====================
    
    @staticmethod
    def _next_trading_day(date_str: str) -> str:
        """
        計算下一個交易日（跳過週末）
        
        注意：這只跳過週末，不處理公眾假期。
        對於假期，依賴 yfinance 返回空數據後的處理邏輯。
        
        Args:
            date_str: 日期字串 (YYYY-MM-DD)
            
        Returns:
            下一個交易日的日期字串
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
        # 跳過週末（5=星期六, 6=星期日）
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        return dt.strftime("%Y-%m-%d")
    
    @staticmethod
    def _is_weekend(date_str: str) -> bool:
        """判斷是否為週末"""
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5
    
    # ==================== 寫入方法 ====================
    
    def _write_ohlcv(self, df: pd.DataFrame):
        """
        線程安全的批量寫入 OHLCV 數據
        
        使用 ON CONFLICT DO UPDATE（upsert）
        如果 (ticker, date) 已存在，則更新
        """
        if df is None or df.empty:
            return
        
        records = df.to_dict("records")
        
        sql = """
            INSERT INTO ohlcv (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                adj_close=excluded.adj_close,
                volume=excluded.volume
        """
        
        # ✅ 使用 Lock 保護寫入（避免 SQLITE_BUSY）
        with self._lock:
            with self._get_connection() as conn:
                conn.executemany(sql, records)
                conn.commit()
    
    def _write_screening(self, df: pd.DataFrame, screening_date: str):
        """線程安全地寫入篩選結果"""
        if df is None or df.empty:
            return
        
        # 準備數據
        records = []
        for _, row in df.iterrows():
            # 處理 failed_criteria（可能為 str 或 list）
            fc = row.get('failed_criteria', '')
            if isinstance(fc, list):
                fc = ', '.join(str(x) for x in fc)
            elif fc is None:
                fc = ''
            
            records.append({
                'screening_date': screening_date,
                'ticker': row.get('ticker', ''),
                'company_name': row.get('company_name', ''),
                'passed': int(row.get('passed', 0)),
                'score': row.get('score'),
                'roe': row.get('roe'),
                'pe': row.get('pe'),
                'pb': row.get('pb'),
                'profit_margin': row.get('profit_margin'),
                'debt_to_equity': row.get('debt_to_equity'),
                'current_ratio': row.get('current_ratio'),
                'free_cash_flow': row.get('free_cash_flow'),
                'revenue_growth': row.get('revenue_growth'),
                'market_cap': row.get('market_cap'),
                'dividend_yield': row.get('dividend_yield'),
                'beta': row.get('beta'),
                'sector': row.get('sector', ''),
                'industry': row.get('industry', ''),
                'currency': row.get('currency', 'HKD'),
                'reason': row.get('reason', ''),
                'failed_criteria': fc,
            })
        
        sql = """
            INSERT INTO screening_results 
            (screening_date, ticker, company_name, passed, score, roe, pe, pb,
             profit_margin, debt_to_equity, current_ratio, free_cash_flow,
             revenue_growth, market_cap, dividend_yield, beta, sector, industry,
             currency, reason, failed_criteria)
            VALUES 
            (:screening_date, :ticker, :company_name, :passed, :score, :roe, :pe, :pb,
             :profit_margin, :debt_to_equity, :current_ratio, :free_cash_flow,
             :revenue_growth, :market_cap, :dividend_yield, :beta, :sector, :industry,
             :currency, :reason, :failed_criteria)
            ON CONFLICT(screening_date, ticker) DO UPDATE SET
                company_name=excluded.company_name,
                passed=excluded.passed,
                score=excluded.score,
                roe=excluded.roe,
                pe=excluded.pe,
                pb=excluded.pb,
                profit_margin=excluded.profit_margin,
                debt_to_equity=excluded.debt_to_equity,
                current_ratio=excluded.current_ratio,
                free_cash_flow=excluded.free_cash_flow,
                revenue_growth=excluded.revenue_growth,
                market_cap=excluded.market_cap,
                dividend_yield=excluded.dividend_yield,
                beta=excluded.beta,
                sector=excluded.sector,
                industry=excluded.industry,
                currency=excluded.currency,
                reason=excluded.reason,
                failed_criteria=excluded.failed_criteria
        """
        
        with self._lock:
            with self._get_connection() as conn:
                conn.executemany(sql, records)
                conn.commit()
    
    # ==================== 查詢方法 ====================
    
    def get_latest_date(self, ticker: str) -> Optional[str]:
        """獲取特定股票在資料庫中的最新日期"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(date) FROM ohlcv WHERE ticker = ?",
                (ticker,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
    
    def get_ticker_date_range(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """獲取股票的日期範圍"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MIN(date), MAX(date) FROM ohlcv WHERE ticker = ?",
                (ticker,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return result[0], result[1]
            return None, None
    
    # ==================== 數據抓取 ====================
    
    def fetch_single_stock(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        retry_times: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        單檔股票下載（帶重試機制）
        
        Args:
            ticker: 股票代碼
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            retry_times: 重試次數
            
        Returns:
            清洗後的 DataFrame，或 None
        """
        for attempt in range(retry_times):
            try:
                stock = yf.Ticker(ticker)
                # auto_adjust=False：保留原始 Close 和 Adj Close
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=False
                )
                
                if df.empty:
                    logger.debug(f"   {ticker}: 無新數據")
                    return None
                
                df = df.reset_index()
                
                # ✅ 時區處理（避免日期偏差）
                date_col = df["Date"]
                if isinstance(date_col.dtype, pd.DatetimeTZDtype):
                    df["date"] = date_col.dt.tz_localize(None)
                else:
                    df["date"] = pd.to_datetime(date_col)
                
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                df["ticker"] = ticker
                
                # 欄位重命名
                rename_map = {
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
                df = df.rename(columns=rename_map)
                
                # 選取必要欄位
                cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
                available = [c for c in cols if c in df.columns]
                df = df[available].copy()
                
                # 數值清理
                for col in self.NUMERIC_COLS:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if "volume" in df.columns:
                    df["volume"] = pd.to_numeric(df["volume"], errors='coerce').fillna(0).astype(int)
                
                # 移除 close 為 NaN 的行
                if "close" in df.columns:
                    df = df.dropna(subset=["close"])
                
                return df
                
            except Exception as e:
                if attempt == retry_times - 1:
                    logger.error(f"❌ {ticker} 抓取失敗 (已重試 {retry_times} 次): {e}")
                    return None
                logger.warning(f"🔄 {ticker} 重試 {attempt+1}/{retry_times}")
                import time
                time.sleep(1.5 * (attempt + 1))
        
        return None
    
    # ==================== 主同步方法 ====================
    
    def sync_data(
        self,
        tickers: List[str],
        max_workers: int = 5,
        full_download_years: int = 10,
        skip_recent_days: int = 7
    ) -> Dict[str, int]:
        """
        多線程增量/全量同步
        
        Args:
            tickers: 股票清單
            max_workers: 並發執行緒數
            full_download_years: 全量下載的年數
            skip_recent_days: 最近 N 天內無數據視為假期
            
        Returns:
            統計字典 {"updated": N, "skipped": N, "failed": N}
        """
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        
        stats = {"updated": 0, "skipped": 0, "failed": 0}
        
        def process_ticker(ticker: str) -> Tuple[str, str, str]:
            try:
                latest = self.get_latest_date(ticker)
                
                if latest:
                    # 增量更新：計算下一個交易日
                    next_trading = self._next_trading_day(latest)
                    
                    if next_trading > today_str:
                        return ticker, "skipped", f"已最新 (最新: {latest})"
                    
                    start_date = next_trading
                else:
                    # 全量下載
                    start_date = (
                        today - timedelta(days=365 * full_download_years)
                    ).strftime("%Y-%m-%d")
                
                # 結束日期：+1 天（yfinance 的 end 不包含）
                end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
                
                df = self.fetch_single_stock(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    self._write_ohlcv(df)
                    return ticker, "updated", f"新增 {len(df)} 筆"
                else:
                    if latest:
                        return ticker, "skipped", "非交易日或無新數據"
                    else:
                        return ticker, "failed", "無法取得數據"
                        
            except Exception as e:
                return ticker, "failed", str(e)
        
        logger.info(f"🔄 開始同步 {len(tickers)} 支股票 (並發: {max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_ticker, t): t for t in tickers}
            
            for future in as_completed(futures):
                ticker, status, msg = future.result()
                stats[status] = stats.get(status, 0) + 1
                
                if status == "failed":
                    logger.warning(f"❌ [{ticker}] {msg}")
                elif status == "updated":
                    logger.info(f"✅ [{ticker}] {msg}")
        
        logger.info(
            f"📊 同步完成: 更新 {stats['updated']}, "
            f"跳過 {stats['skipped']}, 失敗 {stats['failed']}"
        )
        
        return stats
    
    # ==================== 儲存篩選結果 ====================
    
    def save_screening_results(
        self,
        df: pd.DataFrame,
        screening_date: Optional[str] = None
    ) -> int:
        """
        儲存篩選結果
        
        Args:
            df: 篩選結果 DataFrame（來自 BestHKScreener.screen_stocks）
            screening_date: 篩選日期（預設今天）
            
        Returns:
            儲存的記錄數
        """
        if df is None or df.empty:
            logger.warning("❌ 篩選結果為空")
            return 0
        
        screening_date = screening_date or datetime.now().strftime("%Y-%m-%d")
        self._write_screening(df, screening_date)
        
        logger.info(f"✅ 篩選結果已儲存: {len(df)} 筆 ({screening_date})")
        return len(df)
    
    def get_screening_history(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        only_passed: bool = True
    ) -> pd.DataFrame:
        """
        查詢篩選歷史
        
        Args:
            ticker: 股票代碼（可選）
            start_date: 起始日期
            end_date: 結束日期
            only_passed: 只顯示合格的
        """
        query = "SELECT * FROM screening_results WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if start_date:
            query += " AND screening_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND screening_date <= ?"
            params.append(end_date)
        
        if only_passed:
            query += " AND passed = 1"
        
        query += " ORDER BY screening_date DESC, score DESC"
        
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ==================== 單股查詢 ====================
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查詢單一股票 OHLCV 數據
        
        Args:
            ticker: 股票代碼
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            
        Returns:
            DataFrame (Index=Date, Columns=open/high/low/close/adj_close/volume)
        """
        query = """
            SELECT date, open, high, low, close, adj_close, volume
            FROM ohlcv
            WHERE ticker = ?
        """
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.index.name = "Date"
        
        return df
    
    # ==================== Panel Data ====================
    
    def get_panel_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        field: str = "adj_close",
        chunk_size: int = 500
    ) -> pd.DataFrame:
        """
        將數據轉為 Panel 矩陣（Index: Date, Columns: Ticker）
        
        Args:
            tickers: 股票清單
            start_date: 起始日期
            end_date: 結束日期
            field: 欄位名稱
            chunk_size: SQL 分批大小（避免變數上限）
            
        Returns:
            Panel DataFrame
        """
        if field not in self.VALID_FIELDS:
            raise ValueError(f"field 必須是: {self.VALID_FIELDS}")
        
        if not tickers:
            return pd.DataFrame()
        
        all_dfs = []
        
        # 分批查詢（解決 SQLite 變數上限）
        for i in range(0, len(tickers), chunk_size):
            chunk_tickers = tickers[i:i + chunk_size]
            placeholders = ", ".join(["?"] * len(chunk_tickers))
            
            sql = f"""
                SELECT date, ticker, {field}
                FROM ohlcv
                WHERE ticker IN ({placeholders})
                  AND date >= ?
                  AND date <= ?
                ORDER BY date ASC
            """
            params = chunk_tickers + [start_date, end_date]
            
            with self._get_connection() as conn:
                df_chunk = pd.read_sql_query(sql, conn, params=params)
                all_dfs.append(df_chunk)
        
        if not all_dfs or all(df.empty for df in all_dfs):
            return pd.DataFrame()
        
        # 合併並 Pivot
        full_df = pd.concat(all_dfs, ignore_index=True)
        panel_df = full_df.pivot(index="date", columns="ticker", values=field)
        panel_df.index = pd.to_datetime(panel_df.index)
        panel_df = panel_df.sort_index()
        
        return panel_df
    
    # ==================== 工具方法 ====================
    
    def get_data_summary(self) -> pd.DataFrame:
        """獲取數據摘要"""
        sql = """
            SELECT 
                ticker,
                COUNT(*) as days,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM ohlcv
            GROUP BY ticker
            ORDER BY ticker
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(sql, conn)
    
    def get_available_tickers(self) -> List[str]:
        """獲取所有已下載的股票代碼"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker")
            return [row[0] for row in cursor.fetchall()]
    
    def get_table_stats(self) -> Dict[str, int]:
        """獲取所有表格的記錄數"""
        stats = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
        
        return stats
    
    def clear_all(self):
        """清空所有數據（謹慎使用！）"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM ohlcv")
                conn.execute("DELETE FROM screening_results")
                conn.commit()
        logger.warning("🗑️ 已清空所有數據")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # sqlite3 每次連線自動關閉


# ==================== 測試區塊 ====================

if __name__ == "__main__":
    """
    獨立測試 data_engine.py
    執行: python data_engine.py
    """
    print("="*70)
    print("🧪 MarketDatabase 獨立測試")
    print("="*70)
    
    # 使用測試資料庫
    db = MarketDatabase("test_market.db")
    
    try:
        # 測試 1：同步少量股票
        print("\n📥 測試 1: 同步股票數據")
        test_tickers = ["0700.HK", "0005.HK", "9988.HK"]
        
        stats = db.sync_data(
            tickers=test_tickers,
            max_workers=3,
            full_download_years=1
        )
        
        print(f"\n同步結果: {stats}")
        
        # 測試 2：查詢數據
        print("\n📊 測試 2: 查詢 0700.HK")
        with db._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM ohlcv WHERE ticker = '0700.HK' ORDER BY date DESC LIMIT 5",
                conn
            )
        print(df)
        
        # 測試 3：Panel Data
        print("\n📈 測試 3: Panel Data")
        panel = db.get_panel_data(
            tickers=test_tickers,
            start_date="2025-01-01",
            end_date="2026-09-15",
            field="adj_close"
        )
        print(f"Panel 形狀: {panel.shape}")
        print(panel.tail())
        
        # 測試 4：數據摘要
        print("\n📋 測試 4: 數據摘要")
        summary = db.get_data_summary()
        print(summary)
        
        # 測試 5：表格統計
        print("\n📊 測試 5: 表格統計")
        stats = db.get_table_stats()
        print(stats)
        
        print("\n" + "="*70)
        print("✅ 所有測試完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n💾 測試資料庫: test_market.db")
        print("   如需清理: rm test_market.db*")