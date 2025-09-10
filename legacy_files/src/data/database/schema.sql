-- =====================================================
-- BIST Historical Data Database Schema
-- Railway + SQLite/PostgreSQL Compatible
-- =====================================================

-- Stocks master table
CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    name_turkish VARCHAR(100),
    sector VARCHAR(50),
    sector_turkish VARCHAR(50),
    market_cap DECIMAL(20,2),
    market_segment VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Historical price data with technical indicators
CREATE TABLE IF NOT EXISTS historical_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    date_time DATETIME NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- 'daily' or 'hourly'
    
    -- OHLCV Data
    open_price DECIMAL(10,4) NOT NULL,
    high_price DECIMAL(10,4) NOT NULL,
    low_price DECIMAL(10,4) NOT NULL,
    close_price DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    weighted_close DECIMAL(10,4),
    
    -- Technical Indicators
    rsi_14 DECIMAL(8,4),
    
    -- Ichimoku Cloud System
    tenkan_sen DECIMAL(10,4),
    kijun_sen DECIMAL(10,4),
    senkou_span_a DECIMAL(10,4),
    senkou_span_b DECIMAL(10,4),
    chikou_span DECIMAL(10,4),
    
    -- MACD System
    macd_line DECIMAL(10,6),
    macd_signal DECIMAL(10,6),
    
    -- Bollinger Bands
    bollinger_upper DECIMAL(10,4),
    bollinger_middle DECIMAL(10,4),
    bollinger_lower DECIMAL(10,4),
    
    -- Volatility & Trend Indicators
    atr_14 DECIMAL(10,6),
    adx_14 DECIMAL(8,4),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite unique constraint
    UNIQUE(symbol, date_time, timeframe),
    
    -- Foreign key
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_historical_symbol ON historical_data(symbol);
CREATE INDEX IF NOT EXISTS idx_historical_datetime ON historical_data(date_time);
CREATE INDEX IF NOT EXISTS idx_historical_timeframe ON historical_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_historical_symbol_date ON historical_data(symbol, date_time DESC);
CREATE INDEX IF NOT EXISTS idx_historical_symbol_timeframe ON historical_data(symbol, timeframe, date_time DESC);

-- Market overview cache table (for Redis backup)
CREATE TABLE IF NOT EXISTS market_overview (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calculation_date DATE NOT NULL UNIQUE,
    bist_100_value DECIMAL(10,2),
    bist_100_change DECIMAL(8,4),
    bist_30_value DECIMAL(10,2),
    bist_30_change DECIMAL(8,4),
    total_volume BIGINT,
    total_value BIGINT,
    rising_stocks INTEGER,
    falling_stocks INTEGER,
    unchanged_stocks INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sectors summary table
CREATE TABLE IF NOT EXISTS sectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL UNIQUE,
    name_turkish VARCHAR(50),
    stock_count INTEGER DEFAULT 0,
    total_market_cap DECIMAL(20,2),
    avg_performance DECIMAL(8,4),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query performance views
CREATE VIEW IF NOT EXISTS v_latest_prices AS
SELECT 
    h.symbol,
    s.name,
    s.name_turkish,
    h.close_price as last_price,
    h.volume,
    h.date_time as last_updated,
    -- Price change calculation (vs previous day)
    LAG(h.close_price) OVER (
        PARTITION BY h.symbol 
        ORDER BY h.date_time
    ) as prev_close,
    
    -- Technical indicators latest values
    h.rsi_14,
    h.macd_line,
    h.bollinger_upper,
    h.bollinger_lower,
    h.atr_14,
    h.adx_14
FROM historical_data h
JOIN stocks s ON h.symbol = s.symbol
WHERE h.timeframe = 'daily'
AND h.date_time = (
    SELECT MAX(date_time) 
    FROM historical_data h2 
    WHERE h2.symbol = h.symbol 
    AND h2.timeframe = 'daily'
);
