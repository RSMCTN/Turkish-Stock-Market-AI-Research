-- MAMUT_R600 Enhanced Database Schema
-- Yeni Excel dosyaları için genişletilmiş şema

-- Mevcut stock_data tablosunu genişletelim
CREATE TABLE IF NOT EXISTS enhanced_stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    time TIME,
    timeframe VARCHAR(10) NOT NULL, -- '30m', '60m', 'daily'
    
    -- OHLCV Data
    open DECIMAL(15,4),
    high DECIMAL(15,4),
    low DECIMAL(15,4),
    close DECIMAL(15,4),
    volume BIGINT,
    wclose DECIMAL(15,4),
    
    -- Trend Following Indicators
    adx_14 DECIMAL(10,6),
    atr_14 DECIMAL(10,6),
    psar DECIMAL(15,4),
    
    -- Momentum Indicators
    rsi_14 DECIMAL(10,6),
    stochastic_k_5 DECIMAL(10,6),
    stochastic_d_3 DECIMAL(10,6),
    stoccci_20 DECIMAL(10,6),
    stoccci_trigger_20 DECIMAL(10,6),
    
    -- MACD System
    macd_26_12 DECIMAL(10,6),
    macd_trigger_9 DECIMAL(10,6),
    
    -- Bollinger Bands Set 1
    bol_upper_20_2 DECIMAL(15,4),
    bol_middle_20_2 DECIMAL(15,4),
    bol_lower_20_2 DECIMAL(15,4),
    
    -- Bollinger Bands Set 2
    bol_upper_20_2_alt DECIMAL(15,4),
    bol_middle_20_2_alt DECIMAL(15,4),
    bol_lower_20_2_alt DECIMAL(15,4),
    
    -- Ichimoku Cloud Complete
    tenkan_sen DECIMAL(15,4),
    kijun_sen DECIMAL(15,4),
    senkou_span_a DECIMAL(15,4),
    senkou_span_b DECIMAL(15,4),
    chikou_span DECIMAL(15,4),
    
    -- Moving Averages
    wma_50 DECIMAL(15,4),
    
    -- Alligator System
    jaw_13_8 DECIMAL(15,4),
    teeth_8_5 DECIMAL(15,4),
    lips_5_3 DECIMAL(15,4),
    
    -- Advanced Oscillators
    awesome_oscillator_5_7 DECIMAL(10,6),
    acc_dist_oscillator_21_10 DECIMAL(10,6),
    
    -- SuperSmoother Filter
    supersmooth_fr DECIMAL(10,6),
    supersmooth_filt DECIMAL(10,6),
    cs DECIMAL(10,6),
    
    -- MFI Related
    prev_at_14_1_mfi DECIMAL(10,6),
    alpha_14_1_mfi DECIMAL(10,6),
    
    -- Signal System
    signal DECIMAL(10,6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uk_enhanced_stock_data UNIQUE (symbol, date, time, timeframe)
);

-- İndeksler
CREATE INDEX IF NOT EXISTS idx_enhanced_stock_symbol_date ON enhanced_stock_data (symbol, date);
CREATE INDEX IF NOT EXISTS idx_enhanced_stock_timeframe ON enhanced_stock_data (timeframe);
CREATE INDEX IF NOT EXISTS idx_enhanced_stock_date_time ON enhanced_stock_data (date, time);

-- Hızlı sorgular için materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_technical_indicators AS
SELECT DISTINCT ON (symbol, timeframe)
    symbol,
    timeframe,
    date,
    time,
    close,
    rsi_14,
    macd_26_12,
    macd_trigger_9,
    bol_upper_20_2,
    bol_lower_20_2,
    adx_14,
    atr_14,
    volume,
    tenkan_sen,
    kijun_sen,
    stochastic_k_5,
    stochastic_d_3
FROM enhanced_stock_data
ORDER BY symbol, timeframe, date DESC, time DESC;

-- Refresh için function
CREATE OR REPLACE FUNCTION refresh_latest_indicators()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW latest_technical_indicators;
END;
$$ LANGUAGE plpgsql;

-- Otomatik trigger
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_enhanced_stock_data_timestamp
    BEFORE UPDATE ON enhanced_stock_data
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

-- Performance için partitioning (opsiyonel)
-- CREATE TABLE enhanced_stock_data_2025 PARTITION OF enhanced_stock_data
-- FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
