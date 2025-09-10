'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Zap, 
  RefreshCw,
  AlertCircle,
  Database,
  Clock
} from 'lucide-react';
import { railwayAPI, formatPrice, formatVolume } from '@/lib/railway-api';

interface RealTimePriceData {
  symbol: string;
  current_price: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number;
  change_percent: number;
  last_updated: string;
  data_source: string;
  technical_indicators: {
    rsi_14: number | null;
    macd: number | null;
    bollinger_upper: number | null;
    bollinger_middle: number | null;
    bollinger_lower: number | null;
  };
}

interface RealTimePriceFeedProps {
  symbol: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export default function RealTimePriceFeed({ 
  symbol, 
  autoRefresh = true, 
  refreshInterval = 10000 
}: RealTimePriceFeedProps) {
  const [priceData, setPriceData] = useState<RealTimePriceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (symbol) {
      loadRealTimePrice();
    }
  }, [symbol]);

  useEffect(() => {
    if (!autoRefresh || !symbol) return;

    const interval = setInterval(() => {
      loadRealTimePrice(true);
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [symbol, autoRefresh, refreshInterval]);

  const loadRealTimePrice = async (isAutoRefresh = false) => {
    if (!isAutoRefresh) {
      setLoading(true);
    } else {
      setIsRefreshing(true);
    }
    
    setError(null);

    try {
      console.log(`🔄 Loading real-time price for ${symbol}...`);
      
      const data = await railwayAPI.getRealTimePrice(symbol);
      setPriceData(data);
      setLastUpdate(new Date());
      
      console.log(`✅ Real-time price loaded for ${symbol}:`, data);
      
    } catch (error) {
      console.error('❌ Real-time price loading error:', error);
      setError(error instanceof Error ? error.message : 'Failed to load real-time price');
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  };

  const handleManualRefresh = () => {
    loadRealTimePrice();
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <div className="animate-spin w-6 h-6 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Loading real-time price...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-red-200">
        <CardContent className="p-6">
          <div className="flex items-center gap-2 text-red-600 mb-2">
            <AlertCircle className="w-5 h-5" />
            <span className="font-semibold">Price Feed Error</span>
          </div>
          <p className="text-sm text-red-600 mb-3">{error}</p>
          <Button variant="outline" size="sm" onClick={handleManualRefresh}>
            <RefreshCw className="w-4 h-4 mr-1" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!priceData) return null;

  const isPositive = priceData.change >= 0;

  return (
    <Card className={`transition-all duration-300 ${
      isRefreshing ? 'ring-2 ring-blue-500' : ''
    }`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-green-500" />
              {priceData.symbol} Real-Time
              <Badge variant="outline" className="text-xs">
                <Database className="w-3 h-3 mr-1" />
                {priceData.data_source}
              </Badge>
            </CardTitle>
            <CardDescription>
              Live price feed from Railway PostgreSQL
            </CardDescription>
          </div>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={handleManualRefresh}
            disabled={isRefreshing}
            className="ml-2"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        {/* Current Price Display */}
        <div className="grid grid-cols-2 gap-6 mb-6">
          <div className="text-center">
            <div className="text-3xl font-bold mb-2">
              {formatPrice(priceData.current_price)}
            </div>
            <div className={`flex items-center justify-center gap-1 text-sm ${
              isPositive ? 'text-green-600' : 'text-red-600'
            }`}>
              {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              {isPositive ? '+' : ''}{priceData.change.toFixed(2)} 
              ({priceData.change_percent.toFixed(2)}%)
            </div>
          </div>

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Open:</span>
              <span className="font-medium">{formatPrice(priceData.open)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">High:</span>
              <span className="font-medium text-green-600">{formatPrice(priceData.high)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Low:</span>
              <span className="font-medium text-red-600">{formatPrice(priceData.low)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Volume:</span>
              <span className="font-medium">{formatVolume(priceData.volume)}</span>
            </div>
          </div>
        </div>

        {/* Technical Indicators */}
        <div className="border-t pt-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-purple-500" />
            <span className="font-semibold text-sm">Technical Indicators</span>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {priceData.technical_indicators.rsi_14 && (
              <div className="text-center p-2 bg-purple-50 dark:bg-purple-950/30 rounded">
                <div className="text-xs text-gray-600 dark:text-gray-400">RSI (14)</div>
                <div className={`font-bold ${
                  priceData.technical_indicators.rsi_14 > 70 ? 'text-red-600' :
                  priceData.technical_indicators.rsi_14 < 30 ? 'text-green-600' : 'text-blue-600'
                }`}>
                  {priceData.technical_indicators.rsi_14.toFixed(1)}
                </div>
              </div>
            )}
            
            {priceData.technical_indicators.macd && (
              <div className="text-center p-2 bg-blue-50 dark:bg-blue-950/30 rounded">
                <div className="text-xs text-gray-600 dark:text-gray-400">MACD</div>
                <div className={`font-bold ${
                  priceData.technical_indicators.macd > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {priceData.technical_indicators.macd.toFixed(3)}
                </div>
              </div>
            )}
            
            {priceData.technical_indicators.bollinger_middle && (
              <div className="text-center p-2 bg-gray-50 dark:bg-gray-900/30 rounded">
                <div className="text-xs text-gray-600 dark:text-gray-400">Bollinger Mid</div>
                <div className="font-bold text-gray-700 dark:text-gray-300">
                  {formatPrice(priceData.technical_indicators.bollinger_middle)}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Last Update */}
        <div className="mt-4 pt-3 border-t">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              <span>Last Update: {lastUpdate?.toLocaleTimeString('tr-TR')}</span>
            </div>
            {autoRefresh && (
              <div className="flex items-center gap-1">
                <div className={`w-2 h-2 rounded-full ${isRefreshing ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
                <span>Auto-refresh {refreshInterval / 1000}s</span>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
