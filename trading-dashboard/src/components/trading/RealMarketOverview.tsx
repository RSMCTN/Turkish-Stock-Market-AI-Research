'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { TrendingUp, TrendingDown, Activity, BarChart3, AlertCircle, RefreshCw, Clock } from 'lucide-react';

interface MarketOverview {
  bist_100_value: number;
  bist_100_change: number;
  bist_30_value: number;
  bist_30_change: number;
  total_volume: number;
  total_value: number;
  rising_stocks: number;
  falling_stocks: number;
  unchanged_stocks: number;
  last_updated: string;
}

interface BISTStock {
  symbol: string;
  name_turkish: string;
  last_price: number;
  change: number;
  change_percent: number;
  volume: number;
  sector_turkish: string;
}

export default function RealMarketOverview() {
  const [marketOverview, setMarketOverview] = useState<MarketOverview | null>(null);
  const [topPerformers, setTopPerformers] = useState<{
    gainers: BISTStock[];
    losers: BISTStock[];
  }>({
    gainers: [],
    losers: []
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchMarketData = async () => {
    setIsLoading(true);
    setError('');

    try {
      // Fetch market overview - Use Railway production API
      const baseUrl = 'https://bistai001-production.up.railway.app';
      
      const overviewResponse = await fetch(`${baseUrl}/api/bist/market-overview`);
      
      if (!overviewResponse.ok) {
        throw new Error(`HTTP error! status: ${overviewResponse.status}`);
      }
      
      const overviewData = await overviewResponse.json();
      
      if (overviewData.success) {
        setMarketOverview(overviewData.market_overview);
      }

      // Fetch top performers
      const stocksResponse = await fetch(`${baseUrl}/api/bist/all-stocks?limit=50`);
      
      if (stocksResponse.ok) {
        const stocksData = await stocksResponse.json();
        
        if (stocksData.success && stocksData.stocks) {
          const stocks = stocksData.stocks;
          
          // Sort by performance
          const gainers = [...stocks]
            .filter(s => s.change_percent > 0)
            .sort((a, b) => b.change_percent - a.change_percent)
            .slice(0, 5);
          
          const losers = [...stocks]
            .filter(s => s.change_percent < 0)
            .sort((a, b) => a.change_percent - b.change_percent)
            .slice(0, 5);
          
          setTopPerformers({ gainers, losers });
        }
      }

      setLastUpdate(new Date());

    } catch (err) {
      console.error('Failed to fetch market data:', err);
      setError('Failed to load market data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
    
    // Update every 2 minutes
    const interval = setInterval(fetchMarketData, 120000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number) => {
    if (num >= 1000000000) {
      return `${(num / 1000000000).toFixed(1)}B`;
    }
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    }
    return num.toLocaleString('tr-TR');
  };

  const formatPrice = (price: number) => {
    return price.toLocaleString('tr-TR', { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    });
  };

  const getPriceChangeColor = (change: number) => {
    return change >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const getPriceChangeIcon = (change: number) => {
    return change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />;
  };

  const getMarketStatus = () => {
    const now = new Date();
    const hour = now.getHours();
    const isWeekend = now.getDay() === 0 || now.getDay() === 6;
    const isMarketHours = !isWeekend && hour >= 10 && hour <= 18;
    
    if (isWeekend) return { status: 'WEEKEND', text: 'Hafta Sonu', color: 'bg-gray-100 text-gray-700 border-gray-300' };
    if (isMarketHours) return { status: 'OPEN', text: 'Seansda', color: 'bg-green-100 text-green-700 border-green-300' };
    return { status: 'CLOSED', text: 'Seans Kapalı', color: 'bg-red-100 text-red-700 border-red-300' };
  };

  const marketStatus = getMarketStatus();

  if (isLoading && !marketOverview) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Market Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="flex justify-between items-center mb-2">
                  <div className="h-4 bg-gray-200 rounded w-20"></div>
                  <div className="h-4 bg-gray-200 rounded w-16"></div>
                </div>
                <div className="h-3 bg-gray-200 rounded w-full"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error && !marketOverview) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Market Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertCircle className="h-12 w-12 mx-auto mb-2 text-red-300" />
            <p className="text-red-600 font-medium mb-2">Error Loading Market Data</p>
            <p className="text-red-500 text-sm mb-4">{error}</p>
            <Button 
              onClick={fetchMarketData} 
              variant="outline" 
              size="sm"
              className="border-red-300 text-red-600 hover:bg-red-50"
            >
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!marketOverview) return null;

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Market Overview
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge className={`text-xs font-medium ${marketStatus.color}`}>
              {marketStatus.text}
            </Badge>
            <Button
              onClick={fetchMarketData}
              variant="ghost"
              size="sm"
              disabled={isLoading}
              className="h-6 w-6 p-0"
            >
              <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
        <CardDescription className="flex items-center gap-2">
          Real-time BIST market data
          {lastUpdate && (
            <span className="flex items-center gap-1 text-xs">
              <Clock className="h-3 w-3" />
              {lastUpdate.toLocaleTimeString('tr-TR')}
            </span>
          )}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Major Indices */}
        <div className="space-y-3">
          {/* BIST 100 */}
          <div className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-blue-50 to-blue-100/50 border border-blue-200">
            <div>
              <div className="font-semibold text-gray-900">BIST 100</div>
              <div className="text-sm text-gray-500">XU100</div>
            </div>
            <div className="text-right">
              <div className="font-bold text-lg">{marketOverview.bist_100_value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}</div>
              <div className={`text-sm flex items-center gap-1 justify-end ${getPriceChangeColor(marketOverview.bist_100_change)}`}>
                {getPriceChangeIcon(marketOverview.bist_100_change)}
                {marketOverview.bist_100_change >= 0 ? '+' : ''}{marketOverview.bist_100_change.toFixed(2)}%
              </div>
            </div>
          </div>

          {/* BIST 30 */}
          <div className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-yellow-50 to-yellow-100/50 border border-yellow-200">
            <div>
              <div className="font-semibold text-gray-900">BIST 30</div>
              <div className="text-sm text-gray-500">XU030</div>
            </div>
            <div className="text-right">
              <div className="font-bold text-lg">{marketOverview.bist_30_value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}</div>
              <div className={`text-sm flex items-center gap-1 justify-end ${getPriceChangeColor(marketOverview.bist_30_change)}`}>
                {getPriceChangeIcon(marketOverview.bist_30_change)}
                {marketOverview.bist_30_change >= 0 ? '+' : ''}{marketOverview.bist_30_change.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>

        {/* Market Statistics */}
        <div className="border-t pt-4">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-bold text-green-600">
                {marketOverview.rising_stocks}
              </div>
              <div className="text-xs text-gray-500">Yükselen</div>
            </div>
            <div>
              <div className="text-lg font-bold text-red-600">
                {marketOverview.falling_stocks}
              </div>
              <div className="text-xs text-gray-500">Düşen</div>
            </div>
            <div>
              <div className="text-lg font-bold text-gray-600">
                {marketOverview.unchanged_stocks}
              </div>
              <div className="text-xs text-gray-500">Sabit</div>
            </div>
          </div>
        </div>

        {/* Top Performers */}
        {(topPerformers.gainers.length > 0 || topPerformers.losers.length > 0) && (
          <div className="border-t pt-4">
            <div className="grid grid-cols-2 gap-4">
              {/* Top Gainers */}
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
                  <TrendingUp className="h-4 w-4 text-green-600" />
                  En Çok Yükselenler
                </h4>
                <div className="space-y-1">
                  {topPerformers.gainers.slice(0, 3).map((stock, idx) => (
                    <div key={idx} className="flex justify-between text-xs">
                      <span className="font-medium">{stock.symbol}</span>
                      <span className="text-green-600 font-semibold">
                        +{stock.change_percent.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Top Losers */}
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
                  <TrendingDown className="h-4 w-4 text-red-600" />
                  En Çok Düşenler
                </h4>
                <div className="space-y-1">
                  {topPerformers.losers.slice(0, 3).map((stock, idx) => (
                    <div key={idx} className="flex justify-between text-xs">
                      <span className="font-medium">{stock.symbol}</span>
                      <span className="text-red-600 font-semibold">
                        {stock.change_percent.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Volume and Value */}
        <div className="border-t pt-4">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-lg font-bold text-gray-900">
                {formatNumber(marketOverview.total_volume)}
              </div>
              <div className="text-xs text-gray-500">Günlük Hacim</div>
            </div>
            <div>
              <div className="text-lg font-bold text-gray-900">
                {formatNumber(marketOverview.total_value)}
              </div>
              <div className="text-xs text-gray-500">İşlem Değeri</div>
            </div>
          </div>
        </div>

        {/* Market Alert */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-blue-600 mt-0.5" />
            <div>
              <div className="text-sm font-semibold text-blue-900">Market Update</div>
              <div className="text-xs text-blue-700">
                BIST 100 endeksi günün %{Math.abs(marketOverview.bist_100_change).toFixed(1)} 
                {marketOverview.bist_100_change >= 0 ? ' yükselişle' : ' düşüşle'} işlem görüyor.
                {topPerformers.gainers.length + topPerformers.losers.length} hisse analizde.
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
