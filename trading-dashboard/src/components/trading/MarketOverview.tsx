'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, AlertCircle } from 'lucide-react';

interface MarketIndex {
  name: string;
  symbol: string;
  value: number;
  change: number;
  changePercent: number;
  volume?: number;
}

interface MarketData {
  indices: MarketIndex[];
  marketStatus: 'OPEN' | 'CLOSED' | 'PRE_MARKET' | 'AFTER_HOURS';
  tradingSession: string;
  topGainers: MarketIndex[];
  topLosers: MarketIndex[];
}

export default function MarketOverview() {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Generate mock market data
  const generateMarketData = (): MarketData => {
    const currentTime = new Date();
    const hour = currentTime.getHours();
    const isMarketHours = hour >= 9 && hour <= 18;
    
    const indices: MarketIndex[] = [
      {
        name: 'BIST 100',
        symbol: 'XU100',
        value: 8450.75 + (Math.random() - 0.5) * 100,
        change: (Math.random() - 0.5) * 50,
        changePercent: (Math.random() - 0.5) * 3,
        volume: 25600000000 + Math.random() * 5000000000
      },
      {
        name: 'BIST 30',
        symbol: 'XU030',
        value: 9850.25 + (Math.random() - 0.5) * 120,
        change: (Math.random() - 0.5) * 60,
        changePercent: (Math.random() - 0.5) * 3.5,
        volume: 18200000000 + Math.random() * 3000000000
      },
      {
        name: 'BIST Bank',
        symbol: 'XBANK',
        value: 2150.45 + (Math.random() - 0.5) * 80,
        change: (Math.random() - 0.5) * 40,
        changePercent: (Math.random() - 0.5) * 4,
        volume: 8500000000 + Math.random() * 2000000000
      },
      {
        name: 'BIST Tech',
        symbol: 'XTECH',
        value: 3250.80 + (Math.random() - 0.5) * 150,
        change: (Math.random() - 0.5) * 75,
        changePercent: (Math.random() - 0.5) * 5,
        volume: 5200000000 + Math.random() * 1000000000
      }
    ];

    // Recalculate change percent based on change and value
    indices.forEach(index => {
      index.changePercent = (index.change / (index.value - index.change)) * 100;
    });

    const symbols = ['AKBNK', 'GARAN', 'ISCTR', 'THYAO', 'ASELS', 'SISE', 'EREGL', 'PETKM'];
    
    const topGainers: MarketIndex[] = Array.from({ length: 5 }, (_, i) => ({
      name: symbols[Math.floor(Math.random() * symbols.length)],
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      value: 25 + Math.random() * 50,
      change: 1 + Math.random() * 3,
      changePercent: 3 + Math.random() * 7
    }));

    const topLosers: MarketIndex[] = Array.from({ length: 5 }, (_, i) => ({
      name: symbols[Math.floor(Math.random() * symbols.length)],
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      value: 25 + Math.random() * 50,
      change: -(1 + Math.random() * 3),
      changePercent: -(3 + Math.random() * 7)
    }));

    return {
      indices,
      marketStatus: isMarketHours ? 'OPEN' : 'CLOSED',
      tradingSession: isMarketHours ? 'Seansda' : 'Seans Kapalı',
      topGainers: topGainers.sort((a, b) => b.changePercent - a.changePercent),
      topLosers: topLosers.sort((a, b) => a.changePercent - b.changePercent)
    };
  };

  useEffect(() => {
    const loadMarketData = () => {
      setIsLoading(true);
      setTimeout(() => {
        setMarketData(generateMarketData());
        setIsLoading(false);
      }, 1000);
    };

    loadMarketData();
    
    // Update every 30 seconds
    const interval = setInterval(loadMarketData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
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

  if (!marketData) return null;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'OPEN': return 'bg-green-100 text-green-700 border-green-300';
      case 'CLOSED': return 'bg-red-100 text-red-700 border-red-300';
      default: return 'bg-yellow-100 text-yellow-700 border-yellow-300';
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Market Overview
          </CardTitle>
          <Badge className={`text-xs font-medium ${getStatusColor(marketData.marketStatus)}`}>
            {marketData.tradingSession}
          </Badge>
        </div>
        <CardDescription>
          Real-time BIST market indices and performance
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Major Indices */}
        <div className="space-y-3">
          {marketData.indices.map((index) => (
            <div key={index.symbol} className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50 transition-colors">
              <div>
                <div className="font-semibold text-gray-900">{index.name}</div>
                <div className="text-sm text-gray-500">{index.symbol}</div>
              </div>
              <div className="text-right">
                <div className="font-semibold">{index.value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}</div>
                <div className={`text-sm flex items-center gap-1 justify-end ${
                  index.changePercent >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {index.changePercent >= 0 ? (
                    <TrendingUp className="h-3 w-3" />
                  ) : (
                    <TrendingDown className="h-3 w-3" />
                  )}
                  {index.changePercent >= 0 ? '+' : ''}{index.changePercent.toFixed(2)}%
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="border-t pt-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Top Gainers */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
                <TrendingUp className="h-4 w-4 text-green-600" />
                En Çok Yükselenler
              </h4>
              <div className="space-y-1">
                {marketData.topGainers.slice(0, 3).map((stock, idx) => (
                  <div key={idx} className="flex justify-between text-xs">
                    <span className="font-medium">{stock.symbol}</span>
                    <span className="text-green-600 font-semibold">
                      +{stock.changePercent.toFixed(1)}%
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
                {marketData.topLosers.slice(0, 3).map((stock, idx) => (
                  <div key={idx} className="flex justify-between text-xs">
                    <span className="font-medium">{stock.symbol}</span>
                    <span className="text-red-600 font-semibold">
                      {stock.changePercent.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Market Stats */}
        <div className="border-t pt-4">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-lg font-bold text-gray-900">
                {marketData.indices[0].volume ? 
                  `${(marketData.indices[0].volume / 1000000000).toFixed(1)}B` : 
                  '25.6B'
                }
              </div>
              <div className="text-xs text-gray-500">Günlük Hacim</div>
            </div>
            <div>
              <div className="text-lg font-bold text-gray-900">
                {marketData.topGainers.length + marketData.topLosers.length}
              </div>
              <div className="text-xs text-gray-500">Aktif Hisse</div>
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
                BIST 100 endeksi günün %{Math.abs(marketData.indices[0].changePercent).toFixed(1)} 
                {marketData.indices[0].changePercent >= 0 ? ' yükselişle' : ' düşüşle'} işlem görüyor.
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
