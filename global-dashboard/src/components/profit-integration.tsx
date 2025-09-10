'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, TrendingDown, Volume2, Clock } from 'lucide-react';

interface ProfitStock {
  ticker: string;
  symbol: string;
  name: string;
  price: number;
  previous_close: number;
  daily_price_change: number;
  daily_percentage_change: number;
  volume: number;
  timestamp: number;
  logo_url?: string;
}

interface ProfitIntegrationProps {
  selectedStock?: any;
  onPriceUpdate?: (symbol: string, price: number) => void;
}

export function ProfitIntegration({ selectedStock, onPriceUpdate }: ProfitIntegrationProps) {
  const [liveData, setLiveData] = useState<ProfitStock | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch live data for selected stock
  useEffect(() => {
    if (!selectedStock?.symbol) return;

    fetchLiveData(selectedStock.symbol);
    
    // Set up periodic updates (every 30 seconds)
    const interval = setInterval(() => {
      fetchLiveData(selectedStock.symbol);
    }, 30000);

    return () => clearInterval(interval);
  }, [selectedStock]);

  const fetchLiveData = async (symbol: string) => {
    try {
      setLoading(true);
      setError(null);

      // Add .IS suffix if not present for Turkish stocks
      const ticker = symbol.endsWith('.IS') ? symbol : 
                    selectedStock?.region === 'turkey' ? `${symbol}.IS` : symbol;
      
      console.log(`🔍 Fetching live data for: ${ticker}`);
      
      // Call our Next.js API route
      const response = await fetch(`/api/stock/${ticker}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const data = await response.json();
      console.log(`✅ Received live data:`, data);
      
      setLiveData(data);
      setError(null); // Clear any previous errors
      
      // Notify parent component of price update
      if (onPriceUpdate && data.price) {
        onPriceUpdate(symbol, data.price);
      }

    } catch (err: any) {
      console.error('❌ Error fetching live data:', err);
      
      // Determine error message
      let errorMessage = 'Canlı veri alınamadı';
      
      if (err.message?.includes('timeout')) {
        errorMessage = 'API zaman aşımı - tekrar denenecek';
      } else if (err.message?.includes('404')) {
        errorMessage = 'Hisse bulunamadı';
      } else if (err.message?.includes('401')) {
        errorMessage = 'API yetkilendirme hatası';
      }
      
      setError(errorMessage);
      
      // Create fallback data for display
      const fallbackData: ProfitStock = {
        ticker: selectedStock?.ticker || `${symbol}.IS`,
        symbol: symbol,
        name: selectedStock?.name || symbol,
        price: 0,
        previous_close: 0,
        daily_price_change: 0,
        daily_percentage_change: 0,
        volume: 0,
        timestamp: Date.now(),
        logo_url: selectedStock?.region === 'turkey' ? 
                  `https://cdn.profit.com/logo/stocks/${symbol}.IS.png` : 
                  undefined
      };
      
      setLiveData(fallbackData);
    } finally {
      setLoading(false);
    }
  };

  if (!selectedStock) {
    return (
      <Card className="border-slate-700 bg-slate-800/50">
        <CardContent className="p-6 text-center">
          <div className="text-slate-400">
            Canlı fiyat verisi için bir hisse seçin
          </div>
        </CardContent>
      </Card>
    );
  }

  if (loading && !liveData) {
    return (
      <Card className="border-slate-700 bg-slate-800/50">
        <CardContent className="p-6 text-center">
          <div className="animate-spin h-8 w-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
          <div className="text-slate-400">Canlı veriler yükleniyor...</div>
        </CardContent>
      </Card>
    );
  }

  const priceChange = liveData?.daily_price_change || 0;
  const percentChange = liveData?.daily_percentage_change || 0;
  const isPositive = priceChange >= 0;

  return (
    <div className="grid gap-4">
      {/* Main Price Card */}
      <Card className="border-slate-700 bg-slate-800/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-white flex items-center justify-between">
            <div className="flex items-center gap-3">
              {liveData?.logo_url && (
                <img 
                  src={liveData.logo_url} 
                  alt={selectedStock.name}
                  className="w-10 h-10 rounded-full bg-white p-1"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              )}
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold">{selectedStock.symbol}</span>
                  <span className="text-xs bg-slate-600 px-2 py-1 rounded">
                    {selectedStock.currency || 'TRY'}
                  </span>
                </div>
                <div className="text-sm font-normal text-slate-400">
                  {liveData?.name || selectedStock.name}
                </div>
              </div>
            </div>
            
            {loading && (
              <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            )}
          </CardTitle>
          
          {error && (
            <CardDescription className="text-yellow-400 text-xs">
              ⚠️ {error}
            </CardDescription>
          )}
        </CardHeader>

        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Current Price */}
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-3xl font-bold text-white">
                ₺{liveData?.price?.toFixed(2) || '--'}
              </div>
              <div className="text-sm text-slate-400">Güncel Fiyat</div>
            </div>

            {/* Daily Change */}
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className={`text-2xl font-bold flex items-center justify-center gap-1 ${
                isPositive ? 'text-green-400' : 'text-red-400'
              }`}>
                {isPositive ? <TrendingUp className="h-5 w-5" /> : <TrendingDown className="h-5 w-5" />}
                {isPositive ? '+' : ''}₺{priceChange.toFixed(2)}
              </div>
              <div className={`text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                {isPositive ? '+' : ''}{percentChange.toFixed(2)}%
              </div>
            </div>

            {/* Volume */}
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-2xl font-bold text-blue-400 flex items-center justify-center gap-1">
                <Volume2 className="h-5 w-5" />
                {liveData?.volume ? (liveData.volume / 1000000).toFixed(1) + 'M' : '--'}
              </div>
              <div className="text-sm text-slate-400">Hacim</div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div className="flex justify-between items-center p-2 bg-slate-700/30 rounded">
              <span className="text-slate-400">Önceki Kapanış</span>
              <span className="text-white font-mono">
                ₺{liveData?.previous_close?.toFixed(2) || '--'}
              </span>
            </div>
            
            <div className="flex justify-between items-center p-2 bg-slate-700/30 rounded">
              <span className="text-slate-400 flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Son Güncelleme
              </span>
              <span className="text-white font-mono text-xs">
                {liveData?.timestamp ? 
                  new Date(liveData.timestamp).toLocaleTimeString('tr-TR') : 
                  '--:--'
                }
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
