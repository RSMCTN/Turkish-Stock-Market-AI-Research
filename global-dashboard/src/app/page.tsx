'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TradingViewMarketOverview } from '@/components/tradingview/market-overview';
import { GlobalMarketSelector } from '@/components/market-selector';
import { ApiStockSearch } from '@/components/api-stock-search';
import { ProfitIntegration } from '@/components/profit-integration';
import { TrendingUp, TrendingDown, Globe, Flag, Zap, X } from 'lucide-react';

export default function GlobalDashboard() {
  const [selectedMarket, setSelectedMarket] = useState('turkey');
  const [selectedStock, setSelectedStock] = useState<any>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-white text-xl">Loading MAMUT R600 Global Dashboard...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Zap className="h-8 w-8 text-blue-400" />
                <h1 className="text-2xl font-bold text-white">MAMUT R600</h1>
                <span className="text-sm text-slate-400">Global Markets</span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="hidden md:block">
                <ApiStockSearch 
                  onStockSelect={setSelectedStock}
                  onClearSelection={() => setSelectedStock(null)}
                  selectedMarket={selectedMarket}
                  selectedStock={selectedStock}
                />
              </div>
              <GlobalMarketSelector 
                selected={selectedMarket}
                onSelect={setSelectedMarket}
              />
              <div className="text-green-400 text-sm">● Live</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid gap-6">
          
          {/* Mobile Search Bar */}
          <div className="md:hidden">
            <ApiStockSearch 
              onStockSelect={setSelectedStock}
              onClearSelection={() => setSelectedStock(null)}
              selectedMarket={selectedMarket}
              selectedStock={selectedStock}
            />
          </div>

          {/* Selected Stock Info */}
          {selectedStock && (
            <Card className="border-slate-700 bg-slate-800/50">
              <CardHeader>
                <CardTitle className="text-white flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center font-bold text-sm">
                      {selectedStock.symbol.substring(0, 2)}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span>{selectedStock.symbol}</span>
                        <span className="text-xs bg-slate-600 px-2 py-1 rounded">{selectedStock.currency}</span>
                        <span className="text-xs bg-blue-600 px-2 py-1 rounded">{selectedStock.region.toUpperCase()}</span>
                      </div>
                      <div className="text-sm font-normal text-slate-400">{selectedStock.name}</div>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedStock(null)}
                    className="text-slate-400 hover:text-white border-slate-600"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                    <div className="text-green-400 text-sm">Seçildi</div>
                    <div className="text-white font-semibold">✓ Aktif</div>
                  </div>
                  <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
                    <div className="text-blue-400 text-sm">Sektör</div>
                    <div className="text-white font-semibold">{selectedStock.sector}</div>
                  </div>
                  <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3">
                    <div className="text-purple-400 text-sm">Pazar</div>
                    <div className="text-white font-semibold">{selectedStock.market}</div>
                  </div>
                  <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3">
                    <div className="text-yellow-400 text-sm">Ticker</div>
                    <div className="text-white font-semibold">{selectedStock.ticker}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Live Stock Data */}
          {selectedStock && (
            <ProfitIntegration 
              selectedStock={selectedStock}
              onPriceUpdate={(symbol, price) => {
                console.log(`Price update: ${symbol} = ₺${price}`);
              }}
            />
          )}
          
          {/* Welcome & Status Cards */}
          <div className="grid md:grid-cols-3 gap-4">
            <Card className="border-slate-700 bg-slate-800/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2">
                  <Flag className="h-5 w-5 text-red-500" />
                  Turkey Markets
                </CardTitle>
                <CardDescription className="text-slate-400">
                  BIST 100 + 560 additional stocks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold text-green-400">660</span>
                  <div className="flex items-center text-green-400 text-sm">
                    <TrendingUp className="h-4 w-4 mr-1" />
                    Real-time
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-700 bg-slate-800/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2">
                  <Globe className="h-5 w-5 text-blue-500" />
                  Global Markets
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Top 10 international markets
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold text-blue-400">430</span>
                  <div className="flex items-center text-blue-400 text-sm">
                    <TrendingUp className="h-4 w-4 mr-1" />
                    15min updates
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-700 bg-slate-800/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-white">API Status</CardTitle>
                <CardDescription className="text-slate-400">
                  Daily usage & capacity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold text-yellow-400">41.8%</span>
                  <div className="flex items-center text-green-400 text-sm">
                    ● Healthy
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* TradingView Market Overview */}
          <Card className="border-slate-700 bg-slate-800/50">
            <CardHeader>
              <CardTitle className="text-white">🔥 Global Heat Map</CardTitle>
              <CardDescription className="text-slate-400">
                Real-time market overview with TradingView integration
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TradingViewMarketOverview market={selectedMarket} />
            </CardContent>
          </Card>

          {/* Navigation Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="border-slate-700 bg-slate-800/50 hover:bg-slate-700/50 transition-colors cursor-pointer">
              <CardHeader>
                <CardTitle className="text-white text-lg">📊 Advanced Charts</CardTitle>
                <CardDescription className="text-slate-400">
                  Professional trading analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  View Charts
                </Button>
              </CardContent>
            </Card>

            <Card className="border-slate-700 bg-slate-800/50 hover:bg-slate-700/50 transition-colors cursor-pointer">
              <CardHeader>
                <CardTitle className="text-white text-lg">🎯 Sentiment Analysis</CardTitle>
                <CardDescription className="text-slate-400">
                  AI-powered market sentiment
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  View Sentiment
                </Button>
              </CardContent>
            </Card>

            <Card className="border-slate-700 bg-slate-800/50 hover:bg-slate-700/50 transition-colors cursor-pointer">
              <CardHeader>
                <CardTitle className="text-white text-lg">🔍 Market Screener</CardTitle>
                <CardDescription className="text-slate-400">
                  Filter & discover stocks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  Open Screener
                </Button>
              </CardContent>
            </Card>

            <Card className="border-slate-700 bg-slate-800/50 hover:bg-slate-700/50 transition-colors cursor-pointer">
              <CardHeader>
                <CardTitle className="text-white text-lg">📈 Watchlist</CardTitle>
                <CardDescription className="text-slate-400">
                  Personal portfolio tracking
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  Manage List
                </Button>
              </CardContent>
            </Card>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700 bg-slate-900/50 backdrop-blur-md mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-slate-400 text-sm">
            <div>
              MAMUT R600 Global Dashboard - Powered by TradingView & Profit.com
            </div>
            <div className="flex items-center space-x-4">
              <span>🚀 Railway Infrastructure</span>
              <span>📊 150K Daily API Limit</span>
              <span>🌍 Multi-Market Support</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}