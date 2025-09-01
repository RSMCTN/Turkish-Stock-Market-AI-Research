'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Search, TrendingUp, TrendingDown, BarChart3, RefreshCw } from 'lucide-react';

interface BISTStock {
  symbol: string;
  name: string;
  name_turkish: string;
  sector: string;
  sector_turkish: string;
  market_cap: number;
  last_price: number;
  change: number;
  change_percent: number;
  volume: number;
  bist_markets: string[];
  market_segment: string;
  is_active: boolean;
  last_updated: string;
}

interface Sector {
  name: string;
  name_turkish: string;
  companies: string[];
  keywords: string[];
  weight: number;
}

interface RealSymbolSelectorProps {
  selectedSymbol: string;
  onSymbolChange: (symbol: string) => void;
  showSearch?: boolean;
  showFilters?: boolean;
  limit?: number;
}

const RealSymbolSelector = ({ 
  selectedSymbol, 
  onSymbolChange, 
  showSearch = true,
  showFilters = true,
  limit = 50 
}: RealSymbolSelectorProps) => {
  const [stocks, setStocks] = useState<BISTStock[]>([]);
  const [sectors, setSectors] = useState<Record<string, Sector>>({});
  const [filteredStocks, setFilteredStocks] = useState<BISTStock[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSector, setSelectedSector] = useState<string>('all');
  const [selectedMarket, setSelectedMarket] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');

  // Fetch all stocks and sectors
  const fetchData = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      // Try enhanced local data first
      try {
        console.log('🎯 Loading enhanced BIST data...');
        const localResponse = await fetch('/data/working_bist_data.json');
        
        if (localResponse.ok) {
          const data = await localResponse.json();
          console.log(`✅ Loaded ${data.stocks?.length || 0} enhanced stocks from ${Object.keys(data.sectors || {}).length} sectors`);
          
          if (data.stocks && data.sectors) {
            setStocks(data.stocks);
            setFilteredStocks(data.stocks);
            setSectors(data.sectors);
            return; // Success - exit early
          }
        }
      } catch (localErr) {
        console.log('📡 Enhanced data not available, falling back to Railway API...');
      }
      
      // Fallback to Railway API
      const baseUrl = 'https://bistai001-production.up.railway.app';
        
      const stocksResponse = await fetch(`${baseUrl}/api/bist/all-stocks?limit=${limit}`);
      
      if (!stocksResponse.ok) {
        throw new Error(`HTTP error! status: ${stocksResponse.status}`);
      }
      
      const stocksData = await stocksResponse.json();
      
      if (stocksData.success) {
        setStocks(stocksData.stocks);
        setFilteredStocks(stocksData.stocks);
        console.log(`📡 Loaded ${stocksData.stocks?.length || 0} stocks from Railway API`);
      }

      // Fetch sectors from Railway
      const sectorsResponse = await fetch(`${baseUrl}/api/bist/sectors`);
      
      if (sectorsResponse.ok) {
        const sectorsData = await sectorsResponse.json();
        if (sectorsData.success) {
          setSectors(sectorsData.sectors);
        }
      }
      
    } catch (err) {
      console.error('Failed to fetch BIST data:', err);
      setError('Failed to load BIST data. Check console for details.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [limit]);

  // Filter stocks based on search and filters
  useEffect(() => {
    let filtered = stocks;

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(stock => 
        stock.symbol.toLowerCase().includes(query) ||
        stock.name.toLowerCase().includes(query) ||
        stock.name_turkish.toLowerCase().includes(query)
      );
    }

    // Apply sector filter
    if (selectedSector !== 'all') {
      filtered = filtered.filter(stock => 
        stock.sector.toLowerCase() === selectedSector.toLowerCase()
      );
    }

    // Apply market filter
    if (selectedMarket !== 'all') {
      filtered = filtered.filter(stock => 
        stock.bist_markets?.includes(selectedMarket) ||
        stock.market_segment === selectedMarket
      );
    }

    setFilteredStocks(filtered);
  }, [stocks, searchQuery, selectedSector, selectedMarket]);

  const getPriceChangeIcon = (changePercent: number) => {
    if (changePercent > 0) return <TrendingUp className="h-3 w-3 text-green-600" />;
    if (changePercent < 0) return <TrendingDown className="h-3 w-3 text-red-600" />;
    return <BarChart3 className="h-3 w-3 text-gray-500" />;
  };

  const getPriceChangeColor = (changePercent: number) => {
    if (changePercent > 0) return 'text-green-600';
    if (changePercent < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  const formatPrice = (price: number) => {
    return price.toLocaleString('tr-TR', { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    });
  };

  const formatMarketCap = (marketCap: number) => {
    if (marketCap >= 1000000000) {
      return `${(marketCap / 1000000000).toFixed(1)}M ₺`;
    }
    if (marketCap >= 1000000) {
      return `${(marketCap / 1000000).toFixed(0)}M ₺`;
    }
    return `${marketCap.toLocaleString('tr-TR')} ₺`;
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <RefreshCw className="h-4 w-4 animate-spin" />
            Loading BIST stocks...
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-4">
          <div className="text-red-600 text-sm mb-2">{error}</div>
          <Button onClick={fetchData} size="sm" variant="outline">
            Try Again
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardContent className="p-4 space-y-4">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            <h3 className="font-semibold text-gray-900">Hisse Seçimi</h3>
            <Badge variant="outline" className="text-xs">
              {filteredStocks.length} stocks
            </Badge>
          </div>
          <Button onClick={fetchData} size="sm" variant="outline" className="gap-2">
            <RefreshCw className="h-3 w-3" />
            Refresh
          </Button>
        </div>

        {/* Search */}
        {showSearch && (
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search by symbol or company name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        )}

        {/* Filters */}
        {showFilters && (
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-xs font-medium text-gray-700 mb-1">Sector</label>
              <select
                value={selectedSector}
                onChange={(e) => setSelectedSector(e.target.value)}
                className="w-full px-3 py-1.5 border border-gray-300 rounded-md text-sm bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All Sectors</option>
                {Object.entries(sectors).map(([sectorId, sector]) => (
                  <option key={sectorId} value={sector.name}>
                    {sector.name_turkish} ({sector.companies?.length || 0})
                  </option>
                ))}
              </select>
            </div>
            
            <div className="flex-1">
              <label className="block text-xs font-medium text-gray-700 mb-1">Market</label>
              <select
                value={selectedMarket}
                onChange={(e) => setSelectedMarket(e.target.value)}
                className="w-full px-3 py-1.5 border border-gray-300 rounded-md text-sm bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All Markets</option>
                <option value="bist_30">BIST 30</option>
                <option value="bist_50">BIST 50</option>
                <option value="bist_100">BIST 100</option>
                <option value="yildiz_pazar">Yıldız Pazar</option>
                <option value="ana_pazar">Ana Pazar</option>
              </select>
            </div>
          </div>
        )}

        {/* Selected Symbol Info */}
        {selectedSymbol && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-blue-900">{selectedSymbol}</span>
                  {stocks.find(s => s.symbol === selectedSymbol) && (
                    <>
                      <Badge className="bg-blue-100 text-blue-700 text-xs">
                        {stocks.find(s => s.symbol === selectedSymbol)?.sector_turkish}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {stocks.find(s => s.symbol === selectedSymbol)?.market_segment}
                      </Badge>
                    </>
                  )}
                </div>
                {stocks.find(s => s.symbol === selectedSymbol) && (
                  <div className="text-sm text-blue-700 mt-1">
                    {stocks.find(s => s.symbol === selectedSymbol)?.name_turkish}
                  </div>
                )}
              </div>
              {stocks.find(s => s.symbol === selectedSymbol) && (
                <div className="text-right">
                  <div className="font-bold text-blue-900">
                    ₺{formatPrice(stocks.find(s => s.symbol === selectedSymbol)!.last_price)}
                  </div>
                  <div className={`text-sm flex items-center gap-1 ${getPriceChangeColor(stocks.find(s => s.symbol === selectedSymbol)!.change_percent)}`}>
                    {getPriceChangeIcon(stocks.find(s => s.symbol === selectedSymbol)!.change_percent)}
                    {stocks.find(s => s.symbol === selectedSymbol)!.change_percent > 0 ? '+' : ''}
                    {stocks.find(s => s.symbol === selectedSymbol)!.change_percent.toFixed(2)}%
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Stock List */}
        <div className="max-h-64 overflow-y-auto space-y-1">
          {filteredStocks.map((stock) => (
            <div
              key={stock.symbol}
              onClick={() => onSymbolChange(stock.symbol)}
              className={`p-3 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
                selectedSymbol === stock.symbol 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-gray-900">{stock.symbol}</span>
                    <Badge variant="outline" className="text-xs">
                      {stock.sector_turkish}
                    </Badge>
                    {stock.bist_markets?.includes('bist_30') && (
                      <Badge className="bg-yellow-100 text-yellow-800 text-xs">BIST 30</Badge>
                    )}
                  </div>
                  <div className="text-sm text-gray-600 truncate">
                    {stock.name_turkish}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Market Cap: {formatMarketCap(stock.market_cap)}
                  </div>
                </div>
                
                <div className="text-right ml-4">
                  <div className="font-semibold text-gray-900">
                    ₺{formatPrice(stock.last_price)}
                  </div>
                  <div className={`text-sm flex items-center gap-1 justify-end ${getPriceChangeColor(stock.change_percent)}`}>
                    {getPriceChangeIcon(stock.change_percent)}
                    {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Vol: {stock.volume.toLocaleString('tr-TR')}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredStocks.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>No stocks found matching your criteria</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RealSymbolSelector;
