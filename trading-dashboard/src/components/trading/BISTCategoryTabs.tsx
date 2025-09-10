'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  TrendingUp, 
  TrendingDown, 
  Search, 
  Filter, 
  Star, 
  Activity, 
  BarChart3, 
  Zap,
  Database,
  Clock,
  ArrowUpCircle,
  ArrowDownCircle
} from 'lucide-react';

interface BISTStock {
  symbol: string;
  name: string;
  sector: string;
  category: string;
  priority: number;
  latest_price: number;
  latest_date: string | null;
  data_sources: {
    enhanced_available: boolean;
    historical_available: boolean;
    enhanced_records: number;
    historical_records: number;
    api_available: boolean;
  };
  technical_indicators: {
    rsi_14: number | null;
    macd: number | null;
    bollinger_upper: number | null;
    bollinger_middle: number | null;
    bollinger_lower: number | null;
    atr_14: number | null;
    adx_14: number | null;
  };
}

interface CategoryStats {
  category: string;
  total_stocks: number;
  enhanced_data_count: number;
  historical_data_count: number;
  api_available_count: number;
}

interface BISTCategoryTabsProps {
  onStockSelect?: (symbol: string) => void;
  selectedSymbol?: string;
}

export default function BISTCategoryTabs({ onStockSelect, selectedSymbol }: BISTCategoryTabsProps) {
  const [activeCategory, setActiveCategory] = useState<'BIST_30' | 'BIST_50' | 'BIST_100'>('BIST_30');
  const [stocks, setStocks] = useState<{ [key: string]: BISTStock[] }>({});
  const [categoryStats, setCategoryStats] = useState<CategoryStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sectorFilter, setSectorFilter] = useState('all');
  const [sortBy, setSortBy] = useState<'symbol' | 'price' | 'sector' | 'rsi'>('symbol');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const RAILWAY_API = process.env.NODE_ENV === 'development' 
    ? 'http://localhost:8080' 
    : 'https://bistai001-production.up.railway.app';
  
  useEffect(() => {
    loadBISTData();
  }, []);

  const loadBISTData = async () => {
    setLoading(true);
    try {
      console.log('🚀 Loading BIST Category Data from Railway...');
      
      // Load all categories using FIXED endpoints
      const categories = ['BIST_30', 'BIST_50', 'BIST_100'];
      const stockPromises = categories.map(async category => {
        // Try fixed endpoint first, fallback to original
        let response = await fetch(`${RAILWAY_API}/api/bist/stocks-fixed/${category}?limit=200`);
        if (!response.ok) {
          console.warn(`⚠️ Fixed endpoint failed for ${category}, trying original...`);
          response = await fetch(`${RAILWAY_API}/api/bist/stocks/${category}?limit=200`);
        }
        const data = await response.json();
        return { category, data: data.data.stocks };
      });
      
      // Load summary stats
      const summaryPromise = fetch(`${RAILWAY_API}/api/bist/summary`)
        .then(res => res.json())
        .then(data => data.data.category_stats);
      
      const [stockResults, summaryStats] = await Promise.all([
        Promise.all(stockPromises),
        summaryPromise
      ]);
      
      // Process stock data by category
      const stocksByCategory: { [key: string]: BISTStock[] } = {};
      stockResults.forEach(({ category, data }) => {
        stocksByCategory[category] = data;
      });
      
      setStocks(stocksByCategory);
      setCategoryStats(summaryStats);
      
      console.log('✅ BIST Data Loaded:', {
        categories: Object.keys(stocksByCategory),
        totalStocks: Object.values(stocksByCategory).reduce((sum, stocks) => sum + stocks.length, 0),
        stats: summaryStats
      });
      
    } catch (error) {
      console.error('❌ Error loading BIST data:', error);
      
      // Fallback mock data for development
      const mockStock: BISTStock = {
        symbol: 'AKBNK',
        name: 'Akbank T.A.Ş.',
        sector: 'Banking',
        category: 'BIST_30',
        priority: 1,
        latest_price: 72.45,
        latest_date: new Date().toISOString(),
        data_sources: {
          enhanced_available: true,
          historical_available: true,
          enhanced_records: 5000,
          historical_records: 3000,
          api_available: true
        },
        technical_indicators: {
          rsi_14: 65.2,
          macd: 0.15,
          bollinger_upper: 75.0,
          bollinger_middle: 72.0,
          bollinger_lower: 69.0,
          atr_14: 1.25,
          adx_14: 28.5
        }
      };
      
      setStocks({
        BIST_30: Array(30).fill(null).map((_, i) => ({ ...mockStock, symbol: `STOCK${i + 1}` })),
        BIST_50: Array(50).fill(null).map((_, i) => ({ ...mockStock, symbol: `STOCK${i + 1}` })),
        BIST_100: Array(100).fill(null).map((_, i) => ({ ...mockStock, symbol: `STOCK${i + 1}` }))
      });
      
      setCategoryStats([
        { category: 'BIST_30', total_stocks: 30, enhanced_data_count: 10, historical_data_count: 30, api_available_count: 30 },
        { category: 'BIST_50', total_stocks: 50, enhanced_data_count: 17, historical_data_count: 50, api_available_count: 50 },
        { category: 'BIST_100', total_stocks: 100, enhanced_data_count: 39, historical_data_count: 100, api_available_count: 100 }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getFilteredAndSortedStocks = (categoryStocks: BISTStock[]) => {
    let filtered = categoryStocks;
    
    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(stock => 
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        stock.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        stock.sector.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // Sector filter
    if (sectorFilter !== 'all') {
      filtered = filtered.filter(stock => stock.sector === sectorFilter);
    }
    
    // Sorting
    filtered.sort((a, b) => {
      let compareValue = 0;
      
      switch (sortBy) {
        case 'symbol':
          compareValue = a.symbol.localeCompare(b.symbol);
          break;
        case 'price':
          compareValue = a.latest_price - b.latest_price;
          break;
        case 'sector':
          compareValue = a.sector.localeCompare(b.sector);
          break;
        case 'rsi':
          compareValue = (a.technical_indicators.rsi_14 || 0) - (b.technical_indicators.rsi_14 || 0);
          break;
      }
      
      return sortDirection === 'asc' ? compareValue : -compareValue;
    });
    
    return filtered;
  };

  const getUniqueSecors = (categoryStocks: BISTStock[]) => {
    return [...new Set(categoryStocks.map(stock => stock.sector))].sort();
  };

  const getRSIColor = (rsi: number | null) => {
    if (!rsi) return 'text-gray-400';
    if (rsi >= 70) return 'text-red-500';
    if (rsi <= 30) return 'text-green-500';
    return 'text-blue-500';
  };

  const getDataSourceBadges = (stock: BISTStock) => {
    const badges = [];
    
    if (stock.data_sources.enhanced_available) {
      badges.push(
        <Badge key="enhanced" variant="default" className="bg-blue-600 text-xs">
          <Database className="w-3 h-3 mr-1" />
          Enhanced ({stock.data_sources.enhanced_records.toLocaleString()})
        </Badge>
      );
    }
    
    if (stock.data_sources.historical_available) {
      badges.push(
        <Badge key="historical" variant="default" className="bg-purple-600 text-xs">
          <Clock className="w-3 h-3 mr-1" />
          Historical ({stock.data_sources.historical_records.toLocaleString()})
        </Badge>
      );
    }
    
    if (stock.data_sources.api_available) {
      badges.push(
        <Badge key="api" variant="default" className="bg-green-600 text-xs">
          <Zap className="w-3 h-3 mr-1" />
          Real-time
        </Badge>
      );
    }
    
    return badges;
  };

  const renderStocksList = (categoryStocks: BISTStock[]) => {
    const filteredStocks = getFilteredAndSortedStocks(categoryStocks);
    
    return (
      <ScrollArea className="h-[600px] pr-4">
        <div className="space-y-2">
          {filteredStocks.map((stock) => (
            <Card 
              key={stock.symbol} 
              className={`cursor-pointer transition-all duration-200 hover:shadow-lg border-l-4 ${
                selectedSymbol === stock.symbol 
                  ? 'border-l-blue-500 bg-blue-50 dark:bg-blue-950/50' 
                  : 'border-l-gray-200 dark:border-l-gray-700 hover:border-l-blue-300'
              }`}
              onClick={() => onStockSelect?.(stock.symbol)}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  {/* Stock Info */}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h3 className="font-bold text-lg">{stock.symbol}</h3>
                      <Badge variant="outline" className="text-xs">
                        {stock.sector}
                      </Badge>
                      {selectedSymbol === stock.symbol && (
                        <Star className="w-4 h-4 text-yellow-500" fill="currentColor" />
                      )}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {stock.name}
                    </p>
                    <div className="flex flex-wrap gap-1 mb-2">
                      {getDataSourceBadges(stock)}
                    </div>
                  </div>
                  
                  {/* Price & Indicators */}
                  <div className="text-right">
                    <div className="text-xl font-bold mb-2">
                      ₺{stock.latest_price.toFixed(2)}
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {stock.technical_indicators.rsi_14 && (
                        <div className="flex items-center gap-1">
                          <Activity className="w-3 h-3" />
                          <span className={getRSIColor(stock.technical_indicators.rsi_14)}>
                            RSI: {stock.technical_indicators.rsi_14.toFixed(1)}
                          </span>
                        </div>
                      )}
                      
                      {stock.technical_indicators.macd && (
                        <div className="flex items-center gap-1">
                          <BarChart3 className="w-3 h-3" />
                          <span className={stock.technical_indicators.macd > 0 ? 'text-green-500' : 'text-red-500'}>
                            MACD: {stock.technical_indicators.macd.toFixed(3)}
                          </span>
                        </div>
                      )}
                      
                      {stock.technical_indicators.atr_14 && (
                        <div className="flex items-center gap-1 text-xs text-gray-600">
                          ATR: {stock.technical_indicators.atr_14.toFixed(2)}
                        </div>
                      )}
                      
                      {stock.technical_indicators.adx_14 && (
                        <div className="flex items-center gap-1 text-xs text-gray-600">
                          ADX: {stock.technical_indicators.adx_14.toFixed(1)}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    );
  };

  const getCategoryStats = (category: string) => {
    return categoryStats.find(stat => stat.category === category);
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p>Loading BIST categories from Railway database...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-blue-500" />
            BIST Categories - Professional Trading Interface
          </CardTitle>
          <CardDescription>
            Railway PostgreSQL integration with 1.17M+ historical records
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs value={activeCategory} onValueChange={(value) => setActiveCategory(value as typeof activeCategory)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="BIST_30" className="flex items-center gap-2">
                <Star className="w-4 h-4" />
                BIST 30
                <Badge variant="secondary" className="ml-1">
                  {getCategoryStats('BIST_30')?.total_stocks || 30}
                </Badge>
              </TabsTrigger>
              <TabsTrigger value="BIST_50" className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                BIST 50
                <Badge variant="secondary" className="ml-1">
                  {getCategoryStats('BIST_50')?.total_stocks || 50}
                </Badge>
              </TabsTrigger>
              <TabsTrigger value="BIST_100" className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                BIST 100
                <Badge variant="secondary" className="ml-1">
                  {getCategoryStats('BIST_100')?.total_stocks || 100}
                </Badge>
              </TabsTrigger>
            </TabsList>

            {/* Search and Filter Controls */}
            <div className="flex flex-wrap items-center gap-4 my-4 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
              <div className="flex items-center gap-2 flex-1">
                <Search className="w-4 h-4 text-gray-500" />
                <Input
                  placeholder="Search by symbol, name, or sector..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="max-w-xs"
                />
              </div>
              
              <Select value={sectorFilter} onValueChange={setSectorFilter}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Filter by sector" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sectors</SelectItem>
                  {stocks[activeCategory] && getUniqueSecors(stocks[activeCategory]).map(sector => (
                    <SelectItem key={sector} value={sector}>{sector}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Select value={sortBy} onValueChange={(value) => setSortBy(value as typeof sortBy)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="symbol">Symbol</SelectItem>
                  <SelectItem value="price">Price</SelectItem>
                  <SelectItem value="sector">Sector</SelectItem>
                  <SelectItem value="rsi">RSI</SelectItem>
                </SelectContent>
              </Select>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')}
              >
                {sortDirection === 'asc' ? 
                  <ArrowUpCircle className="w-4 h-4" /> : 
                  <ArrowDownCircle className="w-4 h-4" />
                }
              </Button>
            </div>

            {/* Category Stats */}
            {categoryStats.length > 0 && (
              <div className="grid grid-cols-3 gap-4 mb-4">
                {getCategoryStats(activeCategory) && (
                  <>
                    <div className="text-center p-2 bg-blue-50 dark:bg-blue-950/30 rounded">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Enhanced Data</div>
                      <div className="font-bold text-blue-600">
                        {getCategoryStats(activeCategory)?.enhanced_data_count}
                      </div>
                    </div>
                    <div className="text-center p-2 bg-purple-50 dark:bg-purple-950/30 rounded">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Historical Data</div>
                      <div className="font-bold text-purple-600">
                        {getCategoryStats(activeCategory)?.historical_data_count}
                      </div>
                    </div>
                    <div className="text-center p-2 bg-green-50 dark:bg-green-950/30 rounded">
                      <div className="text-sm text-gray-600 dark:text-gray-400">API Available</div>
                      <div className="font-bold text-green-600">
                        {getCategoryStats(activeCategory)?.api_available_count}
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Stock Lists by Category */}
            <TabsContent value="BIST_30">
              {stocks.BIST_30 ? renderStocksList(stocks.BIST_30) : <div>Loading BIST_30 stocks...</div>}
            </TabsContent>
            
            <TabsContent value="BIST_50">
              {stocks.BIST_50 ? renderStocksList(stocks.BIST_50) : <div>Loading BIST_50 stocks...</div>}
            </TabsContent>
            
            <TabsContent value="BIST_100">
              {stocks.BIST_100 ? renderStocksList(stocks.BIST_100) : <div>Loading BIST_100 stocks...</div>}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
