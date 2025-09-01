'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Brain, Clock, Target, AlertTriangle, RefreshCw, Search, ChevronDown } from 'lucide-react';

interface ForecastData {
  timestamp: string;
  actualPrice?: number;
  predictedPrice: number;
  predictedPriceMin: number;  // Minimum tahmin
  predictedPriceMax: number;  // Maximum tahmin
  confidence: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  priceChange: number;
  priceChangePercent: number;
  isMarketOpen: boolean;      // BIST market açık mı?
}

interface NewsImpact {
  headline: string;
  sentiment: number; // -1 to 1
  impact: 'HIGH' | 'MEDIUM' | 'LOW';
  timestamp: string;
}

interface ModelMetrics {
  accuracy: number;
  confidence?: number;
  mse?: number;
  lastUpdated?: string;
  trainingStatus?: 'TRAINED' | 'TRAINING' | 'ERROR';
  lastTraining?: string;
}

interface BISTStock {
  symbol: string;
  name: string;
  name_turkish: string;
  sector: string;
  last_price: number;
  change_percent: number;
}

interface ForecastPanelProps {
  selectedSymbol?: string;
}

export default function ForecastPanel({ selectedSymbol: propSelectedSymbol = 'AKBNK' }: ForecastPanelProps) {
  const [selectedSymbol, setSelectedSymbol] = useState(propSelectedSymbol);
  const [forecastHours, setForecastHours] = useState(8);  // BIST trading day (10:00-18:00)
  const [forecastData, setForecastData] = useState<any>({ predictions: [] });
  const [newsImpact, setNewsImpact] = useState<NewsImpact[]>([]);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [availableStocks, setAvailableStocks] = useState<BISTStock[]>([]);
  const [stocksLoading, setStocksLoading] = useState(true);
  
  // Search states for stock selector
  const [searchQuery, setSearchQuery] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [filteredStocks, setFilteredStocks] = useState<BISTStock[]>([]);

  // Update selectedSymbol when prop changes
  useEffect(() => {
    if (propSelectedSymbol && propSelectedSymbol !== selectedSymbol) {
      setSelectedSymbol(propSelectedSymbol);
    }
  }, [propSelectedSymbol, selectedSymbol]);



  // Fetch available stocks from Railway API
  const fetchAvailableStocks = async () => {
    try {
      setStocksLoading(true);
      const response = await fetch('https://bistai001-production.up.railway.app/api/bist/all-stocks?limit=600');
      
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.stocks) {
          setAvailableStocks(data.stocks);
          
          // Set default symbol to first available stock if current not available
          if (data.stocks.length > 0 && !data.stocks.find((s: BISTStock) => s.symbol === selectedSymbol)) {
            setSelectedSymbol(data.stocks[0].symbol);
          }
        }
      }
    } catch (error) {
      console.error('Failed to fetch available stocks:', error);
      // Fallback to a few default symbols if API fails
      setAvailableStocks([
        { symbol: 'GARAN', name: 'Türkiye Garanti Bankası', name_turkish: 'Garanti BBVA', sector: 'Banking', last_price: 149.6, change_percent: 1.01 },
        { symbol: 'AKBNK', name: 'Akbank T.A.Ş.', name_turkish: 'Akbank', sector: 'Banking', last_price: 69.85, change_percent: 2.34 }
      ]);
    } finally {
      setStocksLoading(false);
    }
  };

  // BIST Market Hours: 10:00 - 18:00 (Mon-Fri)
  const isBISTMarketOpen = (date: Date): boolean => {
    const day = date.getDay(); // 0=Sunday, 1=Monday, ..., 6=Saturday
    const hour = date.getHours();
    const minute = date.getMinutes();
    const timeInMinutes = hour * 60 + minute;
    
    // Weekend check
    if (day === 0 || day === 6) return false;
    
    // Market hours: 10:00 - 18:00
    const marketOpen = 10 * 60;  // 10:00 in minutes
    const marketClose = 18 * 60; // 18:00 in minutes
    
    return timeInMinutes >= marketOpen && timeInMinutes <= marketClose;
  };

  // Generate realistic forecast data
  const generateForecastData = (): ForecastData[] => {
    const data: ForecastData[] = [];
    const currentTime = new Date();
    let basePrice = 25 + Math.random() * 50; // Current price

    // Historical data (last 6 hours)
    for (let i = -6; i <= 0; i++) {
      const timestamp = new Date(currentTime.getTime() + i * 60 * 60 * 1000);
      const actualPrice = basePrice + (Math.random() - 0.5) * 2;
      const isOpen = isBISTMarketOpen(timestamp);
      
      data.push({
        timestamp: timestamp.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
        actualPrice,
        predictedPrice: actualPrice + (Math.random() - 0.5) * 0.5, // Close prediction for historical
        predictedPriceMin: actualPrice * 0.985, // ±1.5% aralık
        predictedPriceMax: actualPrice * 1.015,
        confidence: 0.85 + Math.random() * 0.1,
        signal: 'HOLD',
        priceChange: 0,
        priceChangePercent: 0,
        isMarketOpen: isOpen
      });
      
      basePrice = actualPrice;
    }

    // Future predictions
    for (let i = 1; i <= forecastHours; i++) {
      const timestamp = new Date(currentTime.getTime() + i * 60 * 60 * 1000);
      const isOpen = isBISTMarketOpen(timestamp);
      
      // Trend-based prediction with some noise
      // Market kapalıyken daha az volatilite
      const volatilityMultiplier = isOpen ? 1.0 : 0.3;
      const trend = (Math.random() - 0.5) * 0.02 * volatilityMultiplier; // ±1% trend
      const noise = (Math.random() - 0.5) * 0.01 * volatilityMultiplier; // ±0.5% noise
      const predictedPrice = basePrice * (1 + trend + noise);
      
      // Min-Max aralık (market kapalıyken daha dar)
      const priceVariation = isOpen ? 0.025 : 0.01; // ±2.5% veya ±1%
      const predictedPriceMin = predictedPrice * (1 - priceVariation);
      const predictedPriceMax = predictedPrice * (1 + priceVariation);
      
      const priceChange = predictedPrice - basePrice;
      const priceChangePercent = (priceChange / basePrice) * 100;
      
      // Generate signals based on price movement (sadece market açıkken)
      let signal: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
      if (isOpen) {
        if (priceChangePercent > 2) signal = 'BUY';
        else if (priceChangePercent < -2) signal = 'SELL';
      }
      
      // Confidence decreases over time, market kapalıyken daha düşük
      const baseConfidence = isOpen ? 0.95 : 0.75;
      const confidence = Math.max(0.6, baseConfidence - (i * 0.01));
      
      data.push({
        timestamp: timestamp.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
        predictedPrice: Number(predictedPrice.toFixed(2)),
        predictedPriceMin: Number(predictedPriceMin.toFixed(2)),
        predictedPriceMax: Number(predictedPriceMax.toFixed(2)),
        confidence: Number(confidence.toFixed(2)),
        signal,
        priceChange: Number(priceChange.toFixed(2)),
        priceChangePercent: Number(priceChangePercent.toFixed(2)),
        isMarketOpen: isOpen
      });
      
      basePrice = predictedPrice;
    }

    return data;
  };

  // Generate news sentiment data
  const generateNewsImpact = (): NewsImpact[] => {
    const headlines = [
      `${selectedSymbol} Q3 results exceed expectations`,
      `Analysts upgrade ${selectedSymbol} rating to BUY`,
      `New partnership announcement from ${selectedSymbol}`,
      `Market volatility affects ${selectedSymbol} trading`,
      `${selectedSymbol} CEO announces expansion plans`
    ];

    return headlines.slice(0, 3).map((headline, index) => ({
      headline,
      sentiment: (Math.random() - 0.5) * 2, // -1 to 1
      impact: Math.random() > 0.6 ? 'HIGH' : Math.random() > 0.3 ? 'MEDIUM' : 'LOW' as 'HIGH' | 'MEDIUM' | 'LOW',
      timestamp: new Date(Date.now() - index * 2 * 60 * 60 * 1000).toLocaleString('tr-TR')
    }));
  };

  // Generate model metrics
  const generateModelMetrics = (): ModelMetrics => {
    return {
      accuracy: 0.68 + Math.random() * 0.15, // 68-83% accuracy (realistic for stock prediction)
      mse: 0.02 + Math.random() * 0.03, // Mean squared error
      lastUpdated: new Date(Date.now() - Math.random() * 2 * 60 * 60 * 1000).toLocaleString('tr-TR'),
      trainingStatus: 'TRAINED'
    };
  };

  const fetchForecast = async () => {
    setIsLoading(true);
    
    try {
      // Call the Railway backend forecast endpoint (GET)
      const response = await fetch(`https://bistai001-production.up.railway.app/api/forecast/${selectedSymbol}?hours=${forecastHours}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.predictions && data.predictions.length > 0) {
          // Get the first future prediction (next hour)
          const nextPrediction = data.predictions.find((p: any) => p.actualPrice === null) || data.predictions[0];
          const currentPrice = data.predictions[0].actualPrice || data.predictions[0].predictedPrice;
          
          // Calculate range from predictions
          const prices = data.predictions.map((p: any) => p.predictedPrice);
          const minPrice = Math.min(...prices);
          const maxPrice = Math.max(...prices);
          
          // Transform backend response to frontend format with proper BIST sequential hours
          const bistTradingHours = ['10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00'];
          const transformedPredictions = data.predictions.slice(0, forecastHours).map((p: any, index: number) => {
            
            // Use sequential BIST trading hours
            let bistHour;
            if (index < bistTradingHours.length) {
              bistHour = bistTradingHours[index];
            } else {
              // If we need more hours than trading hours, cycle through or extend to next day
              const cycleIndex = index % bistTradingHours.length;
              bistHour = bistTradingHours[cycleIndex];
            }
            
            // Determine market status based on hour
            const hourNum = parseInt(bistHour.split(':')[0]);
            const isMarketOpen = hourNum >= 10 && hourNum <= 17; // 10:00-17:00 açık, 18:00 kapanış
            
            return {
              time: bistHour,
              predictedPrice: p.predictedPrice,
              actualPrice: p.actualPrice,
              confidence: p.confidence,
              isMarketOpen: isMarketOpen
            };
          });

          setForecastData({
            nextHourPrediction: nextPrediction.predictedPrice,
            range: { min: minPrice, max: maxPrice },
            confidence: nextPrediction.confidence,
            change: nextPrediction.priceChangePercent,
            predictions: transformedPredictions
          });
          
          // Use news impact from API
          setNewsImpact(data.newsImpact.slice(0, 5).map((news: any) => ({
            headline: news.headline,
            sentiment: news.sentiment,
            impact: news.impact,
            timestamp: news.timestamp
          })));
          
          setModelMetrics({
            accuracy: data.modelMetrics.accuracy,
            mse: data.modelMetrics.mse,
            lastUpdated: data.modelMetrics.lastUpdated,
            trainingStatus: data.modelMetrics.trainingStatus
          });
        } else {
          throw new Error('Invalid forecast data structure');
        }
      } else {
        throw new Error('Backend not available');
      }
    } catch (error) {
      console.error('Backend API error:', error);
      setForecastData({
        nextHourPrediction: 0,
        range: { min: 0, max: 0 },
        confidence: 0,
        change: 0,
        predictions: []
      });
      setNewsImpact([]);
      setModelMetrics({
        accuracy: 0,
        confidence: 0,
        lastTraining: 'Error loading model'
      });
    }
    
    setIsLoading(false);
    setLastUpdate(new Date().toLocaleTimeString('tr-TR'));
  };

  // Auto-fetch forecast data on component mount and when symbol/hours change
  useEffect(() => {
    if (selectedSymbol) {
      fetchForecast();
    }
  }, [selectedSymbol, forecastHours]);

  useEffect(() => {
    fetchAvailableStocks();
  }, []);

  // Filter stocks based on search query
  useEffect(() => {
    if (!availableStocks.length) {
      setFilteredStocks([]);
      return;
    }

    if (!searchQuery.trim()) {
      setFilteredStocks(availableStocks);
      return;
    }

    const filtered = availableStocks.filter(stock => 
      stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      stock.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      stock.name_turkish.toLowerCase().includes(searchQuery.toLowerCase())
    );
    
    setFilteredStocks(filtered);
  }, [availableStocks, searchQuery]);

  // Handle stock selection
  const handleStockSelect = (stock: BISTStock) => {
    setSelectedSymbol(stock.symbol);
    setSearchQuery(''); // Clear search after selection
    setIsDropdownOpen(false);
  };

  // Handle search input change
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setIsDropdownOpen(true);
  };

  // Get selected stock info for display
  const getSelectedStock = () => {
    return availableStocks.find(stock => stock.symbol === selectedSymbol);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const dropdown = document.querySelector('[data-dropdown="stock-selector"]');
      if (dropdown && !dropdown.contains(event.target as Node)) {
        setIsDropdownOpen(false);
        setSearchQuery('');
      }
    };

    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isDropdownOpen]);

  useEffect(() => {
    if (availableStocks.length > 0) {
      fetchForecast();
    }
  }, [selectedSymbol, forecastHours, availableStocks]);

  const ForecastTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white border-2 border-gray-200 rounded-lg shadow-xl p-3">
          <p className="font-semibold text-gray-900">{label}</p>
          
          {/* Market Status */}
          <div className="flex items-center gap-2 text-xs mb-2">
            <Badge className={`${
              data.isMarketOpen 
                ? 'bg-green-100 text-green-700' 
                : 'bg-gray-100 text-gray-600'
            }`}>
              {data.isMarketOpen ? 'Market Açık' : 'Market Kapalı'}
            </Badge>
          </div>

          {data.actualPrice && (
            <p className="text-sm">
              <span className="font-medium">Actual:</span> ₺{data.actualPrice.toFixed(2)}
            </p>
          )}
          <p className="text-sm">
            <span className="font-medium">Predicted:</span> ₺{data.predictedPrice?.toFixed(2)}
          </p>
          
          {/* Min-Max Range */}
          {data.predictedPriceMin && data.predictedPriceMax && (
            <p className="text-sm text-gray-600">
              <span className="font-medium">Range:</span> ₺{data.predictedPriceMin.toFixed(2)} - ₺{data.predictedPriceMax.toFixed(2)}
            </p>
          )}
          
          <p className="text-sm">
            <span className="font-medium">Confidence:</span> {(data.confidence * 100).toFixed(0)}%
          </p>
          <p className="text-sm">
            <span className="font-medium">Signal:</span> 
            <span className={`ml-1 font-semibold ${
              data.signal === 'BUY' ? 'text-green-600' : 
              data.signal === 'SELL' ? 'text-red-600' : 'text-gray-600'
            }`}>
              {data.signal}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  // Safely handle forecastData - ensure it's an array
  const predictions = Array.isArray(forecastData?.predictions) ? forecastData.predictions : [];
  
  const currentPrediction = predictions.find((d: any) => !d.actualPrice) || {
    prediction: forecastData?.nextHourPrediction || 0,
    confidence: forecastData?.confidence || 0,
    signal: 'HOLD'
  };
  
  const avgConfidence = predictions.length > 0 
    ? predictions.reduce((sum: number, d: any) => sum + d.confidence, 0) / predictions.length 
    : forecastData?.confidence || 0;
    
  const bullishSignals = predictions.filter((d: any) => d.signal === 'BUY').length;
  const bearishSignals = predictions.filter((d: any) => d.signal === 'SELL').length;

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-6 w-6 text-purple-600" />
              <div>
                <CardTitle className="text-xl">DP-LSTM Price Forecast</CardTitle>
                <CardDescription>AI-powered BIST stock price predictions with news sentiment</CardDescription>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {/* Custom Searchable Stock Selector */}
              <div className="relative w-64" data-dropdown="stock-selector">
                <div 
                  className="flex items-center gap-2 px-3 py-2 border border-gray-300 rounded-md bg-white cursor-pointer hover:border-gray-400 transition-colors"
                  onClick={() => !stocksLoading && setIsDropdownOpen(!isDropdownOpen)}
                >
                  {stocksLoading ? (
                    <span className="text-gray-500 text-sm">Loading...</span>
                  ) : (
                    <>
                      <span className="font-mono text-sm">{selectedSymbol}</span>
                      {getSelectedStock() && (
                        <span className="text-xs text-gray-500">
                          {getSelectedStock()!.change_percent > 0 ? '📈' : 
                           getSelectedStock()!.change_percent < 0 ? '📉' : '➡️'}
                          {getSelectedStock()!.change_percent.toFixed(2)}%
                        </span>
                      )}
                    </>
                  )}
                  <ChevronDown className={`h-4 w-4 text-gray-400 ml-auto transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
                </div>

                {/* Dropdown */}
                {isDropdownOpen && !stocksLoading && (
                  <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-md shadow-lg z-50 max-h-80 flex flex-col">
                    {/* Search Input */}
                    <div className="p-2 border-b">
                      <div className="relative">
                        <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                        <input
                          type="text"
                          value={searchQuery}
                          onChange={handleSearchChange}
                          placeholder="Search stocks..."
                          className="w-full pl-8 pr-3 py-1.5 text-sm border border-gray-200 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          autoFocus
                        />
                      </div>
                    </div>
                    
                    {/* Stock List */}
                    <div className="flex-1 overflow-y-auto">
                      {filteredStocks.length === 0 ? (
                        <div className="p-3 text-sm text-gray-500 text-center">
                          {searchQuery ? 'No stocks found' : 'No stocks available'}
                        </div>
                      ) : (
                        filteredStocks.slice(0, 50).map(stock => (
                          <div
                            key={stock.symbol}
                            className={`px-3 py-2 cursor-pointer hover:bg-gray-100 flex items-center justify-between text-sm border-b border-gray-50 last:border-b-0 ${
                              stock.symbol === selectedSymbol ? 'bg-blue-50' : ''
                            }`}
                            onClick={() => handleStockSelect(stock)}
                          >
                            <div className="flex flex-col flex-1 min-w-0">
                              <span className="font-mono font-medium">{stock.symbol}</span>
                              <span className="text-xs text-gray-500 truncate" title={stock.name}>
                                {stock.name_turkish}
                              </span>
                            </div>
                            <span className={`text-xs px-1 py-0.5 rounded ${
                              stock.change_percent > 0 ? 'bg-green-100 text-green-700' :
                              stock.change_percent < 0 ? 'bg-red-100 text-red-700' : 
                              'bg-gray-100 text-gray-700'
                            }`}>
                              {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                            </span>
                          </div>
                        ))
                      )}
                      
                      {filteredStocks.length > 50 && (
                        <div className="p-2 text-xs text-gray-500 text-center bg-gray-50">
                          Showing first 50 results. Keep typing to narrow down...
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              
              <Select value={forecastHours.toString()} onValueChange={(value) => setForecastHours(parseInt(value))}>
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="4">4H (Yarım gün)</SelectItem>
                  <SelectItem value="8">8H (1 işlem günü)</SelectItem>
                  <SelectItem value="16">16H (2 işlem günü)</SelectItem>
                  <SelectItem value="40">40H (1 hafta)</SelectItem>
                </SelectContent>
              </Select>
              
              <Button 
                size="sm" 
                onClick={fetchForecast} 
                disabled={isLoading}
                className="gap-1"
              >
                <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
                {isLoading ? 'Loading...' : 'Refresh'}
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Next Hour Prediction</p>
                <div className="text-lg font-bold text-blue-600 mb-1">
                  ₺{currentPrediction?.predictedPrice?.toFixed(2) || '0.00'}
                </div>
                <div className="text-xs text-gray-600 mb-1">
                  Range: ₺{currentPrediction?.predictedPriceMin?.toFixed(2) || '0.00'} - ₺{currentPrediction?.predictedPriceMax?.toFixed(2) || '0.00'}
                </div>
                {currentPrediction?.priceChangePercent && (
                  <div className={`text-xs flex items-center gap-1 ${
                    currentPrediction.priceChangePercent >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {currentPrediction.priceChangePercent >= 0 ? (
                      <TrendingUp className="h-3 w-3" />
                    ) : (
                      <TrendingDown className="h-3 w-3" />
                    )}
                    {currentPrediction.priceChangePercent >= 0 ? '+' : ''}{currentPrediction.priceChangePercent.toFixed(1)}%
                  </div>
                )}
                {currentPrediction && (
                  <div className="mt-1">
                    <Badge className={`text-xs ${
                      currentPrediction.isMarketOpen 
                        ? 'bg-green-100 text-green-700 border-green-300' 
                        : 'bg-gray-100 text-gray-700 border-gray-300'
                    }`}>
                      {currentPrediction.isMarketOpen ? 'Market Açık' : 'Market Kapalı'}
                    </Badge>
                  </div>
                )}
              </div>
              <Target className="h-5 w-5 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Model Confidence</p>
                <p className="text-xl font-bold text-green-600">
                  {(avgConfidence * 100).toFixed(0)}%
                </p>
                <p className="text-xs text-gray-500">
                  Accuracy: {modelMetrics ? (modelMetrics.accuracy * 100).toFixed(0) : '0'}%
                </p>
              </div>
              <Brain className="h-5 w-5 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Trading Signals</p>
                <div className="flex gap-2 mt-1">
                  <span className="text-sm font-semibold text-green-600">
                    {bullishSignals} BUY
                  </span>
                  <span className="text-sm font-semibold text-red-600">
                    {bearishSignals} SELL
                  </span>
                </div>
                <p className="text-xs text-gray-500">{forecastHours} saat tahmin</p>
              </div>
              <TrendingUp className="h-5 w-5 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Last Updated</p>
                <p className="text-sm font-semibold text-gray-900">{lastUpdate}</p>
                <Badge 
                  className={`mt-1 text-xs ${
                    modelMetrics?.trainingStatus === 'TRAINED' ? 
                      'bg-green-100 text-green-700 border-green-300' : 
                      'bg-yellow-100 text-yellow-700 border-yellow-300'
                  }`}
                >
                  {modelMetrics?.trainingStatus || 'LOADING'}
                </Badge>
              </div>
              <Clock className="h-5 w-5 text-gray-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Forecast Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            BIST Fiyat Tahmini - {selectedSymbol}
          </CardTitle>
          <CardDescription>
            Dashed gray: Historical prices • Green solid: Market open predictions • Red dashed: Market closed predictions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="h-80 flex items-center justify-center">
              <div className="text-center">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2 text-blue-500" />
                <p className="text-gray-500">Loading DP-LSTM predictions...</p>
              </div>
            </div>
          ) : (
            <div style={{ width: '100%', height: 320 }}>
              <ResponsiveContainer>
                <LineChart data={forecastData.predictions}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fontSize: 11, fill: '#6b7280' }}
                    axisLine={{ stroke: '#d1d5db' }}
                  />
                  <YAxis 
                    domain={['dataMin - 1', 'dataMax + 1']}
                    tick={{ fontSize: 11, fill: '#6b7280' }}
                    axisLine={{ stroke: '#d1d5db' }}
                  />
                  <Tooltip content={<ForecastTooltip />} />
                  
                  {/* Historical actual prices */}
                  <Line 
                    type="monotone" 
                    dataKey="actualPrice" 
                    stroke="#6b7280"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={(props: any) => {
                      const { payload, index } = props;
                      return (
                        <circle 
                          key={`historical-${index}`}
                          cx={props.cx} 
                          cy={props.cy} 
                          r={3}
                          fill={payload?.isMarketOpen ? "#6b7280" : "#d1d5db"}
                          stroke="#fff"
                          strokeWidth={1}
                        />
                      );
                    }}
                    connectNulls={false}
                  />
                  
                  {/* Predicted prices - Market Open */}
                  <Line 
                    type="monotone" 
                    dataKey="predictedPrice" 
                    stroke="#10b981"
                    strokeWidth={3}
                    dot={(props: any) => {
                      const { payload, index } = props;
                      if (!payload?.actualPrice && payload?.isMarketOpen) {
                        return (
                          <circle 
                            key={`predicted-open-${index}`}
                            cx={props.cx} 
                            cy={props.cy} 
                            r={4}
                            fill="#10b981"
                            stroke="#fff"
                            strokeWidth={2}
                          />
                        );
                      }
                      return null;
                    }}
                  />
                  
                  {/* Predicted prices - Market Closed */}
                  <Line 
                    type="monotone" 
                    dataKey="predictedPrice" 
                    stroke="#ef4444"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    dot={(props: any) => {
                      const { payload, index } = props;
                      if (!payload?.actualPrice && !payload?.isMarketOpen) {
                        return (
                          <circle 
                            key={`predicted-closed-${index}`}
                            cx={props.cx} 
                            cy={props.cy} 
                            r={3}
                            fill="#ef4444"
                            stroke="#fff"
                            strokeWidth={1}
                            opacity={0.7}
                          />
                        );
                      }
                      return null;
                    }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      {/* News Sentiment Impact */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            News Sentiment Impact
          </CardTitle>
          <CardDescription>
            Recent financial news affecting {selectedSymbol} price predictions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {newsImpact.map((news, index) => (
              <div key={index} className="border rounded-lg p-3 hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="text-sm font-semibold text-gray-900 flex-1">{news.headline}</h4>
                  <div className="flex items-center gap-2 ml-3">
                    <Badge className={`text-xs ${
                      news.sentiment > 0.3 ? 'bg-green-100 text-green-700 border-green-300' :
                      news.sentiment < -0.3 ? 'bg-red-100 text-red-700 border-red-300' :
                      'bg-gray-100 text-gray-700 border-gray-300'
                    }`}>
                      {news.sentiment > 0.3 ? 'Positive' : news.sentiment < -0.3 ? 'Negative' : 'Neutral'}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {news.impact}
                    </Badge>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>Sentiment Score: {news.sentiment.toFixed(2)}</span>
                  <span>{news.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
