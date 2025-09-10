'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  ComposedChart,
  ReferenceLine
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Zap,
  Database,
  Clock,
  AlertTriangle,
  Target,
  Volume,
  Maximize2
} from 'lucide-react';

interface ChartDataPoint {
  timestamp: string;
  date: string;
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi_14?: number;
  macd_line?: number;
  macd_signal?: number;
  bollinger_upper?: number;
  bollinger_middle?: number;
  bollinger_lower?: number;
  atr_14?: number;
  adx_14?: number;
}

interface ProfessionalStockChartProps {
  symbol: string;
  onTimeframeChange?: (timeframe: string) => void;
}

export default function ProfessionalStockChart({ symbol, onTimeframeChange }: ProfessionalStockChartProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('60min');
  const [chartType, setChartType] = useState<'line' | 'area' | 'candlestick' | 'composed'>('area');
  const [showVolume, setShowVolume] = useState(true);
  const [showIndicators, setShowIndicators] = useState(true);
  const [dataSource, setDataSource] = useState<'enhanced' | 'historical' | 'auto'>('auto');
  const [error, setError] = useState<string | null>(null);

  const RAILWAY_API = process.env.NODE_ENV === 'development' 
    ? 'http://localhost:8080' 
    : 'https://bistai001-production.up.railway.app';

  const timeframes = [
    { value: '60min', label: '60 Min', source: 'historical' },
    { value: 'daily', label: 'Daily', source: 'enhanced' },
    { value: 'günlük', label: 'Günlük', source: 'enhanced' },
    { value: '30min', label: '30 Min', source: 'enhanced' },
    { value: '20min', label: '20 Min', source: 'enhanced' }
  ];

  useEffect(() => {
    if (symbol) {
      loadChartData();
    }
  }, [symbol, selectedTimeframe, dataSource]);

  const loadChartData = async () => {
    setLoading(true);
    setError(null);

    try {
      console.log(`📈 Loading chart data for ${symbol} (${selectedTimeframe})...`);

      const response = await fetch(
        `${RAILWAY_API}/api/bist/historical/${symbol}?timeframe=${selectedTimeframe}&limit=500`
      );

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.message || 'Failed to fetch data');
      }

      const historicalData = result.data.historical_data;
      console.log(`✅ Loaded ${historicalData.length} data points for ${symbol}`);

      // Process and format data
      const processedData = historicalData.map((item: any) => ({
        timestamp: item.timestamp,
        date: new Date(item.timestamp).toLocaleDateString(),
        time: new Date(item.timestamp).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
        open: parseFloat(item.open) || 0,
        high: parseFloat(item.high) || 0,
        low: parseFloat(item.low) || 0,
        close: parseFloat(item.close) || 0,
        volume: parseInt(item.volume) || 0,
        rsi_14: item.rsi_14 ? parseFloat(item.rsi_14) : undefined,
        macd_line: item.macd_line ? parseFloat(item.macd_line) : undefined,
        macd_signal: item.macd_signal ? parseFloat(item.macd_signal) : undefined,
        bollinger_upper: item.bollinger_upper ? parseFloat(item.bollinger_upper) : undefined,
        bollinger_middle: item.bollinger_middle ? parseFloat(item.bollinger_middle) : undefined,
        bollinger_lower: item.bollinger_lower ? parseFloat(item.bollinger_lower) : undefined,
        atr_14: item.atr_14 ? parseFloat(item.atr_14) : undefined,
        adx_14: item.adx_14 ? parseFloat(item.adx_14) : undefined
      })).reverse(); // Reverse to show chronological order

      setChartData(processedData);
      setDataSource(result.data.data_source === 'historical_data' ? 'historical' : 'enhanced');
      
      // Notify parent about timeframe change
      onTimeframeChange?.(selectedTimeframe);

    } catch (error) {
      console.error('❌ Chart data loading error:', error);
      setError(error instanceof Error ? error.message : 'Failed to load chart data');
      
      // Generate mock data as fallback
      const mockData = generateMockChartData();
      setChartData(mockData);
    } finally {
      setLoading(false);
    }
  };

  const generateMockChartData = (): ChartDataPoint[] => {
    const data: ChartDataPoint[] = [];
    let basePrice = 50 + Math.random() * 100;
    const now = new Date();

    for (let i = 100; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
      const volatility = 0.02;
      const change = (Math.random() - 0.5) * volatility * basePrice;
      
      const open = basePrice;
      const close = basePrice + change;
      const high = Math.max(open, close) + Math.random() * 0.01 * basePrice;
      const low = Math.min(open, close) - Math.random() * 0.01 * basePrice;
      
      data.push({
        timestamp: timestamp.toISOString(),
        date: timestamp.toLocaleDateString(),
        time: timestamp.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 1000000) + 100000,
        rsi_14: 30 + Math.random() * 40,
        macd_line: (Math.random() - 0.5) * 2,
        macd_signal: (Math.random() - 0.5) * 1.5,
        bollinger_upper: close + Math.random() * 5,
        bollinger_middle: close,
        bollinger_lower: close - Math.random() * 5,
        atr_14: Math.random() * 3,
        adx_14: 20 + Math.random() * 50
      });
      
      basePrice = close;
    }

    return data;
  };

  const getCurrentPrice = () => {
    if (chartData.length === 0) return 0;
    return chartData[chartData.length - 1]?.close || 0;
  };

  const getPriceChange = () => {
    if (chartData.length < 2) return { change: 0, changePercent: 0 };
    
    const current = chartData[chartData.length - 1]?.close || 0;
    const previous = chartData[chartData.length - 2]?.close || current;
    const change = current - previous;
    const changePercent = previous !== 0 ? (change / previous) * 100 : 0;
    
    return { change, changePercent };
  };

  const getLatestIndicators = () => {
    if (chartData.length === 0) return {};
    const latest = chartData[chartData.length - 1];
    return {
      rsi: latest?.rsi_14,
      macd: latest?.macd_line,
      atr: latest?.atr_14,
      adx: latest?.adx_14
    };
  };

  const formatPrice = (value: number) => `₺${value.toFixed(2)}`;
  const formatVolume = (value: number) => {
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(0)}K`;
    return value.toString();
  };

  const { change, changePercent } = getPriceChange();
  const indicators = getLatestIndicators();

  if (loading) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p>Loading chart data from Railway database...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-500" />
                {symbol} - Professional Chart
                <Badge variant={dataSource === 'historical' ? 'default' : 'secondary'}>
                  <Database className="w-3 h-3 mr-1" />
                  {dataSource === 'historical' ? 'Historical Data' : 'Enhanced Data'}
                </Badge>
              </CardTitle>
              <CardDescription>
                Railway PostgreSQL integration • {chartData.length} data points
              </CardDescription>
            </div>
            
            <div className="flex items-center gap-2">
              <div className="text-right">
                <div className="text-2xl font-bold">
                  {formatPrice(getCurrentPrice())}
                </div>
                <div className={`text-sm flex items-center gap-1 ${
                  change >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {change >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePercent.toFixed(2)}%)
                </div>
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-4 mb-6 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {timeframes.map(tf => (
                  <SelectItem key={tf.value} value={tf.value}>
                    {tf.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={chartType} onValueChange={(value) => setChartType(value as typeof chartType)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="area">Area</SelectItem>
                <SelectItem value="line">Line</SelectItem>
                <SelectItem value="composed">Composed</SelectItem>
              </SelectContent>
            </Select>

            <Button
              variant={showVolume ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowVolume(!showVolume)}
            >
              <Volume className="w-4 h-4 mr-1" />
              Volume
            </Button>

            <Button
              variant={showIndicators ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowIndicators(!showIndicators)}
            >
              <Activity className="w-4 h-4 mr-1" />
              Indicators
            </Button>

            <Button variant="outline" size="sm">
              <Maximize2 className="w-4 h-4 mr-1" />
              Fullscreen
            </Button>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg">
              <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm">{error} (showing mock data)</span>
              </div>
            </div>
          )}

          {/* Main Chart */}
          <div className="h-96 mb-6">
            <ResponsiveContainer width="100%" height="100%">
              {chartType === 'area' ? (
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="time" 
                    fontSize={12}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis 
                    domain={['dataMin - 2', 'dataMax + 2']}
                    fontSize={12}
                    tick={{ fontSize: 10 }}
                    tickFormatter={formatPrice}
                  />
                  <Tooltip 
                    formatter={(value: any, name: string) => [
                      name === 'close' ? formatPrice(value) : value,
                      name === 'close' ? 'Price' : name
                    ]}
                    labelFormatter={(label) => `Time: ${label}`}
                  />
                  <Area
                    type="monotone"
                    dataKey="close"
                    stroke="#3b82f6"
                    fill="url(#colorPrice)"
                    strokeWidth={2}
                  />
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  
                  {showIndicators && chartData.some(d => d.bollinger_upper) && (
                    <>
                      <Line
                        type="monotone"
                        dataKey="bollinger_upper"
                        stroke="#ef4444"
                        strokeWidth={1}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="bollinger_middle"
                        stroke="#6b7280"
                        strokeWidth={1}
                        strokeDasharray="5 5"
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="bollinger_lower"
                        stroke="#22c55e"
                        strokeWidth={1}
                        dot={false}
                      />
                    </>
                  )}
                </AreaChart>
              ) : chartType === 'line' ? (
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" fontSize={12} tick={{ fontSize: 10 }} />
                  <YAxis domain={['dataMin - 2', 'dataMax + 2']} fontSize={12} tick={{ fontSize: 10 }} tickFormatter={formatPrice} />
                  <Tooltip formatter={(value: any) => [formatPrice(value), 'Price']} />
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              ) : (
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" fontSize={12} tick={{ fontSize: 10 }} />
                  <YAxis yAxisId="price" domain={['dataMin - 2', 'dataMax + 2']} fontSize={12} tick={{ fontSize: 10 }} tickFormatter={formatPrice} />
                  {showVolume && (
                    <YAxis yAxisId="volume" orientation="right" fontSize={12} tick={{ fontSize: 10 }} tickFormatter={formatVolume} />
                  )}
                  <Tooltip 
                    formatter={(value: any, name: string) => {
                      if (name === 'volume') return [formatVolume(value), 'Volume'];
                      return [formatPrice(value), name];
                    }}
                  />
                  <Area
                    yAxisId="price"
                    type="monotone"
                    dataKey="close"
                    fill="#3b82f6"
                    stroke="#3b82f6"
                    fillOpacity={0.3}
                  />
                  {showVolume && (
                    <Bar yAxisId="volume" dataKey="volume" fill="#6b7280" opacity={0.3} />
                  )}
                </ComposedChart>
              )}
            </ResponsiveContainer>
          </div>

          {/* Volume Chart */}
          {showVolume && (
            <div className="h-24 mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <XAxis dataKey="time" hide />
                  <YAxis hide />
                  <Tooltip formatter={(value: any) => [formatVolume(value), 'Volume']} />
                  <Bar dataKey="volume" fill="#6b7280" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Technical Indicators */}
          {showIndicators && (
            <Tabs defaultValue="rsi" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="rsi">RSI</TabsTrigger>
                <TabsTrigger value="macd">MACD</TabsTrigger>
                <TabsTrigger value="bollinger">Bollinger</TabsTrigger>
                <TabsTrigger value="summary">Summary</TabsTrigger>
              </TabsList>

              <TabsContent value="rsi" className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <XAxis dataKey="time" fontSize={10} />
                    <YAxis domain={[0, 100]} fontSize={10} />
                    <Tooltip />
                    <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="2 2" />
                    <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="2 2" />
                    <Line type="monotone" dataKey="rsi_14" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </TabsContent>

              <TabsContent value="macd" className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <XAxis dataKey="time" fontSize={10} />
                    <YAxis fontSize={10} />
                    <Tooltip />
                    <ReferenceLine y={0} stroke="#6b7280" />
                    <Line type="monotone" dataKey="macd_line" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="macd_signal" stroke="#ef4444" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </TabsContent>

              <TabsContent value="bollinger" className="space-y-2">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div className="text-center p-2 bg-red-50 dark:bg-red-950/30 rounded">
                    <div className="text-red-600 font-semibold">Upper Band</div>
                    <div>₺{chartData[chartData.length - 1]?.bollinger_upper?.toFixed(2) || 'N/A'}</div>
                  </div>
                  <div className="text-center p-2 bg-gray-50 dark:bg-gray-900/50 rounded">
                    <div className="text-gray-600 font-semibold">Middle Band</div>
                    <div>₺{chartData[chartData.length - 1]?.bollinger_middle?.toFixed(2) || 'N/A'}</div>
                  </div>
                  <div className="text-center p-2 bg-green-50 dark:bg-green-950/30 rounded">
                    <div className="text-green-600 font-semibold">Lower Band</div>
                    <div>₺{chartData[chartData.length - 1]?.bollinger_lower?.toFixed(2) || 'N/A'}</div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="summary" className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-purple-50 dark:bg-purple-950/30 rounded">
                    <div className="text-sm text-gray-600 dark:text-gray-400">RSI (14)</div>
                    <div className="font-bold text-lg">
                      {indicators.rsi?.toFixed(1) || 'N/A'}
                    </div>
                    <div className="text-xs">
                      {!indicators.rsi ? 'No data' :
                        indicators.rsi > 70 ? 'Overbought' :
                        indicators.rsi < 30 ? 'Oversold' : 'Neutral'
                      }
                    </div>
                  </div>
                  
                  <div className="text-center p-3 bg-blue-50 dark:bg-blue-950/30 rounded">
                    <div className="text-sm text-gray-600 dark:text-gray-400">MACD</div>
                    <div className="font-bold text-lg">
                      {indicators.macd?.toFixed(3) || 'N/A'}
                    </div>
                    <div className="text-xs">
                      {!indicators.macd ? 'No data' :
                        indicators.macd > 0 ? 'Bullish' : 'Bearish'
                      }
                    </div>
                  </div>
                  
                  <div className="text-center p-3 bg-orange-50 dark:bg-orange-950/30 rounded">
                    <div className="text-sm text-gray-600 dark:text-gray-400">ATR (14)</div>
                    <div className="font-bold text-lg">
                      {indicators.atr?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-xs">Volatility</div>
                  </div>
                  
                  <div className="text-center p-3 bg-green-50 dark:bg-green-950/30 rounded">
                    <div className="text-sm text-gray-600 dark:text-gray-400">ADX (14)</div>
                    <div className="font-bold text-lg">
                      {indicators.adx?.toFixed(1) || 'N/A'}
                    </div>
                    <div className="text-xs">
                      {!indicators.adx ? 'No data' :
                        indicators.adx > 25 ? 'Strong Trend' : 'Weak Trend'
                      }
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </div>
  );
}