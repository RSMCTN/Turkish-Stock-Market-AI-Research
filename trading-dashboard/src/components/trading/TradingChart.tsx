'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Volume, Activity, Target, AlertTriangle } from 'lucide-react';
import CandlestickChart from './CandlestickChart';

interface ChartData {
  time: string;
  price: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  close: number;
  signal?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
}

interface TradingChartProps {
  symbol?: string;
  timeframe?: string;
}

export default function TradingChart({ symbol = 'AKBNK', timeframe = '1H' }: TradingChartProps) {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState(symbol);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [chartType, setChartType] = useState<'line' | 'area' | 'candlestick'>('area');

  // Popular BIST symbols
  const symbols = [
    'AKBNK', 'GARAN', 'ISCTR', 'THYAO', 'ASELS', 'SISE', 'EREGL', 
    'PETKM', 'ARCELIK', 'MGROS', 'TCELL', 'VAKBN', 'HALKB', 'SKBNK'
  ];

  const timeframes = [
    { value: '5M', label: '5 Min' },
    { value: '15M', label: '15 Min' },
    { value: '1H', label: '1 Hour' },
    { value: '4H', label: '4 Hours' },
    { value: '1D', label: '1 Day' }
  ];

  // Generate realistic trading data
  const generateChartData = (periods: number = 48) => {
    const data: ChartData[] = [];
    const now = new Date();
    let basePrice = 25 + Math.random() * 50; // 25-75 TL range
    
    for (let i = periods; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000); // Hourly data
      
      // Price movement with some volatility
      const change = (Math.random() - 0.5) * 0.04; // ±2% max change
      const newPrice = Math.max(basePrice * (1 + change), 1);
      
      const high = newPrice * (1 + Math.random() * 0.02);
      const low = newPrice * (1 - Math.random() * 0.02);
      const volume = 500000 + Math.random() * 1500000;
      
      // Generate trading signals
      let signal: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
      let confidence = 0.5 + Math.random() * 0.4; // 50-90%
      
      if (Math.random() > 0.85) {
        signal = Math.random() > 0.5 ? 'BUY' : 'SELL';
        confidence = 0.7 + Math.random() * 0.25; // 70-95% for signals
      }
      
      data.push({
        time: time.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
        price: Number(newPrice.toFixed(2)),
        volume: Math.round(volume),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        open: Number(basePrice.toFixed(2)),
        close: Number(newPrice.toFixed(2)),
        signal,
        confidence: Number(confidence.toFixed(2))
      });
      
      basePrice = newPrice;
    }
    
    return data;
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const data = generateChartData();
      setChartData(data);
      setIsLoading(false);
    };
    
    loadData();
  }, [selectedSymbol, selectedTimeframe]);

  // Calculate metrics
  const currentPrice = chartData[chartData.length - 1]?.price || 0;
  const previousPrice = chartData[chartData.length - 2]?.price || 0;
  const priceChange = currentPrice - previousPrice;
  const priceChangePercent = previousPrice ? (priceChange / previousPrice) * 100 : 0;
  
  const dayHigh = Math.max(...chartData.map(d => d.high));
  const dayLow = Math.min(...chartData.map(d => d.low));
  const avgVolume = chartData.reduce((sum, d) => sum + d.volume, 0) / chartData.length;
  
  // Get latest signals
  const signals = chartData.filter(d => d.signal !== 'HOLD').slice(-3);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white border-2 border-gray-200 rounded-lg shadow-xl p-3">
          <p className="font-semibold text-gray-900">{label}</p>
          <p className="text-sm text-gray-700">
            <span className="font-medium">Price:</span> ₺{data.price.toFixed(2)}
          </p>
          <p className="text-sm text-gray-600">
            <span className="font-medium">High:</span> ₺{data.high.toFixed(2)}
          </p>
          <p className="text-sm text-gray-600">
            <span className="font-medium">Low:</span> ₺{data.low.toFixed(2)}
          </p>
          <p className="text-sm text-gray-700">
            <span className="font-medium">Volume:</span> {data.volume.toLocaleString()}
          </p>
          {data.signal !== 'HOLD' && (
            <div className="mt-2 pt-2 border-t border-gray-200">
              <Badge 
                className={`text-xs font-semibold ${
                  data.signal === 'BUY' 
                    ? 'bg-green-100 text-green-800 border border-green-300' 
                    : 'bg-red-100 text-red-800 border border-red-300'
                }`}
              >
                {data.signal} {(data.confidence * 100).toFixed(0)}%
              </Badge>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Loading Chart...</CardTitle>
              <CardDescription>Fetching real-time data</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Chart Header & Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl">{selectedSymbol}</CardTitle>
              <CardDescription>Real-time BIST trading data</CardDescription>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Symbol Selector */}
              <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {symbols.map(sym => (
                    <SelectItem key={sym} value={sym}>{sym}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              {/* Timeframe Selector */}
              <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {timeframes.map(tf => (
                    <SelectItem key={tf.value} value={tf.value}>{tf.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              {/* Chart Type Selector */}
              <Select value={chartType} onValueChange={(value: any) => setChartType(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="line">Line</SelectItem>
                  <SelectItem value="area">Area</SelectItem>
                  <SelectItem value="candlestick">Candlestick</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Price & Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Current Price</p>
                <p className="text-2xl font-bold">₺{currentPrice.toFixed(2)}</p>
                <div className="flex items-center gap-1 mt-1">
                  {priceChangePercent >= 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-600" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-600" />
                  )}
                  <span className={`text-sm font-semibold ${priceChangePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Day Range</p>
                <p className="text-lg font-semibold">₺{dayLow.toFixed(2)} - ₺{dayHigh.toFixed(2)}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Spread: {((dayHigh - dayLow) / dayLow * 100).toFixed(1)}%
                </p>
              </div>
              <Activity className="h-5 w-5 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Volume</p>
                <p className="text-lg font-semibold">{avgVolume.toLocaleString('tr-TR', { maximumFractionDigits: 0 })}</p>
                <p className="text-xs text-muted-foreground mt-1">Last {chartData.length}H</p>
              </div>
              <Volume className="h-5 w-5 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Signals</p>
                <p className="text-lg font-semibold">{signals.length}</p>
                <div className="flex gap-1 mt-1">
                  {signals.slice(0, 2).map((signal, idx) => (
                    <Badge 
                      key={idx} 
                      variant={signal.signal === 'BUY' ? 'default' : 'destructive'}
                      className="text-xs"
                    >
                      {signal.signal}
                    </Badge>
                  ))}
                </div>
              </div>
              <Target className="h-5 w-5 text-green-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Price Chart - {selectedTimeframe}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div style={{ width: '100%', height: 400 }}>
            {chartType === 'candlestick' ? (
              <CandlestickChart data={chartData} height={400} />
            ) : (
              <ResponsiveContainer>
                {chartType === 'line' ? (
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                      axisLine={{ stroke: '#d1d5db' }}
                    />
                    <YAxis 
                      domain={['dataMin - 1', 'dataMax + 1']}
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                      axisLine={{ stroke: '#d1d5db' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#2563eb" 
                      strokeWidth={3}
                      dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, fill: '#1d4ed8' }}
                    />
                  </LineChart>
                ) : (
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                      axisLine={{ stroke: '#d1d5db' }}
                    />
                    <YAxis 
                      domain={['dataMin - 1', 'dataMax + 1']}
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                      axisLine={{ stroke: '#d1d5db' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke="#2563eb"
                      fill="#3b82f6"
                      fillOpacity={0.1}
                      strokeWidth={3}
                    />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Volume Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Volume className="h-5 w-5" />
            Volume Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div style={{ width: '100%', height: 200 }}>
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="time" 
                  tick={{ fontSize: 12, fill: '#6b7280' }}
                  axisLine={{ stroke: '#d1d5db' }}
                />
                <YAxis 
                  tick={{ fontSize: 12, fill: '#6b7280' }}
                  axisLine={{ stroke: '#d1d5db' }}
                />
                <Tooltip />
                <Bar 
                  dataKey="volume" 
                  fill="#6b7280"
                  opacity={0.6}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
