'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { 
  ResponsiveContainer, 
  LineChart, 
  AreaChart, 
  BarChart,
  Line, 
  Area, 
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ReferenceLine,
  ComposedChart
} from 'recharts';
import { 
  BarChart3, 
  TrendingUp, 
  Activity, 
  Volume2, 
  Settings,
  Eye,
  EyeOff
} from 'lucide-react';

interface ChartDataPoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi?: number;
  macd?: number;
  bb_upper?: number;
  bb_lower?: number;
  sma_20?: number;
  ema_50?: number;
  signal?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
}

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'BUY' | 'SELL' | 'HOLD' | 'NEUTRAL';
  weight: number;
  description: string;
  status: string;
}

interface AdvancedChartProps {
  symbol: string;
  data?: ChartDataPoint[];
  indicators?: TechnicalIndicator[];
  isLoading?: boolean;
}

const AdvancedChart = ({ symbol, data = [], indicators = [], isLoading = false }: AdvancedChartProps) => {
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area' | 'volume' | 'renko'>('candlestick');
  const [timeframe, setTimeframe] = useState('1H');
  const [overlayIndicators, setOverlayIndicators] = useState<string[]>(['SMA_20', 'EMA_50']);
  const [oscillatorIndicators, setOscillatorIndicators] = useState<string[]>(['RSI']);
  const [showVolume, setShowVolume] = useState(true);

  // Generate mock advanced chart data
  const generateAdvancedChartData = (): ChartDataPoint[] => {
    const data: ChartDataPoint[] = [];
    const now = new Date();
    let basePrice = 45 + Math.random() * 20;
    
    for (let i = 23; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
      
      // OHLC calculation
      const volatility = 0.02 + Math.random() * 0.02;
      const trend = (Math.random() - 0.5) * 0.01;
      
      const open = basePrice;
      const close = open * (1 + trend + (Math.random() - 0.5) * volatility);
      const high = Math.max(open, close) * (1 + Math.random() * volatility);
      const low = Math.min(open, close) * (1 - Math.random() * volatility);
      const volume = 100000 + Math.random() * 500000;
      
      // Technical indicators
      const rsi = 30 + Math.random() * 40; // 30-70 range
      const macd = (Math.random() - 0.5) * 2;
      const bb_upper = close * 1.02;
      const bb_lower = close * 0.98;
      const sma_20 = close * (0.99 + Math.random() * 0.02);
      const ema_50 = close * (0.98 + Math.random() * 0.04);
      
      // Signal generation
      let signal: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
      if (rsi < 35 && macd > 0) signal = 'BUY';
      else if (rsi > 65 && macd < 0) signal = 'SELL';
      
      data.push({
        time: timestamp.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
        volume: Math.round(volume),
        rsi: Number(rsi.toFixed(1)),
        macd: Number(macd.toFixed(3)),
        bb_upper: Number(bb_upper.toFixed(2)),
        bb_lower: Number(bb_lower.toFixed(2)),
        sma_20: Number(sma_20.toFixed(2)),
        ema_50: Number(ema_50.toFixed(2)),
        signal,
        confidence: 0.7 + Math.random() * 0.25
      });
      
      basePrice = close;
    }
    
    return data;
  };

  const chartData = data.length > 0 ? data : generateAdvancedChartData();

  const toggleOverlayIndicator = (indicator: string) => {
    setOverlayIndicators(prev => 
      prev.includes(indicator) 
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator]
    );
  };

  const toggleOscillatorIndicator = (indicator: string) => {
    setOscillatorIndicators(prev => 
      prev.includes(indicator) 
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator]
    );
  };

  // Custom Candlestick Component
  const Candlestick = (props: any) => {
    const { x, y, width, payload } = props;
    const { open, close, high, low } = payload;
    const isRising = close > open;
    const color = isRising ? '#22c55e' : '#ef4444';
    const candleHeight = Math.abs(y - (y + (open - close) * 10));

    return (
      <g>
        {/* Wick */}
        <line
          x1={x + width / 2}
          y1={y - (high - Math.max(open, close)) * 10}
          x2={x + width / 2}
          y2={y + (Math.min(open, close) - low) * 10}
          stroke={color}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x}
          y={isRising ? y - (close - open) * 10 : y}
          width={width}
          height={candleHeight || 1}
          fill={color}
          stroke={color}
        />
      </g>
    );
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white border-2 border-gray-200 rounded-lg shadow-xl p-4">
          <p className="font-semibold text-gray-900 mb-2">{label}</p>
          
          {/* OHLCV Data */}
          <div className="grid grid-cols-2 gap-3 text-sm mb-3">
            <div>
              <p className="text-gray-600">Open: <span className="font-semibold">₺{data.open?.toFixed(2)}</span></p>
              <p className="text-gray-600">High: <span className="font-semibold">₺{data.high?.toFixed(2)}</span></p>
              <p className="text-gray-600">Low: <span className="font-semibold">₺{data.low?.toFixed(2)}</span></p>
            </div>
            <div>
              <p className="text-gray-600">Close: <span className="font-semibold">₺{data.close?.toFixed(2)}</span></p>
              <p className="text-gray-600">Volume: <span className="font-semibold">{data.volume?.toLocaleString()}</span></p>
            </div>
          </div>

          {/* Technical Indicators */}
          {overlayIndicators.includes('SMA_20') && data.sma_20 && (
            <p className="text-sm text-blue-600">SMA(20): ₺{data.sma_20.toFixed(2)}</p>
          )}
          {overlayIndicators.includes('EMA_50') && data.ema_50 && (
            <p className="text-sm text-purple-600">EMA(50): ₺{data.ema_50.toFixed(2)}</p>
          )}
          
          {oscillatorIndicators.includes('RSI') && data.rsi && (
            <p className="text-sm text-orange-600">RSI: {data.rsi.toFixed(1)}</p>
          )}
          
          {data.signal && data.signal !== 'HOLD' && (
            <div className="mt-2 pt-2 border-t">
              <Badge className={`text-xs ${
                data.signal === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {data.signal} - {(data.confidence * 100).toFixed(0)}%
              </Badge>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  const renderChart = () => {
    switch (chartType) {
      case 'candlestick':
        return (
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} />
            <YAxis domain={['dataMin - 1', 'dataMax + 1']} tick={{ fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            
            {/* Candlesticks */}
            <Bar dataKey="close" fill="#8884d8" shape={<Candlestick />} />
            
            {/* Overlay Indicators */}
            {overlayIndicators.includes('SMA_20') && (
              <Line type="monotone" dataKey="sma_20" stroke="#3b82f6" strokeWidth={1} dot={false} />
            )}
            {overlayIndicators.includes('EMA_50') && (
              <Line type="monotone" dataKey="ema_50" stroke="#8b5cf6" strokeWidth={1} dot={false} />
            )}
            {overlayIndicators.includes('BB_UPPER') && (
              <Line type="monotone" dataKey="bb_upper" stroke="#f59e0b" strokeWidth={1} strokeDasharray="3 3" dot={false} />
            )}
            {overlayIndicators.includes('BB_LOWER') && (
              <Line type="monotone" dataKey="bb_lower" stroke="#f59e0b" strokeWidth={1} strokeDasharray="3 3" dot={false} />
            )}
          </ComposedChart>
        );

      case 'line':
        return (
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} />
            <YAxis domain={['dataMin - 1', 'dataMax + 1']} tick={{ fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="close" stroke="#2563eb" strokeWidth={2} dot={false} />
            
            {overlayIndicators.includes('SMA_20') && (
              <Line type="monotone" dataKey="sma_20" stroke="#3b82f6" strokeWidth={1} dot={false} />
            )}
            {overlayIndicators.includes('EMA_50') && (
              <Line type="monotone" dataKey="ema_50" stroke="#8b5cf6" strokeWidth={1} dot={false} />
            )}
          </LineChart>
        );

      case 'area':
        return (
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} />
            <YAxis domain={['dataMin - 1', 'dataMax + 1']} tick={{ fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="close" stroke="#2563eb" fill="#3b82f6" fillOpacity={0.3} />
          </AreaChart>
        );

      case 'volume':
        return (
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="volume" fill="#6366f1" />
          </BarChart>
        );

      default:
        return null;
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Advanced Chart - {symbol}
          </CardTitle>
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
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Advanced Chart - {symbol}
          </CardTitle>
          
          {/* Chart Controls */}
          <div className="flex items-center gap-3">
            <Select value={chartType} onValueChange={(value: any) => setChartType(value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="candlestick">Candlestick</SelectItem>
                <SelectItem value="line">Line</SelectItem>
                <SelectItem value="area">Area</SelectItem>
                <SelectItem value="volume">Volume</SelectItem>
                <SelectItem value="renko">Renko</SelectItem>
              </SelectContent>
            </Select>

            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger className="w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="5M">5M</SelectItem>
                <SelectItem value="15M">15M</SelectItem>
                <SelectItem value="1H">1H</SelectItem>
                <SelectItem value="4H">4H</SelectItem>
                <SelectItem value="1D">1D</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Indicator Controls */}
        <div className="mb-4 space-y-3">
          
          {/* Overlay Indicators */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Overlay Indicators</p>
            <div className="flex flex-wrap gap-2">
              {['SMA_20', 'EMA_50', 'BB_UPPER', 'BB_LOWER'].map(indicator => (
                <Button
                  key={indicator}
                  variant={overlayIndicators.includes(indicator) ? "default" : "outline"}
                  size="sm"
                  onClick={() => toggleOverlayIndicator(indicator)}
                  className="text-xs gap-1"
                >
                  {overlayIndicators.includes(indicator) ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                  {indicator.replace('_', ' ')}
                </Button>
              ))}
            </div>
          </div>

          {/* Oscillator Indicators */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Oscillator Indicators</p>
            <div className="flex flex-wrap gap-2">
              {['RSI', 'MACD', 'STOCHASTIC'].map(indicator => (
                <Button
                  key={indicator}
                  variant={oscillatorIndicators.includes(indicator) ? "default" : "outline"}
                  size="sm"
                  onClick={() => toggleOscillatorIndicator(indicator)}
                  className="text-xs gap-1"
                >
                  {oscillatorIndicators.includes(indicator) ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                  {indicator}
                </Button>
              ))}
            </div>
          </div>

          {/* Indicator Weights Display */}
          {indicators.length > 0 && (
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm font-medium text-gray-700 mb-2">Active Indicator Weights</p>
              <div className="flex flex-wrap gap-2">
                {indicators.map((indicator, index) => (
                  <Badge 
                    key={index}
                    className={`text-xs ${
                      indicator.signal === 'BUY' ? 'bg-green-100 text-green-800' :
                      indicator.signal === 'SELL' ? 'bg-red-100 text-red-800' : 
                      'bg-gray-100 text-gray-800'
                    }`}
                  >
                    {indicator.name}: {(indicator.weight * 100).toFixed(0)}%
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Main Chart */}
        <div style={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            {renderChart()}
          </ResponsiveContainer>
        </div>

        {/* Volume Chart (if enabled) */}
        {showVolume && chartType !== 'volume' && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-medium text-gray-700">Volume</p>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => setShowVolume(false)}
              >
                <EyeOff className="h-4 w-4" />
              </Button>
            </div>
            <div style={{ width: '100%', height: 100 }}>
              <ResponsiveContainer>
                <BarChart data={chartData}>
                  <XAxis dataKey="time" tick={{ fontSize: 8 }} />
                  <YAxis tick={{ fontSize: 8 }} />
                  <Bar dataKey="volume" fill="#6366f1" opacity={0.7} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Oscillator Charts */}
        {oscillatorIndicators.includes('RSI') && (
          <div className="mt-4">
            <p className="text-sm font-medium text-gray-700 mb-2">RSI (14)</p>
            <div style={{ width: '100%', height: 120 }}>
              <ResponsiveContainer>
                <LineChart data={chartData}>
                  <XAxis dataKey="time" tick={{ fontSize: 8 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 8 }} />
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="2 2" />
                  <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="2 2" />
                  <Line type="monotone" dataKey="rsi" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AdvancedChart;
