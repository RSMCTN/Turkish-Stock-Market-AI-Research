'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
// import { Slider } from '@/components/ui/slider'; // Not available
// import { Switch } from '@/components/ui/switch'; // Not available
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot,
  Brush
} from 'recharts';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Crosshair,
  BarChart3
} from 'lucide-react';

interface EnhancedDataPoint {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi?: number;
  macd?: number;
  pattern?: string;
  signal?: 'BUY' | 'SELL' | null;
  support?: number;
  resistance?: number;
}

interface EnhancedHistoricalChartProps {
  symbol: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  showTechnicalOverlay?: boolean;
  showPatternDetection?: boolean;
  showAlerts?: boolean;
}

export default function EnhancedHistoricalChart({
  symbol,
  autoRefresh = true,
  refreshInterval = 30000,
  showTechnicalOverlay = true,
  showPatternDetection = true,
  showAlerts = true
}: EnhancedHistoricalChartProps) {
  const [data, setData] = useState<EnhancedDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [playbackMode, setPlaybackMode] = useState(false);
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showIndicators, setShowIndicators] = useState({
    rsi: true,
    macd: true,
    supportResistance: true,
    patterns: true,
    signals: true
  });

  // Real data fetching from Railway API
  const fetchRealData = useCallback(async () => {
    try {
      console.log(`📊 EnhancedChart: Fetching real data for ${symbol}`);
      const response = await fetch(`https://bistai001-production.up.railway.app/api/bist/historical/${symbol}?timeframe=60min&limit=200`);
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }
      
      const historicalData = await response.json();
      
      if (!historicalData['60min'] || !historicalData['60min'].data || historicalData['60min'].data.length === 0) {
        console.error(`❌ No real data found for ${symbol}`);
        return [];
      }
      
      const apiData = historicalData['60min'].data;
      console.log(`✅ EnhancedChart: Found ${apiData.length} real records for ${symbol}`);
      
      // Transform API data to chart format
      const points: EnhancedDataPoint[] = apiData.map((item: any, index: number) => {
        // Pattern detection based on price movement
        let pattern: string | undefined;
        let signal: 'BUY' | 'SELL' | null = null;
        
        if (index > 0) {
          const prevItem = apiData[index - 1];
          const priceChange = ((item.close - prevItem.close) / prevItem.close) * 100;
          
          if (priceChange > 3) {
            pattern = 'Bullish Breakout';
            signal = 'BUY';
          } else if (priceChange < -3) {
            pattern = 'Bearish Breakdown';
            signal = 'SELL';
          } else if (Math.abs(priceChange) < 0.5 && item.rsi) {
            if (item.rsi < 30) {
              pattern = 'Oversold';
              signal = 'BUY';
            } else if (item.rsi > 70) {
              pattern = 'Overbought';
              signal = 'SELL';
            }
          }
        }
        
        return {
          datetime: item.datetime,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume,
          rsi: item.rsi,
          macd: item.macd,
          pattern,
          signal,
          support: item.close * 0.98, // 2% below as support
          resistance: item.close * 1.02 // 2% above as resistance
        };
      });
      
      // Sort by datetime to ensure correct chronological order
      return points.sort((a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime());
      
    } catch (error) {
      console.error(`❌ EnhancedChart error for ${symbol}:`, error);
      return [];
    }
  }, [symbol]);

  // Load real data when symbol changes
  useEffect(() => {
    if (symbol) {
      setLoading(true);
      fetchRealData().then((realData) => {
        console.log(`📊 EnhancedChart: Setting ${realData.length} real data points for ${symbol}`);
        setData(realData);
        setPlaybackIndex(realData.length - 1);
        setLoading(false);
      }).catch((error) => {
        console.error(`❌ EnhancedChart: Failed to load data for ${symbol}:`, error);
        setData([]);
        setLoading(false);
      });
    }
  }, [symbol, fetchRealData]);

  // Playback functionality
  useEffect(() => {
    if (playbackMode && isPlaying && data.length > 0) {
      const interval = setInterval(() => {
        setPlaybackIndex(prev => {
          if (prev >= data.length - 1) {
            setIsPlaying(false);
            return data.length - 1;
          }
          return prev + 1;
        });
      }, 1000 / playbackSpeed);

      return () => clearInterval(interval);
    }
  }, [playbackMode, isPlaying, playbackSpeed, data.length]);

  const getCurrentDisplayData = () => {
    if (!playbackMode) return data;
    return data.slice(0, playbackIndex + 1);
  };

  const handlePlayPause = () => {
    if (playbackMode) {
      setIsPlaying(!isPlaying);
    } else {
      setPlaybackMode(true);
      setPlaybackIndex(Math.max(0, Math.floor(data.length * 0.1))); // Start from 10%
      setIsPlaying(true);
    }
  };

  const handleStop = () => {
    setPlaybackMode(false);
    setIsPlaying(false);
    setPlaybackIndex(data.length - 1);
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload as EnhancedDataPoint;
      
      return (
        <div className="bg-white p-4 border rounded-lg shadow-lg">
          <h3 className="font-semibold mb-2">
            {new Date(label).toLocaleString('tr-TR')}
          </h3>
          
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div>Open: <span className="font-semibold">₺{data.open?.toFixed(2)}</span></div>
                <div>High: <span className="font-semibold text-green-600">₺{data.high?.toFixed(2)}</span></div>
              </div>
              <div>
                <div>Low: <span className="font-semibold text-red-600">₺{data.low?.toFixed(2)}</span></div>
                <div>Close: <span className="font-semibold">₺{data.close?.toFixed(2)}</span></div>
              </div>
            </div>
            
            <div className="border-t pt-2">
              <div>Volume: <span className="font-semibold">{data.volume?.toLocaleString()}</span></div>
              {data.rsi && <div>RSI: <span className="font-semibold">{data.rsi.toFixed(1)}</span></div>}
              {data.macd && <div>MACD: <span className="font-semibold">{data.macd.toFixed(3)}</span></div>}
            </div>
            
            {data.pattern && (
              <div className="border-t pt-2">
                <Badge variant="outline" className="text-xs">
                  Pattern: {data.pattern}
                </Badge>
              </div>
            )}
            
            {data.signal && (
              <div className="mt-2">
                <Badge className={data.signal === 'BUY' ? 'bg-green-500' : 'bg-red-500'}>
                  {data.signal} SIGNAL
                </Badge>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Enhanced Historical Chart - {symbol}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center">
            <div className="animate-pulse text-gray-500">Loading enhanced chart data...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const displayData = getCurrentDisplayData();

  return (
    <div className="space-y-4">
      {/* Chart Controls */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Enhanced Chart - {symbol}
            </CardTitle>
            
            {/* Playback Controls */}
            <div className="flex items-center gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setPlaybackIndex(Math.max(0, playbackIndex - 10))}
                disabled={!playbackMode}
              >
                <SkipBack className="h-4 w-4" />
              </Button>
              
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handlePlayPause}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setPlaybackIndex(Math.min(data.length - 1, playbackIndex + 10))}
                disabled={!playbackMode}
              >
                <SkipForward className="h-4 w-4" />
              </Button>
              
              <Button variant="outline" size="sm" onClick={handleStop}>
                Stop
              </Button>
            </div>
          </div>
          
          {playbackMode && (
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium">Speed:</span>
                <input
                  type="range"
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                  max={5}
                  min={0.5}
                  step={0.5}
                  className="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm text-gray-600">{playbackSpeed}x</span>
              </div>
              
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium">Progress:</span>
                <input
                  type="range"
                  value={playbackIndex}
                  onChange={(e) => setPlaybackIndex(parseInt(e.target.value))}
                  max={data.length - 1}
                  min={0}
                  className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm text-gray-600">
                  {playbackIndex + 1} / {data.length}
                </span>
              </div>
            </div>
          )}
        </CardHeader>
        
        <CardContent>
          {/* Indicator Controls */}
          <div className="mb-4 flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showIndicators.rsi}
                onChange={(e) => 
                  setShowIndicators(prev => ({ ...prev, rsi: e.target.checked }))
                }
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="text-sm">RSI</span>
            </div>
            
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showIndicators.patterns}
                onChange={(e) => 
                  setShowIndicators(prev => ({ ...prev, patterns: e.target.checked }))
                }
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="text-sm">Patterns</span>
            </div>
            
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showIndicators.signals}
                onChange={(e) => 
                  setShowIndicators(prev => ({ ...prev, signals: e.target.checked }))
                }
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="text-sm">Signals</span>
            </div>
            
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showIndicators.supportResistance}
                onChange={(e) => 
                  setShowIndicators(prev => ({ ...prev, supportResistance: e.target.checked }))
                }
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="text-sm">Support/Resistance</span>
            </div>
          </div>
          
          {/* Main Chart */}
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                <XAxis 
                  dataKey="datetime"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleDateString('tr-TR', {
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                />
                <YAxis domain={['dataMin - 5', 'dataMax + 5']} tick={{ fontSize: 12 }} />
                <Tooltip content={<CustomTooltip />} />
                
                {/* Support/Resistance Lines */}
                {showIndicators.supportResistance && displayData.length > 0 && (
                  <>
                    <ReferenceLine 
                      y={displayData[displayData.length - 1]?.support} 
                      stroke="#ef4444" 
                      strokeDasharray="5 5" 
                      label="Support"
                    />
                    <ReferenceLine 
                      y={displayData[displayData.length - 1]?.resistance} 
                      stroke="#22c55e" 
                      strokeDasharray="5 5" 
                      label="Resistance"
                    />
                  </>
                )}
                
                {/* Main Price Line */}
                <Line 
                  type="monotone" 
                  dataKey="close" 
                  stroke="#2563eb" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, stroke: '#2563eb', strokeWidth: 2 }}
                />
                
                {/* Pattern Dots */}
                {showIndicators.patterns && displayData.map((point, index) => {
                  if (point.pattern) {
                    return (
                      <ReferenceDot
                        key={`pattern-${index}`}
                        x={point.datetime}
                        y={point.close}
                        r={6}
                        fill="#f59e0b"
                        stroke="#d97706"
                        strokeWidth={2}
                      />
                    );
                  }
                  return null;
                })}
                
                {/* Signal Dots */}
                {showIndicators.signals && displayData.map((point, index) => {
                  if (point.signal) {
                    return (
                      <ReferenceDot
                        key={`signal-${index}`}
                        x={point.datetime}
                        y={point.close}
                        r={8}
                        fill={point.signal === 'BUY' ? '#22c55e' : '#ef4444'}
                        stroke={point.signal === 'BUY' ? '#16a34a' : '#dc2626'}
                        strokeWidth={2}
                      />
                    );
                  }
                  return null;
                })}
                
                <Brush dataKey="datetime" height={30} stroke="#2563eb" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {/* Pattern & Signal Summary */}
          {(showIndicators.patterns || showIndicators.signals) && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              {showIndicators.patterns && (
                <div className="bg-orange-50 p-3 rounded-lg">
                  <h3 className="font-semibold text-orange-800 mb-2 flex items-center gap-1">
                    <AlertTriangle className="h-4 w-4" />
                    Detected Patterns
                  </h3>
                  <div className="space-y-1 text-sm">
                    {displayData
                      .filter(p => p.pattern)
                      .slice(-3)
                      .map((point, idx) => (
                        <div key={idx} className="flex justify-between">
                          <span>{point.pattern}</span>
                          <span className="text-gray-600">
                            {new Date(point.datetime).toLocaleDateString()}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
              
              {showIndicators.signals && (
                <div className="bg-blue-50 p-3 rounded-lg">
                  <h3 className="font-semibold text-blue-800 mb-2 flex items-center gap-1">
                    <Activity className="h-4 w-4" />
                    Trading Signals
                  </h3>
                  <div className="space-y-1 text-sm">
                    {displayData
                      .filter(p => p.signal)
                      .slice(-3)
                      .map((point, idx) => (
                        <div key={idx} className="flex justify-between">
                          <Badge 
                            className={`text-xs ${
                              point.signal === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                            }`}
                          >
                            {point.signal}
                          </Badge>
                          <span className="text-gray-600">
                            ₺{point.close.toFixed(2)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
