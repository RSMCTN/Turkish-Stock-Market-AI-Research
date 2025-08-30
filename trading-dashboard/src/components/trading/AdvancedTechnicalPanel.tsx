"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, TrendingDown, Activity, Zap, 
  Target, AlertCircle, BarChart, LineChart,
  Layers, PieChart, ArrowUpDown, Signal
} from 'lucide-react';

interface TechnicalData {
  symbol: string;
  date: string;
  time: string;
  timeframe: string;
  
  // Basic OHLCV
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  
  // Advanced Technical Indicators
  rsi_14: number;
  macd_26_12: number;
  macd_trigger_9: number;
  atr_14: number;
  adx_14: number;
  
  // Stochastic
  stochastic_k_5: number;
  stochastic_d_3: number;
  stoccci_20: number;
  
  // Bollinger Bands
  bol_upper_20_2: number;
  bol_middle_20_2: number;
  bol_lower_20_2: number;
  
  // Ichimoku Cloud
  tenkan_sen: number;
  kijun_sen: number;
  senkou_span_a: number;
  senkou_span_b: number;
  chikou_span: number;
  
  // Alligator System
  jaw_13_8: number;
  teeth_8_5: number;
  lips_5_3: number;
  
  // Advanced Oscillators
  awesome_oscillator_5_7: number;
  supersmooth_fr: number;
  supersmooth_filt: number;
}

interface AdvancedTechnicalPanelProps {
  selectedSymbol: string;
  apiBaseUrl: string;
}

const AdvancedTechnicalPanel: React.FC<AdvancedTechnicalPanelProps> = ({
  selectedSymbol,
  apiBaseUrl
}) => {
  const [technicalData, setTechnicalData] = useState<TechnicalData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'30m' | '60m' | 'daily'>('60m');

  useEffect(() => {
    if (selectedSymbol) {
      fetchAdvancedTechnicalData();
    }
  }, [selectedSymbol, selectedTimeframe]);

  const fetchAdvancedTechnicalData = async () => {
    if (!selectedSymbol) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // CORS fix: Always use Railway production API for advanced technical data
      const PRODUCTION_API = 'https://bistai001-production.up.railway.app';
      
      const response = await fetch(
        `${PRODUCTION_API}/api/advanced-technical/${selectedSymbol}?timeframe=${selectedTimeframe}`,
        { 
          method: 'GET',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        }
      );
      
      if (!response.ok) {
        throw new Error(`Railway API Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Map Railway API response to frontend format
      const mappedData: TechnicalData = {
        symbol: data.symbol || selectedSymbol,
        date: data.date || new Date().toISOString().split('T')[0],
        time: data.time || new Date().toTimeString().slice(0, 5),
        timeframe: data.timeframe || selectedTimeframe,
        
        // OHLCV
        open: data.open || 0,
        high: data.high || 0,
        low: data.low || 0,
        close: data.close || 0,
        volume: data.volume || 0,
        
        // Technical Indicators
        rsi_14: data.rsi_14 || 50,
        macd_26_12: data.macd_26_12 || 0,
        macd_trigger_9: data.macd_trigger_9 || 0,
        atr_14: data.atr_14 || 1,
        adx_14: data.adx_14 || 25,
        
        // Stochastic
        stochastic_k_5: data.stochastic_k_5 || 50,
        stochastic_d_3: data.stochastic_d_3 || 50,
        stoccci_20: data.stoccci_20 || 0,
        
        // Bollinger Bands
        bol_upper_20_2: data.bol_upper_20_2 || data.close * 1.02 || 50,
        bol_middle_20_2: data.bol_middle_20_2 || data.close || 48,
        bol_lower_20_2: data.bol_lower_20_2 || data.close * 0.98 || 46,
        
        // Ichimoku Cloud
        tenkan_sen: data.tenkan_sen || data.close || 48,
        kijun_sen: data.kijun_sen || data.close * 0.99 || 47,
        senkou_span_a: data.senkou_span_a || data.close * 1.01 || 49,
        senkou_span_b: data.senkou_span_b || data.close * 0.97 || 46,
        chikou_span: data.chikou_span || data.close * 0.98 || 47,
        
        // Alligator System
        jaw_13_8: data.jaw_13_8 || data.close * 0.99 || 47,
        teeth_8_5: data.teeth_8_5 || data.close * 1.005 || 48.2,
        lips_5_3: data.lips_5_3 || data.close * 1.002 || 48.1,
        
        // Advanced Oscillators
        awesome_oscillator_5_7: data.awesome_oscillator_5_7 || 0,
        supersmooth_fr: data.supersmooth_fr || 0.5,
        supersmooth_filt: data.supersmooth_filt || 0.6
      };
      
      setTechnicalData(mappedData);
      
    } catch (err) {
      console.error('Advanced technical data fetch error:', err);
      setError(err instanceof Error ? err.message : 'Railway API connection failed');
      
      // Fallback to mock data
      setTechnicalData(generateMockTechnicalData());
    } finally {
      setLoading(false);
    }
  };

  const generateMockTechnicalData = (): TechnicalData => ({
    symbol: selectedSymbol,
    date: new Date().toISOString().split('T')[0],
    time: new Date().toTimeString().slice(0, 5),
    timeframe: selectedTimeframe,
    open: 45.20,
    high: 47.85,
    low: 44.10,
    close: 46.50,
    volume: 2850000,
    rsi_14: 58.7,
    macd_26_12: 0.23,
    macd_trigger_9: 0.18,
    atr_14: 1.85,
    adx_14: 28.5,
    stochastic_k_5: 72.3,
    stochastic_d_3: 68.9,
    stoccci_20: 0.15,
    bol_upper_20_2: 48.50,
    bol_middle_20_2: 46.00,
    bol_lower_20_2: 43.50,
    tenkan_sen: 46.25,
    kijun_sen: 45.80,
    senkou_span_a: 46.02,
    senkou_span_b: 44.90,
    chikou_span: 45.10,
    jaw_13_8: 45.60,
    teeth_8_5: 46.10,
    lips_5_3: 46.30,
    awesome_oscillator_5_7: 0.35,
    supersmooth_fr: 0.78,
    supersmooth_filt: 0.82
  });

  const getSignalColor = (value: number, thresholds: { overbought: number; oversold: number }) => {
    if (value >= thresholds.overbought) return 'text-red-600 bg-red-50';
    if (value <= thresholds.oversold) return 'text-green-600 bg-green-50';
    return 'text-blue-600 bg-blue-50';
  };

  const getSignalIcon = (value: number, thresholds: { overbought: number; oversold: number }) => {
    if (value >= thresholds.overbought) return <TrendingDown className="h-4 w-4" />;
    if (value <= thresholds.oversold) return <TrendingUp className="h-4 w-4" />;
    return <Activity className="h-4 w-4" />;
  };

  const getTrendStrength = (adx: number) => {
    if (adx >= 50) return { text: 'Çok Güçlü', color: 'bg-green-500', value: 90 };
    if (adx >= 25) return { text: 'Güçlü', color: 'bg-blue-500', value: 70 };
    if (adx >= 20) return { text: 'Orta', color: 'bg-yellow-500', value: 50 };
    return { text: 'Zayıf', color: 'bg-gray-400', value: 25 };
  };

  const getIchimokuSignal = (data: TechnicalData) => {
    const price = data.close;
    const cloud_top = Math.max(data.senkou_span_a, data.senkou_span_b);
    const cloud_bottom = Math.min(data.senkou_span_a, data.senkou_span_b);
    
    if (price > cloud_top) return { signal: 'Boğa', color: 'text-green-600', icon: <TrendingUp className="h-4 w-4" /> };
    if (price < cloud_bottom) return { signal: 'Ayı', color: 'text-red-600', icon: <TrendingDown className="h-4 w-4" /> };
    return { signal: 'Bulut İçi', color: 'text-yellow-600', icon: <Activity className="h-4 w-4" /> };
  };

  const getMACDSignal = (macd: number, trigger: number) => {
    const diff = macd - trigger;
    if (diff > 0.1) return { signal: 'Güçlü Alım', color: 'text-green-600', strength: 'high' };
    if (diff > 0) return { signal: 'Alım', color: 'text-green-500', strength: 'medium' };
    if (diff < -0.1) return { signal: 'Güçlü Satım', color: 'text-red-600', strength: 'high' };
    if (diff < 0) return { signal: 'Satım', color: 'text-red-500', strength: 'medium' };
    return { signal: 'Nötr', color: 'text-gray-500', strength: 'low' };
  };

  if (loading) {
    return (
      <Card className="h-[600px]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart className="h-5 w-5" />
            Gelişmiş Teknik Analiz - {selectedSymbol}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-full">
          <div className="text-center">
            <Activity className="h-8 w-8 animate-spin mx-auto mb-2" />
            <p>Teknik veriler yükleniyor...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="h-[600px]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <AlertCircle className="h-5 w-5" />
            Veri Hatası
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-full">
          <div className="text-center text-red-600">
            <p>Teknik veri alınamadı: {error}</p>
            <p className="text-sm mt-2 text-gray-500">Demo veri gösteriliyor</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!technicalData) return null;

  const ichimokuSignal = getIchimokuSignal(technicalData);
  const trendStrength = getTrendStrength(technicalData.adx_14);
  const macdSignal = getMACDSignal(technicalData.macd_26_12, technicalData.macd_trigger_9);

  return (
    <Card className="h-[700px] overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart className="h-5 w-5 text-blue-600" />
            Gelişmiş Teknik Analiz
            <Badge variant="outline">{selectedSymbol}</Badge>
          </CardTitle>
          <div className="flex gap-1">
            {(['30m', '60m', 'daily'] as const).map(tf => (
              <Badge
                key={tf}
                variant={selectedTimeframe === tf ? 'default' : 'outline'}
                className="cursor-pointer text-xs"
                onClick={() => setSelectedTimeframe(tf)}
              >
                {tf}
              </Badge>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 h-full overflow-y-auto">
        <Tabs defaultValue="momentum" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="momentum">Momentum</TabsTrigger>
            <TabsTrigger value="trend">Trend</TabsTrigger>
            <TabsTrigger value="volatility">Volatilite</TabsTrigger>
            <TabsTrigger value="cloud">Bulut Analizi</TabsTrigger>
          </TabsList>

          <TabsContent value="momentum" className="space-y-4 mt-4">
            {/* RSI Panel */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Signal className="h-4 w-4" />
                  RSI (14) - Momentum Analizi
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4">
                  <div className={`px-3 py-2 rounded-lg ${getSignalColor(technicalData.rsi_14, { overbought: 70, oversold: 30 })}`}>
                    <div className="flex items-center gap-2">
                      {getSignalIcon(technicalData.rsi_14, { overbought: 70, oversold: 30 })}
                      <span className="font-bold text-lg">{technicalData.rsi_14.toFixed(1)}</span>
                    </div>
                  </div>
                  <div className="flex-1">
                    <Progress value={technicalData.rsi_14} className="h-2" />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Aşırı Satım (30)</span>
                      <span>Nötr (50)</span>
                      <span>Aşırı Alım (70)</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* MACD Panel */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <LineChart className="h-4 w-4" />
                  MACD Sistemi
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500">MACD (26,12)</div>
                    <div className="font-bold text-lg">{technicalData.macd_26_12.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Signal (9)</div>
                    <div className="font-bold text-lg">{technicalData.macd_trigger_9.toFixed(4)}</div>
                  </div>
                </div>
                <div className={`mt-3 px-3 py-2 rounded-lg ${macdSignal.color} bg-opacity-10 border`}>
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4" />
                    <span className="font-semibold">{macdSignal.signal}</span>
                    <Badge variant={macdSignal.strength === 'high' ? 'default' : 'outline'}>
                      {macdSignal.strength}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Stochastic Panel */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <ArrowUpDown className="h-4 w-4" />
                  Stochastic Oscillator
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div className={`p-3 rounded-lg ${getSignalColor(technicalData.stochastic_k_5, { overbought: 80, oversold: 20 })}`}>
                    <div className="text-xs opacity-75">%K (5)</div>
                    <div className="font-bold text-lg">{technicalData.stochastic_k_5.toFixed(1)}</div>
                  </div>
                  <div className={`p-3 rounded-lg ${getSignalColor(technicalData.stochastic_d_3, { overbought: 80, oversold: 20 })}`}>
                    <div className="text-xs opacity-75">%D (3)</div>
                    <div className="font-bold text-lg">{technicalData.stochastic_d_3.toFixed(1)}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="trend" className="space-y-4 mt-4">
            {/* ADX Panel */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  ADX (14) - Trend Gücü
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4">
                  <div className="text-center">
                    <div className="font-bold text-2xl">{technicalData.adx_14.toFixed(1)}</div>
                    <div className="text-xs text-gray-500">ADX Değeri</div>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-4 h-4 rounded-full ${trendStrength.color}`}></div>
                      <span className="font-semibold">{trendStrength.text} Trend</span>
                    </div>
                    <Progress value={trendStrength.value} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Alligator System */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Layers className="h-4 w-4" />
                  Alligator Sistemi
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Jaw (13,8)</span>
                    <span className="font-bold text-blue-600">₺{technicalData.jaw_13_8.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Teeth (8,5)</span>
                    <span className="font-bold text-green-600">₺{technicalData.teeth_8_5.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Lips (5,3)</span>
                    <span className="font-bold text-red-600">₺{technicalData.lips_5_3.toFixed(2)}</span>
                  </div>
                </div>
                <div className="mt-3 p-2 bg-gray-50 rounded text-xs">
                  <strong>Alligator Durumu:</strong> {
                    technicalData.lips_5_3 > technicalData.teeth_8_5 && technicalData.teeth_8_5 > technicalData.jaw_13_8
                      ? "🟢 Trend Yukarı" : technicalData.lips_5_3 < technicalData.teeth_8_5 && technicalData.teeth_8_5 < technicalData.jaw_13_8
                      ? "🔴 Trend Aşağı" : "🟡 Uyku Modunda"
                  }
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="volatility" className="space-y-4 mt-4">
            {/* ATR Panel */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  ATR (14) - Volatilite Analizi
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-2xl font-bold text-orange-600">₺{technicalData.atr_14.toFixed(2)}</div>
                    <div className="text-xs text-gray-500">Ortalama True Range</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Volatilite Seviyesi</div>
                    <div className={`px-2 py-1 rounded text-xs font-semibold ${
                      technicalData.atr_14 > 2.0 ? 'bg-red-100 text-red-700' :
                      technicalData.atr_14 > 1.0 ? 'bg-yellow-100 text-yellow-700' :
                      'bg-green-100 text-green-700'
                    }`}>
                      {technicalData.atr_14 > 2.0 ? 'Yüksek' : technicalData.atr_14 > 1.0 ? 'Orta' : 'Düşük'}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Bollinger Bands */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <PieChart className="h-4 w-4" />
                  Bollinger Bands (20,2)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Üst Band</span>
                    <span className="font-bold text-red-600">₺{technicalData.bol_upper_20_2.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Orta Band (SMA)</span>
                    <span className="font-bold text-blue-600">₺{technicalData.bol_middle_20_2.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Alt Band</span>
                    <span className="font-bold text-green-600">₺{technicalData.bol_lower_20_2.toFixed(2)}</span>
                  </div>
                </div>
                <div className="mt-3 p-2 bg-gray-50 rounded text-xs">
                  <strong>BB Pozisyonu:</strong> {
                    technicalData.close > technicalData.bol_upper_20_2 ? "🔴 Üst Bantta" :
                    technicalData.close < technicalData.bol_lower_20_2 ? "🟢 Alt Bantta" :
                    "🟡 Bantlar Arası"
                  }
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="cloud" className="space-y-4 mt-4">
            {/* Ichimoku Cloud */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Layers className="h-4 w-4" />
                  Ichimoku Cloud Analizi
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs text-gray-500">Tenkan-sen</div>
                      <div className="font-bold">₺{technicalData.tenkan_sen.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Kijun-sen</div>
                      <div className="font-bold">₺{technicalData.kijun_sen.toFixed(2)}</div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs text-gray-500">Senkou Span A</div>
                      <div className="font-bold text-green-600">₺{technicalData.senkou_span_a.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Senkou Span B</div>
                      <div className="font-bold text-red-600">₺{technicalData.senkou_span_b.toFixed(2)}</div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-xs text-gray-500">Chikou Span</div>
                    <div className="font-bold">₺{technicalData.chikou_span.toFixed(2)}</div>
                  </div>
                  
                  <div className={`p-3 rounded-lg border ${ichimokuSignal.color} bg-opacity-10`}>
                    <div className="flex items-center gap-2">
                      {ichimokuSignal.icon}
                      <span className="font-semibold">Ichimoku Sinyali: {ichimokuSignal.signal}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Advanced Oscillators */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Signal className="h-4 w-4" />
                  Gelişmiş Osilatörler
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Awesome Oscillator</span>
                    <span className={`font-bold ${technicalData.awesome_oscillator_5_7 > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {technicalData.awesome_oscillator_5_7.toFixed(4)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">SuperSmooth Filter</span>
                    <span className="font-bold text-blue-600">{technicalData.supersmooth_filt.toFixed(4)}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">StocCCI (20)</span>
                    <span className={`font-bold ${technicalData.stoccci_20 > 0.5 ? 'text-green-600' : 'text-red-600'}`}>
                      {technicalData.stoccci_20.toFixed(4)}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default AdvancedTechnicalPanel;
