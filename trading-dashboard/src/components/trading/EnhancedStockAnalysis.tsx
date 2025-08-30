'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  TrendingUp, 
  TrendingDown, 
  Globe, 
  BarChart3, 
  PieChart, 
  Target,
  Calendar,
  DollarSign,
  Activity,
  Users,
  Building2
} from 'lucide-react';

interface EnhancedStockData {
  // Basic Info
  symbol: string;
  name: string;
  sector: string;
  market: string;
  
  // Prices & Changes
  lastPrice: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  
  // Volume Data
  volume: number;
  volumeTL: number;
  avgVolume7d: number;
  avgVolume30d: number;
  avgVolume52w: number;
  avgVolumeYear: number;
  
  // Performance Metrics
  perf7d: number;
  perf30d: number;
  perfYear: number;
  perfWeek: number;
  perfMonth: number;
  perf5y: number;
  
  // Price Ranges
  high7d: number;
  low7d: number;
  high30d: number;
  low30d: number;
  highYear: number;
  lowYear: number;
  high52w: number;
  low52w: number;
  high5y: number;
  low5y: number;
  
  // Financial Ratios
  pe: number;
  pb: number;
  marketCap: number;
  marketCapUSD: number;
  marketCapEUR: number;
  netIncome: number;
  netDebt: number;
  
  // Foreign Ownership
  foreignOwnership: number;
  foreignWeeklyChange: number;
  foreignMonthlyChange: number;
  foreignYearlyChange: number;
  
  // Index Weights
  xu030Weight: number;
  xu050Weight: number;
  xu100Weight: number;
  xutumWeight: number;
  
  // Other
  publicFloat: number;
  institutionOwnership: number;
}

interface EnhancedStockAnalysisProps {
  selectedSymbol?: string;
}

// Load real enhanced stock data from JSON
const getEnhancedStockData = (symbol: string, enhancedData?: any): EnhancedStockData | null => {
  if (!enhancedData?.stocks) {
    return null;
  }
  
  const stock = enhancedData.stocks.find((s: any) => s.symbol === symbol);
  if (!stock) {
    return null;
  }
  
  return {
    symbol: stock.symbol,
    name: stock.name,
    sector: stock.sector,
    market: stock.market,
    
    lastPrice: stock.lastPrice || 0,
    change: stock.change || 0,
    changePercent: stock.changePercent || 0,
    high: stock.high || stock.lastPrice,
    low: stock.low || stock.lastPrice,
    
    volume: stock.volume || 0,
    volumeTL: stock.volumeTL || 0,
    avgVolume7d: stock.avgVolume7d || 0,
    avgVolume30d: stock.avgVolume30d || 0,
    avgVolume52w: stock.avgVolume52w || 0,
    avgVolumeYear: stock.avgVolumeYear || 0,
    
    perf7d: stock.perf7d || 0,
    perf30d: stock.perf30d || 0,
    perfYear: stock.perfYear || 0,
    perfWeek: stock.perfWeek || 0,
    perfMonth: stock.perfMonth || 0,
    perf5y: stock.perf5y || 0,
    
    high7d: stock.high7d || stock.lastPrice,
    low7d: stock.low7d || stock.lastPrice,
    high30d: stock.high30d || stock.lastPrice,
    low30d: stock.low30d || stock.lastPrice,
    highYear: stock.highYear || stock.lastPrice,
    lowYear: stock.lowYear || stock.lastPrice,
    high52w: stock.high52w || stock.lastPrice,
    low52w: stock.low52w || stock.lastPrice,
    high5y: stock.high5y || stock.lastPrice,
    low5y: stock.low5y || stock.lastPrice,
    
    pe: stock.pe || 0,
    pb: stock.pb || 0,
    marketCap: stock.marketCap || 0,
    marketCapUSD: stock.marketCapUSD || 0,
    marketCapEUR: stock.marketCapEUR || 0,
    netIncome: stock.netIncome || 0,
    netDebt: stock.netDebt || 0,
    
    foreignOwnership: stock.foreignOwnership || 0,
    foreignWeeklyChange: stock.foreignWeeklyChange || 0,
    foreignMonthlyChange: stock.foreignMonthlyChange || 0,
    foreignYearlyChange: stock.foreignYearlyChange || 0,
    
    xu030Weight: stock.xu030Weight || 0,
    xu050Weight: stock.xu050Weight || 0,
    xu100Weight: stock.xu100Weight || 0,
    xutumWeight: stock.xutumWeight || 0,
    
    publicFloat: stock.publicFloat || 0,
    institutionOwnership: stock.institutionOwnership || 0
  };
};

const formatNumber = (num: number, decimals = 0): string => {
  if (Math.abs(num) >= 1e9) return (num / 1e9).toFixed(1) + 'B';
  if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + 'M';
  if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toFixed(decimals);
};

const getChangeColor = (value: number): string => {
  if (value > 0) return 'text-green-600';
  if (value < 0) return 'text-red-600';
  return 'text-gray-600';
};

const getChangeIcon = (value: number) => {
  if (value > 0) return <TrendingUp className="h-4 w-4 text-green-600" />;
  if (value < 0) return <TrendingDown className="h-4 w-4 text-red-600" />;
  return <Activity className="h-4 w-4 text-gray-600" />;
};

export default function EnhancedStockAnalysis({ selectedSymbol = 'GARAN' }: EnhancedStockAnalysisProps) {
  const [stockData, setStockData] = useState<EnhancedStockData | null>(null);
  const [enhancedData, setEnhancedData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Load enhanced BIST data
  useEffect(() => {
    const loadEnhancedData = async () => {
      try {
        const response = await fetch('/data/enhanced_bist_data.json');
        if (response.ok) {
          const data = await response.json();
          setEnhancedData(data);
          console.log(`📊 Enhanced BIST data loaded: ${data.stocks?.length || 0} stocks with full metrics`);
        } else {
          console.warn('⚠️ Could not load enhanced BIST data');
        }
      } catch (error) {
        console.error('❌ Error loading enhanced BIST data:', error);
      }
    };

    loadEnhancedData();
  }, []);

  useEffect(() => {
    // Load stock-specific data when symbol changes
    const loadStockData = async () => {
      setLoading(true);
      
      if (enhancedData) {
        const data = getEnhancedStockData(selectedSymbol, enhancedData);
        setStockData(data);
        
        if (data) {
          console.log(`✅ Loaded enhanced data for ${selectedSymbol}: ${data.name}`);
        } else {
          console.warn(`⚠️ No enhanced data found for ${selectedSymbol}`);
        }
      }
      
      setLoading(false);
    };

    loadStockData();
  }, [selectedSymbol, enhancedData]);

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map(i => (
          <Card key={i} className="animate-pulse">
            <CardContent className="h-32 bg-gray-100 rounded"></CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!stockData) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold text-blue-800">
                {stockData.symbol} - Kapsamlı Analiz
              </CardTitle>
              <p className="text-blue-600">{stockData.name}</p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-gray-800">₺{stockData.lastPrice.toFixed(2)}</div>
              <div className={`flex items-center gap-1 ${getChangeColor(stockData.change)}`}>
                {getChangeIcon(stockData.change)}
                <span>{stockData.change > 0 ? '+' : ''}{stockData.change.toFixed(2)}</span>
                <span>({stockData.changePercent > 0 ? '+' : ''}{stockData.changePercent.toFixed(2)}%)</span>
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Genel</TabsTrigger>
          <TabsTrigger value="performance">Performans</TabsTrigger>
          <TabsTrigger value="volume">Hacim</TabsTrigger>
          <TabsTrigger value="ownership">Sahiplik</TabsTrigger>
          <TabsTrigger value="ratios">Oranlar</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Building2 className="h-4 w-4" />
                  Şirket Bilgileri
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Sektör:</span>
                  <Badge variant="outline">{stockData.sector}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Pazar:</span>
                  <span className="text-sm font-medium">{stockData.market}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Piyasa Değeri:</span>
                  <span className="text-sm font-medium">₺{formatNumber(stockData.marketCap)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Halka Açıklık:</span>
                  <span className="text-sm font-medium">{stockData.publicFloat.toFixed(1)}%</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Fiyat Aralıkları
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-600">
                    <span>52 Hafta</span>
                    <span>₺{stockData.low52w.toFixed(2)} - ₺{stockData.high52w.toFixed(2)}</span>
                  </div>
                  <Progress 
                    value={((stockData.lastPrice - stockData.low52w) / (stockData.high52w - stockData.low52w)) * 100} 
                    className="h-2"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-600">
                    <span>Bu Yıl</span>
                    <span>₺{stockData.lowYear.toFixed(2)} - ₺{stockData.highYear.toFixed(2)}</span>
                  </div>
                  <Progress 
                    value={((stockData.lastPrice - stockData.lowYear) / (stockData.highYear - stockData.lowYear)) * 100} 
                    className="h-2"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-600">
                    <span>30 Gün</span>
                    <span>₺{stockData.low30d.toFixed(2)} - ₺{stockData.high30d.toFixed(2)}</span>
                  </div>
                  <Progress 
                    value={((stockData.lastPrice - stockData.low30d) / (stockData.high30d - stockData.low30d)) * 100} 
                    className="h-2"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <PieChart className="h-4 w-4" />
                  İndeks Ağırlıkları
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">XU030:</span>
                  <span className="text-sm font-medium">{stockData.xu030Weight.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">XU050:</span>
                  <span className="text-sm font-medium">{stockData.xu050Weight.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">XU100:</span>
                  <span className="text-sm font-medium">{stockData.xu100Weight.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">XUTUM:</span>
                  <span className="text-sm font-medium">{stockData.xutumWeight.toFixed(2)}%</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {[
              { label: '7 Gün', value: stockData.perf7d, period: '7d' },
              { label: '30 Gün', value: stockData.perf30d, period: '30d' },
              { label: 'Bu Hafta', value: stockData.perfWeek, period: 'week' },
              { label: 'Bu Ay', value: stockData.perfMonth, period: 'month' },
              { label: 'Bu Yıl', value: stockData.perfYear, period: 'year' },
              { label: '5 Yıl', value: stockData.perf5y, period: '5y' }
            ].map((perf, index) => (
              <Card key={index}>
                <CardContent className="pt-4">
                  <div className="text-center">
                    <div className="text-sm text-gray-600 mb-1">{perf.label}</div>
                    <div className={`text-lg font-bold ${getChangeColor(perf.value)}`}>
                      {perf.value > 0 ? '+' : ''}{perf.value.toFixed(2)}%
                    </div>
                    <div className="mt-2">
                      {getChangeIcon(perf.value)}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Volume Tab */}
        <TabsContent value="volume" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Hacim Analizi
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span>Günlük Hacim:</span>
                  <span className="font-bold">{formatNumber(stockData.volume)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Günlük Hacim (TL):</span>
                  <span className="font-bold">₺{formatNumber(stockData.volumeTL)}</span>
                </div>
                <div className="space-y-2">
                  <div className="text-sm font-medium">Ortalama Hacimler:</div>
                  <div className="pl-4 space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span>7 Günlük Ort.:</span>
                      <span>{formatNumber(stockData.avgVolume7d)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>30 Günlük Ort.:</span>
                      <span>{formatNumber(stockData.avgVolume30d)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>52 Haftalık Ort.:</span>
                      <span>{formatNumber(stockData.avgVolume52w)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Yıllık Ort.:</span>
                      <span>{formatNumber(stockData.avgVolumeYear)}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Hacim Karşılaştırması</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>7 Günlük Ortalamaya Göre</span>
                      <span className="font-medium">
                        {stockData.avgVolume7d > 0 ? 
                          ((stockData.volume / stockData.avgVolume7d) * 100).toFixed(0) + '%' : 
                          'N/A'
                        }
                      </span>
                    </div>
                    <Progress 
                      value={stockData.avgVolume7d > 0 ? 
                        Math.min(((stockData.volume / stockData.avgVolume7d) * 100), 200) : 
                        0
                      } 
                      className="h-2"
                    />
                    <div className="text-xs text-gray-500 mt-1">
                      {stockData.volume > stockData.avgVolume7d ? '↑ Ortalamanın üzerinde' : '↓ Ortalamanın altında'}
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>30 Günlük Ortalamaya Göre</span>
                      <span className="font-medium">
                        {stockData.avgVolume30d > 0 ? 
                          ((stockData.volume / stockData.avgVolume30d) * 100).toFixed(0) + '%' : 
                          'N/A'
                        }
                      </span>
                    </div>
                    <Progress 
                      value={stockData.avgVolume30d > 0 ? 
                        Math.min(((stockData.volume / stockData.avgVolume30d) * 100), 200) : 
                        0
                      } 
                      className="h-2"
                    />
                    <div className="text-xs text-gray-500 mt-1">
                      {stockData.volume > stockData.avgVolume30d ? '↑ Ortalamanın üzerinde' : '↓ Ortalamanın altında'}
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>52 Haftalık Ortalamaya Göre</span>
                      <span className="font-medium">
                        {stockData.avgVolume52w > 0 ? 
                          ((stockData.volume / stockData.avgVolume52w) * 100).toFixed(0) + '%' : 
                          'N/A'
                        }
                      </span>
                    </div>
                    <Progress 
                      value={stockData.avgVolume52w > 0 ? 
                        Math.min(((stockData.volume / stockData.avgVolume52w) * 100), 200) : 
                        0
                      } 
                      className="h-2"
                    />
                    <div className="text-xs text-gray-500 mt-1">
                      {stockData.volume > stockData.avgVolume52w ? '↑ Ortalamanın üzerinde' : '↓ Ortalamanın altında'}
                    </div>
                  </div>
                  
                  {/* Volume Insights */}
                  <div className="border-t pt-3 mt-3">
                    <div className="text-sm font-medium text-gray-700 mb-2">Hacim Analizi</div>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className={`p-2 rounded ${stockData.volume > stockData.avgVolume7d ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                        <div className="font-medium">Kısa Vadeli</div>
                        <div>{stockData.volume > stockData.avgVolume7d ? 'Aktif' : 'Sakin'}</div>
                      </div>
                      <div className={`p-2 rounded ${stockData.volume > stockData.avgVolume52w ? 'bg-blue-50 text-blue-700' : 'bg-gray-50 text-gray-700'}`}>
                        <div className="font-medium">Uzun Vadeli</div>
                        <div>{stockData.volume > stockData.avgVolume52w ? 'Yüksek' : 'Normal'}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Ownership Tab */}
        <TabsContent value="ownership" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  Yabancı Sahiplik
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">
                    {stockData.foreignOwnership.toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Mevcut Yabancı Payi</div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Haftalık Değişim:</span>
                    <div className={`flex items-center gap-1 ${getChangeColor(stockData.foreignWeeklyChange)}`}>
                      {getChangeIcon(stockData.foreignWeeklyChange)}
                      <span className="text-sm font-medium">
                        {stockData.foreignWeeklyChange > 0 ? '+' : ''}{stockData.foreignWeeklyChange.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Aylık Değişim:</span>
                    <div className={`flex items-center gap-1 ${getChangeColor(stockData.foreignMonthlyChange)}`}>
                      {getChangeIcon(stockData.foreignMonthlyChange)}
                      <span className="text-sm font-medium">
                        {stockData.foreignMonthlyChange > 0 ? '+' : ''}{stockData.foreignMonthlyChange.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Yıllık Değişim:</span>
                    <div className={`flex items-center gap-1 ${getChangeColor(stockData.foreignYearlyChange)}`}>
                      {getChangeIcon(stockData.foreignYearlyChange)}
                      <span className="text-sm font-medium">
                        {stockData.foreignYearlyChange > 0 ? '+' : ''}{stockData.foreignYearlyChange.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5" />
                  Sahiplik Yapısı
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Yabancı</span>
                      <span>{stockData.foreignOwnership.toFixed(1)}%</span>
                    </div>
                    <Progress value={stockData.foreignOwnership} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Kurumsal</span>
                      <span>{stockData.institutionOwnership.toFixed(1)}%</span>
                    </div>
                    <Progress value={stockData.institutionOwnership} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Halka Açıklık</span>
                      <span>{stockData.publicFloat.toFixed(1)}%</span>
                    </div>
                    <Progress value={stockData.publicFloat} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Ratios Tab */}
        <TabsContent value="ratios" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="h-5 w-5" />
                  Değerleme Oranları
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-sm text-gray-600">F/K (P/E)</div>
                    <div className="text-xl font-bold text-blue-600">{stockData.pe.toFixed(1)}</div>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-sm text-gray-600">PD/DD (P/B)</div>
                    <div className="text-xl font-bold text-green-600">{stockData.pb.toFixed(2)}</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Piyasa Değeri (TL):</span>
                    <span className="font-medium">₺{formatNumber(stockData.marketCap)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Piyasa Değeri ($):</span>
                    <span className="font-medium">${formatNumber(stockData.marketCapUSD)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Piyasa Değeri (€):</span>
                    <span className="font-medium">€{formatNumber(stockData.marketCapEUR)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="h-5 w-5" />
                  Finansal Durum
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Net Kar:</span>
                    <span className={`font-medium ${getChangeColor(stockData.netIncome)}`}>
                      ₺{formatNumber(stockData.netIncome)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Net Borç:</span>
                    <span className="font-medium">₺{formatNumber(stockData.netDebt)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Borç/Özkaynak:</span>
                    <span className="font-medium">
                      {(stockData.netDebt / (stockData.marketCap * 0.6)).toFixed(2)}
                    </span>
                  </div>
                </div>
                
                <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                  <div className="text-xs text-gray-600 mb-2">Finansal Sağlık Skoru</div>
                  <Progress 
                    value={Math.max(0, Math.min(100, 60 + (stockData.netIncome / stockData.marketCap) * 1000))}
                    className="h-2"
                  />
                  <div className="text-xs text-gray-500 mt-1 text-center">
                    {Math.round(Math.max(0, Math.min(100, 60 + (stockData.netIncome / stockData.marketCap) * 1000)))}/100
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
