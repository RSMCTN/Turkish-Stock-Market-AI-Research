'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Building2, TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';
import { useState, useEffect } from 'react';

// Dynamic company data loading from working_bist_data.json
const getCompanyData = (symbol: string, bistData?: any) => {
  // If BIST data loaded, use it
  if (bistData && bistData.stocks) {
    const stock = bistData.stocks.find((s: any) => s.symbol === symbol);
    if (stock) {
      return {
        name: stock.name,
        sector: stock.sector,
        marketCap: Math.round(stock.market_cap / 1000000), // Convert to Million TL
        pe: stock.pe_ratio,
        pb: stock.pb_ratio,
        roe: stock.roe || 0, // Some stocks might not have ROE
        debtEquity: stock.debt_equity || 0,
        week52High: stock.week_52_high || stock.last_price * 1.2,
        week52Low: stock.week_52_low || stock.last_price * 0.8,
        volume: stock.volume,
        lastPrice: stock.last_price,
        change: stock.change,
        changePercent: stock.change_percent,
        bistMarkets: stock.bist_markets || [],
        description: `${stock.sector} sektöründe faaliyet gösteren şirket`
      };
    }
  }

  // Fallback: existing hard-coded data for backwards compatibility
  const companyData: { [key: string]: any } = {
    'AKSEN': {
      name: 'AKSA ENERJI URETIM',
      sector: 'ELEKTRIK',
      marketCap: 47900, // Million TL
      pe: 28.92,
      pb: 0.98,
      roe: 8.62,
      debtEquity: 36572,
      week52High: 43.74,
      week52Low: 29.92,
      volume: 6109521,
      description: 'Elektrik üretimi ve enerji sektöründe faaliyet gösteren şirket'
    },
    'ASTOR': {
      name: 'ASTOR ENERJI',
      sector: 'METALESYA',
      marketCap: 113472, // Million TL
      pe: 18.89,
      pb: 4.34,
      roe: 11.88,
      debtEquity: -6964,
      week52High: 128.84,
      week52Low: 64.27,
      volume: 6981245,
      description: 'Metal eşya ve enerji sektöründe üretim yapan şirket'
    },
    'GARAN': {
      name: 'T. GARANTI BANKASI',
      sector: 'BANKA',
      marketCap: 612360, // Million TL
      pe: 6.07,
      pb: 1.62,
      roe: 0.0,
      debtEquity: 0,
      week52High: 154.5,
      week52Low: 93.73,
      volume: 14669824,
      description: 'Türkiye\'nin önde gelen özel bankalarından biri'
    },
    'THYAO': {
      name: 'TURK HAVA YOLLARI',
      sector: 'ULASTIRMA',
      marketCap: 469200, // Million TL
      pe: 4.64,
      pb: 0.62,
      roe: 5.63,
      debtEquity: 335241,
      week52High: 346.25,
      week52Low: 249.2,
      volume: 20691413,
      description: 'Türkiye\'nin bayrak taşıyıcı havayolu şirketi'
    },
    'TUPRS': {
      name: 'TUPRAS',
      sector: 'KIMYA',
      marketCap: 329482, // Million TL
      pe: 14.25,
      pb: 1.07,
      roe: 5.37,
      debtEquity: -48981,
      week52High: 174.9,
      week52Low: 116.36,
      volume: 13549874,
      description: 'Türkiye\'nin en büyük petrol rafinerisi'
    },
    'BRSAN': {
      name: 'BORUSAN MANNESMANN',
      sector: 'METALESYA',
      marketCap: 18970, // Million TL
      pe: 0.0,
      pb: 1.94,
      roe: 0.0,
      debtEquity: 2840,
      week52High: 520.0,
      week52Low: 310.0,
      volume: 385642,
      description: 'Çelik boru ve metal ürünleri üreticisi'
    },
    'AKBNK': {
      name: 'AKBANK',
      sector: 'BANKA',
      marketCap: 361620, // Million TL
      pe: 5.89,
      pb: 1.20,
      roe: 0.0,
      debtEquity: 0,
      week52High: 75.2,
      week52Low: 45.86,
      volume: 18745632,
      description: 'Türkiye\'nin önde gelen özel bankalarından biri'
    }
  };

  return companyData[symbol] || {
    name: symbol,
    sector: 'Diğer',
    marketCap: 0,
    pe: 0,
    pb: 0,
    roe: 0,
    debtEquity: 0,
    week52High: 0,
    week52Low: 0,
    volume: 0,
    description: 'Şirket bilgileri yükleniyor...'
  };
};

interface CompanyInfoCardProps {
  selectedSymbol?: string;
}

export default function CompanyInfoCard({ selectedSymbol = 'GARAN' }: CompanyInfoCardProps) {
  const [bistData, setBistData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadBistData = async () => {
      try {
        // Try loading from local public data first
        const response = await fetch('/data/working_bist_data.json');
        if (response.ok) {
          const data = await response.json();
          setBistData(data);
          console.log(`📊 BIST data loaded: ${data.stocks?.length || 0} stocks for CompanyInfoCard`);
        } else {
          console.warn('⚠️ Could not load local BIST data');
        }
      } catch (error) {
        console.error('❌ Error loading BIST data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadBistData();
  }, []);

  const companyData = getCompanyData(selectedSymbol, bistData);
  
  // Use real current price from BIST data or fallback
  const currentPrice = companyData.lastPrice || 50.0;
  
  const week52Performance = companyData.week52High > 0 ? 
    ((currentPrice - companyData.week52Low) / (companyData.week52High - companyData.week52Low)) * 100 : 50;

  if (loading) {
    return (
      <Card className="bg-gradient-to-br from-indigo-50 to-cyan-50 border-indigo-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5 text-indigo-600" />
            Şirket Bilgileri - {selectedSymbol}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto mb-4"></div>
            <p className="text-indigo-600">Şirket bilgileri yükleniyor...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gradient-to-br from-indigo-50 to-cyan-50 border-indigo-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Building2 className="h-5 w-5 text-indigo-600" />
          Şirket Bilgileri - {selectedSymbol}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge className="bg-indigo-100 text-indigo-700">{companyData.sector}</Badge>
          <Badge variant="outline">Temel Analiz</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Company Name & Description */}
          <div className="p-3 bg-white/60 rounded-lg">
            <h3 className="font-semibold text-indigo-800 mb-1">{companyData.name}</h3>
            <p className="text-sm text-slate-600">{companyData.description}</p>
          </div>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center p-3 bg-white/40 rounded-lg">
              <div className="text-sm text-slate-600">Piyasa Değeri</div>
              <div className="text-lg font-bold text-indigo-700">
                {companyData.marketCap > 1000 ? 
                  `₺${(companyData.marketCap / 1000).toFixed(1)}B` : 
                  `₺${companyData.marketCap}M`}
              </div>
            </div>
            <div className="text-center p-3 bg-white/40 rounded-lg">
              <div className="text-sm text-slate-600">Günlük Hacim</div>
              <div className="text-lg font-bold text-slate-800">
                {companyData.volume > 1000000 ? 
                  `${(companyData.volume / 1000000).toFixed(1)}M` : 
                  `${Math.round(companyData.volume / 1000)}K`}
              </div>
            </div>
          </div>

          {/* Fundamental Ratios */}
          <div className="space-y-2">
            <div className="text-sm font-medium text-slate-700">Temel Oranlar:</div>
            <div className="grid grid-cols-3 gap-2">
              <div className="flex items-center justify-between p-2 bg-white/40 rounded text-sm">
                <span>F/K:</span>
                <span className={`font-medium ${companyData.pe > 0 && companyData.pe < 15 ? 'text-green-600' : companyData.pe > 25 ? 'text-red-600' : 'text-slate-700'}`}>
                  {companyData.pe > 0 ? companyData.pe.toFixed(1) : 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between p-2 bg-white/40 rounded text-sm">
                <span>PD/DD:</span>
                <span className={`font-medium ${companyData.pb > 0 && companyData.pb < 2 ? 'text-green-600' : companyData.pb > 3 ? 'text-red-600' : 'text-slate-700'}`}>
                  {companyData.pb > 0 ? companyData.pb.toFixed(2) : 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between p-2 bg-white/40 rounded text-sm">
                <span>ROE:</span>
                <span className={`font-medium ${companyData.roe > 15 ? 'text-green-600' : companyData.roe < 5 ? 'text-red-600' : 'text-slate-700'}`}>
                  {companyData.roe > 0 ? `${companyData.roe.toFixed(1)}%` : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          {/* 52 Week Performance */}
          <div className="p-3 bg-white/40 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">52 Haftalık Performans</span>
              <span className="text-sm text-slate-600">
                {companyData.week52Low} - {companyData.week52High} TL
              </span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2 mb-1">
              <div 
                className="bg-gradient-to-r from-red-400 via-yellow-400 to-green-400 h-2 rounded-full"
                style={{ width: `${Math.min(100, Math.max(0, week52Performance))}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>Düşük</span>
              <span className="font-medium">{currentPrice} TL ({week52Performance.toFixed(1)}%)</span>
              <span>Yüksek</span>
            </div>
          </div>

          {/* Quick Analysis */}
          <div className="p-3 bg-gradient-to-r from-white/60 to-indigo-100/60 rounded-lg border-l-4 border-indigo-400">
            <div className="text-sm font-medium text-slate-700 mb-1">Hızlı Değerlendirme</div>
            <div className="flex items-center gap-2">
              {companyData.pe > 0 && companyData.pe < 15 && (
                <Badge className="bg-green-100 text-green-700 text-xs">
                  <TrendingUp className="h-3 w-3 mr-1" />
                  Düşük F/K
                </Badge>
              )}
              {companyData.pb > 0 && companyData.pb < 1.5 && (
                <Badge className="bg-blue-100 text-blue-700 text-xs">
                  <BarChart3 className="h-3 w-3 mr-1" />
                  Düşük PD/DD
                </Badge>
              )}
              {week52Performance > 70 && (
                <Badge className="bg-red-100 text-red-700 text-xs">
                  <TrendingDown className="h-3 w-3 mr-1" />
                  52W Zirvede
                </Badge>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
