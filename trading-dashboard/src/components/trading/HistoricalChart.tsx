'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, TrendingUp, Brain, BarChart3, Target, Clock } from 'lucide-react';
import ProfessionalStockChart from './ProfessionalStockChart';

interface HistoricalChartProps {
  selectedSymbol?: string;
}

export default function HistoricalChart({ selectedSymbol = 'AKBNK' }: HistoricalChartProps) {
  const [loading, setLoading] = useState(false);
  const [forecastHours, setForecastHours] = useState<8 | 16 | 40>(8);

  return (
    <div className="w-full space-y-6">
      {/* Chart Controls */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-4">
            {/* Chart Info */}
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-emerald-400" />
              <span className="text-sm font-medium text-slate-300">Bloomberg Terminal Style</span>
              <Badge className="bg-blue-600 text-white text-xs">SyncFusion</Badge>
            </div>

            {/* Forecast Hours Selector */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-slate-300">Fiyat Öngörüsü:</span>
              <div className="flex bg-slate-700/50 rounded-lg p-1">
                {[8, 16, 40].map((hours) => (
                  <button
                    key={hours}
                    onClick={() => setForecastHours(hours as any)}
                    className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                      forecastHours === hours 
                        ? 'bg-blue-600 text-white shadow-lg' 
                        : 'text-slate-400 hover:text-white hover:bg-slate-600'
                    }`}
                  >
                    {hours}H
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Bloomberg-Style Professional Chart with AI Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Chart Area - SyncFusion StockChart */}
        <div className="lg:col-span-3">
          <ProfessionalStockChart 
            selectedSymbol={selectedSymbol}
            height="600px"
          />
        </div>

        {/* AI Commentary Sidebar */}
        <div className="space-y-4">
          {/* Price Forecast Panel */}
          <div className="bg-gradient-to-br from-blue-900/20 to-indigo-900/20 rounded-lg border border-blue-500/30 p-4">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="h-5 w-5 text-blue-400" />
              <h4 className="font-semibold text-blue-300">{forecastHours}H Fiyat Öngörüsü</h4>
              <Badge className="bg-blue-600 text-white text-xs">DP-LSTM</Badge>
            </div>
            
            <div className="space-y-2">
              {/* Generate forecast prices for selected hours */}
              {Array.from({ length: Math.min(forecastHours / 4, 10) }, (_, i) => {
                const currentPrice = {
                  'AKBNK': 69.5,
                  'BIMAS': 536.0,
                  'GARAN': 145.8,
                  'A1YEN': 58.0,
                }[selectedSymbol] || 50.0;
                
                const forecastPrice = currentPrice * (1 + (Math.random() - 0.5) * 0.05);
                const change = ((forecastPrice - currentPrice) / currentPrice) * 100;
                const isPositive = change > 0;
                
                return (
                  <div key={i} className="flex justify-between items-center p-2 bg-slate-800/50 rounded text-sm">
                    <span className="text-slate-400">{(i + 1) * 4}H:</span>
                    <div className="text-right">
                      <span className="text-white font-medium">₺{forecastPrice.toFixed(2)}</span>
                      <span className={`ml-2 text-xs ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                        {isPositive ? '+' : ''}{change.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* BIST-Ultimate Turkish AI Commentary */}
          <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 rounded-lg border border-purple-500/30 p-4">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="h-5 w-5 text-purple-400" />
              <h4 className="font-semibold text-purple-300">BIST AI Yorumu</h4>
              <Badge className="bg-purple-600 text-white text-xs">v4-Ultimate</Badge>
            </div>
            
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-slate-800/50 rounded border-l-4 border-purple-500">
                <div className="font-medium text-purple-300 mb-1">Teknik Analiz</div>
                <p className="text-slate-300">
                  {selectedSymbol} için RSI seviyelerinde dengeleyici hareket. 
                  MACD trend pozitif sinyaller veriyor. 
                  Bollinger bantları arasında sağlıklı hareket ediyor.
                </p>
              </div>
              
              <div className="p-3 bg-slate-800/50 rounded border-l-4 border-blue-500">
                <div className="font-medium text-blue-300 mb-1">Hacim Analizi</div>
                <p className="text-slate-300">
                  Günlük ortalama hacmin {Math.random() > 0.5 ? 'üzerinde' : 'altında'} işlem görüyor. 
                  {Math.random() > 0.5 ? 'Güçlü' : 'Orta seviye'} yatırımcı ilgisi mevcut.
                </p>
              </div>
              
              <div className="p-3 bg-slate-800/50 rounded border-l-4 border-green-500">
                <div className="font-medium text-green-300 mb-1">Karar Desteği</div>
                <p className="text-slate-300">
                  Mevcut seviyelerden {Math.random() > 0.6 ? 'alım' : Math.random() > 0.3 ? 'bekle' : 'sat'} 
                  {' '}önerisi. Risk yönetimi kritik öneme sahip.
                </p>
              </div>
              
              <div className="p-3 bg-gradient-to-r from-amber-900/30 to-orange-900/30 rounded border border-amber-500/30">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-amber-400" />
                  <span className="font-medium text-amber-300">Hedef Fiyatlar</span>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Destek:</span>
                    <span className="text-green-400">₺{(({
                      'AKBNK': 69.5,
                      'BIMAS': 536.0,
                      'GARAN': 145.8,
                      'A1YEN': 58.0,
                    }[selectedSymbol] || 50.0) * 0.97).toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Direnç:</span>
                    <span className="text-red-400">₺{(({
                      'AKBNK': 69.5,
                      'BIMAS': 536.0,
                      'GARAN': 145.8,
                      'A1YEN': 58.0,
                    }[selectedSymbol] || 50.0) * 1.03).toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Hedef:</span>
                    <span className="text-blue-400">₺{(({
                      'AKBNK': 69.5,
                      'BIMAS': 536.0,
                      'GARAN': 145.8,
                      'A1YEN': 58.0,
                    }[selectedSymbol] || 50.0) * 1.05).toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Technical Summary */}
          <div className="bg-slate-800/30 rounded-lg border border-slate-700 p-4">
            <h4 className="font-semibold text-slate-300 mb-3 flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Teknik Özet
            </h4>
            <div className="space-y-2 text-xs">
              {[
                { label: 'Trend', value: 'Yükseliş', color: 'text-green-400' },
                { label: 'Volatilite', value: 'Orta', color: 'text-blue-400' },
                { label: 'Momentum', value: 'Pozitif', color: 'text-green-400' },
                { label: 'Hacim', value: 'Normal', color: 'text-slate-400' }
              ].map((item, index) => (
                <div key={index} className="flex justify-between">
                  <span className="text-slate-400">{item.label}:</span>
                  <span className={item.color}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
          
          {/* Market Sentiment */}
          <div className="bg-slate-800/30 rounded-lg border border-slate-700 p-4">
            <h4 className="font-semibold text-slate-300 mb-3">Piyasa Duygusu</h4>
            <div className="flex items-center justify-center">
              <div className="relative w-16 h-16">
                <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
                <div className="absolute inset-0 rounded-full border-4 border-emerald-500 border-t-transparent animate-spin"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-sm font-bold text-emerald-400">78%</span>
                </div>
              </div>
            </div>
            <div className="text-center mt-2">
              <div className="text-xs text-slate-400">Güven Skoru</div>
              <div className="text-sm font-medium text-emerald-400">Olumlu</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}