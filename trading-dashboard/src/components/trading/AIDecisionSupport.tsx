'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Brain, TrendingUp, AlertCircle, CheckCircle, XCircle, Target, Zap, Activity, TrendingDown } from 'lucide-react';

// Enhanced company data with comprehensive analysis
const getEnhancedCompanyData = (symbol: string) => {
  const companyData: { [key: string]: any } = {
    'AKSEN': {
      name: 'AKSA ENERJI URETIM',
      sector: 'ELEKTRIK',
      marketCap: 47900,
      pe: 28.92,
      pb: 0.98,
      roe: 8.62,
      debtEquity: 36572,
      week52High: 43.74,
      week52Low: 29.92,
      volume: 6109521,
      volumeTL: 240538725,
      weekPerf: -2.1,
      monthPerf: 5.8,
      month3Perf: 12.3,
      yearPerf: 45.2,
      netDebt: 125000,
      currentPrice: 39.06
    },
    'ASTOR': {
      name: 'ASTOR ENERJI',
      sector: 'METALESYA',
      marketCap: 113472,
      pe: 18.89,
      pb: 4.34,
      roe: 11.88,
      debtEquity: -6964,
      week52High: 128.84,
      week52Low: 64.27,
      volume: 6981245,
      volumeTL: 789456123,
      weekPerf: 3.2,
      monthPerf: 8.7,
      month3Perf: 22.1,
      yearPerf: 67.3,
      netDebt: -45000,
      currentPrice: 113.7
    },
    'GARAN': {
      name: 'T. GARANTI BANKASI',
      sector: 'BANKA',
      marketCap: 612360,
      pe: 6.07,
      pb: 1.62,
      roe: 18.2,
      debtEquity: 0,
      week52High: 154.5,
      week52Low: 93.73,
      volume: 14669824,
      volumeTL: 2134567890,
      weekPerf: 1.5,
      monthPerf: -3.2,
      month3Perf: 15.6,
      yearPerf: 34.8,
      netDebt: 0,
      currentPrice: 145.8
    },
    'THYAO': {
      name: 'TURK HAVA YOLLARI',
      sector: 'ULASTIRMA',
      marketCap: 469200,
      pe: 4.64,
      pb: 0.62,
      roe: 5.63,
      debtEquity: 335241,
      week52High: 346.25,
      week52Low: 249.2,
      volume: 20691413,
      volumeTL: 7034879640,
      weekPerf: -1.8,
      monthPerf: 12.4,
      month3Perf: 8.9,
      yearPerf: 28.7,
      netDebt: 8500000,
      currentPrice: 340.0
    },
    'TUPRS': {
      name: 'TUPRAS',
      sector: 'KIMYA',
      marketCap: 329482,
      pe: 14.25,
      pb: 1.07,
      roe: 5.37,
      debtEquity: -48981,
      week52High: 174.9,
      week52Low: 116.36,
      volume: 13549874,
      volumeTL: 2317637654,
      weekPerf: 2.3,
      monthPerf: -1.7,
      month3Perf: 18.2,
      yearPerf: 41.5,
      netDebt: -125000,
      currentPrice: 171.0
    }
  };

  return companyData[symbol] || {
    name: symbol,
    sector: 'Diğer',
    marketCap: 0,
    pe: 0,
    pb: 0,
    roe: 0,
    currentPrice: 50.0
  };
};

// AI decision logic with detailed reasoning
const getAIDecision = (company: any) => {
  const decisions = [];
  let totalScore = 0;
  let confidence = 0;

  // Valuation Analysis
  if (company.pe > 0 && company.pe < 12) {
    decisions.push({
      factor: 'Değerleme',
      criterion: `F/K oranı ${company.pe.toFixed(1)} (düşük)`,
      impact: 'Pozitif',
      weight: 25,
      explanation: 'Hisse senedi düşük F/K ile ucuz görünüyor'
    });
    totalScore += 25;
  } else if (company.pe > 20) {
    decisions.push({
      factor: 'Değerleme',
      criterion: `F/K oranı ${company.pe.toFixed(1)} (yüksek)`,
      impact: 'Negatif',
      weight: -15,
      explanation: 'Hisse senedi yüksek F/K ile pahalı görünüyor'
    });
    totalScore -= 15;
  }

  // Market Sentiment & Momentum Analysis (Türkiye piyasası için daha önemli)
  if (company.monthPerf > 10) {
    decisions.push({
      factor: 'Piyasa Momentumu',
      criterion: `Son ay +${company.monthPerf?.toFixed(1)}% (güçlü)`,
      impact: 'Pozitif',
      weight: 25,
      explanation: 'Güçlü aylık momentum - piyasa ilgisi yüksek'
    });
    totalScore += 25;
  } else if (company.monthPerf < -15) {
    decisions.push({
      factor: 'Piyasa Momentumu',
      criterion: `Son ay ${company.monthPerf?.toFixed(1)}% (zayıf)`,
      impact: 'Negatif',
      weight: -15,
      explanation: 'Negatif momentum - piyasa ilgisi düşük'
    });
    totalScore -= 15;
  }

  // NOTE: Karlılık Türkiye'de enflasyon muhasebesi nedeniyle yanıltıcı olabilir

  // Teknik Momentum (3 aylık)
  if (company.month3Perf > 20) {
    decisions.push({
      factor: 'Teknik Momentum',
      criterion: `3 ay +${company.month3Perf?.toFixed(1)}% (çok güçlü)`,
      impact: 'Pozitif',
      weight: 20,
      explanation: 'Çok güçlü teknik momentum - trend devam edebilir'
    });
    totalScore += 20;
  } else if (company.month3Perf < -20) {
    decisions.push({
      factor: 'Teknik Momentum',
      criterion: `3 ay ${company.month3Perf?.toFixed(1)}% (çok zayıf)`,
      impact: 'Negatif',
      weight: -10,
      explanation: 'Güçlü negatif momentum, dikkatli olunmalı'
    });
    totalScore -= 10;
  }

  // Piyasa İlgisi & Likidite (Türkiye için kritik)
  if (company.volumeTL > 100000000) {
    decisions.push({
      factor: 'Piyasa İlgisi',
      criterion: `Yüksek işlem hacmi (₺${(company.volumeTL/1000000).toFixed(0)}M)`,
      impact: 'Pozitif',
      weight: 15,
      explanation: 'Yüksek piyasa ilgisi - likidite çok iyi, kolayca işlem yapılabilir'
    });
    totalScore += 15;
  } else if (company.volumeTL < 10000000) {
    decisions.push({
      factor: 'Piyasa İlgisi',
      criterion: `Düşük işlem hacmi (₺${(company.volumeTL/1000000).toFixed(1)}M)`,
      impact: 'Negatif',
      weight: -8,
      explanation: 'Düşük likidite - alım satımda zorluk çekilebilir'
    });
    totalScore -= 8;
  }

  // BIST Endeks Üyeliği (Türkiye için önemli)  
  if (company.marketCap > 500000) {
    decisions.push({
      factor: 'Kurumsallık',
      criterion: `Büyük şirket (₺${(company.marketCap/1000).toFixed(0)}B piyasa değeri)`,
      impact: 'Pozitif',
      weight: 12,
      explanation: 'Büyük şirket - kurumsal yatırımcı ilgisi, istikrar'
    });
    totalScore += 12;
  }

  // Enflasyon Hedge Faktörü (Türkiye özel)
  if (company.sector === 'BANKA' || company.sector === 'KIMYA' || company.sector === 'METALESYA') {
    decisions.push({
      factor: 'Enflasyon Korunması',
      criterion: `${company.sector} sektörü enflasyon hedge'i`,
      impact: 'Pozitif',
      weight: 8,
      explanation: 'Enflasyon ortamında değer koruma potansiyeli yüksek sektör'
    });
    totalScore += 8;
  }

  // Calculate confidence based on number of factors
  confidence = Math.min(95, Math.max(60, decisions.length * 15 + Math.abs(totalScore)));

  // Final decision
  let finalDecision, decisionColor, actionPlan;
  if (totalScore >= 30) {
    finalDecision = 'GÜÇLÜ AL';
    decisionColor = 'green';
    actionPlan = 'Pozisyon açılabilir, risk toleransına göre %3-5 ağırlık verilebilir';
  } else if (totalScore >= 15) {
    finalDecision = 'AL';
    decisionColor = 'blue';
    actionPlan = 'Küçük pozisyon açılabilir, gelişmeleri takip edin';
  } else if (totalScore >= -5) {
    finalDecision = 'İZLE';
    decisionColor = 'yellow';
    actionPlan = 'Mevcut pozisyonu koruyun, gelişmeleri yakından takip edin';
  } else if (totalScore >= -20) {
    finalDecision = 'SAT';
    decisionColor = 'orange';
    actionPlan = 'Pozisyon azaltın veya çıkış stratejisi geliştirin';
  } else {
    finalDecision = 'GÜÇLÜ SAT';
    decisionColor = 'red';
    actionPlan = 'Mevcut pozisyonları kapatın, yeni pozisyon açmayın';
  }

  return {
    decision: finalDecision,
    color: decisionColor,
    score: totalScore,
    confidence,
    factors: decisions,
    actionPlan
  };
};

interface AIDecisionSupportProps {
  selectedSymbol?: string;
}

export default function AIDecisionSupport({ selectedSymbol = 'GARAN' }: AIDecisionSupportProps) {
  const [realTimeAnalysis, setRealTimeAnalysis] = useState<any>(null);
  const [priceData, setPriceData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  
  const companyData = getEnhancedCompanyData(selectedSymbol);
  const aiDecision = getAIDecision(companyData);

  // Real-time comprehensive analysis
  useEffect(() => {
    const performAdvancedAnalysis = async () => {
      setLoading(true);
      try {
        // COMPREHENSIVE ANALYSIS API CALL
        const baseUrl = 'https://bistai001-production.up.railway.app';
        
        console.log(`🔥 ADVANCED ANALYSIS BAŞLATIYOR: ${selectedSymbol}`);
        console.log(`📊 VERİ KAYNAKLARI: Historical (60min+daily), LSTM, Technical, KAP, Sentiment, Risk`);
        
        try {
          // Real comprehensive analysis API
          const response = await fetch(`${baseUrl}/api/comprehensive-analysis/${selectedSymbol}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json'
            }
          });
          
          if (response.ok) {
            const result = await response.json();
            console.log(`✅ GERÇEK ANALİZ TAMAMLANDI: ${result.data.calculations_performed}+ hesaplama`);
            
            const analysis = result.data.analysis;
            const advancedAnalysis = {
              priceTargets: {
                support: analysis.price_targets.support,
                resistance: analysis.price_targets.resistance,
                target: analysis.price_targets.target_1d,
                stopLoss: analysis.price_targets.stop_loss
              },
              technicalSignals: analysis.technical_signals,
              riskMetrics: analysis.risk_metrics,
              kapImpact: analysis.kap_impact,
              sentimentScore: analysis.sentiment_score,
              positionSizing: analysis.position_sizing,
              finalDecision: analysis.final_decision,
              dataSourcesCount: result.data.data_sources_count,
              calculationsPerformed: result.data.calculations_performed
            };
            
            setRealTimeAnalysis(advancedAnalysis);
            console.log(`🎯 FİNAL KARAR: ${analysis.final_decision.decision} (Güven: %${analysis.final_decision.confidence})`);
            
          } else {
            throw new Error(`API error: ${response.status}`);
          }
          
        } catch (apiError) {
          console.log(`⚠️ API henüz hazır değil, mock analiz kullanılıyor...`);
          
          // Fallback to enhanced mock analysis
          const mockAnalysis = {
            priceTargets: {
              support: companyData.currentPrice * 0.92,
              resistance: companyData.currentPrice * 1.08,
              target: companyData.currentPrice * (1 + (Math.random() * 0.10 + 0.02)),
              stopLoss: companyData.currentPrice * 0.88
            },
            technicalSignals: [
              { 
                timeframe: '1H', 
                signal: Math.random() > 0.5 ? 'BUY' : 'HOLD', 
                strength: Math.random() * 0.5 + 0.4,
                rsi: Math.random() * 40 + 30,
                confidence: Math.random() * 0.3 + 0.6
              },
              { 
                timeframe: '4H', 
                signal: Math.random() > 0.6 ? 'BUY' : 'HOLD', 
                strength: Math.random() * 0.4 + 0.3,
                rsi: Math.random() * 40 + 30,
                confidence: Math.random() * 0.3 + 0.6
              },
              { 
                timeframe: '1D', 
                signal: Math.random() > 0.4 ? 'BUY' : 'SELL', 
                strength: Math.random() * 0.6 + 0.2,
                rsi: Math.random() * 40 + 30,
                confidence: Math.random() * 0.3 + 0.7
              }
            ],
            riskMetrics: {
              volatility: Math.random() * 0.3 + 0.15,
              beta: Math.random() * 1.0 + 0.7,
              var_95: companyData.currentPrice * (Math.random() * 0.08 + 0.03),
              sharpe_ratio: Math.random() * 1.5 + 0.5,
              risk_score: Math.random() * 40 + 30
            },
            dataSourcesCount: 6,
            calculationsPerformed: 150,
            isMock: true
          };
          
          setRealTimeAnalysis(mockAnalysis);
        }
        
      } catch (error) {
        console.error('❌ Advanced analysis completely failed:', error);
      } finally {
        setLoading(false);
      }
    };
    
    performAdvancedAnalysis();
  }, [selectedSymbol]);

  return (
    <Card className="bg-gradient-to-br from-purple-50 to-indigo-50 border-purple-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-purple-600" />
          AI Karar Destek - BIST Odaklı - {selectedSymbol}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge 
            className={`${
              aiDecision.color === 'green' ? 'bg-green-100 text-green-700' :
              aiDecision.color === 'blue' ? 'bg-blue-100 text-blue-700' :
              aiDecision.color === 'yellow' ? 'bg-yellow-100 text-yellow-700' :
              aiDecision.color === 'orange' ? 'bg-orange-100 text-orange-700' :
              'bg-red-100 text-red-700'
            }`}
          >
            {aiDecision.decision}
          </Badge>
          <Badge variant="outline" className="text-xs">
            Güven: %{aiDecision.confidence}
          </Badge>
          <Badge variant="outline" className="text-xs">
            Skor: {aiDecision.score > 0 ? '+' : ''}{aiDecision.score}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          
          {/* AI Decision Summary */}
          <div className={`p-4 rounded-lg border-l-4 ${
            aiDecision.color === 'green' ? 'bg-green-50 border-green-400' :
            aiDecision.color === 'blue' ? 'bg-blue-50 border-blue-400' :
            aiDecision.color === 'yellow' ? 'bg-yellow-50 border-yellow-400' :
            aiDecision.color === 'orange' ? 'bg-orange-50 border-orange-400' :
            'bg-red-50 border-red-400'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <Target className={`h-4 w-4 ${
                aiDecision.color === 'green' ? 'text-green-600' :
                aiDecision.color === 'blue' ? 'text-blue-600' :
                aiDecision.color === 'yellow' ? 'text-yellow-600' :
                aiDecision.color === 'orange' ? 'text-orange-600' :
                'text-red-600'
              }`} />
              <span className="font-medium">AI Kararı: {aiDecision.decision}</span>
            </div>
            <p className="text-sm text-slate-700 mb-2">{aiDecision.actionPlan}</p>
            <div className="flex items-center gap-4 text-xs text-slate-600">
              <span>Güven Seviyesi: %{aiDecision.confidence}</span>
              <span>Toplam Skor: {aiDecision.score > 0 ? '+' : ''}{aiDecision.score}</span>
              <span>{aiDecision.factors.length} faktör analiz edildi</span>
            </div>
          </div>

          {/* Decision Factors Analysis */}
          <div>
            <h3 className="font-medium text-purple-800 mb-3 flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Karar Faktörleri Analizi
            </h3>
            
            <div className="space-y-3">
              {aiDecision.factors.map((factor, index) => (
                <div key={index} className="p-3 bg-white/60 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{factor.factor}</span>
                      {factor.impact === 'Pozitif' ? 
                        <CheckCircle className="h-3 w-3 text-green-500" /> :
                        <XCircle className="h-3 w-3 text-red-500" />
                      }
                    </div>
                    <Badge 
                      className={`text-xs ${
                        factor.impact === 'Pozitif' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {factor.weight > 0 ? '+' : ''}{factor.weight}
                    </Badge>
                  </div>
                  <div className="text-sm text-slate-600 mb-1">{factor.criterion}</div>
                  <div className="text-xs text-slate-500">{factor.explanation}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Score Breakdown */}
          <div className="p-3 bg-white/60 rounded-lg">
            <h4 className="font-medium text-sm text-slate-700 mb-2">Skor Dağılımı</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Pozitif Faktörler:</span>
                <span className="text-green-600 font-medium">
                  +{aiDecision.factors.filter(f => f.weight > 0).reduce((sum, f) => sum + f.weight, 0)}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>Negatif Faktörler:</span>
                <span className="text-red-600 font-medium">
                  {aiDecision.factors.filter(f => f.weight < 0).reduce((sum, f) => sum + f.weight, 0)}
                </span>
              </div>
              <div className="border-t pt-2">
                <div className="flex items-center justify-between text-sm font-medium">
                  <span>Net Skor:</span>
                  <span className={aiDecision.score > 0 ? 'text-green-600' : 'text-red-600'}>
                    {aiDecision.score > 0 ? '+' : ''}{aiDecision.score}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Visual Score Bar */}
            <div className="mt-3">
              <div className="w-full bg-slate-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-300 ${
                    aiDecision.score > 0 ? 'bg-gradient-to-r from-green-400 to-green-600' : 'bg-gradient-to-r from-red-400 to-red-600'
                  }`}
                  style={{ 
                    width: `${Math.min(100, Math.max(0, (Math.abs(aiDecision.score) / 50) * 100))}%`,
                    marginLeft: aiDecision.score < 0 ? 'auto' : '0'
                  }}
                />
              </div>
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>-50 (Güçlü Sat)</span>
                <span>0 (Nötr)</span>
                <span>+50 (Güçlü Al)</span>
              </div>
            </div>
          </div>

          {/* Advanced Price Analysis */}
          {!loading && realTimeAnalysis && (
            <div className="space-y-3">
              <h3 className="font-medium text-purple-800 mb-3 flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Gelişmiş Fiyat Analizi
              </h3>
              
              {/* Price Targets */}
              <div className="grid grid-cols-2 gap-3">
                <div className="text-center p-3 bg-green-50 rounded-lg border">
                  <div className="text-xs text-green-600">Hedef Fiyat</div>
                  <div className="font-bold text-green-700">
                    ₺{(realTimeAnalysis?.priceTargets?.target || companyData.currentPrice).toFixed(2)}
                  </div>
                  <div className="text-xs text-green-500">
                    +{(((((realTimeAnalysis?.priceTargets?.target || companyData.currentPrice) / companyData.currentPrice) - 1) * 100)).toFixed(1)}%
                  </div>
                </div>
                <div className="text-center p-3 bg-red-50 rounded-lg border">
                  <div className="text-xs text-red-600">Stop Loss</div>
                  <div className="font-bold text-red-700">
                    ₺{(realTimeAnalysis?.priceTargets?.stopLoss || companyData.currentPrice * 0.95).toFixed(2)}
                  </div>
                  <div className="text-xs text-red-500">
                    {(((((realTimeAnalysis?.priceTargets?.stopLoss || companyData.currentPrice * 0.95) / companyData.currentPrice) - 1) * 100)).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Multi-Timeframe Signals */}
              <div>
                <div className="font-medium text-sm text-slate-700 mb-2">Multi-Timeframe Analizi:</div>
                <div className="space-y-2">
                  {(realTimeAnalysis?.technicalSignals || []).map((signal: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-white/60 rounded text-sm">
                      <span>{signal.timeframe}</span>
                      <div className="flex items-center gap-2">
                        <Badge className={`text-xs ${
                          signal.signal === 'BUY' ? 'bg-green-100 text-green-700' :
                          signal.signal === 'SELL' ? 'bg-red-100 text-red-700' :
                          'bg-yellow-100 text-yellow-700'
                        }`}>
                          {signal.signal}
                        </Badge>
                        <div className="w-16 bg-slate-200 rounded-full h-1">
                          <div 
                            className={`h-1 rounded-full ${
                              signal.strength > 0.7 ? 'bg-green-500' :
                              signal.strength > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${signal.strength * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Risk Metrics */}
              <div className="p-3 bg-orange-50 rounded-lg border">
                <div className="font-medium text-sm text-orange-800 mb-2">Risk Metrikleri:</div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center">
                    <div className="text-orange-600">Volatilite</div>
                    <div className="font-medium">{((realTimeAnalysis?.riskMetrics?.volatility || 0) * 100).toFixed(1)}%</div>
                  </div>
                  <div className="text-center">
                    <div className="text-orange-600">Beta</div>
                    <div className="font-medium">{(realTimeAnalysis?.riskMetrics?.beta || 0).toFixed(2)}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-orange-600">VaR (95%)</div>
                    <div className="font-medium">₺{(realTimeAnalysis?.riskMetrics?.var95 || 0).toFixed(2)}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="text-center py-4">
              <Activity className="h-6 w-6 animate-spin mx-auto mb-2 text-purple-600" />
              <div className="text-sm text-slate-600">Gelişmiş analiz hesaplanıyor...</div>
            </div>
          )}

          {/* System Performance Info */}
          {!loading && realTimeAnalysis && (
            <div className="p-3 bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg">
              <div className="text-xs text-purple-700 font-medium mb-1">
                🚀 Kapsamlı Analiz Sistemi
                {realTimeAnalysis?.isMock && <span className="text-orange-600"> (Mock Mode)</span>}
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-purple-600">
                <div>
                  <strong>Veri Kaynakları:</strong> {realTimeAnalysis?.dataSourcesCount || 6}
                </div>
                <div>
                  <strong>Hesaplamalar:</strong> {realTimeAnalysis?.calculationsPerformed || 150}+
                </div>
              </div>
              <div className="text-xs text-purple-600 mt-1">
                • <strong>Historical Data:</strong> 60min + günlük veriler<br/>
                • <strong>LSTM Tahminleri:</strong> Fiyat hedefleri<br/>
                • <strong>Teknik Analiz:</strong> Multi-timeframe RSI, MACD, Bollinger<br/>
                • <strong>KAP Analysis:</strong> Son duyuru etkisi<br/>
                • <strong>Sentiment:</strong> Piyasa duygu analizi<br/>
                • <strong>Risk Management:</strong> VaR, Beta, Position sizing
              </div>
            </div>
          )}

          {/* Türkiye Piyasası Notu */}
          <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
            <div className="text-xs text-indigo-700 font-medium mb-1">📍 BIST Piyasası Özel Yaklaşım</div>
            <div className="text-xs text-indigo-600">
              • <strong>Enflasyon muhasebesi</strong> dikkate alınarak karlılık düşük ağırlık<br/>
              • <strong>Piyasa momentumu</strong> ve likidite Türkiye için kritik<br/>
              • <strong>KAP duyuruları</strong> sentiment analizine dahil<br/>
              • <strong>Multi-timeframe</strong> 1H, 4H, 1D sinyalleri<br/>
              {realTimeAnalysis?.isMock && 
                <span className="text-orange-600">• <strong>Backend API entegrasyonu</strong> geliştirme aşamasında</span>
              }
            </div>
          </div>

        </div>
      </CardContent>
    </Card>
  );
}
