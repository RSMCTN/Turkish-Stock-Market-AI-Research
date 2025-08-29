'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Calculator, TrendingUp, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

// Sector averages and benchmarks
const getSectorBenchmarks = (sector: string) => {
  const benchmarks: { [key: string]: any } = {
    'BANKA': {
      avgPE: 8.5,
      avgPB: 1.4,
      avgROE: 12.0,
      avgDebtEquity: 0, // Banks don't use traditional debt ratios
      description: 'Bankacılık sektörü'
    },
    'ELEKTRIK': {
      avgPE: 18.0,
      avgPB: 1.8,
      avgROE: 8.5,
      avgDebtEquity: 15000,
      description: 'Elektrik ve enerji sektörü'
    },
    'KIMYA': {
      avgPE: 15.5,
      avgPB: 1.6,
      avgROE: 9.2,
      avgDebtEquity: 25000,
      description: 'Kimya ve petrokimya sektörü'
    },
    'ULASTIRMA': {
      avgPE: 12.0,
      avgPB: 1.2,
      avgROE: 7.8,
      avgDebtEquity: 180000,
      description: 'Ulaştırma ve lojistik sektörü'
    },
    'METALESYA': {
      avgPE: 16.0,
      avgPB: 2.1,
      avgROE: 8.8,
      avgDebtEquity: 12000,
      description: 'Metal eşya ve makine sektörü'
    }
  };

  return benchmarks[sector] || {
    avgPE: 15.0,
    avgPB: 1.8,
    avgROE: 8.0,
    avgDebtEquity: 20000,
    description: 'Genel sektör ortalaması'
  };
};

// Company data (same as CompanyInfoCard)
const getCompanyData = (symbol: string) => {
  const companyData: { [key: string]: any } = {
    'AKSEN': {
      name: 'AKSA ENERJI URETIM',
      sector: 'ELEKTRIK',
      pe: 28.92,
      pb: 0.98,
      roe: 8.62,
      debtEquity: 36572,
      revenue: 241803, // Million TL (T.HACİM)
      netIncome: 8355, // Estimated from market cap and PE
    },
    'ASTOR': {
      name: 'ASTOR ENERJI',
      sector: 'METALESYA',
      pe: 18.89,
      pb: 4.34,
      roe: 11.88,
      debtEquity: -6964,
      revenue: 794913,
      netIncome: 6010,
    },
    'GARAN': {
      name: 'T. GARANTI BANKASI',
      sector: 'BANKA',
      pe: 6.07,
      pb: 1.62,
      roe: 0.0,
      debtEquity: 0,
      revenue: 2145638,
      netIncome: 100896,
    },
    'THYAO': {
      name: 'TURK HAVA YOLLARI',
      sector: 'ULASTIRMA',
      pe: 4.64,
      pb: 0.62,
      roe: 5.63,
      debtEquity: 335241,
      revenue: 7060371,
      netIncome: 101121,
    },
    'TUPRS': {
      name: 'TUPRAS',
      sector: 'KIMYA',
      pe: 14.25,
      pb: 1.07,
      roe: 5.37,
      debtEquity: -48981,
      revenue: 2331123,
      netIncome: 23128,
    }
  };

  return companyData[symbol] || {
    name: symbol,
    sector: 'Diğer',
    pe: 0,
    pb: 0,
    roe: 0,
    debtEquity: 0,
    revenue: 0,
    netIncome: 0
  };
};

// Investment recommendation logic
const getInvestmentRecommendation = (company: any, benchmark: any) => {
  let score = 0;
  let reasons = [];

  // P/E Analysis
  if (company.pe > 0) {
    if (company.pe < benchmark.avgPE * 0.8) {
      score += 2;
      reasons.push("F/K oranı sektör ortalamasının altında");
    } else if (company.pe > benchmark.avgPE * 1.3) {
      score -= 1;
      reasons.push("F/K oranı sektör ortalamasının üzerinde");
    }
  }

  // P/B Analysis  
  if (company.pb > 0) {
    if (company.pb < benchmark.avgPB * 0.9) {
      score += 1;
      reasons.push("PD/DD oranı makul seviyelerde");
    } else if (company.pb > benchmark.avgPB * 1.5) {
      score -= 1;
      reasons.push("PD/DD oranı yüksek");
    }
  }

  // ROE Analysis
  if (company.roe > benchmark.avgROE * 1.2) {
    score += 2;
    reasons.push("ROE sektör ortalamasının üzerinde");
  } else if (company.roe < benchmark.avgROE * 0.6) {
    score -= 1;
    reasons.push("ROE düşük");
  }

  // Determine recommendation
  if (score >= 3) return { rating: 'AL', color: 'green', reasons };
  if (score >= 1) return { rating: 'TUTAN', color: 'blue', reasons };
  if (score >= -1) return { rating: 'İZLE', color: 'yellow', reasons };
  return { rating: 'SAT', color: 'red', reasons };
};

interface FundamentalAnalysisProps {
  selectedSymbol?: string;
}

export default function FundamentalAnalysis({ selectedSymbol = 'GARAN' }: FundamentalAnalysisProps) {
  const companyData = getCompanyData(selectedSymbol);
  const benchmark = getSectorBenchmarks(companyData.sector);
  const recommendation = getInvestmentRecommendation(companyData, benchmark);

  return (
    <Card className="bg-gradient-to-br from-emerald-50 to-teal-50 border-emerald-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calculator className="h-5 w-5 text-emerald-600" />
          Temel Analiz - {selectedSymbol}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge className="bg-emerald-100 text-emerald-700">{companyData.sector}</Badge>
          <Badge 
            className={`${
              recommendation.color === 'green' ? 'bg-green-100 text-green-700' :
              recommendation.color === 'blue' ? 'bg-blue-100 text-blue-700' :
              recommendation.color === 'yellow' ? 'bg-yellow-100 text-yellow-700' :
              'bg-red-100 text-red-700'
            }`}
          >
            {recommendation.rating}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          
          {/* Sector Comparison */}
          <div className="p-3 bg-white/60 rounded-lg">
            <h3 className="font-medium text-emerald-800 mb-3">Sektör Karşılaştırması</h3>
            
            <div className="space-y-3">
              {/* P/E Comparison */}
              {companyData.pe > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">F/K Oranı</span>
                    <span className="text-sm font-medium">
                      {companyData.pe.toFixed(1)} / {benchmark.avgPE.toFixed(1)}
                    </span>
                  </div>
                  <Progress 
                    value={Math.min(100, (companyData.pe / benchmark.avgPE) * 50)} 
                    className="h-2"
                  />
                </div>
              )}

              {/* P/B Comparison */}
              {companyData.pb > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">PD/DD Oranı</span>
                    <span className="text-sm font-medium">
                      {companyData.pb.toFixed(2)} / {benchmark.avgPB.toFixed(2)}
                    </span>
                  </div>
                  <Progress 
                    value={Math.min(100, (companyData.pb / benchmark.avgPB) * 50)} 
                    className="h-2"
                  />
                </div>
              )}

              {/* ROE Comparison */}
              {companyData.roe > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">ROE</span>
                    <span className="text-sm font-medium">
                      {companyData.roe.toFixed(1)}% / {benchmark.avgROE.toFixed(1)}%
                    </span>
                  </div>
                  <Progress 
                    value={Math.min(100, (companyData.roe / benchmark.avgROE) * 50)} 
                    className="h-2"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Financial Health Indicators */}
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center p-3 bg-white/40 rounded-lg">
              <div className="text-sm text-slate-600 mb-1">Karlılık</div>
              <div className="flex items-center justify-center gap-1">
                {companyData.roe > benchmark.avgROE ? 
                  <CheckCircle className="h-4 w-4 text-green-500" /> :
                  <XCircle className="h-4 w-4 text-red-500" />
                }
                <span className={`font-medium text-sm ${
                  companyData.roe > benchmark.avgROE ? 'text-green-600' : 'text-red-600'
                }`}>
                  {companyData.roe > benchmark.avgROE ? 'İYİ' : 'ZAYIF'}
                </span>
              </div>
            </div>
            
            <div className="text-center p-3 bg-white/40 rounded-lg">
              <div className="text-sm text-slate-600 mb-1">Değerleme</div>
              <div className="flex items-center justify-center gap-1">
                {(companyData.pe > 0 && companyData.pe < benchmark.avgPE && 
                  companyData.pb > 0 && companyData.pb < benchmark.avgPB) ? 
                  <CheckCircle className="h-4 w-4 text-green-500" /> :
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                }
                <span className={`font-medium text-sm ${
                  (companyData.pe > 0 && companyData.pe < benchmark.avgPE) ? 'text-green-600' : 'text-yellow-600'
                }`}>
                  {(companyData.pe > 0 && companyData.pe < benchmark.avgPE) ? 'UCUZ' : 'NORMAL'}
                </span>
              </div>
            </div>
          </div>

          {/* Investment Recommendation */}
          <div className={`p-3 rounded-lg border-l-4 ${
            recommendation.color === 'green' ? 'bg-green-50 border-green-400' :
            recommendation.color === 'blue' ? 'bg-blue-50 border-blue-400' :
            recommendation.color === 'yellow' ? 'bg-yellow-50 border-yellow-400' :
            'bg-red-50 border-red-400'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className={`h-4 w-4 ${
                recommendation.color === 'green' ? 'text-green-600' :
                recommendation.color === 'blue' ? 'text-blue-600' :
                recommendation.color === 'yellow' ? 'text-yellow-600' :
                'text-red-600'
              }`} />
              <span className="font-medium text-sm">Yatırım Önerisi: {recommendation.rating}</span>
            </div>
            <ul className="text-xs space-y-1">
              {recommendation.reasons.map((reason, index) => (
                <li key={index} className="text-slate-600">• {reason}</li>
              ))}
            </ul>
          </div>

          {/* Key Financial Metrics */}
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center p-2 bg-white/40 rounded text-xs">
              <div className="text-slate-600">Ciro</div>
              <div className="font-medium">
                {companyData.revenue > 1000 ? 
                  `₺${(companyData.revenue / 1000).toFixed(1)}B` : 
                  `₺${companyData.revenue}M`}
              </div>
            </div>
            <div className="text-center p-2 bg-white/40 rounded text-xs">
              <div className="text-slate-600">Net Kar</div>
              <div className="font-medium">
                {companyData.netIncome > 1000 ? 
                  `₺${(companyData.netIncome / 1000).toFixed(1)}B` : 
                  `₺${companyData.netIncome}M`}
              </div>
            </div>
            <div className="text-center p-2 bg-white/40 rounded text-xs">
              <div className="text-slate-600">Kar Marjı</div>
              <div className="font-medium">
                {companyData.revenue > 0 ? 
                  `${((companyData.netIncome / companyData.revenue) * 100).toFixed(1)}%` : 
                  'N/A'}
              </div>
            </div>
          </div>

        </div>
      </CardContent>
    </Card>
  );
}
