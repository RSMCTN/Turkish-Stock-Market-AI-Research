'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Calendar, 
  DollarSign,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Play
} from 'lucide-react';

interface StockAnalysis {
  symbol: string;
  currentPrice: number;
  analysis_timestamp: string;
  fiveDayPredictions: Array<{
    day: number;
    date: string;
    predictedPrice: number;
    dailyLow: number;
    dailyHigh: number;
    volatility: number;
  }>;
  priceRangeWeekly: {
    minPrice: number;
    maxPrice: number;
    avgPrice: number;
  };
  entryPoint: {
    recommendedPrice: number;
    timing: 'IMMEDIATE' | 'WAIT_FOR_DIP';
    confidence: number;
    reason: string;
  };
  exitPoint: {
    targetPrice: number;
    expectedDay: number;
    expectedDate: string;
    probability: number;
    stopLoss: number;
  };
  profitabilityAnalysis: {
    expectedReturn: number;
    riskRewardRatio: number;
    investmentGrade: 'HIGH' | 'MEDIUM' | 'LOW';
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD';
  };
  technicalSummary: {
    trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    momentum: 'STRONG' | 'MODERATE' | 'WEAK';
    volatility: 'HIGH' | 'MEDIUM' | 'LOW';
    volume: 'ABOVE_AVERAGE' | 'NORMAL' | 'BELOW_AVERAGE';
  };
}

interface PortfolioSummary {
  totalSymbols: number;
  averageExpectedReturn: number;
  strongBuyCount: number;
  buyCount: number;
  holdCount: number;
  highRiskCount: number;
  analysisTimestamp: string;
}

const BulkAnalysis = () => {
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [analyses, setAnalyses] = useState<StockAnalysis[]>([]);
  const [portfolioSummary, setPortfolioSummary] = useState<PortfolioSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedAnalysis, setExpandedAnalysis] = useState<string | null>(null);

  // Real BIST symbols from backend
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  
  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const response = await fetch('https://bistai001-production.up.railway.app/api/bist/all-stocks?limit=100');
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.stocks) {
            setAvailableSymbols(data.stocks.map((stock: any) => stock.symbol));
          }
        }
      } catch (error) {
        console.error('Failed to fetch symbols:', error);
        // Fallback to a few symbols if API fails
        setAvailableSymbols(['GARAN', 'AKBNK', 'ISCTR', 'THYAO', 'ASELS']);
      }
    };
    
    fetchSymbols();
  }, []);

  const toggleSymbol = (symbol: string) => {
    setSelectedSymbols(prev => 
      prev.includes(symbol) 
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    );
  };

  const runBulkAnalysis = async () => {
    if (selectedSymbols.length === 0) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('https://bistai001-production.up.railway.app/api/bulk-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selectedSymbols)
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnalyses(data.bulkAnalysis);
        setPortfolioSummary(data.portfolioSummary);
      }
    } catch (error) {
      console.error('Bulk analysis failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'STRONG_BUY': return 'bg-green-100 text-green-800 border-green-300';
      case 'BUY': return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'HOLD': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return 'text-green-600';
      case 'MEDIUM': return 'text-yellow-600';
      case 'HIGH': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'BULLISH': return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'BEARISH': return <TrendingDown className="h-4 w-4 text-red-600" />;
      default: return <div className="h-4 w-4 rounded-full bg-gray-400"></div>;
    }
  };

  return (
    <div className="space-y-6">
      
      {/* Stock Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Bulk Stock Analysis
          </CardTitle>
          <CardDescription>
            Select multiple BIST stocks for comprehensive analysis including 5-day predictions, entry/exit points, and profitability assessments
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            
            {/* Symbol Grid */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-medium text-gray-700">
                  Select Symbols ({selectedSymbols.length}/{availableSymbols.length})
                </p>
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => setSelectedSymbols([])}
                  >
                    Clear All
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => setSelectedSymbols([...availableSymbols.slice(0, 10)])}
                  >
                    Top 10
                  </Button>
                </div>
              </div>
              
              <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 mb-4">
                {availableSymbols.map(symbol => (
                  <div 
                    key={symbol}
                    className={`flex items-center space-x-2 p-2 border rounded cursor-pointer transition-all hover:shadow-sm ${
                      selectedSymbols.includes(symbol) 
                        ? 'bg-blue-50 border-blue-300' 
                        : 'bg-white border-gray-200 hover:bg-gray-50'
                    }`}
                    onClick={() => toggleSymbol(symbol)}
                  >
                    <Checkbox 
                      checked={selectedSymbols.includes(symbol)}
                      className="pointer-events-none"
                    />
                    <label className="text-sm font-medium cursor-pointer">
                      {symbol}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {/* Run Analysis Button */}
            <div className="flex justify-center">
              <Button 
                onClick={runBulkAnalysis}
                disabled={selectedSymbols.length === 0 || isLoading}
                size="lg"
                className="gap-2"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Analyze {selectedSymbols.length} Symbols
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Portfolio Summary */}
      {portfolioSummary && (
        <Card>
          <CardHeader>
            <CardTitle>Portfolio Analysis Summary</CardTitle>
            <CardDescription>
              Overview of all analyzed symbols - {new Date(portfolioSummary.analysisTimestamp).toLocaleString()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{portfolioSummary.totalSymbols}</p>
                <p className="text-sm text-gray-600">Total Symbols</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{portfolioSummary.averageExpectedReturn}%</p>
                <p className="text-sm text-gray-600">Avg Return</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">{portfolioSummary.strongBuyCount}</p>
                <p className="text-sm text-gray-600">Strong Buy</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{portfolioSummary.buyCount}</p>
                <p className="text-sm text-gray-600">Buy</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-yellow-600">{portfolioSummary.holdCount}</p>
                <p className="text-sm text-gray-600">Hold</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-red-600">{portfolioSummary.highRiskCount}</p>
                <p className="text-sm text-gray-600">High Risk</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Individual Analysis Results */}
      {analyses.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Individual Analysis Results</h3>
          
          {analyses.map((analysis) => (
            <Card key={analysis.symbol} className="overflow-hidden">
              <CardHeader 
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => setExpandedAnalysis(
                  expandedAnalysis === analysis.symbol ? null : analysis.symbol
                )}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getTrendIcon(analysis.technicalSummary.trend)}
                    <div>
                      <CardTitle className="text-lg">{analysis.symbol}</CardTitle>
                      <p className="text-sm text-gray-600">₺{analysis.currentPrice.toFixed(2)}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    <Badge className={getRecommendationColor(analysis.profitabilityAnalysis.recommendation)}>
                      {analysis.profitabilityAnalysis.recommendation.replace('_', ' ')}
                    </Badge>
                    
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Expected Return</p>
                      <p className={`font-bold ${
                        analysis.profitabilityAnalysis.expectedReturn > 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {analysis.profitabilityAnalysis.expectedReturn > 0 ? '+' : ''}{analysis.profitabilityAnalysis.expectedReturn}%
                      </p>
                    </div>
                  </div>
                </div>
              </CardHeader>

              {expandedAnalysis === analysis.symbol && (
                <CardContent className="pt-0">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    
                    {/* 5-Day Predictions */}
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-900 flex items-center gap-2">
                        <Calendar className="h-4 w-4" />
                        5-Day Price Predictions
                      </h4>
                      <div className="space-y-2">
                        {analysis.fiveDayPredictions.map((pred) => (
                          <div key={pred.day} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <div>
                              <p className="text-sm font-medium">Day {pred.day} - {pred.date}</p>
                              <p className="text-xs text-gray-600">Vol: {pred.volatility}%</p>
                            </div>
                            <div className="text-right">
                              <p className="font-semibold">₺{pred.predictedPrice.toFixed(2)}</p>
                              <p className="text-xs text-gray-600">
                                ₺{pred.dailyLow.toFixed(2)} - ₺{pred.dailyHigh.toFixed(2)}
                              </p>
                            </div>
                          </div>
                        ))}
                        
                        <div className="mt-3 p-3 bg-blue-50 rounded border">
                          <p className="text-sm font-medium text-blue-900">Weekly Range Summary</p>
                          <div className="grid grid-cols-3 gap-2 mt-1 text-sm">
                            <div>
                              <p className="text-blue-600">Min: ₺{analysis.priceRangeWeekly.minPrice.toFixed(2)}</p>
                            </div>
                            <div>
                              <p className="text-blue-600">Avg: ₺{analysis.priceRangeWeekly.avgPrice.toFixed(2)}</p>
                            </div>
                            <div>
                              <p className="text-blue-600">Max: ₺{analysis.priceRangeWeekly.maxPrice.toFixed(2)}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Entry/Exit Analysis */}
                    <div className="space-y-4">
                      <h4 className="font-semibold text-gray-900 flex items-center gap-2">
                        <Target className="h-4 w-4" />
                        Entry/Exit Strategy
                      </h4>
                      
                      {/* Entry Point */}
                      <div className="p-3 border rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          <p className="font-medium text-gray-900">Entry Point</p>
                        </div>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Recommended Price:</span>
                            <span className="font-semibold">₺{analysis.entryPoint.recommendedPrice.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Timing:</span>
                            <Badge variant="outline" className="text-xs">
                              {analysis.entryPoint.timing.replace('_', ' ')}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Confidence:</span>
                            <span className="font-semibold">{(analysis.entryPoint.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        <p className="text-xs text-gray-600 mt-2 italic">
                          {analysis.entryPoint.reason}
                        </p>
                      </div>

                      {/* Exit Point */}
                      <div className="p-3 border rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <DollarSign className="h-4 w-4 text-blue-600" />
                          <p className="font-medium text-gray-900">Exit Strategy</p>
                        </div>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Target Price:</span>
                            <span className="font-semibold text-green-600">₺{analysis.exitPoint.targetPrice.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Expected Date:</span>
                            <span className="font-semibold">{analysis.exitPoint.expectedDate}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Probability:</span>
                            <span className="font-semibold">{(analysis.exitPoint.probability * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Stop Loss:</span>
                            <span className="font-semibold text-red-600">₺{analysis.exitPoint.stopLoss.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>

                      {/* Profitability */}
                      <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp className="h-4 w-4 text-green-600" />
                          <p className="font-medium text-green-900">Profitability Analysis</p>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <p className="text-green-700">Expected Return:</p>
                            <p className="font-bold text-green-800">{analysis.profitabilityAnalysis.expectedReturn}%</p>
                          </div>
                          <div>
                            <p className="text-green-700">Risk/Reward:</p>
                            <p className="font-bold text-green-800">{analysis.profitabilityAnalysis.riskRewardRatio}</p>
                          </div>
                          <div>
                            <p className="text-green-700">Grade:</p>
                            <Badge className={`text-xs ${analysis.profitabilityAnalysis.investmentGrade === 'HIGH' ? 'bg-green-200 text-green-800' : 'bg-yellow-200 text-yellow-800'}`}>
                              {analysis.profitabilityAnalysis.investmentGrade}
                            </Badge>
                          </div>
                          <div>
                            <p className="text-green-700">Risk Level:</p>
                            <span className={`font-semibold ${getRiskColor(analysis.profitabilityAnalysis.riskLevel)}`}>
                              {analysis.profitabilityAnalysis.riskLevel}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Technical Summary */}
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="font-medium text-gray-900 mb-2">Technical Summary</p>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Trend:</span>
                            <span className="font-semibold">{analysis.technicalSummary.trend}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Momentum:</span>
                            <span className="font-semibold">{analysis.technicalSummary.momentum}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Volatility:</span>
                            <span className="font-semibold">{analysis.technicalSummary.volatility}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Volume:</span>
                            <span className="font-semibold">{analysis.technicalSummary.volume.replace('_', ' ')}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              )}
            </Card>
          ))}
        </div>
      )}

      {isLoading && (
        <Card>
          <CardContent className="py-12">
            <div className="text-center space-y-4">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
              <p className="text-lg font-medium">Analyzing {selectedSymbols.length} symbols...</p>
              <p className="text-sm text-gray-600">
                Generating 5-day predictions, entry/exit points, and profitability analysis
              </p>
              <Progress value={33} className="w-64 mx-auto" />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default BulkAnalysis;
