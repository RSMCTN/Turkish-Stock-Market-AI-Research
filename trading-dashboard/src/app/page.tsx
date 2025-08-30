'use client';

import { useState, useEffect } from 'react';
import { Brain, LineChart, Zap, Activity, TrendingUp, AlertCircle, BarChart3, Target } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// Academic Components
import AcademicPredictionPanel from '@/components/trading/AcademicPredictionPanel';
import LiveKAPFeed from '@/components/trading/LiveKAPFeed';
import AcademicMetricsDashboard from '@/components/trading/AcademicMetricsDashboard';
import ComponentContributionChart from '@/components/trading/ComponentContributionChart';
import HuggingFaceModelPanel from '@/components/trading/HuggingFaceModelPanel';
import CompanyInfoCard from '@/components/trading/CompanyInfoCard';
import FundamentalAnalysis from '@/components/trading/FundamentalAnalysis';
import AIDecisionSupport from '@/components/trading/AIDecisionSupport';
import EnhancedStockAnalysis from '@/components/trading/EnhancedStockAnalysis';
import HistoricalChart from '@/components/trading/HistoricalChart';
import AICommentaryPanel from '@/components/trading/AICommentaryPanel';
import AIChatPanel from '@/components/trading/AIChatPanel';

// Traditional Components
import ForecastPanel from '@/components/trading/ForecastPanel';
import RealMarketOverview from '@/components/trading/RealMarketOverview';
import AdvancedIndicators from '@/components/trading/AdvancedIndicators';
import AdvancedNewsSentiment from '@/components/trading/AdvancedNewsSentiment';
import RealSymbolSelector from '@/components/trading/RealSymbolSelector';

export default function AcademicDashboard() {
  const [systemStatus, setSystemStatus] = useState({
    dpLstm: 'active',
    sentimentArma: 'active', 
    kapFeed: 'active',
    huggingFace: 'active',
    differentialPrivacy: 'active'
  });

  const [selectedSymbol, setSelectedSymbol] = useState('GARAN');
  const [indicators, setIndicators] = useState([]);

  // Mock user for demo
  const user = {
    name: "Academic Researcher",
    email: "researcher@academic.edu"
  };

  useEffect(() => {
    // Initialize system health checks
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      // Mock system health check - endpoint doesn't exist yet
      setSystemStatus({
        dpLstm: 'active',
        sentimentArma: 'active', 
        kapFeed: 'active',
        huggingFace: 'active',
        differentialPrivacy: 'active'
      });
    } catch (error) {
      console.error('System health check failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Academic Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm shadow-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-800 to-purple-800 bg-clip-text text-transparent">
                    Academic Trading Research Framework
                  </h1>
                  <p className="text-sm text-slate-600">Differential Privacy LSTM • sentimentARMA • KAP Integration</p>
                </div>
              </div>
              <Badge className="bg-gradient-to-r from-green-100 to-blue-100 text-green-700 border-green-200">
                🎓 Research Mode
              </Badge>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-slate-700">
                <div className="font-medium">{user.name}</div>
                <div className="text-xs text-slate-500">Academic Dashboard</div>
              </div>
              <Button variant="outline" size="sm">
                Export Research
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Academic Dashboard */}
      <main className="container mx-auto px-6 py-8">
        
        {/* System Status Overview */}
        <div className="mb-8 grid grid-cols-2 md:grid-cols-5 gap-4">
          {Object.entries(systemStatus).map(([component, status]) => (
            <Card key={component} className="bg-white/70 border-slate-200">
              <CardContent className="p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-slate-500 capitalize">{component.replace(/([A-Z])/g, ' $1')}</p>
                    <div className="flex items-center gap-1 mt-1">
                      <div className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-green-500' : 'bg-red-500'}`} />
                      <span className="text-sm font-medium text-slate-700">{status}</span>
                    </div>
                  </div>
                  <Activity className={`h-4 w-4 ${status === 'active' ? 'text-green-500' : 'text-red-500'}`} />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Academic Research Tabs */}
        <Tabs defaultValue="academic" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-white/80 border border-slate-200">
            <TabsTrigger 
              value="academic" 
              className="flex items-center gap-2 data-[state=active]:bg-blue-100 data-[state=active]:text-blue-800"
            >
              <Brain className="h-4 w-4" />
              Academic Research
            </TabsTrigger>
            <TabsTrigger 
              value="enhanced" 
              className="flex items-center gap-2 data-[state=active]:bg-orange-100 data-[state=active]:text-orange-800"
            >
              <Target className="h-4 w-4" />
              Kapsamlı Analiz
            </TabsTrigger>
            <TabsTrigger 
              value="traditional" 
              className="flex items-center gap-2 data-[state=active]:bg-green-100 data-[state=active]:text-green-800"
            >
              <LineChart className="h-4 w-4" />
              Traditional Trading
            </TabsTrigger>
            <TabsTrigger 
              value="integrated" 
              className="flex items-center gap-2 data-[state=active]:bg-purple-100 data-[state=active]:text-purple-800"
            >
              <Zap className="h-4 w-4" />
              Integrated View
            </TabsTrigger>
          </TabsList>

          {/* Academic Research Tab */}
          <TabsContent value="academic" className="space-y-6">
            {/* Symbol Selector for Academic Research */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={true}
              limit={600}
            />

            <div className="grid lg:grid-cols-2 gap-6">
              {/* Academic Prediction System */}
              <AcademicPredictionPanel selectedSymbol={selectedSymbol} />
              
              {/* HuggingFace Production Model */}
              <HuggingFaceModelPanel selectedSymbol={selectedSymbol} />
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* Live KAP Feed */}
              <LiveKAPFeed selectedSymbol={selectedSymbol} />
              
              {/* Academic Metrics */}
              <AcademicMetricsDashboard selectedSymbol={selectedSymbol} />
              
              {/* Component Contributions */}
              <ComponentContributionChart selectedSymbol={selectedSymbol} />
            </div>

            {/* Company Analysis Section */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Company Info Card */}
              <CompanyInfoCard selectedSymbol={selectedSymbol} />
              
              {/* Fundamental Analysis */}
              <FundamentalAnalysis selectedSymbol={selectedSymbol} />
            </div>

            {/* AI Decision Support System */}
            <AIDecisionSupport selectedSymbol={selectedSymbol} />

            {/* Academic Research Info */}
            <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Research Framework Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600 mb-1">≥75%</div>
                    <div className="text-sm text-slate-600">Model Accuracy</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600 mb-1">Real-time</div>
                    <div className="text-sm text-slate-600">KAP Integration</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600 mb-1">Active</div>
                    <div className="text-sm text-slate-600">DP Privacy</div>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-white/60 rounded-lg">
                  <p className="text-sm text-slate-700">
                    <strong>Research Title:</strong> "Diferansiyel Gizlilikten Esinlenen LSTM ile Finansal Haberleri ve 
                    Değerleri Kullanarak İsabet Oranı Yüksek Hisse Senedi Tahmini"
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Traditional Trading Tab */}
          <TabsContent value="traditional" className="space-y-6">
            <div className="space-y-6">
              {/* Traditional DP-LSTM Forecast */}
              <ForecastPanel />

              {/* Market Overview */}
              <div className="grid lg:grid-cols-2 gap-6">
                <RealMarketOverview />
                <Card>
                  <CardHeader>
                    <CardTitle>Traditional Analysis Tools</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Button variant="outline" className="w-full justify-start">
                        <BarChart3 className="h-4 w-4 mr-2" />
                        Technical Indicators
                      </Button>
                      <Button variant="outline" className="w-full justify-start">
                        <TrendingUp className="h-4 w-4 mr-2" />
                        Advanced Charts
                      </Button>
                      <Button variant="outline" className="w-full justify-start">
                        <AlertCircle className="h-4 w-4 mr-2" />
                        News Sentiment
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Advanced Indicators */}
              <AdvancedIndicators symbol={selectedSymbol} indicators={indicators} />
            </div>
          </TabsContent>

          {/* Enhanced Analysis Tab */}
          <TabsContent value="enhanced" className="space-y-6">
            <div className="space-y-6">
              {/* Symbol Selector for Enhanced Analysis */}
              <RealSymbolSelector 
                selectedSymbol={selectedSymbol}
                onSymbolChange={setSelectedSymbol}
              />

              {/* Enhanced Stock Analysis */}
              <EnhancedStockAnalysis selectedSymbol={selectedSymbol} />

              {/* Historical Chart with Technical Indicators */}
              <HistoricalChart selectedSymbol={selectedSymbol} />

              {/* AI Commentary Panel - 5 Day Hourly Forecasts */}
              <AICommentaryPanel selectedSymbol={selectedSymbol} />

              {/* AI Chat Panel - Interactive Assistant */}
              <AIChatPanel selectedSymbol={selectedSymbol} />
            </div>
          </TabsContent>

          {/* Integrated View Tab */}
          <TabsContent value="integrated" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Combined Academic + Traditional */}
              <Card className="bg-gradient-to-br from-blue-50 to-purple-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5" />
                    Integrated Prediction Engine
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center p-3 bg-white/60 rounded">
                      <span className="text-sm font-medium">DP-LSTM Weight</span>
                      <Badge>35%</Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white/60 rounded">
                      <span className="text-sm font-medium">sentimentARMA Weight</span>
                      <Badge>30%</Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white/60 rounded">
                      <span className="text-sm font-medium">KAP Impact Weight</span>
                      <Badge>20%</Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white/60 rounded">
                      <span className="text-sm font-medium">HuggingFace Model</span>
                      <Badge>15%</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Real-time Integration Status */}
              <Card className="bg-gradient-to-br from-green-50 to-blue-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Live Data Integration
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {[
                      { source: 'BIST Real-time Data', status: 'Connected', delay: '50ms' },
                      { source: 'KAP Announcements', status: 'Monitoring', delay: '1min' },
                      { source: 'News Sentiment Feed', status: 'Processing', delay: '30s' },
                      { source: 'Technical Indicators', status: 'Computing', delay: '10s' }
                    ].map((item, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-white/60 rounded">
                        <div>
                          <div className="text-sm font-medium">{item.source}</div>
                          <div className="text-xs text-slate-500">{item.status}</div>
                        </div>
                        <Badge variant="outline" className="text-xs">{item.delay}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Comprehensive Research Summary */}
            <Card className="bg-gradient-to-r from-slate-50 to-blue-50 border-slate-200">
              <CardHeader>
                <CardTitle className="text-center">🎯 Complete Academic Framework</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-4 gap-4 text-center">
                  <div className="p-4 bg-white/70 rounded-lg">
                    <Brain className="h-8 w-8 mx-auto text-blue-600 mb-2" />
                    <div className="font-semibold text-blue-800">DP-LSTM</div>
                    <div className="text-xs text-slate-600">Core ML Model</div>
                  </div>
                  <div className="p-4 bg-white/70 rounded-lg">
                    <TrendingUp className="h-8 w-8 mx-auto text-green-600 mb-2" />
                    <div className="font-semibold text-green-800">sentimentARMA</div>
                    <div className="text-xs text-slate-600">Sentiment + ARMA</div>
                  </div>
                  <div className="p-4 bg-white/70 rounded-lg">
                    <AlertCircle className="h-8 w-8 mx-auto text-purple-600 mb-2" />
                    <div className="font-semibold text-purple-800">KAP Feed</div>
                    <div className="text-xs text-slate-600">Real-time News</div>
                  </div>
                  <div className="p-4 bg-white/70 rounded-lg">
                    <BarChart3 className="h-8 w-8 mx-auto text-orange-600 mb-2" />
                    <div className="font-semibold text-orange-800">DP Privacy</div>
                    <div className="text-xs text-slate-600">Privacy Layer</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Footer Research Info */}
        <div className="mt-8 text-center text-sm text-slate-600 bg-white/50 p-4 rounded-lg border">
          <p className="font-medium mb-1">🎓 Academic Research Dashboard</p>
          <p>
            Integrating <strong>5 Data Sources</strong> • <strong>4 ML Models</strong> • <strong>Real-time Processing</strong>
            • <strong>Differential Privacy</strong> • <strong>Turkish NLP</strong>
          </p>
        </div>
      </main>
    </div>
  );
}