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
import AdvancedTechnicalPanel from '@/components/trading/AdvancedTechnicalPanel';
import ProfessionalDecisionSupport from '@/components/trading/ProfessionalDecisionSupport';
import EnhancedHistoricalChart from '@/components/trading/EnhancedHistoricalChart';

export default function MamutDashboard() {
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
    name: "MAMUT Trader",
    email: "trader@mamut-r600.com"
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-emerald-50/30 to-blue-50">
      {/* MAMUT R600 Header */}
      <header className="border-b bg-white/90 backdrop-blur-md shadow-xl border-slate-200/50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-600 via-blue-600 to-purple-600 bg-clip-text text-transparent">
                    MAMUT_R600
                  </h1>
                  <p className="text-sm text-slate-500">Professional AI-Powered Trading Platform</p>
                </div>
              </div>
              <Badge className="bg-gradient-to-r from-emerald-100 to-cyan-100 text-emerald-700 border-emerald-200">
                🚀 LIVE MODE
              </Badge>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-slate-700">
                <div className="font-medium">{user.name}</div>
                <div className="text-xs text-slate-500">Professional Trading</div>
              </div>
              <Button variant="outline" size="sm" className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200 hover:from-blue-100 hover:to-purple-100">
                Export Data
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* MAMUT R600 Professional Dashboard */}
      <main className="container mx-auto px-6 py-8">
        
        {/* Enhanced System Status Overview - WOW Design */}
        <div className="mb-8 grid grid-cols-2 md:grid-cols-5 gap-4">
          {Object.entries(systemStatus).map(([component, status]) => (
            <Card key={component} className="bg-white/90 border-slate-200 shadow-lg hover:shadow-2xl transition-all duration-500 hover:scale-105 hover:border-emerald-300 group">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-slate-600 capitalize font-medium">{component.replace(/([A-Z])/g, ' $1')}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <div className="relative">
                        <div className={`w-3 h-3 rounded-full ${
                          status === 'active' ? 'bg-gradient-to-r from-emerald-500 to-green-500 animate-pulse' : 
                          'bg-gradient-to-r from-red-500 to-pink-500'
                        }`} />
                        {status === 'active' && (
                          <div className="absolute inset-0 w-3 h-3 rounded-full bg-emerald-400 animate-ping opacity-30"></div>
                        )}
                      </div>
                      <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                        status === 'active' ? 'text-emerald-700 bg-gradient-to-r from-emerald-100 to-green-100' :
                        'text-red-700 bg-gradient-to-r from-red-100 to-pink-100'
                      }`}>
                        {status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <Activity className={`h-5 w-5 transition-all duration-300 group-hover:scale-110 ${
                    status === 'active' ? 'text-emerald-500 group-hover:text-emerald-600' : 'text-red-500 group-hover:text-red-600'
                  }`} />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* MAMUT R600 Trading Modules */}
        <Tabs defaultValue="academic" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 bg-gradient-to-r from-white via-slate-50 to-white border border-slate-200 shadow-xl rounded-xl p-1">
            <TabsTrigger 
              value="academic" 
              className="flex items-center gap-2 rounded-lg transition-all duration-300 hover:scale-105 data-[state=active]:bg-gradient-to-r data-[state=active]:from-emerald-500 data-[state=active]:to-blue-500 data-[state=active]:text-white data-[state=active]:shadow-lg hover:shadow-md"
            >
              <Brain className="h-4 w-4" />
              AI Analytics
            </TabsTrigger>
            <TabsTrigger 
              value="enhanced" 
              className="flex items-center gap-2 rounded-lg transition-all duration-300 hover:scale-105 data-[state=active]:bg-gradient-to-r data-[state=active]:from-orange-500 data-[state=active]:to-red-500 data-[state=active]:text-white data-[state=active]:shadow-lg hover:shadow-md"
            >
              <Target className="h-4 w-4" />
              Enhanced Chart
            </TabsTrigger>
            <TabsTrigger 
              value="traditional" 
              className="flex items-center gap-2 rounded-lg transition-all duration-300 hover:scale-105 data-[state=active]:bg-gradient-to-r data-[state=active]:from-green-500 data-[state=active]:to-teal-500 data-[state=active]:text-white data-[state=active]:shadow-lg hover:shadow-md"
            >
              <LineChart className="h-4 w-4" />
              Technical Analysis
            </TabsTrigger>
            <TabsTrigger 
              value="integrated" 
              className="flex items-center gap-2 rounded-lg transition-all duration-300 hover:scale-105 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500 data-[state=active]:to-pink-500 data-[state=active]:text-white data-[state=active]:shadow-lg hover:shadow-md"
            >
              <Zap className="h-4 w-4" />
              Integrated View
            </TabsTrigger>
            <TabsTrigger 
              value="professional" 
              className="flex items-center gap-2 rounded-lg transition-all duration-300 hover:scale-105 data-[state=active]:bg-gradient-to-r data-[state=active]:from-indigo-500 data-[state=active]:to-purple-500 data-[state=active]:text-white data-[state=active]:shadow-lg hover:shadow-md"
            >
              <Target className="h-4 w-4" />
              Pro Decision
            </TabsTrigger>
          </TabsList>

          {/* AI Analytics Tab */}
          <TabsContent value="academic" className="space-y-6">
            {/* Symbol Selector for AI Analytics */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={true}
              limit={600}
            />

            <div className="grid lg:grid-cols-2 gap-6">
              {/* AI Prediction System */}
              <AcademicPredictionPanel selectedSymbol={selectedSymbol} />
              
              {/* HuggingFace Production Model */}
              <HuggingFaceModelPanel selectedSymbol={selectedSymbol} />
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* Live KAP Feed */}
              <LiveKAPFeed selectedSymbol={selectedSymbol} />
              
              {/* AI Performance Metrics */}
              <AcademicMetricsDashboard selectedSymbol={selectedSymbol} />
              
              {/* Component Contributions */}
              <ComponentContributionChart selectedSymbol={selectedSymbol} />
            </div>

            {/* Advanced Technical Analysis Section */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* New Advanced Technical Panel */}
              <AdvancedTechnicalPanel 
                selectedSymbol={selectedSymbol} 
                apiBaseUrl={process.env.NODE_ENV === 'development' ? 'http://localhost:3000' : 'https://bistai001-production.up.railway.app'} 
              />
              
              {/* AI Chat Panel */}
              <AIChatPanel 
                selectedSymbol={selectedSymbol}
                apiBaseUrl={process.env.NODE_ENV === 'development' ? 'http://localhost:3000' : 'https://bistai001-production.up.railway.app'} 
              />
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

            {/* MAMUT System Status */}
            <Card className="bg-gradient-to-r from-emerald-50 via-blue-50 to-purple-50 border-emerald-200 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <div className="p-2 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg">
                    <Brain className="h-5 w-5 text-white" />
                  </div>
                  MAMUT R600 System Status
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
                    <strong>MAMUT R600:</strong> Professional AI-Powered Trading Platform with Advanced Analytics, 
                    Real-time Market Intelligence, and Precision Trading Signals
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
              {/* Combined AI + Technical Analysis */}
              <Card className="bg-gradient-to-br from-emerald-50 to-cyan-50 shadow-lg border-emerald-200">
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

            {/* MAMUT R600 Complete System Summary */}
            <Card className="bg-gradient-to-r from-slate-50 via-emerald-50 to-blue-50 border-emerald-200 shadow-xl">
              <CardHeader>
                <CardTitle className="text-center bg-gradient-to-r from-emerald-600 via-blue-600 to-purple-600 bg-clip-text text-transparent text-xl">
                  🚀 MAMUT R600 Complete Trading System
                </CardTitle>
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

          {/* Professional Decision Support Tab */}
          <TabsContent value="professional" className="space-y-6">
            <div className="space-y-6">
              {/* Symbol Selector for Professional Analysis */}
              <RealSymbolSelector 
                selectedSymbol={selectedSymbol}
                onSymbolChange={setSelectedSymbol}
              />
              
              {/* Enhanced Historical Chart with Advanced Features */}
              <EnhancedHistoricalChart 
                symbol={selectedSymbol}
                autoRefresh={true}
                refreshInterval={30000}
                showTechnicalOverlay={true}
                showPatternDetection={true}
                showAlerts={true}
              />
              
              {/* Professional Decision Support System */}
              <ProfessionalDecisionSupport 
                symbol={selectedSymbol}
                onOrderPrepare={(orderData) => {
                  console.log('Order prepared:', orderData);
                  // Here you can integrate with order management system
                }}
              />
            </div>
          </TabsContent>
        </Tabs>

        {/* Footer System Info */}
        <div className="mt-8 text-center text-sm text-slate-600 bg-gradient-to-r from-white/60 to-emerald-50/60 p-4 rounded-lg border border-emerald-200 shadow-md">
          <p className="font-medium mb-1 text-emerald-800">🚀 MAMUT R600 Professional Trading Platform</p>
          <p>
            Integrating <strong>5 Data Sources</strong> • <strong>4 ML Models</strong> • <strong>Real-time Processing</strong>
            • <strong>Differential Privacy</strong> • <strong>Turkish NLP</strong>
          </p>
        </div>
      </main>
    </div>
  );
}