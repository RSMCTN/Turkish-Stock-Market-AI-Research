'use client';

import { useState, useEffect } from 'react';
import { Brain, LineChart, Zap, Activity, TrendingUp, AlertCircle, BarChart3, Target, RefreshCw } from 'lucide-react';
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

  const [selectedSymbol, setSelectedSymbol] = useState('AKBNK');
  const [indicators, setIndicators] = useState([]);
  const [selectedTool, setSelectedTool] = useState<'indicators' | 'charts' | 'sentiment' | null>(null);

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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900/40 to-indigo-900/60">
      {/* MAMUT R600 Header */}
      <header className="border-b bg-slate-800/95 backdrop-blur-md shadow-xl border-slate-700/50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                {/* MAMUT R600 Premium Logo */}
                <img 
                  src="/mamut-logo.png" 
                  alt="MAMUT R600 - Professional Trading Platform"
                  className="h-12 md:h-14 w-auto object-contain hover:scale-105 transition-all duration-300 drop-shadow-lg cursor-pointer"
                  loading="lazy"
                />
                <div className="ml-2 hidden sm:block">
                  <h1 className="text-xl md:text-2xl font-bold bg-gradient-to-r from-amber-400 via-yellow-400 to-orange-400 bg-clip-text text-transparent">
                    MAMUT R600
                  </h1>
                  <p className="text-xs md:text-sm text-slate-300 font-medium">Professional AI-Powered Trading Platform</p>
                </div>
              </div>
              <Badge className="bg-gradient-to-r from-emerald-600 to-cyan-600 text-white border-emerald-500 shadow-lg animate-pulse">
                🚀 LIVE MODE
              </Badge>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-slate-300">
                <div className="font-medium text-white">{user.name}</div>
                <div className="text-xs text-slate-400">Professional Trading</div>
              </div>
              <Button variant="outline" size="sm" className="bg-gradient-to-r from-blue-600 to-purple-600 text-white border-blue-500 hover:from-blue-700 hover:to-purple-700 hover:shadow-lg transition-all">
                Export Data
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* MAMUT R600 Professional Dashboard */}
      <main className="container mx-auto px-6 py-8">
        
        {/* KOMPAKT System Status - Daha Az Yorucu */}
        <div className="mb-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-emerald-400" />
                  <h3 className="text-sm font-semibold text-white">System Status</h3>
                </div>
                <div className="flex items-center gap-2">
                  {Object.entries(systemStatus).map(([component, status]) => (
                    <div key={component} className="flex items-center gap-1">
                      <div className={`w-2 h-2 rounded-full ${
                        status === 'active' ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'
                      }`} />
                      <span className="text-xs text-slate-300 capitalize">
                        {component.replace(/([A-Z])/g, ' $1').slice(0, 8)}...
                      </span>
                    </div>
                  ))}
                  <Badge className="bg-emerald-600 text-white text-xs ml-2">ALL ACTIVE</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
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
            {/* KOMPAKT RealSymbolSelector - 600 Hisse + Search */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={false}  // Kompakt için filter'ları kapat
              limit={600}
              compact={true}  // Kompakt görünüm
            />

            {/* ANA AI PANELS - Daha Düzenli Layout */}
            <div className="space-y-6">
              {/* Üst: En Önemli AI Paneller */}
              <div className="grid lg:grid-cols-2 gap-6">
                <AcademicPredictionPanel selectedSymbol={selectedSymbol} />
                <HuggingFaceModelPanel selectedSymbol={selectedSymbol} />
              </div>

              {/* Orta: KAP & Metrics & Technical */}
              <div className="grid lg:grid-cols-3 gap-4">
                <LiveKAPFeed selectedSymbol={selectedSymbol} />
                <AcademicMetricsDashboard selectedSymbol={selectedSymbol} />
                <ComponentContributionChart selectedSymbol={selectedSymbol} />
              </div>

              {/* Alt: Company & Decision Support */}
              <div className="grid lg:grid-cols-3 gap-4">
                <CompanyInfoCard selectedSymbol={selectedSymbol} />
                <FundamentalAnalysis selectedSymbol={selectedSymbol} />
                <AdvancedTechnicalPanel 
                  selectedSymbol={selectedSymbol} 
                  apiBaseUrl={process.env.NODE_ENV === 'development' ? 'http://localhost:3000' : 'https://bistai001-production.up.railway.app'} 
                />
              </div>

              {/* Son: AI Decision Support (Full Width - Çok Önemli) */}
              <AIDecisionSupport selectedSymbol={selectedSymbol} />
            </div>
          </TabsContent>

          {/* Traditional Trading Tab */}
          <TabsContent value="traditional" className="space-y-6">
            {/* KOMPAKT RealSymbolSelector - 600 Hisse + Search */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={false}  // Kompakt için filter'ları kapat
              limit={600}
              compact={true}  // Kompakt görünüm
            />
            <div className="space-y-6">
              {/* Traditional DP-LSTM Forecast */}
              <ForecastPanel selectedSymbol={selectedSymbol} />

              {/* Market Overview */}
              <div className="grid lg:grid-cols-2 gap-6">
                <RealMarketOverview />
                <Card>
                  <CardHeader>
                    <CardTitle>Traditional Analysis Tools</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Button 
                        variant={selectedTool === 'indicators' ? 'default' : 'outline'} 
                        className="w-full justify-start"
                        onClick={() => setSelectedTool(selectedTool === 'indicators' ? null : 'indicators')}
                      >
                        <BarChart3 className="h-4 w-4 mr-2" />
                        Technical Indicators
                      </Button>
                      <Button 
                        variant={selectedTool === 'charts' ? 'default' : 'outline'} 
                        className="w-full justify-start"
                        onClick={() => setSelectedTool(selectedTool === 'charts' ? null : 'charts')}
                      >
                        <TrendingUp className="h-4 w-4 mr-2" />
                        Advanced Charts
                      </Button>
                      <Button 
                        variant={selectedTool === 'sentiment' ? 'default' : 'outline'} 
                        className="w-full justify-start"
                        onClick={() => setSelectedTool(selectedTool === 'sentiment' ? null : 'sentiment')}
                      >
                        <AlertCircle className="h-4 w-4 mr-2" />
                        News Sentiment
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Traditional Tools Content */}
              {selectedTool === 'indicators' && (
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <BarChart3 className="h-5 w-5 mr-2" />
                      Technical Indicators - {selectedSymbol}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <AdvancedIndicators symbol={selectedSymbol} indicators={indicators} />
                  </CardContent>
                </Card>
              )}

              {selectedTool === 'charts' && (
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <TrendingUp className="h-5 w-5 mr-2" />
                      Advanced Charts - {selectedSymbol}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <HistoricalChart 
                      selectedSymbol={selectedSymbol}
                    />
                  </CardContent>
                </Card>
              )}

              {selectedTool === 'sentiment' && (
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <AlertCircle className="h-5 w-5 mr-2" />
                      News Sentiment Analysis - {selectedSymbol}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Live KAP Feed */}
                      <div className="space-y-4">
                        <h3 className="font-semibold text-lg">📰 Live KAP Feed</h3>
                        <LiveKAPFeed selectedSymbol={selectedSymbol} />
                      </div>
                      
                      {/* Academic Metrics */}
                      <div className="space-y-4">
                        <h3 className="font-semibold text-lg">📊 Sentiment Metrics</h3>
                        <AcademicMetricsDashboard selectedSymbol={selectedSymbol} />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Advanced Indicators (shown when no tool is selected) */}
              {!selectedTool && (
                <AdvancedIndicators symbol={selectedSymbol} indicators={indicators} />
              )}
            </div>
          </TabsContent>

          {/* Enhanced Analysis Tab */}
          <TabsContent value="enhanced" className="space-y-6">
            {/* KOMPAKT RealSymbolSelector - 600 Hisse + Search */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={false}  // Kompakt için filter'ları kapat
              limit={600}
              compact={true}  // Kompakt görünüm
            />
            <div className="space-y-6">

              {/* Enhanced Stock Analysis */}
              <EnhancedStockAnalysis selectedSymbol={selectedSymbol} />

              {/* Historical Chart with Technical Indicators (Chart Kaldırıldı - Sadece Forecast Panels) */}
              <HistoricalChart selectedSymbol={selectedSymbol} />

              {/* AI Commentary Panel - 5 Day Hourly Forecasts */}
              <AICommentaryPanel selectedSymbol={selectedSymbol} />
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