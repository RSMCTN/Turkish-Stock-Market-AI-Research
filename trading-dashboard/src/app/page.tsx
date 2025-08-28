'use client';

// Auth0 temporarily disabled for development
import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, Users, Shield, BarChart3, ArrowRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import TradingChart from '@/components/trading/TradingChart';
import SignalsPanel from '@/components/trading/SignalsPanel';
import RealMarketOverview from '@/components/trading/RealMarketOverview';
import PortfolioSummary from '@/components/trading/PortfolioSummary';
import OrdersPositions from '@/components/trading/OrdersPositions';
import ForecastPanel from '@/components/trading/ForecastPanel';
import AdvancedIndicators from '@/components/trading/AdvancedIndicators';
import AdvancedNewsSentiment from '@/components/trading/AdvancedNewsSentiment';
import AdvancedChart from '@/components/trading/AdvancedChart';
import BulkAnalysis from '@/components/trading/BulkAnalysis';
import RealSymbolSelector from '@/components/trading/RealSymbolSelector';

interface MarketStats {
  totalSymbols: number;
  activeSignals: number;
  totalUsers: number;
  systemHealth: string;
}

export default function Home() {
  // Mock user data for development (Auth0 disabled temporarily)
  const user = {
    name: "Demo User",
    email: "demo@bisttrading.com",
    picture: "https://via.placeholder.com/40",
    sub: "demo|12345"
  };
  const error = null;
  const isLoading = false;
  
  const [marketStats, setMarketStats] = useState<MarketStats>({
    totalSymbols: 0,
    activeSignals: 0,
    totalUsers: 0,
    systemHealth: 'Loading...'
  });

  const [activeTab, setActiveTab] = useState('forecast');
  const [selectedSymbol, setSelectedSymbol] = useState('GARAN');
  const [indicators, setIndicators] = useState([]);

  useEffect(() => {
    // Fetch market stats
    fetchMarketStats();
  }, []);

  useEffect(() => {
    // Fetch indicators when symbol changes
    fetchIndicators();
  }, [selectedSymbol]);

  const fetchIndicators = async () => {
    try {
      const baseUrl = 'https://bistai001-production.up.railway.app';
      const response = await fetch(`${baseUrl}/api/forecast/${selectedSymbol}?hours=24`);
      const data = await response.json();
      
      if (data.technicalIndicators) {
        setIndicators(data.technicalIndicators);
      }
    } catch (error) {
      console.error('Failed to fetch indicators:', error);
      setIndicators([]);
    }
  };

  const fetchMarketStats = async () => {
    try {
      // Use Railway production API
      const baseUrl = 'https://bistai001-production.up.railway.app';
      const response = await fetch(`${baseUrl}/api/bist/all-stocks?limit=5`);
      const data = await response.json();
      
      if (data.success) {
        setMarketStats({
          totalSymbols: data.total,
          activeSignals: data.stocks?.length || 0, // Real active symbols count
          totalUsers: 150, // Static for now  
          systemHealth: 'Operational'
        });
      }
    } catch (error) {
      console.error('Failed to fetch market stats:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-red-500">Authentication Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Something went wrong with authentication. Please try again.</p>
            <Button className="mt-4" onClick={() => window.location.reload()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!user) {
    // Landing page for non-authenticated users
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
        <div className="container mx-auto px-4 py-16">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4">
              🚀 BIST DP-LSTM Trading System v1.0
            </Badge>
            <h1 className="text-4xl lg:text-6xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent mb-6">
              Advanced Trading Dashboard
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Professional BIST stock analysis with Machine Learning signals, 
              real-time data, and advanced risk management.
            </p>
            
            <div className="flex gap-4 justify-center flex-col sm:flex-row">
              <Button size="lg" asChild>
                <a href="/api/auth/login">
                  <Shield className="mr-2 h-5 w-5" />
                  Secure Login
                </a>
              </Button>
              <Button variant="outline" size="lg">
                View Demo
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
            <Card className="text-center">
              <CardHeader>
                <BarChart3 className="h-8 w-8 mx-auto text-primary mb-2" />
                <CardTitle className="text-lg">600+ BIST Stocks</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Complete BIST market coverage with real-time data
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <Activity className="h-8 w-8 mx-auto text-green-500 mb-2" />
                <CardTitle className="text-lg">AI Signals</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  DP-LSTM powered trading signals with high accuracy
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <TrendingUp className="h-8 w-8 mx-auto text-blue-500 mb-2" />
                <CardTitle className="text-lg">Real-time Analytics</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Live market data with advanced technical indicators
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <Shield className="h-8 w-8 mx-auto text-purple-500 mb-2" />
                <CardTitle className="text-lg">Enterprise Security</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Auth0 powered authentication with role-based access
                </p>
              </CardContent>
            </Card>
          </div>

          {/* System Stats Preview */}
          <Card className="max-w-4xl mx-auto">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Live System Status
                <Badge variant={marketStats.systemHealth === 'Operational' ? 'default' : 'destructive'}>
                  {marketStats.systemHealth}
                </Badge>
              </CardTitle>
              <CardDescription>
                Real-time system metrics and market overview
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary mb-1">
                    {marketStats.totalSymbols.toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">Active Symbols</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-500 mb-1">
                    {marketStats.activeSignals.toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">Active Signals</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-500 mb-1">
                    {marketStats.totalUsers.toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">Active Users</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Footer */}
          <div className="text-center mt-16 text-sm text-muted-foreground">
            <p>
              🔗 <strong>Integrations:</strong> MatriksIQ API • Railway Redis • Auth0 Security • HuggingFace ML
            </p>
            <p className="mt-2">
              ⚖️ <strong>Disclaimer:</strong> This system is for educational and research purposes only. 
              Always consult qualified financial advisors.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Authenticated user dashboard
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="border-b bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900">BIST Trading Dashboard</h1>
              <Badge className="bg-blue-100 text-blue-700 border border-blue-300">Professional</Badge>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-700">
                Welcome, <span className="font-medium text-gray-900">{user.name}</span>
              </div>
              <Button variant="outline" size="sm" asChild className="border-gray-300 text-gray-700 hover:bg-gray-50">
                <a href="/api/auth/logout">Logout</a>
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Advanced Trading Dashboard */}
      <main className="container mx-auto px-4 py-6">
        
        {/* Tab Navigation */}
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
              {[
                { id: 'forecast', label: 'DP-LSTM Forecast', icon: TrendingUp },
                { id: 'advanced-chart', label: 'Advanced Charts', icon: BarChart3 },
                { id: 'indicators', label: 'Technical Indicators', icon: Activity },
                { id: 'news-sentiment', label: 'News Sentiment', icon: TrendingDown },
                { id: 'bulk-analysis', label: 'Bulk Analysis', icon: Users }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 bg-blue-50'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-2 px-4 border-b-2 font-medium text-sm flex items-center gap-2 rounded-t-lg transition-all`}
                >
                  <tab.icon className="h-4 w-4" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'forecast' && (
          <div className="space-y-6">
            {/* DP-LSTM FORECAST PANEL (MAIN FOCUS) */}
            <ForecastPanel />

            {/* Market Overview + Portfolio Summary */}
            <div className="grid lg:grid-cols-2 gap-6">
              <RealMarketOverview />
              <PortfolioSummary />
            </div>

            {/* Orders/Positions + Trading Signals */}
            <div className="grid lg:grid-cols-2 gap-6">
              <OrdersPositions />
              <SignalsPanel />
            </div>
          </div>
        )}

        {activeTab === 'advanced-chart' && (
          <div className="space-y-6">
            {/* Real Symbol Selector */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={true}
              limit={600}
            />
            <AdvancedChart symbol={selectedSymbol} />
          </div>
        )}

        {activeTab === 'indicators' && (
          <div className="space-y-6">
            {/* Real Symbol Selector */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={true}
              limit={600}
            />
            <AdvancedIndicators symbol={selectedSymbol} indicators={indicators} />
          </div>
        )}

        {activeTab === 'news-sentiment' && (
          <div className="space-y-6">
            {/* Real Symbol Selector */}
            <RealSymbolSelector
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
              showSearch={true}
              showFilters={true}
              limit={600}
            />
            <AdvancedNewsSentiment symbol={selectedSymbol} />
          </div>
        )}

        {activeTab === 'bulk-analysis' && (
          <div className="space-y-6">
            <BulkAnalysis />
          </div>
        )}

        {/* Quick Stats Bar (Always Visible) */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-white/50 border-gray-200">
            <CardContent className="p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Total Symbols</p>
                  <p className="text-lg font-bold text-gray-900">{marketStats.totalSymbols.toLocaleString()}</p>
                </div>
                <BarChart3 className="h-5 w-5 text-blue-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/50 border-gray-200">
            <CardContent className="p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Active Signals</p>
                  <p className="text-lg font-bold text-green-600">{marketStats.activeSignals}</p>
                </div>
                <Activity className="h-5 w-5 text-green-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/50 border-gray-200">
            <CardContent className="p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">System Status</p>
                  <p className="text-lg font-bold text-blue-600">{marketStats.systemHealth}</p>
                </div>
                <Shield className="h-5 w-5 text-blue-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/50 border-gray-200">
            <CardContent className="p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">BIST 100</p>
                  <p className="text-lg font-bold text-green-600">+2.4%</p>
                </div>
                <TrendingUp className="h-5 w-5 text-green-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
