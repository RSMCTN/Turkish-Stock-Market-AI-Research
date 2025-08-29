'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Target, TrendingUp, TrendingDown, Clock, Star, AlertTriangle, Activity, Zap } from 'lucide-react';

interface TradingSignal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  confidence: number;
  price: number;
  targetPrice: number;
  stopLoss: number;
  timeframe: string;
  reason: string;
  timestamp: Date;
  status: 'ACTIVE' | 'EXECUTED' | 'EXPIRED';
}

export default function SignalsPanel() {
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Generate mock signals
  const generateSignals = (): TradingSignal[] => {
    const symbols = ['AKBNK', 'GARAN', 'ISCTR', 'THYAO', 'ASELS', 'SISE', 'EREGL'];
    const reasons = [
      'Technical breakout detected',
      'RSI oversold condition',
      'Moving average crossover',
      'Volume surge detected',
      'Support/resistance level',
      'Momentum divergence',
      'News sentiment positive',
      'Institution accumulation'
    ];

    return Array.from({ length: 8 }, (_, i) => {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const action = (Math.random() > 0.6 ? 'BUY' : 'SELL') as 'BUY' | 'SELL';
      const price = 20 + Math.random() * 60;
      const confidence = 0.65 + Math.random() * 0.3;
      
      return {
        id: `signal-${i}`,
        symbol,
        action,
        confidence: Number(confidence.toFixed(2)),
        price: Number(price.toFixed(2)),
        targetPrice: Number((price * (action === 'BUY' ? 1.05 + Math.random() * 0.05 : 0.95 - Math.random() * 0.05)).toFixed(2)),
        stopLoss: Number((price * (action === 'BUY' ? 0.97 - Math.random() * 0.02 : 1.03 + Math.random() * 0.02)).toFixed(2)),
        timeframe: ['1H', '4H', '1D'][Math.floor(Math.random() * 3)],
        reason: reasons[Math.floor(Math.random() * reasons.length)],
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
        status: ['ACTIVE', 'ACTIVE', 'ACTIVE', 'EXECUTED'][Math.floor(Math.random() * 4)] as 'ACTIVE' | 'EXECUTED'
      };
    }).sort((a, b) => b.confidence - a.confidence);
  };

  useEffect(() => {
    const loadSignals = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setSignals(generateSignals());
      setIsLoading(false);
    };
    
    loadSignals();
    
    // Auto refresh every 30 seconds
    const interval = setInterval(loadSignals, 30000);
    return () => clearInterval(interval);
  }, []);

  const getSignalIcon = (action: 'BUY' | 'SELL') => {
    return action === 'BUY' ? 
      <TrendingUp className="h-4 w-4 text-green-600" /> : 
      <TrendingDown className="h-4 w-4 text-red-600" />;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.85) return 'text-green-500';
    if (confidence >= 0.75) return 'text-blue-500';
    if (confidence >= 0.65) return 'text-yellow-500';
    return 'text-gray-500';
  };

  const getTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return timestamp.toLocaleDateString('tr-TR');
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Loading Signals...
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="animate-pulse">
                  <div className="flex items-center justify-between mb-2">
                    <div className="h-4 bg-muted rounded w-20"></div>
                    <div className="h-4 bg-muted rounded w-12"></div>
                  </div>
                  <div className="h-3 bg-muted rounded w-full mb-1"></div>
                  <div className="h-3 bg-muted rounded w-3/4"></div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const activeSignals = signals.filter(s => s.status === 'ACTIVE');
  const executedSignals = signals.filter(s => s.status === 'EXECUTED');

  return (
    <div className="space-y-6">
      {/* Active Signals */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Active Signals
            </div>
            <Badge variant="secondary">{activeSignals.length}</Badge>
          </CardTitle>
          <CardDescription>
            AI-generated trading signals with confidence scores
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {activeSignals.slice(0, 5).map((signal) => (
              <div key={signal.id} className="border rounded-lg p-3 hover:bg-muted/50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getSignalIcon(signal.action)}
                    <span className="font-semibold">{signal.symbol}</span>
                    <Badge 
                      className={`text-xs font-semibold ${
                        signal.action === 'BUY' 
                          ? 'bg-green-100 text-green-700 border border-green-300' 
                          : 'bg-red-100 text-red-700 border border-red-300'
                      }`}
                    >
                      {signal.action}
                    </Badge>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-semibold ${getConfidenceColor(signal.confidence)}`}>
                      {(signal.confidence * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {signal.timeframe}
                    </div>
                  </div>
                </div>
                
                <div className="text-xs text-muted-foreground mb-2">
                  {signal.reason}
                </div>
                
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Price:</span>
                    <div className="font-medium">₺{signal.price}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Target:</span>
                    <div className="font-semibold text-green-700">₺{signal.targetPrice}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Stop:</span>
                    <div className="font-semibold text-red-700">₺{signal.stopLoss}</div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between mt-2 pt-2 border-t">
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {getTimeAgo(signal.timestamp)}
                  </div>
                  <Button size="sm" variant="outline" className="h-6 text-xs">
                    Details
                  </Button>
                </div>
              </div>
            ))}
            
            {activeSignals.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Target className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No active signals at the moment</p>
                <p className="text-xs">New signals will appear here automatically</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Signal Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Today's Signals</span>
              <span className="font-semibold">{signals.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Success Rate</span>
              <span className="font-semibold text-green-500">73.2%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Avg Confidence</span>
              <span className="font-semibold">{(signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length * 100).toFixed(0)}%</span>
            </div>
            <Separator />
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-green-600">{activeSignals.filter(s => s.action === 'BUY').length}</div>
                <div className="text-xs text-muted-foreground">BUY Signals</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-red-600">{activeSignals.filter(s => s.action === 'SELL').length}</div>
                <div className="text-xs text-muted-foreground">SELL Signals</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Quick Actions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button className="w-full" variant="outline" size="sm">
            <Target className="mr-2 h-4 w-4" />
            View All Signals
          </Button>
          <Button className="w-full" variant="outline" size="sm">
            <Activity className="mr-2 h-4 w-4" />
            Signal History
          </Button>
          <Button className="w-full" variant="outline" size="sm">
            <Star className="mr-2 h-4 w-4" />
            Watchlist
          </Button>
          <Button className="w-full" variant="outline" size="sm">
            <AlertTriangle className="mr-2 h-4 w-4" />
            Alerts & Notifications
          </Button>
        </CardContent>
      </Card>

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">System Status</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div className="flex justify-between items-center">
            <span>DP-LSTM Models:</span>
            <Badge className="text-xs bg-green-50 text-green-700 border border-green-300">
              Online
            </Badge>
          </div>
          <div className="flex justify-between items-center">
            <span>Real-time Data:</span>
            <Badge className="text-xs bg-green-50 text-green-700 border border-green-300">
              Active
            </Badge>
          </div>
          <div className="flex justify-between items-center">
            <span>Signal Generation:</span>
            <Badge className="text-xs bg-blue-50 text-blue-700 border border-blue-300">
              Running
            </Badge>
          </div>
          <div className="flex justify-between items-center">
            <span>Redis Cache:</span>
            <Badge className="text-xs bg-green-50 text-green-700 border border-green-300">
              Connected
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
