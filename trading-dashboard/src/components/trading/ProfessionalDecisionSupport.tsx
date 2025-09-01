'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  AlertCircle,
  Target,
  DollarSign,
  BarChart3,
  Shield,
  Zap
} from 'lucide-react';

interface DecisionData {
  final_decision: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  key_factors: string[];
  technical_score: number;
  fundamental_score: number;
  sentiment_score: number;
  price_target: number;
  stop_loss: number;
  position_size_recommendation: number;
}

interface ProfessionalDecisionSupportProps {
  symbol: string;
  onOrderPrepare?: (orderData: any) => void;
}

export default function ProfessionalDecisionSupport({ symbol, onOrderPrepare }: ProfessionalDecisionSupportProps) {
  const [decisionData, setDecisionData] = useState<DecisionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (symbol) {
      fetchDecisionData();
    }
  }, [symbol]);

  const fetchDecisionData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Mock advanced decision data - replace with real API call
      const mockData: DecisionData = {
        final_decision: Math.random() > 0.6 ? 'BUY' : Math.random() > 0.3 ? 'HOLD' : 'SELL',
        confidence: Math.random() * 30 + 70, // 70-100%
        risk_level: ['LOW', 'MEDIUM', 'HIGH'][Math.floor(Math.random() * 3)] as 'LOW' | 'MEDIUM' | 'HIGH',
        key_factors: [
          'RSI indicates oversold conditions',
          'Strong support level at current price',
          'Positive earnings momentum',
          'Sector rotation favoring this stock',
          'Technical breakout pattern forming'
        ].slice(0, Math.floor(Math.random() * 3) + 2),
        technical_score: Math.random() * 40 + 60,
        fundamental_score: Math.random() * 40 + 60,
        sentiment_score: Math.random() * 40 + 60,
        price_target: Math.random() * 20 + 100,
        stop_loss: Math.random() * 10 + 85,
        position_size_recommendation: Math.random() * 5 + 2
      };

      await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate API call
      setDecisionData(mockData);
      
    } catch (err) {
      setError('Failed to load decision analysis');
      console.error('Decision data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleOrderPrepare = () => {
    if (decisionData && onOrderPrepare) {
      const orderData = {
        symbol,
        action: decisionData.final_decision,
        price_target: decisionData.price_target,
        stop_loss: decisionData.stop_loss,
        position_size: decisionData.position_size_recommendation,
        confidence: decisionData.confidence,
        timestamp: new Date().toISOString()
      };
      onOrderPrepare(orderData);
    }
  };

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'BUY': return 'bg-green-500';
      case 'SELL': return 'bg-red-500';
      case 'HOLD': return 'bg-yellow-500';
      default: return 'bg-gray-500';
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

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-500" />
            Professional Decision Support
          </CardTitle>
          <CardDescription>AI-powered trading analysis for {symbol}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            <div className="h-4 bg-gray-200 rounded w-2/3"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !decisionData) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            Decision Support Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600">{error || 'No decision data available'}</p>
          <Button onClick={fetchDecisionData} className="mt-4">
            Retry Analysis
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Main Decision Card */}
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-blue-500" />
              Professional Decision Support
            </div>
            <Badge className={`${getDecisionColor(decisionData.final_decision)} text-white`}>
              {decisionData.final_decision}
            </Badge>
          </CardTitle>
          <CardDescription>AI Analysis for {symbol}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Confidence and Risk */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Confidence</span>
                <span className="text-sm text-gray-600">{decisionData.confidence.toFixed(1)}%</span>
              </div>
              <Progress value={decisionData.confidence} className="h-2" />
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Risk Level</span>
                <span className={`text-sm font-semibold ${getRiskColor(decisionData.risk_level)}`}>
                  {decisionData.risk_level}
                </span>
              </div>
              <div className="flex items-center">
                <Shield className={`h-4 w-4 mr-1 ${getRiskColor(decisionData.risk_level)}`} />
                <span className="text-xs text-gray-500">Risk Assessment</span>
              </div>
            </div>
          </div>

          <Separator />

          {/* Analysis Scores */}
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <BarChart3 className="h-6 w-6 mx-auto mb-2 text-blue-500" />
              <div className="text-lg font-semibold">{decisionData.technical_score.toFixed(0)}</div>
              <div className="text-xs text-gray-500">Technical</div>
            </div>
            <div className="text-center">
              <Activity className="h-6 w-6 mx-auto mb-2 text-green-500" />
              <div className="text-lg font-semibold">{decisionData.fundamental_score.toFixed(0)}</div>
              <div className="text-xs text-gray-500">Fundamental</div>
            </div>
            <div className="text-center">
              <TrendingUp className="h-6 w-6 mx-auto mb-2 text-purple-500" />
              <div className="text-lg font-semibold">{decisionData.sentiment_score.toFixed(0)}</div>
              <div className="text-xs text-gray-500">Sentiment</div>
            </div>
          </div>

          <Separator />

          {/* Price Targets */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <Target className="h-4 w-4 text-green-600" />
                <span className="text-sm font-medium text-green-800">Price Target</span>
              </div>
              <div className="text-lg font-semibold text-green-700">
                ₺{decisionData.price_target.toFixed(2)}
              </div>
            </div>
            <div className="bg-red-50 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <AlertCircle className="h-4 w-4 text-red-600" />
                <span className="text-sm font-medium text-red-800">Stop Loss</span>
              </div>
              <div className="text-lg font-semibold text-red-700">
                ₺{decisionData.stop_loss.toFixed(2)}
              </div>
            </div>
          </div>

          <Separator />

          {/* Key Factors */}
          <div>
            <h3 className="font-semibold mb-3">Key Analysis Factors</h3>
            <ul className="space-y-2">
              {decisionData.key_factors.map((factor, index) => (
                <li key={index} className="flex items-start gap-2 text-sm">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>

          <Separator />

          {/* Action Button */}
          <div className="flex gap-3">
            <Button 
              onClick={handleOrderPrepare}
              className="flex-1"
              variant={decisionData.final_decision === 'BUY' ? 'default' : 
                      decisionData.final_decision === 'SELL' ? 'destructive' : 'secondary'}
            >
              <DollarSign className="h-4 w-4 mr-2" />
              Prepare {decisionData.final_decision} Order
            </Button>
            <Button variant="outline" onClick={fetchDecisionData}>
              Refresh Analysis
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Position Sizing Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Position Sizing</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Recommended Position Size</span>
              <span className="text-lg font-semibold text-blue-700">
                {decisionData.position_size_recommendation.toFixed(1)}%
              </span>
            </div>
            <p className="text-xs text-blue-600">
              Based on risk assessment and portfolio optimization
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
