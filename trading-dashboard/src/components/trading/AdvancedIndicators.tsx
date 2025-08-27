'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { TrendingUp, TrendingDown, Activity, Target, AlertCircle } from 'lucide-react';

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'BUY' | 'SELL' | 'HOLD' | 'NEUTRAL';
  weight: number;
  description: string;
  status: string;
}

interface AdvancedIndicatorsProps {
  symbol: string;
  indicators?: TechnicalIndicator[];
  isLoading?: boolean;
}

const AdvancedIndicators = ({ symbol, indicators = [], isLoading = false }: AdvancedIndicatorsProps) => {
  const [selectedIndicator, setSelectedIndicator] = useState<TechnicalIndicator | null>(null);

  const getIndicatorIcon = (name: string) => {
    switch (name) {
      case 'RSI': return <Activity className="h-4 w-4" />;
      case 'MACD': return <TrendingUp className="h-4 w-4" />;
      case 'BOLLINGER_BANDS': return <Target className="h-4 w-4" />;
      case 'STOCHASTIC': return <TrendingDown className="h-4 w-4" />;
      default: return <AlertCircle className="h-4 w-4" />;
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'bg-green-100 text-green-800 border-green-300';
      case 'SELL': return 'bg-red-100 text-red-800 border-red-300';
      case 'HOLD': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'NEUTRAL': return 'bg-gray-100 text-gray-800 border-gray-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getStatusColor = (status: string) => {
    if (status.includes('BULLISH') || status.includes('ABOVE') || status === 'OVERBOUGHT') {
      return 'text-green-600';
    } else if (status.includes('BEARISH') || status.includes('BELOW') || status === 'OVERSOLD') {
      return 'text-red-600';
    }
    return 'text-gray-600';
  };

  const formatIndicatorValue = (name: string, value: number) => {
    switch (name) {
      case 'RSI':
      case 'STOCHASTIC':
        return `${value.toFixed(1)}`;
      case 'MACD':
        return `${value.toFixed(3)}`;
      case 'BOLLINGER_BANDS':
        return `${(value * 100).toFixed(1)}%`;
      case 'VOLUME_WEIGHTED':
        return `${value.toFixed(2)}x`;
      default:
        return value.toFixed(2);
    }
  };

  const bullishCount = indicators.filter(ind => ind.signal === 'BUY').length;
  const bearishCount = indicators.filter(ind => ind.signal === 'SELL').length;
  const overallSentiment = bullishCount > bearishCount ? 'BULLISH' : bearishCount > bullishCount ? 'BEARISH' : 'NEUTRAL';

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Technical Indicators - {symbol}
          </CardTitle>
          <CardDescription>Loading advanced technical analysis...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1,2,3,4,5].map(i => (
              <div key={i} className="animate-pulse">
                <div className="flex justify-between items-center mb-2">
                  <div className="h-4 bg-gray-200 rounded w-20"></div>
                  <div className="h-4 bg-gray-200 rounded w-16"></div>
                </div>
                <div className="h-2 bg-gray-200 rounded w-full"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Technical Indicators - {symbol}
        </CardTitle>
        <CardDescription>Advanced technical analysis with indicator weights</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        
        {/* Overall Sentiment */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div>
            <p className="text-sm text-gray-600">Overall Sentiment</p>
            <div className="flex items-center gap-2 mt-1">
              <Badge className={`${
                overallSentiment === 'BULLISH' ? 'bg-green-100 text-green-800' :
                overallSentiment === 'BEARISH' ? 'bg-red-100 text-red-800' : 
                'bg-gray-100 text-gray-800'
              }`}>
                {overallSentiment}
              </Badge>
              <span className="text-xs text-gray-500">
                {bullishCount} Bullish • {bearishCount} Bearish
              </span>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Weighted Score</p>
            <p className="text-xl font-bold text-gray-900">
              {indicators.reduce((acc, ind) => {
                const score = ind.signal === 'BUY' ? ind.weight : ind.signal === 'SELL' ? -ind.weight : 0;
                return acc + score;
              }, 0).toFixed(2)}
            </p>
          </div>
        </div>

        {/* Individual Indicators */}
        <div className="space-y-4">
          {indicators.map((indicator, index) => (
            <div 
              key={index}
              className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                selectedIndicator?.name === indicator.name ? 'ring-2 ring-blue-200 border-blue-300' : ''
              }`}
              onClick={() => setSelectedIndicator(selectedIndicator?.name === indicator.name ? null : indicator)}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getIndicatorIcon(indicator.name)}
                  <div>
                    <p className="font-medium text-gray-900">{indicator.name.replace('_', ' ')}</p>
                    <p className="text-xs text-gray-500">{indicator.description}</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <Badge className={getSignalColor(indicator.signal)}>
                    {indicator.signal}
                  </Badge>
                  <div className="text-right">
                    <p className="font-semibold">{formatIndicatorValue(indicator.name, indicator.value)}</p>
                    <p className="text-xs text-gray-500">Weight: {(indicator.weight * 100).toFixed(0)}%</p>
                  </div>
                </div>
              </div>

              {/* Weight Progress Bar */}
              <div className="mb-2">
                <Progress 
                  value={indicator.weight * 100} 
                  className="h-2"
                />
              </div>

              {/* Status */}
              <div className="flex justify-between items-center">
                <p className={`text-sm font-medium ${getStatusColor(indicator.status)}`}>
                  {indicator.status.replace('_', ' ')}
                </p>
                <p className="text-xs text-gray-500">
                  Impact: {indicator.signal === 'BUY' ? '+' : indicator.signal === 'SELL' ? '-' : '±'}
                  {(indicator.weight * (indicator.signal === 'HOLD' ? 0 : 100)).toFixed(0)}%
                </p>
              </div>

              {/* Expanded Details */}
              {selectedIndicator?.name === indicator.name && (
                <div className="mt-3 pt-3 border-t bg-gray-50 rounded p-3">
                  <h4 className="font-medium text-gray-900 mb-2">Detailed Analysis</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Current Value:</p>
                      <p className="font-semibold">{formatIndicatorValue(indicator.name, indicator.value)}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Signal Strength:</p>
                      <p className="font-semibold">{indicator.signal === 'BUY' || indicator.signal === 'SELL' ? 'Strong' : 'Weak'}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Market Status:</p>
                      <p className="font-semibold">{indicator.status.replace('_', ' ')}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Reliability:</p>
                      <p className="font-semibold">{indicator.weight > 0.2 ? 'High' : 'Medium'}</p>
                    </div>
                  </div>
                  
                  {/* Trading Recommendation */}
                  <div className="mt-3 p-2 bg-white rounded border">
                    <p className="text-xs font-medium text-gray-700 mb-1">Trading Recommendation:</p>
                    <p className="text-sm text-gray-600">
                      {indicator.signal === 'BUY' && `Strong ${indicator.name} signal suggests upward momentum. Consider entry positions.`}
                      {indicator.signal === 'SELL' && `${indicator.name} indicates potential downward pressure. Consider exit strategy.`}
                      {indicator.signal === 'HOLD' && `${indicator.name} shows neutral signals. Monitor for changes.`}
                      {indicator.signal === 'NEUTRAL' && `${indicator.name} is in neutral territory. Wait for clearer signals.`}
                    </p>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {indicators.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <AlertCircle className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>No technical indicators available</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AdvancedIndicators;
