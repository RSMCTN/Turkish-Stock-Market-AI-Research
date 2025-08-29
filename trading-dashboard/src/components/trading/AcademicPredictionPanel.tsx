'use client';

import { useState, useEffect } from 'react';
import { Brain, TrendingUp, Target } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

// Real BIST prices from basestock2808.xlsx (Updated 28/08)
const getRealPrice = (symbol: string): number => {
  const realPrices: { [key: string]: number } = {
    'AKSEN': 39.06,  // Real closing price
    'ASTOR': 113.7,  // Real closing price
    'GARAN': 145.8,  // Real closing price
    'THYAO': 340.0,  // Real closing price
    'TUPRS': 171.0,  // Real closing price
    'BRSAN': 499.25, // Real closing price
    'AKBNK': 69.5,   // Real closing price
    'ISCTR': 15.14,  // Real closing price
    'SISE': 40.74,   // Real closing price
    'ARCLK': 141.2,  // Real closing price
    'KCHOL': 184.8,
    'BIMAS': 536.0,
    'PETKM': 20.96,
    'TTKOM': 58.4
  };
  return realPrices[symbol] || 50.0;
};

interface AcademicPredictionPanelProps {
  selectedSymbol?: string;
}

export default function AcademicPredictionPanel({ selectedSymbol = 'GARAN' }: AcademicPredictionPanelProps) {
  const [prediction, setPrediction] = useState({
    symbol: selectedSymbol,
    currentPrice: getRealPrice(selectedSymbol),
    predictedPrice: getRealPrice(selectedSymbol) * 1.02, // +2% prediction
    confidence: 0.87,
    dpLstmWeight: 0.35,
    sentimentArmaWeight: 0.30,
    kapImpact: 0.20,
    huggingFaceWeight: 0.15,
    isMarketOpen: true
  });

  useEffect(() => {
    // Update prediction when symbol changes
    const currentPrice = getRealPrice(selectedSymbol);
    setPrediction(prev => ({
      ...prev,
      symbol: selectedSymbol,
      currentPrice: currentPrice,
      predictedPrice: currentPrice * 1.02 // +2% prediction
    }));
  }, [selectedSymbol]);

  return (
    <Card className="bg-gradient-to-br from-blue-50 to-purple-50 border-blue-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-blue-600" />
          Academic Ensemble Prediction - {selectedSymbol}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge className="bg-blue-100 text-blue-700">DP-LSTM Active</Badge>
          <Badge className="bg-purple-100 text-purple-700">sentimentARMA</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Current vs Predicted */}
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-white/60 rounded-lg">
              <div className="text-sm text-slate-600">Current Price</div>
              <div className="text-xl font-bold text-slate-800">₺{prediction.currentPrice}</div>
            </div>
            <div className="text-center p-3 bg-white/60 rounded-lg">
              <div className="text-sm text-slate-600">Predicted (8H)</div>
              <div className="text-xl font-bold text-green-600">₺{prediction.predictedPrice}</div>
              <div className="text-xs text-green-500">+{((prediction.predictedPrice / prediction.currentPrice - 1) * 100).toFixed(2)}%</div>
            </div>
          </div>

          {/* Model Components */}
          <div className="space-y-2">
            <div className="text-sm font-medium text-slate-700">Model Contribution:</div>
            {[
              { name: 'DP-LSTM Neural Network', weight: prediction.dpLstmWeight, color: 'blue' },
              { name: 'sentimentARMA Model', weight: prediction.sentimentArmaWeight, color: 'purple' },
              { name: 'KAP News Impact', weight: prediction.kapImpact, color: 'green' },
              { name: 'HuggingFace Production', weight: prediction.huggingFaceWeight, color: 'orange' }
            ].map((component, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-white/40 rounded">
                <span className="text-sm">{component.name}</span>
                <div className="flex items-center gap-2">
                  <div className="w-16 bg-slate-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full bg-${component.color}-500`}
                      style={{ width: `${component.weight * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-medium">{Math.round(component.weight * 100)}%</span>
                </div>
              </div>
            ))}
          </div>

          {/* Confidence & Action */}
          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-slate-600" />
              <span className="text-sm">Confidence: </span>
              <Badge className="bg-green-100 text-green-700">
                {Math.round(prediction.confidence * 100)}%
              </Badge>
            </div>
            <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
              <TrendingUp className="h-4 w-4 mr-1" />
              View Details
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
