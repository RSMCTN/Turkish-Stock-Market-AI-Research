'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PieChart } from 'lucide-react';

interface ComponentContributionChartProps {
  selectedSymbol?: string;
}

export default function ComponentContributionChart({ selectedSymbol = 'GARAN' }: ComponentContributionChartProps) {
  return (
    <Card className="bg-gradient-to-br from-orange-50 to-red-50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <PieChart className="h-5 w-5 text-orange-600" />
          Model Contributions - {selectedSymbol}
        </CardTitle>
        <Badge className="w-fit bg-orange-100 text-orange-700">Real-time</Badge>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {[
            { component: 'DP-LSTM', contribution: 35, color: 'blue' },
            { component: 'sentimentARMA', contribution: 30, color: 'purple' },
            { component: 'KAP Impact', contribution: 20, color: 'green' },
            { component: 'HuggingFace', contribution: 15, color: 'orange' }
          ].map((item, index) => (
            <div key={index} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{item.component}</span>
                <span className="text-sm font-bold">{item.contribution}%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    item.color === 'blue' ? 'bg-blue-500' :
                    item.color === 'purple' ? 'bg-purple-500' :
                    item.color === 'green' ? 'bg-green-500' : 'bg-orange-500'
                  }`}
                  style={{ width: `${item.contribution}%` }}
                />
              </div>
            </div>
          ))}
          
          <div className="mt-4 pt-3 border-t border-slate-200">
            <div className="text-center">
              <div className="text-lg font-bold text-slate-800">Ensemble Score</div>
              <div className="text-2xl font-bold text-green-600">87.3%</div>
              <div className="text-xs text-slate-500">Combined accuracy</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
