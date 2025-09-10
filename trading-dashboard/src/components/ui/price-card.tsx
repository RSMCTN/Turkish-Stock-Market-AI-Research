'use client';

import { TrendingUp, TrendingDown } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

interface PriceCardProps {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  className?: string;
}

export default function PriceCard({ 
  symbol, 
  price, 
  change, 
  changePercent, 
  className = '' 
}: PriceCardProps) {
  const isPositive = change >= 0;
  
  return (
    <Card className={`hover:shadow-lg transition-all duration-300 hover:scale-105 cursor-pointer ${className}`}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-bold text-lg">{symbol}</h3>
            <p className="text-2xl font-bold text-slate-900">₺{price.toFixed(2)}</p>
          </div>
          <div className={`flex items-center gap-2 ${isPositive ? 'text-emerald-600' : 'text-red-600'}`}>
            {isPositive ? <TrendingUp className="h-6 w-6" /> : <TrendingDown className="h-6 w-6" />}
            <div className="text-right">
              <div className="font-bold">{isPositive ? '+' : ''}{change.toFixed(2)}</div>
              <div className="text-sm">{isPositive ? '+' : ''}{changePercent.toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
