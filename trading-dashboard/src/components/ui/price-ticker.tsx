'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface PriceTickerProps {
  symbol: string;
  initialPrice?: number;
  className?: string;
}

export function PriceTicker({ symbol, initialPrice = 50.25, className = '' }: PriceTickerProps) {
  const [price, setPrice] = useState(initialPrice);
  const [change, setChange] = useState(0);
  const [isPositive, setIsPositive] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      const changeAmount = (Math.random() - 0.5) * 1.0; // ±0.5 TL change
      
      setPrice(prevPrice => {
        const newPrice = Math.max(prevPrice + changeAmount, 1);
        const percentChange = ((newPrice - prevPrice) / prevPrice) * 100;
        setChange(percentChange);
        setIsPositive(changeAmount >= 0);
        return newPrice;
      });
    }, 2000 + Math.random() * 3000); // Random interval between 2-5 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className="flex items-center space-x-1">
        <span className="font-bold text-lg">₺{price.toFixed(2)}</span>
        <div className={`flex items-center space-x-1 text-sm ${
          isPositive ? 'text-green-600' : 'text-red-600'
        }`}>
          {isPositive ? (
            <TrendingUp className="h-3 w-3" />
          ) : (
            <TrendingDown className="h-3 w-3" />
          )}
          <span>{isPositive ? '+' : ''}{change.toFixed(2)}%</span>
        </div>
      </div>
    </div>
  );
}
