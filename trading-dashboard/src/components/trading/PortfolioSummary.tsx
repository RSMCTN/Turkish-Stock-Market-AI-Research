'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { Wallet, TrendingUp, TrendingDown, DollarSign, Target, Eye, Plus } from 'lucide-react';

interface PortfolioHolding {
  symbol: string;
  name: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  totalValue: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  weight: number;
}

interface PortfolioData {
  totalValue: number;
  totalCost: number;
  totalPL: number;
  totalPLPercent: number;
  availableCash: number;
  holdings: PortfolioHolding[];
  dayChange: number;
  dayChangePercent: number;
}

export default function PortfolioSummary() {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Generate mock portfolio data
  const generatePortfolioData = (): PortfolioData => {
    const symbols = [
      { symbol: 'AKBNK', name: 'Akbank' },
      { symbol: 'GARAN', name: 'Garanti BBVA' },
      { symbol: 'ISCTR', name: 'İş Bankası' },
      { symbol: 'THYAO', name: 'THY' },
      { symbol: 'ASELS', name: 'Aselsan' },
      { symbol: 'SISE', name: 'Şişe Cam' }
    ];

    let totalValue = 0;
    let totalCost = 0;
    
    const holdings: PortfolioHolding[] = symbols.slice(0, 6).map(stock => {
      const quantity = Math.floor(Math.random() * 1000) + 100;
      const averagePrice = 20 + Math.random() * 60;
      const currentPrice = averagePrice * (0.85 + Math.random() * 0.3); // ±15% from avg price
      const cost = quantity * averagePrice;
      const value = quantity * currentPrice;
      
      totalValue += value;
      totalCost += cost;
      
      return {
        symbol: stock.symbol,
        name: stock.name,
        quantity,
        averagePrice: Number(averagePrice.toFixed(2)),
        currentPrice: Number(currentPrice.toFixed(2)),
        totalValue: Number(value.toFixed(2)),
        unrealizedPL: Number((value - cost).toFixed(2)),
        unrealizedPLPercent: Number(((value - cost) / cost * 100).toFixed(2)),
        weight: 0 // Will be calculated below
      };
    });

    // Calculate weights
    holdings.forEach(holding => {
      holding.weight = Number((holding.totalValue / totalValue * 100).toFixed(1));
    });

    const totalPL = totalValue - totalCost;
    const availableCash = 50000 + Math.random() * 100000;

    return {
      totalValue: Number(totalValue.toFixed(2)),
      totalCost: Number(totalCost.toFixed(2)),
      totalPL: Number(totalPL.toFixed(2)),
      totalPLPercent: Number((totalPL / totalCost * 100).toFixed(2)),
      availableCash: Number(availableCash.toFixed(2)),
      holdings: holdings.sort((a, b) => b.weight - a.weight),
      dayChange: Number((totalValue * (Math.random() - 0.5) * 0.05).toFixed(2)),
      dayChangePercent: Number(((Math.random() - 0.5) * 5).toFixed(2))
    };
  };

  useEffect(() => {
    const loadPortfolio = () => {
      setIsLoading(true);
      setTimeout(() => {
        setPortfolio(generatePortfolioData());
        setIsLoading(false);
      }, 1000);
    };

    loadPortfolio();
    
    // Update every 30 seconds
    const interval = setInterval(loadPortfolio, 30000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wallet className="h-5 w-5" />
            Portfolio Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-gray-200 rounded w-3/4"></div>
            <div className="h-6 bg-gray-200 rounded w-1/2"></div>
            <div className="space-y-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-4 bg-gray-200 rounded"></div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!portfolio) return null;

  // Prepare pie chart data
  const pieData = portfolio.holdings.map(holding => ({
    name: holding.symbol,
    value: holding.weight,
    fill: `hsl(${Math.random() * 360}, 70%, 50%)`
  }));

  // Colors for pie chart
  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white border-2 border-gray-200 rounded-lg shadow-xl p-2">
          <p className="font-semibold">{data.name}</p>
          <p className="text-sm text-gray-600">{data.value}% of portfolio</p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Wallet className="h-5 w-5" />
            Portfolio Summary
          </CardTitle>
          <Button size="sm" variant="outline" className="gap-1">
            <Eye className="h-3 w-3" />
            Detay
          </Button>
        </div>
        <CardDescription>
          Real-time portfolio performance and allocation
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Portfolio Value */}
        <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Total Portfolio Value</span>
            <Badge className="bg-blue-100 text-blue-700 border border-blue-300">
              Live
            </Badge>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            ₺{portfolio.totalValue.toLocaleString('tr-TR')}
          </div>
          <div className="flex items-center gap-4 mt-2 text-sm">
            <div className={`flex items-center gap-1 ${
              portfolio.totalPL >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {portfolio.totalPL >= 0 ? (
                <TrendingUp className="h-4 w-4" />
              ) : (
                <TrendingDown className="h-4 w-4" />
              )}
              <span className="font-semibold">
                {portfolio.totalPL >= 0 ? '+' : ''}₺{portfolio.totalPL.toLocaleString('tr-TR')}
              </span>
              <span>
                ({portfolio.totalPLPercent >= 0 ? '+' : ''}{portfolio.totalPLPercent}%)
              </span>
            </div>
          </div>
        </div>

        {/* Day Performance */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white border rounded-lg p-3">
            <div className="text-xs text-gray-500 mb-1">Today's Change</div>
            <div className={`text-lg font-semibold ${
              portfolio.dayChange >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {portfolio.dayChange >= 0 ? '+' : ''}₺{portfolio.dayChange.toLocaleString('tr-TR')}
            </div>
            <div className={`text-xs ${
              portfolio.dayChangePercent >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {portfolio.dayChangePercent >= 0 ? '+' : ''}{portfolio.dayChangePercent}%
            </div>
          </div>
          
          <div className="bg-white border rounded-lg p-3">
            <div className="text-xs text-gray-500 mb-1">Available Cash</div>
            <div className="text-lg font-semibold text-gray-900">
              ₺{portfolio.availableCash.toLocaleString('tr-TR')}
            </div>
            <div className="text-xs text-gray-500">Ready to invest</div>
          </div>
        </div>

        {/* Portfolio Allocation Chart */}
        <div className="border rounded-lg p-3">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Asset Allocation</h4>
          <div className="flex items-center gap-4">
            <div style={{ width: 80, height: 80 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={20}
                    outerRadius={35}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div className="flex-1 space-y-1">
              {portfolio.holdings.slice(0, 4).map((holding, index) => (
                <div key={holding.symbol} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-2 h-2 rounded-full" 
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    />
                    <span className="font-medium">{holding.symbol}</span>
                  </div>
                  <span className="font-semibold">{holding.weight}%</span>
                </div>
              ))}
              {portfolio.holdings.length > 4 && (
                <div className="text-xs text-gray-500">
                  +{portfolio.holdings.length - 4} more stocks
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Top Holdings */}
        <div className="border rounded-lg p-3">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Top Holdings</h4>
          <div className="space-y-2">
            {portfolio.holdings.slice(0, 4).map((holding) => (
              <div key={holding.symbol} className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-sm">{holding.symbol}</div>
                  <div className="text-xs text-gray-500">
                    {holding.quantity} shares • ₺{holding.currentPrice}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-semibold">
                    ₺{holding.totalValue.toLocaleString('tr-TR')}
                  </div>
                  <div className={`text-xs ${
                    holding.unrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {holding.unrealizedPL >= 0 ? '+' : ''}₺{Math.abs(holding.unrealizedPL).toLocaleString('tr-TR')}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-2 gap-2">
          <Button size="sm" className="gap-1">
            <Plus className="h-3 w-3" />
            Buy More
          </Button>
          <Button size="sm" variant="outline" className="gap-1">
            <Target className="h-3 w-3" />
            Rebalance
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
