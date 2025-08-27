'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ShoppingCart, Target, Clock, X, CheckCircle, AlertCircle, TrendingUp, TrendingDown, Plus } from 'lucide-react';

interface Order {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  orderType: 'MARKET' | 'LIMIT' | 'STOP';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'PENDING' | 'PARTIAL' | 'FILLED' | 'CANCELLED';
  filledQuantity: number;
  timestamp: Date;
  value: number;
}

interface Position {
  id: string;
  symbol: string;
  name: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  dayChange: number;
  dayChangePercent: number;
  marketValue: number;
}

export default function OrdersPositions() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Generate mock orders
  const generateOrders = (): Order[] => {
    const symbols = ['AKBNK', 'GARAN', 'ISCTR', 'THYAO', 'ASELS', 'SISE'];
    const orderTypes = ['MARKET', 'LIMIT', 'STOP'] as const;
    const statuses = ['PENDING', 'PARTIAL', 'FILLED', 'CANCELLED'] as const;
    
    return Array.from({ length: 8 }, (_, i) => {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const type = Math.random() > 0.5 ? 'BUY' : 'SELL';
      const orderType = orderTypes[Math.floor(Math.random() * orderTypes.length)];
      const quantity = Math.floor(Math.random() * 500) + 50;
      const price = orderType !== 'MARKET' ? 20 + Math.random() * 60 : undefined;
      const filledQuantity = Math.floor(Math.random() * quantity);
      
      return {
        id: `order-${i}`,
        symbol,
        type,
        orderType,
        quantity,
        price: price ? Number(price.toFixed(2)) : undefined,
        stopPrice: orderType === 'STOP' ? Number((price! * 0.95).toFixed(2)) : undefined,
        status: statuses[Math.floor(Math.random() * statuses.length)],
        filledQuantity,
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
        value: Number((quantity * (price || 25)).toFixed(2))
      };
    }).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  };

  // Generate mock positions
  const generatePositions = (): Position[] => {
    const stocks = [
      { symbol: 'AKBNK', name: 'Akbank' },
      { symbol: 'GARAN', name: 'Garanti BBVA' },
      { symbol: 'ISCTR', name: 'İş Bankası' },
      { symbol: 'THYAO', name: 'THY' },
      { symbol: 'ASELS', name: 'Aselsan' },
      { symbol: 'SISE', name: 'Şişe Cam' }
    ];
    
    return stocks.slice(0, 6).map((stock, i) => {
      const quantity = Math.floor(Math.random() * 800) + 100;
      const averagePrice = 20 + Math.random() * 60;
      const currentPrice = averagePrice * (0.85 + Math.random() * 0.3);
      const unrealizedPL = (currentPrice - averagePrice) * quantity;
      const dayChange = currentPrice * (Math.random() - 0.5) * 0.1;
      
      return {
        id: `pos-${i}`,
        symbol: stock.symbol,
        name: stock.name,
        quantity,
        averagePrice: Number(averagePrice.toFixed(2)),
        currentPrice: Number(currentPrice.toFixed(2)),
        unrealizedPL: Number(unrealizedPL.toFixed(2)),
        unrealizedPLPercent: Number((unrealizedPL / (averagePrice * quantity) * 100).toFixed(2)),
        dayChange: Number(dayChange.toFixed(2)),
        dayChangePercent: Number((dayChange / currentPrice * 100).toFixed(2)),
        marketValue: Number((currentPrice * quantity).toFixed(2))
      };
    }).sort((a, b) => Math.abs(b.unrealizedPL) - Math.abs(a.unrealizedPL));
  };

  useEffect(() => {
    const loadData = () => {
      setIsLoading(true);
      setTimeout(() => {
        setOrders(generateOrders());
        setPositions(generatePositions());
        setIsLoading(false);
      }, 1000);
    };

    loadData();
    
    // Update every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getOrderStatusIcon = (status: string) => {
    switch (status) {
      case 'PENDING': return <Clock className="h-3 w-3 text-yellow-600" />;
      case 'PARTIAL': return <AlertCircle className="h-3 w-3 text-blue-600" />;
      case 'FILLED': return <CheckCircle className="h-3 w-3 text-green-600" />;
      case 'CANCELLED': return <X className="h-3 w-3 text-red-600" />;
      default: return <Clock className="h-3 w-3 text-gray-600" />;
    }
  };

  const getOrderStatusColor = (status: string) => {
    switch (status) {
      case 'PENDING': return 'bg-yellow-100 text-yellow-700 border-yellow-300';
      case 'PARTIAL': return 'bg-blue-100 text-blue-700 border-blue-300';
      case 'FILLED': return 'bg-green-100 text-green-700 border-green-300';
      case 'CANCELLED': return 'bg-red-100 text-red-700 border-red-300';
      default: return 'bg-gray-100 text-gray-700 border-gray-300';
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('tr-TR', { 
      hour: '2-digit', 
      minute: '2-digit'
    });
  };

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShoppingCart className="h-5 w-5" />
            Orders & Positions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="flex justify-between items-center">
                <div className="space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-20"></div>
                  <div className="h-3 bg-gray-200 rounded w-16"></div>
                </div>
                <div className="h-4 bg-gray-200 rounded w-16"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const activeOrders = orders.filter(order => order.status === 'PENDING' || order.status === 'PARTIAL');
  const totalPositionValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
  const totalUnrealizedPL = positions.reduce((sum, pos) => sum + pos.unrealizedPL, 0);

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <ShoppingCart className="h-5 w-5" />
            Orders & Positions
          </CardTitle>
          <Button size="sm" className="gap-1">
            <Plus className="h-3 w-3" />
            New Order
          </Button>
        </div>
        <CardDescription>
          Active orders and current positions
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <Tabs defaultValue="positions" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="positions" className="gap-1">
              <Target className="h-3 w-3" />
              Positions ({positions.length})
            </TabsTrigger>
            <TabsTrigger value="orders" className="gap-1">
              <Clock className="h-3 w-3" />
              Orders ({activeOrders.length})
            </TabsTrigger>
          </TabsList>
          
          {/* Positions Tab */}
          <TabsContent value="positions" className="space-y-3">
            {/* Position Summary */}
            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-3">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-600">Total Position Value</span>
                <span className="font-semibold">₺{totalPositionValue.toLocaleString('tr-TR')}</span>
              </div>
              <div className="flex justify-between items-center text-sm mt-1">
                <span className="text-gray-600">Unrealized P&L</span>
                <span className={`font-semibold ${totalUnrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {totalUnrealizedPL >= 0 ? '+' : ''}₺{totalUnrealizedPL.toLocaleString('tr-TR')}
                </span>
              </div>
            </div>

            {/* Positions List */}
            <ScrollArea className="h-80">
              <div className="space-y-2">
                {positions.map((position) => (
                  <div key={position.id} className="border rounded-lg p-3 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <div className="font-semibold text-sm">{position.symbol}</div>
                        <div className="text-xs text-gray-500">{position.name}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-semibold">
                          ₺{position.marketValue.toLocaleString('tr-TR')}
                        </div>
                        <div className={`text-xs flex items-center gap-1 justify-end ${
                          position.dayChangePercent >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {position.dayChangePercent >= 0 ? (
                            <TrendingUp className="h-3 w-3" />
                          ) : (
                            <TrendingDown className="h-3 w-3" />
                          )}
                          {position.dayChangePercent >= 0 ? '+' : ''}{position.dayChangePercent.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <span className="text-gray-500">Quantity:</span>
                        <div className="font-medium">{position.quantity}</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Avg Price:</span>
                        <div className="font-medium">₺{position.averagePrice}</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Current:</span>
                        <div className="font-medium">₺{position.currentPrice}</div>
                      </div>
                    </div>
                    
                    <div className="mt-2 pt-2 border-t flex justify-between items-center">
                      <div className="text-xs text-gray-500">
                        Unrealized P&L
                      </div>
                      <div className={`text-sm font-semibold ${
                        position.unrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {position.unrealizedPL >= 0 ? '+' : ''}₺{Math.abs(position.unrealizedPL).toLocaleString('tr-TR')}
                        <span className="text-xs ml-1">
                          ({position.unrealizedPL >= 0 ? '+' : ''}{position.unrealizedPLPercent.toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
          
          {/* Orders Tab */}
          <TabsContent value="orders" className="space-y-3">
            {/* Active Orders Summary */}
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-600">Active Orders</span>
                <span className="font-semibold">{activeOrders.length}</span>
              </div>
              <div className="flex justify-between items-center text-sm mt-1">
                <span className="text-gray-600">Total Value</span>
                <span className="font-semibold">
                  ₺{activeOrders.reduce((sum, order) => sum + order.value, 0).toLocaleString('tr-TR')}
                </span>
              </div>
            </div>

            {/* Orders List */}
            <ScrollArea className="h-80">
              <div className="space-y-2">
                {orders.map((order) => (
                  <div key={order.id} className="border rounded-lg p-3 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge className={`text-xs ${
                          order.type === 'BUY' 
                            ? 'bg-green-100 text-green-700 border-green-300' 
                            : 'bg-red-100 text-red-700 border-red-300'
                        }`}>
                          {order.type}
                        </Badge>
                        <span className="font-semibold text-sm">{order.symbol}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={`text-xs ${getOrderStatusColor(order.status)}`}>
                          <div className="flex items-center gap-1">
                            {getOrderStatusIcon(order.status)}
                            {order.status}
                          </div>
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 text-xs mb-2">
                      <div>
                        <span className="text-gray-500">Type:</span>
                        <div className="font-medium">{order.orderType}</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Quantity:</span>
                        <div className="font-medium">{order.quantity}</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Price:</span>
                        <div className="font-medium">
                          {order.price ? `₺${order.price}` : 'Market'}
                        </div>
                      </div>
                    </div>
                    
                    {order.status === 'PARTIAL' && (
                      <div className="text-xs text-blue-600 mb-2">
                        Filled: {order.filledQuantity}/{order.quantity} ({((order.filledQuantity / order.quantity) * 100).toFixed(1)}%)
                      </div>
                    )}
                    
                    <div className="flex justify-between items-center text-xs pt-2 border-t">
                      <span className="text-gray-500">{formatTime(order.timestamp)}</span>
                      <span className="font-semibold">₺{order.value.toLocaleString('tr-TR')}</span>
                    </div>
                    
                    {(order.status === 'PENDING' || order.status === 'PARTIAL') && (
                      <div className="mt-2 pt-2 border-t">
                        <Button size="sm" variant="destructive" className="w-full h-6 text-xs">
                          Cancel Order
                        </Button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
