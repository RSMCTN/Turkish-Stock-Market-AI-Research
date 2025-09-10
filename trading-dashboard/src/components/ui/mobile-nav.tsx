'use client';

import { useState } from 'react';
import { Menu, X, Home, Brain, BarChart3, Target, PieChart } from 'lucide-react';
import { Button } from './button';
import { Badge } from './badge';

interface MobileNavProps {
  activeView: string;
  onViewChange: (view: string) => void;
  selectedSymbol: string;
  currentPrice: number;
}

const navigationItems = [
  { id: 'overview', label: 'Overview', icon: Home, color: 'from-emerald-500 to-cyan-500' },
  { id: 'ai-analytics', label: 'AI Analytics', icon: Brain, color: 'from-blue-500 to-purple-500' },
  { id: 'technical', label: 'Technical', icon: BarChart3, color: 'from-orange-500 to-red-500' },
  { id: 'professional', label: 'Professional', icon: Target, color: 'from-indigo-500 to-purple-500' },
  { id: 'analysis', label: 'Analysis', icon: PieChart, color: 'from-green-500 to-teal-500' },
];

export function MobileNav({ activeView, onViewChange, selectedSymbol, currentPrice }: MobileNavProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleViewChange = (viewId: string) => {
    onViewChange(viewId);
    setIsOpen(false);
  };

  return (
    <>
      {/* Mobile Header */}
      <div className="lg:hidden flex items-center justify-between p-4 bg-white shadow-sm border-b">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsOpen(true)}
        >
          <Menu className="h-5 w-5" />
        </Button>
        
        <div className="flex items-center space-x-3">
          <div className="text-center">
            <p className="text-sm font-medium">{selectedSymbol}</p>
            <p className="text-lg font-bold text-emerald-600">₺{currentPrice.toFixed(2)}</p>
          </div>
          <Badge className="bg-emerald-100 text-emerald-700">LIVE</Badge>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black bg-opacity-50" onClick={() => setIsOpen(false)} />
          
          {/* Mobile Menu */}
          <div className="absolute top-0 left-0 h-full w-80 bg-white shadow-xl">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <img src="/mamut-logo.png" alt="MAMUT R600" className="h-8 w-auto" />
                <div>
                  <h1 className="text-lg font-bold bg-gradient-to-r from-emerald-600 to-blue-600 bg-clip-text text-transparent">
                    MAMUT R600
                  </h1>
                  <p className="text-xs text-slate-500">Trading Platform</p>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setIsOpen(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            
            <nav className="p-6 space-y-3">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Button
                    key={item.id}
                    variant={activeView === item.id ? 'default' : 'ghost'}
                    className={`w-full justify-start text-left ${
                      activeView === item.id 
                        ? `bg-gradient-to-r ${item.color} text-white shadow-lg` 
                        : 'hover:bg-slate-100'
                    }`}
                    onClick={() => handleViewChange(item.id)}
                  >
                    <Icon className="h-4 w-4 mr-3" />
                    {item.label}
                  </Button>
                );
              })}
            </nav>
          </div>
        </div>
      )}
    </>
  );
}
