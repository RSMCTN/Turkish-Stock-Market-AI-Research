'use client';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useState } from 'react';
import { ChevronDown, Flag, Globe, MapPin } from 'lucide-react';

interface GlobalMarketSelectorProps {
  selected: string;
  onSelect: (market: string) => void;
}

const markets = [
  {
    id: 'turkey',
    name: 'Turkey',
    flag: '🇹🇷',
    description: 'BIST Markets - Real Data',
    stocks: 229,
    icon: Flag,
    available: true,
    status: '✅ Live'
  },
  {
    id: 'usa',
    name: 'United States', 
    flag: '🇺🇸',
    description: 'NYSE, NASDAQ - Coming Soon',
    stocks: 0,
    icon: MapPin,
    available: false,
    status: '🚧 Soon'
  },
  {
    id: 'europe',
    name: 'Europe',
    flag: '🇪🇺', 
    description: 'XETRA, LSE - Coming Soon',
    stocks: 0,
    icon: Globe,
    available: false,
    status: '🚧 Soon'
  },
  {
    id: 'asia',
    name: 'Asia Pacific',
    flag: '🌏',
    description: 'TSE, HKEX - Coming Soon', 
    stocks: 0,
    icon: Globe,
    available: false,
    status: '🚧 Soon'
  },
  {
    id: 'global',
    name: 'Global Mix',
    flag: '🌍',
    description: 'Worldwide - Coming Soon',
    stocks: 0,
    icon: Globe,
    available: false,
    status: '🚧 Soon'
  }
];

export function GlobalMarketSelector({ selected, onSelect }: GlobalMarketSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  const selectedMarket = markets.find(m => m.id === selected) || markets[0];

  return (
    <div className="relative">
      <Button
        variant="outline"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 bg-slate-800 border-slate-600 text-white hover:bg-slate-700"
      >
        <span className="text-lg">{selectedMarket.flag}</span>
        <span>{selectedMarket.name}</span>
        <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </Button>

      {isOpen && (
        <Card className="absolute top-full mt-2 right-0 z-50 w-72 border-slate-600 bg-slate-800 shadow-xl">
          <div className="p-2 space-y-1">
            {markets.map((market) => {
              const Icon = market.icon;
              return (
                <button
                  key={market.id}
                  onClick={() => {
                    if (market.available) {
                      onSelect(market.id);
                      setIsOpen(false);
                    }
                  }}
                  disabled={!market.available}
                  className={`w-full text-left p-3 rounded-md transition-colors flex items-center space-x-3 ${
                    market.available 
                      ? 'hover:bg-slate-700' 
                      : 'opacity-60 cursor-not-allowed'
                  } ${
                    selected === market.id ? 'bg-slate-700 border border-blue-500' : ''
                  }`}
                >
                  <span className="text-2xl">{market.flag}</span>
                  <div className="flex-1">
                    <div className={`font-medium ${market.available ? 'text-white' : 'text-slate-400'}`}>
                      {market.name}
                    </div>
                    <div className="text-sm text-slate-400">{market.description}</div>
                  </div>
                  <div className="text-right">
                    <div className={`text-xs ${market.available ? 'text-blue-400' : 'text-slate-500'}`}>
                      {market.status}
                    </div>
                    <div className={`text-sm ${market.available ? 'text-blue-400' : 'text-slate-500'}`}>
                      {market.stocks > 0 ? market.stocks : '-'}
                    </div>
                    <div className="text-xs text-slate-500">stocks</div>
                  </div>
                </button>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}
