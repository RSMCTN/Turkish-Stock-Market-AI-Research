'use client';

import { useState, useEffect, useRef } from 'react';
import { Search, X, TrendingUp, Building2, Globe } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface Stock {
  symbol: string;
  ticker: string;
  name: string;
  sector: string;
  market: string;
  currency: string;
  region: string;
  search_text: string;
}

interface StockSearchProps {
  onSelectStock?: (stock: Stock) => void;
  placeholder?: string;
  className?: string;
  selectedMarket?: string; // Filter by market
  onClearSelection?: () => void; // Clear selection callback
}

export function StockSearch({ 
  onSelectStock, 
  placeholder = "Hisse arayın... (AKBNK, AAPL, GOOGL)", 
  className = "",
  selectedMarket = "all",
  onClearSelection
}: StockSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Stock[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [allStocks, setAllStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Load stock data on component mount
  useEffect(() => {
    loadStockData();
  }, []);

  // Search when query changes
  useEffect(() => {
    if (query.trim() === '') {
      setResults([]);
      setSelectedIndex(-1);
      return;
    }

    const searchResults = performSearch(query);
    setResults(searchResults);
    setSelectedIndex(-1);
    setIsOpen(searchResults.length > 0);
  }, [query, allStocks]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen || results.length === 0) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, results.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, -1));
          break;
        case 'Enter':
          e.preventDefault();
          if (selectedIndex >= 0 && results[selectedIndex]) {
            handleSelectStock(results[selectedIndex]);
          }
          break;
        case 'Escape':
          setIsOpen(false);
          setSelectedIndex(-1);
          break;
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen, results, selectedIndex]);

  const loadStockData = async () => {
    try {
      setLoading(true);
      
      // Try to load global stocks data first
      try {
        const response = await fetch('/global_stocks_data.json');
        if (response.ok) {
          const data = await response.json();
          setAllStocks(data.all_stocks);
          setLoading(false);
          return;
        }
      } catch (error) {
        console.log('Global stocks data not found, trying Turkish only');
      }

      // Fallback to Turkish stocks only
      try {
        const response = await fetch('/profit_search_data.json');
        if (response.ok) {
          const data = await response.json();
          // Add region field to Turkish stocks
          const turkishStocks = data.map((stock: any) => ({
            ...stock,
            region: 'turkey'
          }));
          setAllStocks(turkishStocks);
          setLoading(false);
          return;
        }
      } catch (error) {
        console.log('Turkish data not found, using fallback');
      }

      // Final fallback: Create mock data
      const fallbackStocks = generateFallbackStocks();
      setAllStocks(fallbackStocks);
      setLoading(false);

    } catch (error) {
      console.error('Error loading stock data:', error);
      setLoading(false);
    }
  };

  const generateFallbackStocks = (): Stock[] => {
    // Popular Turkish stocks as fallback
    const popularStocks = [
      { symbol: 'AKBNK', name: 'Akbank T.A.Ş.', sector: 'Bankacılık', market: 'BIST' },
      { symbol: 'GARAN', name: 'Türkiye Garanti Bankası A.Ş.', sector: 'Bankacılık', market: 'BIST' },
      { symbol: 'ISCTR', name: 'Türkiye İş Bankası A.Ş.', sector: 'Bankacılık', market: 'BIST' },
      { symbol: 'YKBNK', name: 'Yapı ve Kredi Bankası A.Ş.', sector: 'Bankacılık', market: 'BIST' },
      { symbol: 'VAKBN', name: 'Türkiye Vakıflar Bankası T.A.O.', sector: 'Bankacılık', market: 'BIST' },
      { symbol: 'TUPRS', name: 'Tüpraş-Türkiye Petrol Rafinerileri A.Ş.', sector: 'Petrol & Kimya', market: 'BIST' },
      { symbol: 'THYAO', name: 'Türk Hava Yolları A.O.', sector: 'Taşımacılık', market: 'BIST' },
      { symbol: 'TCELL', name: 'Turkcell İletişim Hizmetleri A.Ş.', sector: 'Telekomünikasyon', market: 'BIST' },
      { symbol: 'ASELS', name: 'Aselsan Elektronik Sanayi ve Ticaret A.Ş.', sector: 'Teknoloji', market: 'BIST' },
      { symbol: 'KCHOL', name: 'Koç Holding A.Ş.', sector: 'Holding', market: 'BIST' }
    ];

    return popularStocks.map(stock => ({
      ...stock,
      ticker: `${stock.symbol}.IS`,
      currency: 'TRY',
      region: 'turkey',
      search_text: `${stock.symbol} ${stock.name} ${stock.sector}`.toLowerCase()
    }));
  };

  const performSearch = (searchQuery: string): Stock[] => {
    const query = searchQuery.toLowerCase().trim();
    
    if (query.length < 1) return [];

    // Filter by selected market first
    let filteredStocks = allStocks;
    if (selectedMarket && selectedMarket !== 'all' && selectedMarket !== 'global') {
      filteredStocks = allStocks.filter(stock => stock.region === selectedMarket);
    }

    // Advanced scoring system
    const results: Array<{score: number; stock: Stock}> = [];

    filteredStocks.forEach(stock => {
      const symbol = stock.symbol.toLowerCase();
      const name = stock.name.toLowerCase();
      const searchText = stock.search_text.toLowerCase();
      
      let score = 0;
      
      // Perfect symbol match (highest score)
      if (symbol === query) {
        score += 100;
      }
      // Symbol starts with query
      else if (symbol.startsWith(query)) {
        score += 50;
      }
      // Query in symbol
      else if (symbol.includes(query)) {
        score += 30;
      }
      
      // Name matches
      if (name.includes(query)) {
        score += 20;
        if (name.startsWith(query)) {
          score += 10;
        }
      }
      
      // Search text match
      if (searchText.includes(query)) {
        score += 10;
      }
      
      // Only include if there's some match
      if (score > 0) {
        results.push({score, stock});
      }
    });

    // Sort by score (descending) and return top results
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map(result => result.stock);
  };

  const handleSelectStock = (stock: Stock) => {
    setQuery(stock.symbol);
    setIsOpen(false);
    setSelectedIndex(-1);
    onSelectStock?.(stock);
  };

  const clearSearch = () => {
    setQuery('');
    setResults([]);
    setIsOpen(false);
    setSelectedIndex(-1);
    onClearSelection?.(); // Notify parent to clear selection
    inputRef.current?.focus();
  };

  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query && setIsOpen(results.length > 0)}
          placeholder={placeholder}
          className="w-full pl-10 pr-10 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          disabled={loading}
        />
        {query && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearSearch}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0 text-slate-400 hover:text-white"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
        {loading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
          </div>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <Card className="absolute top-full mt-2 w-full z-50 border-slate-600 bg-slate-800 shadow-xl max-h-96 overflow-hidden">
          <div className="p-2">
            <div className="text-xs text-slate-400 px-2 py-1 mb-2">
              {results.length} sonuç bulundu
            </div>
            <div ref={resultsRef} className="space-y-1 max-h-80 overflow-y-auto">
              {results.map((stock, index) => (
                <button
                  key={stock.symbol}
                  onClick={() => handleSelectStock(stock)}
                  className={`w-full text-left p-3 rounded-md transition-colors flex items-center space-x-3 ${
                    index === selectedIndex 
                      ? 'bg-blue-600 text-white' 
                      : 'hover:bg-slate-700 text-slate-200'
                  }`}
                >
                  <div className="flex-shrink-0">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                      index === selectedIndex ? 'bg-blue-800' : 'bg-slate-600'
                    }`}>
                      {stock.symbol.substring(0, 2)}
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono font-semibold">{stock.symbol}</span>
                      <span className="text-xs bg-slate-600 px-2 py-1 rounded">
                        {stock.currency}
                      </span>
                    </div>
                    <div className="text-sm opacity-75 truncate">
                      {stock.name}
                    </div>
                    <div className="flex items-center space-x-4 text-xs opacity-60 mt-1">
                      <span className="flex items-center space-x-1">
                        <Building2 className="h-3 w-3" />
                        <span>{stock.sector}</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <Globe className="h-3 w-3" />
                        <span>{stock.market}</span>
                      </span>
                    </div>
                  </div>
                  <div className="flex-shrink-0">
                    <TrendingUp className="h-4 w-4 opacity-40" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Loading state */}
      {loading && (
        <div className="absolute top-full mt-2 w-full">
          <Card className="border-slate-600 bg-slate-800 p-4 text-center">
            <div className="text-slate-400">Hisse verileri yükleniyor...</div>
          </Card>
        </div>
      )}

      {/* No results */}
      {isOpen && !loading && query && results.length === 0 && (
        <Card className="absolute top-full mt-2 w-full border-slate-600 bg-slate-800 p-4 text-center">
          <div className="text-slate-400">
            "<span className="text-white">{query}</span>" için sonuç bulunamadı
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Hisse kodu, şirket adı veya sektör ile arayın
          </div>
        </Card>
      )}
    </div>
  );
}
