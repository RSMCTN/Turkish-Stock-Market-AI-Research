"use client"

import { useState, useEffect, useCallback } from 'react'
import { Search, Loader2, TrendingUp, TrendingDown, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface ApiStock {
  symbol: string
  name: string
  price?: number
  change?: number
  change_percent?: number
  volume?: number
  currency?: string
  market_cap?: number
}

interface ApiStockSearchProps {
  selectedMarket: string
  onStockSelect: (stock: ApiStock | null) => void
  selectedStock: ApiStock | null
  onClearSelection?: () => void
}

// Cache for API results (30 seconds TTL)
const apiCache = new Map<string, { data: ApiStock[], timestamp: number }>()
const CACHE_TTL = 30 * 1000 // 30 seconds

export function ApiStockSearch({ selectedMarket, onStockSelect, selectedStock, onClearSelection }: ApiStockSearchProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [searchResults, setSearchResults] = useState<ApiStock[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showResults, setShowResults] = useState(false)

  // Search API function
  const searchStocks = useCallback(async (query: string, market: string) => {
    if (!query || query.length < 1) {
      return []
    }
    
    console.log('🚀 API Call - Searching:', query)

    // Check cache first
    const cacheKey = `${query}-${market}`
    const cached = apiCache.get(cacheKey)
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      console.log('📝 Cache hit for:', cacheKey)
      return cached.data
    }

    try {
      console.log('🔍 API Search:', query, 'Market:', market)
      
      // Use Profit.com search endpoint
      const searchUrl = '/api/search-stocks'
      const response = await fetch(searchUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query, 
          market: market.toLowerCase(),
          limit: 10 
        })
      })

      if (!response.ok) {
        console.error('Search API error:', response.status)
        return []
      }

      const data = await response.json()
      const results = data.results || []

      // Cache results
      apiCache.set(cacheKey, { data: results, timestamp: Date.now() })
      console.log('✅ Found', results.length, 'stocks for:', query)

      return results

    } catch (error) {
      console.error('Search error:', error)
      return []
    }
  }, [])

  // Debounced search effect
  useEffect(() => {
    const timeoutId = setTimeout(async () => {
      if (searchTerm.length >= 1) {
        setIsLoading(true)
        setShowResults(true)
        
        console.log('🔍 Searching for:', searchTerm, 'in', selectedMarket)
        
        try {
          const results = await searchStocks(searchTerm, selectedMarket)
          console.log('📊 Search results:', results.length, 'found')
          setSearchResults(results)
        } catch (error) {
          console.error('Search failed:', error)
          setSearchResults([])
        } finally {
          setIsLoading(false)
        }
      } else {
        setSearchResults([])
        setShowResults(false)
      }
    }, 200) // 200ms debounce

    return () => clearTimeout(timeoutId)
  }, [searchTerm, selectedMarket, searchStocks])

  // Handle stock selection
  const handleStockSelect = (stock: ApiStock) => {
    onStockSelect(stock)
    setSearchTerm(stock.symbol)
    setShowResults(false)
    console.log('📊 Selected stock:', stock.symbol, stock.name)
  }

  // Clear search
  const clearSearch = () => {
    setSearchTerm("")
    setSearchResults([])
    setShowResults(false)
    onStockSelect(null)
    if (onClearSelection) {
      onClearSelection()
    }
  }

  // Format price display
  const formatPrice = (price?: number, currency = 'TRY') => {
    if (!price) return 'N/A'
    return `${price.toFixed(2)} ${currency}`
  }

  // Format change display
  const formatChange = (change?: number, changePercent?: number) => {
    if (!change && !changePercent) return null
    
    const isPositive = (change || 0) >= 0
    const Icon = isPositive ? TrendingUp : TrendingDown
    const colorClass = isPositive ? 'text-green-600' : 'text-red-600'
    
    return (
      <div className={`flex items-center ${colorClass}`}>
        <Icon className="h-3 w-3 mr-1" />
        <span className="text-xs">
          {changePercent ? `${changePercent.toFixed(2)}%` : `${change?.toFixed(2)}`}
        </span>
      </div>
    )
  }

  return (
    <div className="relative w-full">
      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
        <Input
          type="text"
          placeholder={`Search ${selectedMarket} stocks... (e.g., ZOREN, zorlu, garanti, AKBNK)`}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10 pr-10 text-white bg-slate-800 border-slate-600 placeholder:text-slate-400 focus:border-blue-500 focus:ring-blue-500"
        />
        
        {/* Loading/Clear button */}
        <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
          ) : searchTerm ? (
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={clearSearch}
              className="h-6 w-6 p-0 hover:bg-gray-100"
            >
              <X className="h-3 w-3" />
            </Button>
          ) : null}
        </div>
      </div>

      {/* Market filter info */}
      <div className="text-xs text-gray-500 mt-1">
        🔍 Searching in: <span className="font-medium">{selectedMarket}</span> market
        {selectedMarket === 'Turkey' && (
          <span className="ml-2">• Real-time from Profit.com API 🚀</span>
        )}
      </div>

      {/* Search Results Dropdown */}
      {showResults && (
        <div className="absolute z-50 w-full mt-1 bg-slate-800 border border-slate-600 rounded-md shadow-lg max-h-80 overflow-y-auto">
          {isLoading ? (
            <div className="p-4 text-center">
              <Loader2 className="h-5 w-5 animate-spin mx-auto" />
              <p className="text-sm text-gray-500 mt-2">Searching stocks...</p>
            </div>
          ) : searchResults.length > 0 ? (
            <div className="py-2">
              <div className="px-3 py-1 text-xs font-medium text-slate-400 border-b border-slate-600">
                Found {searchResults.length} stocks
              </div>
              {searchResults.map((stock, index) => (
                <button
                  key={`${stock.symbol}-${index}`}
                  onClick={() => handleStockSelect(stock)}
                  className="w-full px-3 py-2 text-left hover:bg-slate-700 border-b border-slate-600 last:border-b-0"
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center">
                        <span className="font-medium text-white">{stock.symbol}</span>
                        <span className="ml-2 text-xs bg-blue-600 text-white px-2 py-1 rounded">
                          {selectedMarket}
                        </span>
                      </div>
                      <div className="text-sm text-slate-300 mt-1">
                        {stock.name || 'N/A'}
                      </div>
                      {stock.volume && (
                        <div className="text-xs text-slate-400 mt-1">
                          Volume: {stock.volume.toLocaleString()}
                        </div>
                      )}
                    </div>
                    
                    <div className="text-right ml-2">
                      <div className="font-medium text-white">
                        {formatPrice(stock.price, stock.currency)}
                      </div>
                      {formatChange(stock.change, stock.change_percent)}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          ) : searchTerm.length >= 2 ? (
            <div className="p-4 text-center">
              <p className="text-sm text-slate-400">
                No stocks found for "{searchTerm}" in {selectedMarket} market
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Try a different search term or symbol
              </p>
            </div>
          ) : null}
        </div>
      )}

      {/* Selected Stock Info */}
      {selectedStock && (
        <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <div className="flex items-center">
                <h3 className="font-medium text-gray-900">{selectedStock.symbol}</h3>
                <span className="ml-2 text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                  {selectedMarket}
                </span>
                {onClearSelection && (
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={onClearSelection}
                    className="ml-2 h-6 w-6 p-0 hover:bg-blue-100"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </div>
              <p className="text-sm text-gray-600 mt-1">{selectedStock.name}</p>
              
              {/* API Data Warning */}
              <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
                <div className="flex items-center text-yellow-700">
                  <span>⚠️</span>
                  <span className="ml-1 font-medium">API Veri Uyarısı:</span>
                </div>
                <div className="text-yellow-600 mt-1">
                  Profit.com API'sinden gelen veriler gerçek zamanlı olmayabilir. 
                  {selectedStock.symbol === 'CWENE' && (
                    <span className="block mt-1 font-medium">
                      CWENE için API verisi doğrulanmalı (201.5 TL vs gerçek ~17.33 TL)
                    </span>
                  )}
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mt-2 text-xs">
                {selectedStock.price && (
                  <div>
                    <span className="text-gray-500">API Price:</span>
                    <span className="ml-1 font-medium">
                      {formatPrice(selectedStock.price, selectedStock.currency)}
                    </span>
                  </div>
                )}
                {selectedStock.volume && (
                  <div>
                    <span className="text-gray-500">Volume:</span>
                    <span className="ml-1 font-medium">
                      {selectedStock.volume.toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </div>
            
            <div className="text-right">
              <div className="text-lg font-bold text-gray-900">
                {formatPrice(selectedStock.price, selectedStock.currency)}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                (API Data - Verify)
              </div>
              {formatChange(selectedStock.change, selectedStock.change_percent)}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
