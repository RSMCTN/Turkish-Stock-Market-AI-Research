'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Newspaper, TrendingUp, TrendingDown, Calendar, ExternalLink, Filter, RefreshCw } from 'lucide-react';

interface NewsItem {
  headline: string;
  sentiment: number;
  impact: 'HIGH' | 'MEDIUM' | 'LOW';
  source: string;
  timestamp: string;
  confidence: number;
  category: 'EARNINGS' | 'ANALYST_RATING' | 'PARTNERSHIP' | 'REGULATORY' | 'EXPANSION' | 'MARKET_NEWS';
}

interface AdvancedNewsSentimentProps {
  symbol: string;
  newsItems?: NewsItem[];
  isLoading?: boolean;
}

const AdvancedNewsSentiment = ({ symbol, newsItems: propNewsItems = [], isLoading: propIsLoading = false }: AdvancedNewsSentimentProps) => {
  const [newsItems, setNewsItems] = useState<NewsItem[]>(propNewsItems);
  const [isLoading, setIsLoading] = useState(propIsLoading);
  const [error, setError] = useState<string>('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [expandedNews, setExpandedNews] = useState<number | null>(null);

  const categories = ['ALL', 'EARNINGS', 'ANALYST_RATING', 'PARTNERSHIP', 'REGULATORY', 'EXPANSION', 'MARKET_NEWS'];

  // Fetch news sentiment data
  const fetchNewsSentiment = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch(`https://bistai001-production.up.railway.app/api/forecast/${symbol}?horizon=24`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.newsImpact) {
        // Convert API response to our NewsItem format
        const convertedNews: NewsItem[] = data.newsImpact.map((item: any, index: number) => ({
          headline: item.headline || item.title || `News Item ${index + 1}`,
          sentiment: item.sentiment || item.score || 0,
          impact: item.impact?.toUpperCase() || 'MEDIUM',
          source: item.source || 'Unknown',
          timestamp: item.timestamp || item.published_at || new Date().toISOString(),
          confidence: item.confidence || 0.8,
          category: item.category?.toUpperCase() || 'MARKET_NEWS'
        }));
        
        setNewsItems(convertedNews);
        setLastUpdated(new Date());
      } else {
        setNewsItems([]);
        setError('No news sentiment data available');
      }
    } catch (err) {
      console.error('Failed to fetch news sentiment:', err);
      setError('Failed to load news sentiment data');
      setNewsItems([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (symbol && propNewsItems.length === 0) {
      fetchNewsSentiment();
    }
  }, [symbol]);

  const handleRefresh = () => {
    fetchNewsSentiment();
  };
  
  const filteredNews = selectedCategory === 'ALL' 
    ? newsItems 
    : newsItems.filter(item => item.category === selectedCategory);

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.3) return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (sentiment < -0.3) return <TrendingDown className="h-4 w-4 text-red-600" />;
    return <div className="h-4 w-4 rounded-full bg-gray-400"></div>;
  };

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.5) return 'bg-green-100 text-green-800 border-green-300';
    if (sentiment > 0) return 'bg-green-50 text-green-700 border-green-200';
    if (sentiment < -0.5) return 'bg-red-100 text-red-800 border-red-300';
    if (sentiment < 0) return 'bg-red-50 text-red-700 border-red-200';
    return 'bg-gray-100 text-gray-800 border-gray-300';
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'HIGH': return 'bg-purple-100 text-purple-800 border-purple-300';
      case 'MEDIUM': return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'LOW': return 'bg-gray-100 text-gray-800 border-gray-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      'EARNINGS': 'bg-yellow-100 text-yellow-800',
      'ANALYST_RATING': 'bg-blue-100 text-blue-800',
      'PARTNERSHIP': 'bg-green-100 text-green-800',
      'REGULATORY': 'bg-red-100 text-red-800',
      'EXPANSION': 'bg-purple-100 text-purple-800',
      'MARKET_NEWS': 'bg-gray-100 text-gray-800'
    };
    return colors[category as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const avgSentiment = newsItems.length > 0 
    ? newsItems.reduce((sum, item) => sum + item.sentiment, 0) / newsItems.length 
    : 0;

  const avgConfidence = newsItems.length > 0
    ? newsItems.reduce((sum, item) => sum + item.confidence, 0) / newsItems.length
    : 0;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Newspaper className="h-5 w-5" />
            News Sentiment Analysis - {symbol}
          </CardTitle>
          <CardDescription>Loading sentiment analysis...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1,2,3,4,5].map(i => (
              <div key={i} className="animate-pulse border rounded-lg p-4">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2 mb-2"></div>
                <div className="flex gap-2">
                  <div className="h-6 bg-gray-200 rounded w-16"></div>
                  <div className="h-6 bg-gray-200 rounded w-20"></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Newspaper className="h-5 w-5" />
              News Sentiment Analysis - {symbol}
            </CardTitle>
            <CardDescription>
              Advanced news sentiment analysis with source tracking
              {lastUpdated && (
                <span className="ml-2 text-xs text-gray-500">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </span>
              )}
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isLoading}
            className="gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            {isLoading ? 'Loading...' : 'Refresh'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        
        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="flex items-center justify-center gap-2 mb-1">
              {getSentimentIcon(avgSentiment)}
              <p className="text-sm text-gray-600">Overall Sentiment</p>
            </div>
            <p className="text-xl font-bold text-gray-900">
              {avgSentiment > 0 ? '+' : ''}{avgSentiment.toFixed(3)}
            </p>
            <Badge className={getSentimentColor(avgSentiment)}>
              {avgSentiment > 0.3 ? 'POSITIVE' : avgSentiment < -0.3 ? 'NEGATIVE' : 'NEUTRAL'}
            </Badge>
          </div>
          
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">News Count</p>
            <p className="text-xl font-bold text-gray-900">{filteredNews.length}</p>
            <p className="text-xs text-gray-500">
              {newsItems.filter(n => n.impact === 'HIGH').length} High Impact
            </p>
          </div>
          
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">Avg Confidence</p>
            <p className="text-xl font-bold text-gray-900">{(avgConfidence * 100).toFixed(0)}%</p>
            <p className="text-xs text-gray-500">Analysis Reliability</p>
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2">
          <Filter className="h-4 w-4 mt-2 text-gray-500" />
          {categories.map(category => (
            <Button
              key={category}
              variant={selectedCategory === category ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedCategory(category)}
              className="text-xs"
            >
              {category.replace('_', ' ')}
              {category !== 'ALL' && (
                <span className="ml-1 bg-white/20 px-1 rounded">
                  {newsItems.filter(n => n.category === category).length}
                </span>
              )}
            </Button>
          ))}
        </div>

        {/* News Items */}
        <div className="space-y-3">
          {filteredNews.map((news, index) => (
            <div 
              key={index}
              className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                expandedNews === index ? 'ring-2 ring-blue-200 border-blue-300' : ''
              }`}
              onClick={() => setExpandedNews(expandedNews === index ? null : index)}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 mb-1 leading-tight">
                    {news.headline}
                  </h4>
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <Calendar className="h-3 w-3" />
                    {news.timestamp} • {news.source}
                  </div>
                </div>
                <div className="flex items-center gap-1 ml-4">
                  {getSentimentIcon(news.sentiment)}
                </div>
              </div>

              {/* Badges */}
              <div className="flex flex-wrap gap-2 mb-2">
                <Badge className={getSentimentColor(news.sentiment)}>
                  {news.sentiment > 0 ? '+' : ''}{news.sentiment.toFixed(3)} Sentiment
                </Badge>
                <Badge className={getImpactColor(news.impact)}>
                  {news.impact} Impact
                </Badge>
                <Badge className={getCategoryColor(news.category)}>
                  {news.category.replace('_', ' ')}
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {(news.confidence * 100).toFixed(0)}% Confidence
                </Badge>
              </div>

              {/* Expanded Content */}
              {expandedNews === index && (
                <div className="mt-4 pt-4 border-t bg-gray-50 rounded p-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                      <h5 className="font-medium text-gray-900 mb-2">Sentiment Analysis</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Sentiment Score:</span>
                          <span className="font-semibold">{news.sentiment.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Impact Level:</span>
                          <span className="font-semibold">{news.impact}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Confidence:</span>
                          <span className="font-semibold">{(news.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="font-medium text-gray-900 mb-2">News Details</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Source:</span>
                          <span className="font-semibold">{news.source}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Category:</span>
                          <span className="font-semibold">{news.category.replace('_', ' ')}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Published:</span>
                          <span className="font-semibold">{news.timestamp}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Trading Impact */}
                  <div className="p-3 bg-white rounded border">
                    <h5 className="font-medium text-gray-900 mb-2">Trading Impact Assessment</h5>
                    <p className="text-sm text-gray-600">
                      {news.sentiment > 0.5 && "Strong positive sentiment may drive buying interest and upward price momentum."}
                      {news.sentiment > 0 && news.sentiment <= 0.5 && "Mild positive sentiment suggests cautious optimism in the market."}
                      {news.sentiment < -0.5 && "Strong negative sentiment may trigger selling pressure and downward price movement."}
                      {news.sentiment < 0 && news.sentiment >= -0.5 && "Mild negative sentiment indicates market uncertainty or concern."}
                      {news.sentiment === 0 && "Neutral sentiment suggests minimal immediate impact on stock price."}
                    </p>
                  </div>

                  <div className="mt-3 flex justify-end">
                    <Button variant="outline" size="sm" className="gap-2">
                      <ExternalLink className="h-3 w-3" />
                      View Source
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {error && (
          <div className="text-center py-8">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <Newspaper className="h-12 w-12 mx-auto mb-2 text-red-300" />
              <p className="text-red-600 font-medium mb-2">Error Loading News Sentiment</p>
              <p className="text-red-500 text-sm">{error}</p>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleRefresh} 
                className="mt-3 border-red-300 text-red-600 hover:bg-red-50"
              >
                Try Again
              </Button>
            </div>
          </div>
        )}

        {!error && filteredNews.length === 0 && !isLoading && (
          <div className="text-center py-8 text-gray-500">
            <Newspaper className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>No news items found for {selectedCategory === 'ALL' ? symbol : selectedCategory}</p>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleRefresh} 
              className="mt-3"
            >
              Load News
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AdvancedNewsSentiment;
