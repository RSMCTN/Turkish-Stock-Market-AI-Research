"""
Multi-Source Turkish Financial News Crawler
Scrapes financial news from major Turkish news sources
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from newspaper import Article, Config
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import time
from urllib.parse import urljoin, urlparse
import hashlib


class TurkishNewsCrawler:
    """
    Crawls Turkish financial news from multiple sources
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Configure newspaper for Turkish
        self.newspaper_config = Config()
        self.newspaper_config.language = 'tr'
        self.newspaper_config.memoize_articles = False
        self.newspaper_config.fetch_images = False
        
        # News sources configuration
        self.news_sources = self._configure_news_sources()
        
        # Rate limiting
        self.last_request_times = {}
        self.min_delay_between_requests = 2.0  # 2 seconds between requests per domain
        
        self.logger.info(f"Turkish News Crawler initialized with {len(self.news_sources)} sources")
    
    def _configure_news_sources(self) -> Dict[str, Dict[str, Any]]:
        """Configure Turkish financial news sources"""
        return {
            'aa_finansal': {
                'name': 'Anadolu Ajansı - Finansal',
                'rss_url': 'https://www.aa.com.tr/tr/rss/default?cat=ekonomi',
                'base_url': 'https://www.aa.com.tr',
                'selectors': {
                    'title': 'h1.detail-title',
                    'content': '.detail-text p',
                    'date': '.detail-date',
                },
                'encoding': 'utf-8',
                'priority': 'high',
            },
            
            'bloomberght': {
                'name': 'Bloomberg HT',
                'rss_url': 'https://www.bloomberght.com/rss',
                'base_url': 'https://www.bloomberght.com',
                'selectors': {
                    'title': 'h1',
                    'content': '.article-content p',
                    'date': '.article-date',
                },
                'encoding': 'utf-8',
                'priority': 'high',
            },
            
            'dunya': {
                'name': 'Dünya Gazetesi',
                'rss_url': 'https://www.dunya.com/rss/economy.xml',
                'base_url': 'https://www.dunya.com',
                'selectors': {
                    'title': 'h1.detail-title',
                    'content': '.detail-spot, .detail-text p',
                    'date': '.detail-date',
                },
                'encoding': 'utf-8',
                'priority': 'medium',
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit_request(self, domain: str):
        """Apply rate limiting per domain"""
        current_time = time.time()
        
        if domain in self.last_request_times:
            time_since_last = current_time - self.last_request_times[domain]
            if time_since_last < self.min_delay_between_requests:
                sleep_time = self.min_delay_between_requests - time_since_last
                await asyncio.sleep(sleep_time)
        
        self.last_request_times[domain] = time.time()
    
    async def fetch_rss_feed(self, source_key: str) -> List[Dict[str, Any]]:
        """Fetch RSS feed from a news source"""
        source = self.news_sources[source_key]
        rss_url = source['rss_url']
        
        try:
            domain = urlparse(rss_url).netloc
            await self._rate_limit_request(domain)
            
            async with self.session.get(rss_url) as response:
                if response.status != 200:
                    self.logger.warning(f"RSS fetch failed for {source_key}: {response.status}")
                    return []
                
                rss_content = await response.text()
                
            # Parse RSS feed
            feed = feedparser.parse(rss_content)
            
            articles = []
            for entry in feed.entries[:10]:  # Limit to 10 most recent
                try:
                    # Extract article metadata
                    article_data = {
                        'title': entry.get('title', '').strip(),
                        'url': entry.get('link', '').strip(),
                        'summary': entry.get('summary', '').strip(),
                        'published': self._parse_date(entry.get('published', '')),
                        'source': source_key,
                        'source_name': source['name'],
                        'priority': source.get('priority', 'medium'),
                        'content': '',  # Will be fetched separately
                        'raw_html': '',
                        'hash': self._generate_article_hash(entry.get('link', ''), entry.get('title', ''))
                    }
                    
                    if article_data['title'] and article_data['url']:
                        articles.append(article_data)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing RSS entry from {source_key}: {str(e)}")
                    continue
            
            self.logger.info(f"Fetched {len(articles)} articles from {source_key} RSS")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching RSS from {source_key}: {str(e)}")
            return []
    
    async def extract_article_content(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract full article content from URL"""
        url = article_data['url']
        source_key = article_data['source']
        
        try:
            domain = urlparse(url).netloc
            await self._rate_limit_request(domain)
            
            # Try newspaper3k first
            try:
                article = Article(url, config=self.newspaper_config)
                article.download()
                article.parse()
                
                if article.text and len(article.text.strip()) > 100:
                    article_data['content'] = article.text.strip()
                    article_data['extraction_method'] = 'newspaper3k'
                    return article_data
                    
            except Exception as e:
                self.logger.debug(f"Newspaper3k failed for {url}: {str(e)}")
            
            # Fallback: Simple content extraction
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract all paragraphs
                    paragraphs = soup.find_all('p')
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if text and len(text) > 20:
                            content_parts.append(text)
                    
                    if content_parts:
                        article_data['content'] = ' '.join(content_parts)
                        article_data['extraction_method'] = 'simple_paragraphs'
                
                return article_data
                
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return article_data
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats from RSS feeds"""
        if not date_str:
            return datetime.now()
        
        # Common date formats in RSS feeds
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S GMT',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return datetime.now()
    
    def _generate_article_hash(self, url: str, title: str) -> str:
        """Generate unique hash for article deduplication"""
        content = f"{url}|{title}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    async def crawl_single_source(self, source_key: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """Crawl a single news source"""
        if source_key not in self.news_sources:
            self.logger.error(f"Unknown news source: {source_key}")
            return []
        
        self.logger.info(f"Crawling single source: {source_key}")
        
        # Fetch RSS
        articles = await self.fetch_rss_feed(source_key)
        articles = articles[:max_articles]
        
        # Extract content
        completed_articles = []
        for article in articles:
            try:
                article_with_content = await self.extract_article_content(article)
                if article_with_content.get('content'):
                    completed_articles.append(article_with_content)
            except Exception as e:
                self.logger.error(f"Failed to extract content: {str(e)}")
        
        self.logger.info(f"Crawled {len(completed_articles)} articles from {source_key}")
        return completed_articles


async def test_news_crawler():
    """Test function for news crawler"""
    
    print("📰 Testing Turkish Financial News Crawler...")
    print("=" * 60)
    
    logging.basicConfig(level=logging.INFO)
    crawler = TurkishNewsCrawler(max_concurrent=3)
    
    async with crawler:
        # Test single source
        print("Testing single source (AA Finansal)...")
        aa_articles = await crawler.crawl_single_source('aa_finansal', max_articles=2)
        
        print(f"✅ Fetched {len(aa_articles)} articles from AA Finansal")
        
        for i, article in enumerate(aa_articles, 1):
            print(f"\n{i}. {article['title'][:80]}...")
            print(f"   Source: {article['source_name']}")
            print(f"   Published: {article['published']}")
            print(f"   Content length: {len(article.get('content', ''))} chars")
            if article.get('content'):
                print(f"   Content preview: {article['content'][:100]}...")
        
        print(f"\n🎉 News crawler test completed!")


if __name__ == "__main__":
    asyncio.run(test_news_crawler())
