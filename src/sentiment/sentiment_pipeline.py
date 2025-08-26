"""
Real-Time Sentiment Processing Pipeline
Integrates news crawling, sentiment analysis, and entity extraction
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import our components
from src.sentiment.turkish_vader import TurkishVaderAnalyzer
from src.sentiment.entity_extractor import BISTEntityExtractor
from src.data.collectors.news_crawler import TurkishNewsCrawler
from src.data.storage.schemas import NewsData, Symbol, MarketData, Base


class SentimentPipeline:
    """
    Real-time sentiment processing pipeline
    Orchestrates news collection, sentiment analysis, and entity extraction
    """
    
    def __init__(self, database_url: str = "sqlite:///mamut_r600_sentiment.db"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.sentiment_analyzer = TurkishVaderAnalyzer()
        self.entity_extractor = BISTEntityExtractor()
        self.news_crawler = TurkishNewsCrawler(max_concurrent=3)
        
        # Processing stats
        self.stats = {
            'total_articles_processed': 0,
            'articles_with_entities': 0,
            'last_run': None,
        }
        
        self.logger.info("Sentiment Pipeline initialized")
    
    async def run_pipeline(self, max_articles_per_source: int = 5, 
                          save_to_db: bool = True) -> Dict[str, Any]:
        """Run complete sentiment processing pipeline"""
        
        self.logger.info("🚀 Starting sentiment processing pipeline...")
        start_time = datetime.now()
        
        try:
            # Phase 1: Crawl news (simplified for demo)
            self.logger.info("Phase 1: Crawling news articles...")
            
            async with self.news_crawler:
                articles = await self.news_crawler.crawl_single_source('aa_finansal', max_articles_per_source)
            
            self.logger.info(f"Collected {len(articles)} articles")
            
            if not articles:
                return {'success': False, 'message': 'No articles found', 'stats': self.stats}
            
            # Phase 2: Process articles
            self.logger.info("Phase 2: Processing sentiment and entities...")
            
            processed_results = []
            
            for article in articles:
                try:
                    result = self._process_single_article(article)
                    if result:
                        processed_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing article: {str(e)}")
                    continue
            
            # Phase 3: Save to database
            saved_count = 0
            if save_to_db and processed_results:
                saved_count = self._save_to_database(processed_results)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Generate summaries
            entity_summary = self._generate_entity_summary(processed_results)
            sentiment_summary = self._generate_sentiment_summary(processed_results)
            
            return {
                'success': True,
                'execution_time_seconds': execution_time,
                'articles_crawled': len(articles),
                'articles_processed': len(processed_results),
                'articles_saved': saved_count,
                'processed_articles': processed_results[:3],  # First 3 for preview
                'entity_summary': entity_summary,
                'sentiment_summary': sentiment_summary
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False, 
                'error': str(e), 
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def _process_single_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single article through sentiment and entity analysis"""
        
        if not article.get('content') or len(article['content']) < 50:
            return None
        
        try:
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(article['content'])
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(article['content'])
            
            return {
                'title': article['title'],
                'url': article['url'],
                'source': article['source'],
                'published_at': article['published'],
                'content_length': len(article['content']),
                'sentiment': {
                    'compound': sentiment_result['compound'],
                    'positive': sentiment_result['pos'],
                    'negative': sentiment_result['neg'],
                    'neutral': sentiment_result['neu'],
                    'confidence': sentiment_result['confidence']
                },
                'entities': [
                    {
                        'symbol': e.symbol,
                        'matched_text': e.matched_text,
                        'confidence': e.confidence,
                    }
                    for e in entities
                ],
                'entity_count': len(entities),
                'processed_at': datetime.now(),
            }
            
        except Exception as e:
            self.logger.error(f"Error processing article: {str(e)}")
            return None
    
    def _save_to_database(self, processed_articles: List[Dict[str, Any]]) -> int:
        """Save processed articles to database"""
        
        session = self.Session()
        saved_count = 0
        
        try:
            for article in processed_articles:
                try:
                    # Check if article already exists
                    existing = session.query(NewsData).filter_by(url=article['url']).first()
                    
                    if existing:
                        continue
                    
                    # Create NewsData entry
                    news_entry = NewsData(
                        title=article['title'],
                        content="Content processed",  # Simplified
                        source=article['source'],
                        url=article['url'],
                        published_at=article['published_at'],
                        compound_score=article['sentiment']['compound'],
                        positive_score=article['sentiment']['positive'],
                        negative_score=article['sentiment']['negative'],
                        neutral_score=article['sentiment']['neutral'],
                        symbols=','.join([e['symbol'] for e in article['entities']]) if article['entities'] else None,
                        language='tr',
                        is_processed=True
                    )
                    
                    session.add(news_entry)
                    saved_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error saving article: {str(e)}")
                    continue
            
            session.commit()
            
        except Exception as e:
            self.logger.error(f"Database save error: {str(e)}")
            session.rollback()
        finally:
            session.close()
        
        return saved_count
    
    def _generate_entity_summary(self, processed_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of entities found"""
        
        entity_mentions = {}
        
        for article in processed_articles:
            for entity in article['entities']:
                symbol = entity['symbol']
                if symbol not in entity_mentions:
                    entity_mentions[symbol] = 0
                entity_mentions[symbol] += 1
        
        top_entities = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_unique_entities': len(entity_mentions),
            'top_mentioned_entities': [
                {'symbol': symbol, 'mentions': count}
                for symbol, count in top_entities
            ]
        }
    
    def _generate_sentiment_summary(self, processed_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate sentiment analysis summary"""
        
        if not processed_articles:
            return {'error': 'No articles to analyze'}
        
        sentiments = [article['sentiment']['compound'] for article in processed_articles]
        
        return {
            'total_articles': len(processed_articles),
            'average_sentiment': round(sum(sentiments) / len(sentiments), 3),
            'positive_articles': len([s for s in sentiments if s > 0.05]),
            'negative_articles': len([s for s in sentiments if s < -0.05]),
            'neutral_articles': len([s for s in sentiments if -0.05 <= s <= 0.05]),
        }


async def test_sentiment_pipeline():
    """Test function for sentiment pipeline"""
    
    print("🔄 Testing Sentiment Processing Pipeline...")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create pipeline
    pipeline = SentimentPipeline("sqlite:///test_sentiment.db")
    
    # Run pipeline
    results = await pipeline.run_pipeline(max_articles_per_source=3, save_to_db=True)
    
    if results['success']:
        print(f"\n✅ Pipeline executed successfully!")
        print(f"   Execution time: {results['execution_time_seconds']:.1f} seconds")
        print(f"   Articles processed: {results['articles_processed']}")
        print(f"   Articles saved: {results['articles_saved']}")
        
        # Display sentiment summary
        sentiment = results['sentiment_summary']
        print(f"\n📊 Sentiment Summary:")
        print(f"   Average sentiment: {sentiment['average_sentiment']}")
        print(f"   Positive: {sentiment['positive_articles']}, Negative: {sentiment['negative_articles']}, Neutral: {sentiment['neutral_articles']}")
        
        # Display entity summary
        entities = results['entity_summary']
        print(f"\n🏢 Entities found: {entities['total_unique_entities']}")
        for entity in entities['top_mentioned_entities']:
            print(f"   • {entity['symbol']}: {entity['mentions']} mentions")
        
        # Display sample articles
        print(f"\n📰 Sample Articles:")
        for i, article in enumerate(results['processed_articles'], 1):
            print(f"\n{i}. {article['title'][:70]}...")
            print(f"   Sentiment: {article['sentiment']['compound']:.3f}")
            if article['entities']:
                print(f"   Entities: {[e['symbol'] for e in article['entities']]}")
    
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Pipeline test completed!")


if __name__ == "__main__":
    asyncio.run(test_sentiment_pipeline())
