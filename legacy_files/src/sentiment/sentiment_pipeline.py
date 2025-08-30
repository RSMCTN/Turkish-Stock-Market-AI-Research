"""
Real-Time Sentiment Processing Pipeline
Integrates news crawling, sentiment analysis, and entity extraction
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import json

# Import our components
from .turkish_vader import TurkishVaderAnalyzer
from .entity_extractor import BISTEntityExtractor
from ..data.collectors.news_crawler import TurkishNewsCrawler
from ..data.storage.schemas import NewsData, Symbol, MarketData, Base


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
        self.news_crawler = TurkishNewsCrawler(max_concurrent=5)
        
        # Processing stats
        self.stats = {
            'total_articles_processed': 0,
            'articles_with_entities': 0,
            'unique_entities_found': 0,
            'positive_sentiment_articles': 0,
            'negative_sentiment_articles': 0,
            'neutral_sentiment_articles': 0,
            'last_run': None,
        }
        
        self.logger.info("Sentiment Pipeline initialized")
    
    async def run_pipeline(self, max_articles_per_source: int = 10, 
                          save_to_db: bool = True) -> Dict[str, Any]:
        """
        Run complete sentiment processing pipeline
        
        Args:
            max_articles_per_source: Maximum articles to process per news source
            save_to_db: Whether to save results to database
            
        Returns:
            Processing results and statistics
        """
        
        self.logger.info("🚀 Starting sentiment processing pipeline...")
        start_time = datetime.now()
        
        try:
            # Phase 1: Crawl news
            self.logger.info("Phase 1: Crawling news articles...")
            
            async with self.news_crawler:
                articles = await self.news_crawler.crawl_all_sources(max_articles_per_source)
            
            self.logger.info(f"Collected {len(articles)} articles from news sources")
            
            if not articles:
                self.logger.warning("No articles collected, pipeline stopping")
                return {'success': False, 'message': 'No articles found', 'stats': self.stats}
            
            # Phase 2: Process articles (sentiment + entities)
            self.logger.info("Phase 2: Processing sentiment and entities...")
            
            processed_results = []
            
            for i, article in enumerate(articles, 1):
                try:
                    result = await self._process_single_article(article)
                    if result:
                        processed_results.append(result)
                        
                    # Progress logging
                    if i % 5 == 0:
                        self.logger.info(f"Processed {i}/{len(articles)} articles...")
                        
                except Exception as e:
                    self.logger.error(f"Error processing article {i}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully processed {len(processed_results)} articles")
            
            # Phase 3: Save to database
            if save_to_db and processed_results:
                self.logger.info("Phase 3: Saving results to database...")
                saved_count = self._save_to_database(processed_results)
                self.logger.info(f"Saved {saved_count} articles to database")
            
            # Phase 4: Update statistics
            self._update_statistics(processed_results)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Compile results
            pipeline_results = {
                'success': True,
                'execution_time_seconds': execution_time,
                'articles_crawled': len(articles),
                'articles_processed': len(processed_results),
                'articles_saved': saved_count if save_to_db else 0,
                'stats': self.stats.copy(),
                'processed_articles': processed_results[:5],  # First 5 for preview
                'entity_summary': self._generate_entity_summary(processed_results),
                'sentiment_summary': self._generate_sentiment_summary(processed_results)
            }
            
            self.logger.info(f"✅ Pipeline completed successfully in {execution_time:.1f} seconds")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False, 
                'error': str(e), 
                'stats': self.stats,
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    async def _process_single_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single article through sentiment and entity analysis"""
        
        if not article.get('content') or len(article['content']) < 50:
            return None
        
        try:
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(article['content'])
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(article['content'])
            
            # Combine results
            processed_article = {
                'title': article['title'],
                'url': article['url'],
                'source': article['source'],
                'source_name': article['source_name'],
                'published_at': article['published'],
                'content': article['content'],
                'content_length': len(article['content']),
                
                # Sentiment analysis results
                'sentiment': {
                    'compound': sentiment_result['compound'],
                    'positive': sentiment_result['pos'],
                    'negative': sentiment_result['neg'],
                    'neutral': sentiment_result['neu'],
                    'confidence': sentiment_result['confidence']
                },
                
                # Entity extraction results
                'entities': [
                    {
                        'symbol': e.symbol,
                        'matched_text': e.matched_text,
                        'confidence': e.confidence,
                        'company_info': self.entity_extractor.company_database.get(e.symbol, {})
                    }
                    for e in entities
                ],
                'entity_count': len(entities),
                
                # Processing metadata
                'processed_at': datetime.now(),
                'hash': article.get('hash', ''),
                'extraction_method': article.get('extraction_method', 'unknown')
            }
            
            return processed_article
            
        except Exception as e:
            self.logger.error(f"Error processing article '{article.get('title', 'Unknown')}': {str(e)}")
            return None
    
    def _save_to_database(self, processed_articles: List[Dict[str, Any]]) -> int:
        """Save processed articles to database"""
        
        session = self.Session()
        saved_count = 0
        
        try:
            for article in processed_articles:
                try:
                    # Check if article already exists
                    existing = session.query(NewsData).filter_by(
                        url=article['url']
                    ).first()
                    
                    if existing:
                        continue  # Skip duplicates
                    
                    # Create NewsData entry
                    news_entry = NewsData(
                        title=article['title'],
                        content=article['content'],
                        source=article['source'],
                        url=article['url'],
                        published_at=article['published_at'],
                        scraped_at=article['processed_at'],
                        
                        # Entity symbols (comma-separated)
                        symbols=','.join([e['symbol'] for e in article['entities']]) if article['entities'] else None,
                        
                        # Sentiment scores
                        compound_score=article['sentiment']['compound'],
                        positive_score=article['sentiment']['positive'],
                        negative_score=article['sentiment']['negative'],
                        neutral_score=article['sentiment']['neutral'],
                        
                        # Additional metadata
                        language='tr',
                        is_processed=True
                    )
                    
                    session.add(news_entry)
                    saved_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error saving article to DB: {str(e)}")
                    session.rollback()
                    continue
            
            session.commit()
            
        except Exception as e:
            self.logger.error(f"Database save error: {str(e)}")
            session.rollback()
        finally:
            session.close()
        
        return saved_count
    
    def _update_statistics(self, processed_articles: List[Dict[str, Any]]):
        """Update processing statistics"""
        
        self.stats['total_articles_processed'] += len(processed_articles)
        self.stats['last_run'] = datetime.now()
        
        entities_found = set()
        
        for article in processed_articles:
            # Count articles with entities
            if article['entity_count'] > 0:
                self.stats['articles_with_entities'] += 1
            
            # Count unique entities
            for entity in article['entities']:
                entities_found.add(entity['symbol'])
            
            # Count sentiment categories
            compound = article['sentiment']['compound']
            if compound > 0.05:
                self.stats['positive_sentiment_articles'] += 1
            elif compound < -0.05:
                self.stats['negative_sentiment_articles'] += 1
            else:
                self.stats['neutral_sentiment_articles'] += 1
        
        self.stats['unique_entities_found'] = len(entities_found)
    
    def _generate_entity_summary(self, processed_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of entities found"""
        
        entity_mentions = {}
        entity_sentiments = {}
        
        for article in processed_articles:
            article_sentiment = article['sentiment']['compound']
            
            for entity in article['entities']:
                symbol = entity['symbol']
                
                # Count mentions
                if symbol not in entity_mentions:
                    entity_mentions[symbol] = 0
                entity_mentions[symbol] += 1
                
                # Track sentiment for each entity
                if symbol not in entity_sentiments:
                    entity_sentiments[symbol] = []
                entity_sentiments[symbol].append(article_sentiment)
        
        # Calculate average sentiment per entity
        entity_avg_sentiment = {}
        for symbol, sentiments in entity_sentiments.items():
            entity_avg_sentiment[symbol] = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Sort by mention count
        top_entities = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_unique_entities': len(entity_mentions),
            'top_mentioned_entities': [
                {
                    'symbol': symbol,
                    'mentions': count,
                    'avg_sentiment': round(entity_avg_sentiment.get(symbol, 0), 3),
                    'company_name': self.entity_extractor.company_database.get(symbol, {}).get('full_name', symbol)
                }
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
            'most_positive': round(max(sentiments), 3),
            'most_negative': round(min(sentiments), 3),
            'sentiment_distribution': {
                'very_positive': len([s for s in sentiments if s > 0.5]),
                'positive': len([s for s in sentiments if 0.05 < s <= 0.5]),
                'neutral': len([s for s in sentiments if -0.05 <= s <= 0.05]),
                'negative': len([s for s in sentiments if -0.5 <= s < -0.05]),
                'very_negative': len([s for s in sentiments if s < -0.5])
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        return self.stats.copy()
    
    async def run_continuous_pipeline(self, interval_minutes: int = 30, max_articles: int = 5):
        """Run pipeline continuously at specified intervals"""
        
        self.logger.info(f"Starting continuous pipeline (every {interval_minutes} minutes)")
        
        while True:
            try:
                self.logger.info("Running scheduled pipeline execution...")
                results = await self.run_pipeline(max_articles_per_source=max_articles)
                
                if results['success']:
                    self.logger.info(f"Pipeline execution completed: {results['articles_processed']} articles processed")
                else:
                    self.logger.error(f"Pipeline execution failed: {results.get('error', 'Unknown error')}")
                
                # Wait for next execution
                self.logger.info(f"Waiting {interval_minutes} minutes until next execution...")
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Continuous pipeline stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Continuous pipeline error: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


async def test_sentiment_pipeline():
    """Test function for sentiment pipeline"""
    
    print("🔄 Testing Sentiment Processing Pipeline...")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create pipeline
    pipeline = SentimentPipeline("sqlite:///test_sentiment_pipeline.db")
    
    # Run pipeline with limited articles
    results = await pipeline.run_pipeline(max_articles_per_source=3, save_to_db=True)
    
    if results['success']:
        print(f"\n✅ Pipeline executed successfully!")
        print(f"   Execution time: {results['execution_time_seconds']:.1f} seconds")
        print(f"   Articles crawled: {results['articles_crawled']}")
        print(f"   Articles processed: {results['articles_processed']}")
        print(f"   Articles saved: {results['articles_saved']}")
        
        # Display sentiment summary
        sentiment_summary = results['sentiment_summary']
        print(f"\n📊 Sentiment Summary:")
        print(f"   Average sentiment: {sentiment_summary['average_sentiment']}")
        print(f"   Positive articles: {sentiment_summary['positive_articles']}")
        print(f"   Negative articles: {sentiment_summary['negative_articles']}")
        print(f"   Neutral articles: {sentiment_summary['neutral_articles']}")
        
        # Display entity summary
        entity_summary = results['entity_summary']
        print(f"\n🏢 Entity Summary:")
        print(f"   Unique entities found: {entity_summary['total_unique_entities']}")
        
        if entity_summary['top_mentioned_entities']:
            print("   Top mentioned companies:")
            for entity in entity_summary['top_mentioned_entities'][:5]:
                print(f"     • {entity['symbol']} ({entity['company_name']}): {entity['mentions']} mentions, avg sentiment: {entity['avg_sentiment']}")
        
        # Display sample processed articles
        if results['processed_articles']:
            print(f"\n📰 Sample Processed Articles:")
            for i, article in enumerate(results['processed_articles'][:2], 1):
                print(f"\n{i}. {article['title'][:80]}...")
                print(f"   Source: {article['source_name']}")
                print(f"   Sentiment: {article['sentiment']['compound']:.3f} (confidence: {article['sentiment']['confidence']:.3f})")
                if article['entities']:
                    entities_str = ', '.join([f"{e['symbol']}({e['confidence']:.2f})" for e in article['entities']])
                    print(f"   Entities: {entities_str}")
    
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Pipeline test completed!")


if __name__ == "__main__":
    asyncio.run(test_sentiment_pipeline())
