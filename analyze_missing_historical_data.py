#!/usr/bin/env python3
"""
Railway PostgreSQL Missing Historical Data Analysis
Tespit edilecek eksik hisseler için detaylı analiz
"""

import psycopg2
import psycopg2.extras
import pandas as pd
from typing import Dict, List, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MissingDataAnalyzer:
    def __init__(self):
        # Railway PostgreSQL connection
        self.database_url = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"
        
    def load_bist_indexes(self) -> Dict[str, Set[str]]:
        """Load BIST index files"""
        indexes = {}
        
        try:
            df_100 = pd.read_excel("BIST_100.xlsx")
            indexes['BIST_100'] = set(df_100.iloc[:, 0].dropna().astype(str).str.upper())
            
            df_50 = pd.read_excel("BIST_50.xlsx")
            indexes['BIST_50'] = set(df_50.iloc[:, 0].dropna().astype(str).str.upper())
            
            df_30 = pd.read_excel("BIST_30.xlsx") 
            indexes['BIST_30'] = set(df_30.iloc[:, 0].dropna().astype(str).str.upper())
            
            logger.info(f"✅ Loaded indexes: BIST_100({len(indexes['BIST_100'])}), BIST_50({len(indexes['BIST_50'])}), BIST_30({len(indexes['BIST_30'])})")
            return indexes
            
        except Exception as e:
            logger.error(f"❌ Failed to load BIST index files: {e}")
            return {}

    def analyze_railway_data_coverage(self):
        """Railway database'deki veri coverage analizi"""
        logger.info("🔍 Analyzing Railway database coverage...")
        
        conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        cursor = conn.cursor()
        
        try:
            # Enhanced_stock_data coverage
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT date) as unique_dates,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT timeframe) as timeframes_count
                FROM enhanced_stock_data
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            enhanced_data = {row['symbol']: dict(row) for row in cursor.fetchall()}
            
            # Historical_data coverage
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT DATE(date_time)) as unique_dates,
                    MIN(date_time) as earliest_date,
                    MAX(date_time) as latest_date
                FROM historical_data
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            historical_data = {row['symbol']: dict(row) for row in cursor.fetchall()}
            
            logger.info(f"📊 Enhanced data coverage: {len(enhanced_data)} symbols")
            logger.info(f"📊 Historical data coverage: {len(historical_data)} symbols")
            
            return enhanced_data, historical_data
            
        finally:
            cursor.close()
            conn.close()

    def find_missing_stocks(self, indexes: Dict[str, Set[str]], enhanced_data: dict, historical_data: dict):
        """Eksik hisseleri tespit et"""
        logger.info("🎯 Finding missing stocks for each BIST index...")
        
        missing_analysis = {}
        
        for index_name, index_symbols in indexes.items():
            # Enhanced data'da olan hisseler
            enhanced_available = set(enhanced_data.keys()) & index_symbols
            enhanced_missing = index_symbols - enhanced_available
            
            # Historical data'da olan hisseler  
            historical_available = set(historical_data.keys()) & index_symbols
            historical_missing = index_symbols - historical_available
            
            # Her iki data source'da da olan hisseler (FULL READY)
            fully_available = enhanced_available & historical_available
            
            # Sadece enhanced'da olan (PARTIAL READY)
            enhanced_only = enhanced_available - historical_available
            
            # Sadece historical'da olan (PARTIAL READY)
            historical_only = historical_available - enhanced_available
            
            # Her iki data source'da da eksik (MISSING)
            completely_missing = enhanced_missing & historical_missing
            
            missing_analysis[index_name] = {
                'total_expected': len(index_symbols),
                'fully_available': fully_available,
                'enhanced_only': enhanced_only,
                'historical_only': historical_only,
                'completely_missing': completely_missing,
                'enhanced_missing': enhanced_missing,
                'historical_missing': historical_missing
            }
            
            # Log summary
            logger.info(f"\n📊 {index_name} ANALYSIS:")
            logger.info(f"   📋 Total Expected: {len(index_symbols)}")
            logger.info(f"   ✅ Fully Available: {len(fully_available)}")
            logger.info(f"   🔶 Enhanced Only: {len(enhanced_only)}")
            logger.info(f"   🔷 Historical Only: {len(historical_only)}")
            logger.info(f"   ❌ Completely Missing: {len(completely_missing)}")
            logger.info(f"   📊 Enhanced Missing: {len(enhanced_missing)}")
            logger.info(f"   📊 Historical Missing: {len(historical_missing)}")
            
        return missing_analysis

    def generate_detailed_missing_report(self, indexes: Dict[str, Set[str]], enhanced_data: dict, historical_data: dict, missing_analysis: dict):
        """Detaylı eksik hisse raporu oluştur"""
        logger.info("📋 Generating detailed missing stocks report...")
        
        # Prepare comprehensive data
        all_symbols = set()
        for symbols in indexes.values():
            all_symbols.update(symbols)
            
        detailed_data = []
        
        for symbol in sorted(all_symbols):
            # Check which indexes this symbol belongs to
            in_bist_100 = symbol in indexes['BIST_100']
            in_bist_50 = symbol in indexes['BIST_50'] 
            in_bist_30 = symbol in indexes['BIST_30']
            
            # Check data availability
            has_enhanced = symbol in enhanced_data
            has_historical = symbol in historical_data
            
            # Get data details
            enhanced_records = enhanced_data.get(symbol, {}).get('total_records', 0)
            enhanced_dates = enhanced_data.get(symbol, {}).get('unique_dates', 0)
            enhanced_earliest = enhanced_data.get(symbol, {}).get('earliest_date', 'N/A')
            enhanced_latest = enhanced_data.get(symbol, {}).get('latest_date', 'N/A')
            enhanced_timeframes = enhanced_data.get(symbol, {}).get('timeframes_count', 0)
            
            historical_records = historical_data.get(symbol, {}).get('total_records', 0)
            historical_dates = historical_data.get(symbol, {}).get('unique_dates', 0)
            historical_earliest = historical_data.get(symbol, {}).get('earliest_date', 'N/A')
            historical_latest = historical_data.get(symbol, {}).get('latest_date', 'N/A')
            
            # Status determination
            if has_enhanced and has_historical:
                status = "FULL_READY"
            elif has_enhanced and not has_historical:
                status = "ENHANCED_ONLY"
            elif not has_enhanced and has_historical:
                status = "HISTORICAL_ONLY"
            else:
                status = "MISSING"
            
            detailed_data.append({
                'symbol': symbol,
                'status': status,
                'in_bist_100': in_bist_100,
                'in_bist_50': in_bist_50,
                'in_bist_30': in_bist_30,
                'has_enhanced': has_enhanced,
                'has_historical': has_historical,
                'enhanced_records': enhanced_records,
                'enhanced_dates': enhanced_dates,
                'enhanced_earliest': enhanced_earliest,
                'enhanced_latest': enhanced_latest,
                'enhanced_timeframes': enhanced_timeframes,
                'historical_records': historical_records,
                'historical_dates': historical_dates,
                'historical_earliest': historical_earliest,
                'historical_latest': historical_latest
            })
            
        # Create DataFrame
        df = pd.DataFrame(detailed_data)
        
        # Export to Excel with multiple sheets
        output_file = "bist_missing_data_analysis.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main analysis
            df.to_excel(writer, sheet_name='Complete_Analysis', index=False)
            
            # Missing stocks only
            missing_df = df[df['status'] == 'MISSING']
            missing_df.to_excel(writer, sheet_name='Missing_Stocks', index=False)
            
            # Enhanced only stocks  
            enhanced_only_df = df[df['status'] == 'ENHANCED_ONLY']
            enhanced_only_df.to_excel(writer, sheet_name='Enhanced_Only', index=False)
            
            # Historical only stocks
            historical_only_df = df[df['status'] == 'HISTORICAL_ONLY']
            historical_only_df.to_excel(writer, sheet_name='Historical_Only', index=False)
            
            # Summary by index
            for index_name in ['BIST_100', 'BIST_50', 'BIST_30']:
                index_col = f'in_{index_name.lower()}'
                index_df = df[df[index_col] == True]
                index_df.to_excel(writer, sheet_name=f'{index_name}_Analysis', index=False)
        
        logger.info(f"✅ Detailed analysis saved: {output_file}")
        
        # Print critical missing stocks
        logger.info("\n🚨 CRITICAL MISSING STOCKS (No Historical Data):")
        
        for index_name in ['BIST_30', 'BIST_50', 'BIST_100']:
            index_col = f'in_{index_name.lower()}'
            index_missing = df[(df[index_col] == True) & (df['status'] == 'MISSING')]
            
            if not index_missing.empty:
                logger.info(f"\n❌ {index_name} - Missing Stocks ({len(index_missing)}):")
                for _, row in index_missing.iterrows():
                    logger.info(f"   • {row['symbol']}")
            else:
                logger.info(f"\n✅ {index_name} - No completely missing stocks!")
                
        logger.info("\n🔶 NEEDS HISTORICAL DATA (Enhanced Only):")
        
        for index_name in ['BIST_30', 'BIST_50', 'BIST_100']:
            index_col = f'in_{index_name.lower()}'
            index_enhanced_only = df[(df[index_col] == True) & (df['status'] == 'ENHANCED_ONLY')]
            
            if not index_enhanced_only.empty:
                logger.info(f"\n🔶 {index_name} - Enhanced Only ({len(index_enhanced_only)}):")
                for _, row in index_enhanced_only.iterrows():
                    logger.info(f"   • {row['symbol']} - {row['enhanced_records']:,} records")
                    
        return df

    def create_missing_stocks_list(self, df):
        """Eksik hisseler için action list oluştur"""
        logger.info("\n🎯 CREATING ACTION PLAN FOR MISSING DATA:")
        
        # Completely missing stocks
        completely_missing = df[df['status'] == 'MISSING']
        
        # Historical data missing (enhanced only)
        historical_missing = df[df['status'] == 'ENHANCED_ONLY']
        
        action_plan = {
            'completely_missing': list(completely_missing['symbol'].unique()),
            'historical_missing': list(historical_missing['symbol'].unique())
        }
        
        logger.info(f"\n📋 ACTION PLAN:")
        logger.info(f"   🚨 Completely Missing: {len(action_plan['completely_missing'])} stocks")
        logger.info(f"   🔶 Historical Missing: {len(action_plan['historical_missing'])} stocks")
        logger.info(f"   📊 Total Action Required: {len(action_plan['completely_missing']) + len(action_plan['historical_missing'])} stocks")
        
        # Create user-friendly lists
        logger.info(f"\n🎯 EKSIK HİSSELER LİSTESİ:")
        
        if action_plan['completely_missing']:
            logger.info(f"\n❌ TAM EKSİK HİSSELER ({len(action_plan['completely_missing'])}):")
            logger.info("   (Bu hisselerin hem enhanced hem historical verisi eksik)")
            for i, symbol in enumerate(action_plan['completely_missing'], 1):
                logger.info(f"   {i:2d}. {symbol}")
                
        if action_plan['historical_missing']:
            logger.info(f"\n🔶 TARİHSEL VERİ EKSİK HİSSELER ({len(action_plan['historical_missing'])}):")
            logger.info("   (Bu hisselerin sadece historical verisi eksik, enhanced var)")
            for i, symbol in enumerate(action_plan['historical_missing'], 1):
                logger.info(f"   {i:2d}. {symbol}")
                
        return action_plan

    def run_complete_analysis(self):
        """Complete missing data analysis"""
        logger.info("🚀 RUNNING COMPLETE MISSING DATA ANALYSIS")
        logger.info("=" * 70)
        
        # Load BIST indexes
        indexes = self.load_bist_indexes()
        if not indexes:
            return False
            
        # Analyze Railway coverage
        enhanced_data, historical_data = self.analyze_railway_data_coverage()
        
        # Find missing stocks
        missing_analysis = self.find_missing_stocks(indexes, enhanced_data, historical_data)
        
        # Generate detailed report
        detailed_df = self.generate_detailed_missing_report(indexes, enhanced_data, historical_data, missing_analysis)
        
        # Create action plan
        action_plan = self.create_missing_stocks_list(detailed_df)
        
        logger.info("\n✅ ANALYSIS COMPLETED!")
        logger.info("📁 Check: bist_missing_data_analysis.xlsx")
        
        return action_plan

def main():
    analyzer = MissingDataAnalyzer()
    action_plan = analyzer.run_complete_analysis()
    
    if action_plan:
        print(f"\n🎯 SUMMARY:")
        print(f"❌ Completely Missing: {len(action_plan['completely_missing'])} stocks")
        print(f"🔶 Historical Missing: {len(action_plan['historical_missing'])} stocks")
        print(f"📊 Total Action Required: {len(action_plan['completely_missing']) + len(action_plan['historical_missing'])} stocks")

if __name__ == "__main__":
    main()
