#!/usr/bin/env python3
"""
INCREMENTAL IMPORT SYSTEM - MAMUT R600
====================================
Smart import system to add only NEW symbols to avoid duplicates
Prevents re-importing existing A-series data while adding B,C,D... series
"""

import sqlite3
import logging
from pathlib import Path
from excel_to_database_importer import ExcelToDatabaseImporter
from typing import Set, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IncrementalImporter:
    """Smart importer to add only new symbols without duplicates"""
    
    def __init__(self, db_path: str = "enhanced_bist_data.db", excel_dir: str = "data/New_excell_Graph_Sample"):
        self.db_path = db_path
        self.excel_dir = Path(excel_dir)
        self.schema_path = "enhanced_database_schema.sql"
        
        # Initialize database importer
        self.importer = ExcelToDatabaseImporter(db_path)
        
    def get_existing_symbols(self) -> Set[str]:
        """Get symbols already in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT DISTINCT symbol FROM enhanced_stock_data")
            existing_symbols = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            logger.info(f"📊 Found {len(existing_symbols)} existing symbols in database")
            return existing_symbols
        except Exception as e:
            logger.warning(f"Database not found or empty, will create new: {e}")
            return set()
    
    def get_available_symbols(self) -> Set[str]:
        """Get all symbols available in Excel files"""
        excel_files = list(self.excel_dir.glob("*.xlsx"))
        available_symbols = set()
        
        for file in excel_files:
            symbol = file.stem.split('_')[0]
            available_symbols.add(symbol)
        
        logger.info(f"📁 Found {len(available_symbols)} unique symbols in Excel files")
        return available_symbols
    
    def get_new_symbols(self) -> Set[str]:
        """Get symbols that are not yet in database"""
        existing = self.get_existing_symbols()
        available = self.get_available_symbols()
        
        new_symbols = available - existing
        
        if new_symbols:
            logger.info(f"🆕 New symbols to import: {len(new_symbols)}")
            for symbol in sorted(list(new_symbols))[:20]:  # Show first 20
                logger.info(f"  ✨ {symbol}")
            if len(new_symbols) > 20:
                logger.info(f"  ... and {len(new_symbols) - 20} more")
        else:
            logger.info("✅ No new symbols found, all are already imported")
        
        return new_symbols
    
    def get_files_for_symbols(self, symbols: Set[str]) -> List[Path]:
        """Get Excel files for specific symbols"""
        files_to_import = []
        
        for symbol in symbols:
            # Find all files for this symbol (30m, 60m, daily)
            symbol_files = list(self.excel_dir.glob(f"{symbol}_*.xlsx"))
            files_to_import.extend(symbol_files)
        
        return sorted(files_to_import)
    
    def import_new_symbols(self) -> dict:
        """Import only new symbols to avoid duplicates"""
        logger.info("🚀 INCREMENTAL IMPORT STARTING...")
        logger.info("="*60)
        
        # Database is already initialized in constructor
        logger.info("✅ Database ready")
        
        # Get new symbols
        new_symbols = self.get_new_symbols()
        
        if not new_symbols:
            return {
                "success": True,
                "message": "No new symbols to import",
                "imported_files": 0,
                "imported_records": 0
            }
        
        # Get files for new symbols only
        files_to_import = self.get_files_for_symbols(new_symbols)
        
        logger.info(f"📂 Files to import: {len(files_to_import)}")
        
        # Import new files
        total_records = 0
        imported_files = 0
        failed_files = 0
        
        for file_path in files_to_import:
            try:
                records_imported = self.importer.process_single_file(file_path)
                total_records += records_imported
                imported_files += 1
                
                # Log progress every 10 files
                if imported_files % 10 == 0:
                    logger.info(f"📈 Progress: {imported_files}/{len(files_to_import)} files processed")
                    
            except Exception as e:
                logger.error(f"❌ Failed to import {file_path.name}: {e}")
                failed_files += 1
        
        # Final statistics
        logger.info("🎉 INCREMENTAL IMPORT COMPLETED!")
        logger.info("="*50)
        logger.info(f"✅ New symbols added: {len(new_symbols)}")
        logger.info(f"📂 Files imported: {imported_files}")
        logger.info(f"📊 Records imported: {total_records:,}")
        logger.info(f"❌ Failed files: {failed_files}")
        
        # Show database stats
        self.importer.show_database_stats()
        
        return {
            "success": True,
            "new_symbols_count": len(new_symbols),
            "imported_files": imported_files,
            "imported_records": total_records,
            "failed_files": failed_files,
            "new_symbols": list(new_symbols)
        }
    
    def show_import_preview(self):
        """Show what would be imported without actually importing"""
        logger.info("🔍 INCREMENTAL IMPORT PREVIEW")
        logger.info("="*40)
        
        existing = self.get_existing_symbols()
        available = self.get_available_symbols()
        new_symbols = available - existing
        
        logger.info(f"📊 Current database symbols: {len(existing)}")
        logger.info(f"📁 Available Excel symbols: {len(available)}")
        logger.info(f"🆕 New symbols to add: {len(new_symbols)}")
        
        if new_symbols:
            files_to_import = self.get_files_for_symbols(new_symbols)
            logger.info(f"📂 Files to import: {len(files_to_import)}")
            
            # Estimate records
            estimated_records = len(files_to_import) * 4000  # Average records per file
            logger.info(f"📊 Estimated new records: ~{estimated_records:,}")
            
            # Show sample of new symbols
            logger.info("🎯 Sample new symbols:")
            for symbol in sorted(list(new_symbols))[:10]:
                logger.info(f"  ✨ {symbol}")
        
        return len(new_symbols)

def main():
    """Main execution"""
    importer = IncrementalImporter()
    
    print("""
🚀 INCREMENTAL IMPORT SYSTEM - MAMUT R600
========================================
Smart import to add ONLY new symbols (B,C,D... series)
Prevents duplicate imports of existing A-series data
    """)
    
    # Show preview first
    new_count = importer.show_import_preview()
    
    if new_count == 0:
        print("\n✅ No new symbols to import!")
        return
    
    # Ask user confirmation
    print(f"\n❓ Found {new_count} new symbols to import.")
    choice = input("Proceed with incremental import? (y/n): ").lower().strip()
    
    if choice == 'y':
        result = importer.import_new_symbols()
        
        if result["success"]:
            print(f"""
🎉 SUCCESS! INCREMENTAL IMPORT COMPLETED
========================================
✅ New symbols: {result['new_symbols_count']}
📂 Files imported: {result['imported_files']}
📊 Records added: {result['imported_records']:,}
❌ Failed files: {result['failed_files']}

🔥 Database now contains both A-series and new symbols!
            """)
        else:
            print(f"❌ Import failed: {result.get('error', 'Unknown error')}")
    else:
        print("🔄 Import cancelled by user")

if __name__ == "__main__":
    main()
