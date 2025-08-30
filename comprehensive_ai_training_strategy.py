#!/usr/bin/env python3
"""
COMPREHENSIVE AI TRAINING STRATEGY - MAMUT R600
==============================================
Advanced AI training strategy for 840,782+ BIST records with 42 technical indicators
Multi-model approach: Turkish Q&A + DP-LSTM + Technical Analysis + Sentiment

Training Data Size: 840K+ records, 69 symbols, 24+ years, 4 timeframes
Technical Indicators: 35+ advanced indicators per record
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import asyncio
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfiguration:
    """Training configuration for all AI models"""
    
    # Data Configuration
    database_path: str = "enhanced_bist_data.db"
    partition_dir: str = "data/partitions"
    training_data_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Model Configuration
    turkish_bert_model: str = "dbmdz/bert-base-turkish-cased"
    huggingface_token: str = "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"
    wandb_token: str = "ecb17bf7bef21def85810e689203d279a59839ff"
    
    # Training Parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_sequence_length: int = 512
    
    # Technical Analysis Configuration
    lookback_window: int = 60  # 60 periods for technical analysis
    prediction_horizon: int = 5  # Predict next 5 periods
    
    # Model Output Paths
    output_dir: str = "trained_models"
    turkish_qa_model_name: str = "rsmctn/bist-turkish-qa-enhanced-v2"
    dp_lstm_model_name: str = "rsmctn/bist-dp-lstm-enhanced-v2"
    technical_model_name: str = "rsmctn/bist-technical-analysis-v2"

class BISTDataProcessor:
    """Advanced data processor for BIST AI training"""
    
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.conn = sqlite3.connect(config.database_path)
        
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training data statistics"""
        stats = {}
        
        # Basic statistics
        total_records = self.conn.execute("SELECT COUNT(*) FROM enhanced_stock_data").fetchone()[0]
        symbols = self.conn.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data").fetchone()[0]
        
        # Date range
        date_range = self.conn.execute("""
            SELECT MIN(date) as min_date, MAX(date) as max_date 
            FROM enhanced_stock_data
        """).fetchone()
        
        # Timeframe distribution
        timeframes = self.conn.execute("""
            SELECT timeframe, COUNT(*) as count 
            FROM enhanced_stock_data 
            GROUP BY timeframe
        """).fetchall()
        
        # Technical indicator completeness
        technical_completeness = self.conn.execute("""
            SELECT 
                AVG(CASE WHEN rsi_14 > 0 THEN 1 ELSE 0 END) * 100 as rsi_completeness,
                AVG(CASE WHEN macd_26_12 != 0 THEN 1 ELSE 0 END) * 100 as macd_completeness,
                AVG(CASE WHEN bol_upper_20_2 > 0 THEN 1 ELSE 0 END) * 100 as bollinger_completeness,
                AVG(CASE WHEN tenkan_sen > 0 THEN 1 ELSE 0 END) * 100 as ichimoku_completeness
            FROM enhanced_stock_data
        """).fetchone()
        
        stats = {
            'total_records': total_records,
            'symbols': symbols,
            'date_range': date_range,
            'timeframes': dict(timeframes),
            'technical_completeness': {
                'rsi': round(technical_completeness[0], 1),
                'macd': round(technical_completeness[1], 1),
                'bollinger': round(technical_completeness[2], 1),
                'ichimoku': round(technical_completeness[3], 1)
            }
        }
        
        return stats
    
    def prepare_turkish_qa_dataset(self) -> List[Dict[str, str]]:
        """Prepare Turkish Q&A dataset from BIST data"""
        logger.info("🔍 Preparing Turkish Q&A dataset...")
        
        qa_pairs = []
        
        # Get sample data for different scenarios
        samples = self.conn.execute("""
            SELECT symbol, date, close, rsi_14, macd_26_12, bol_upper_20_2, bol_lower_20_2,
                   tenkan_sen, kijun_sen, atr_14, adx_14, volume
            FROM enhanced_stock_data 
            WHERE rsi_14 > 0 AND macd_26_12 != 0 
            ORDER BY date DESC, symbol
            LIMIT 5000
        """).fetchall()
        
        for sample in samples:
            symbol, date, close, rsi, macd, bol_upper, bol_lower, tenkan, kijun, atr, adx, volume = sample
            
            # Generate diverse Q&A pairs
            
            # RSI based questions
            if rsi > 70:
                qa_pairs.append({
                    "question": f"{symbol} hissesi aşırı alım bölgesinde mi?",
                    "context": f"{symbol} hissesi {date} tarihinde ₺{close:.2f} fiyatında, RSI değeri {rsi:.1f}. RSI 70'in üzerinde olduğu için aşırı alım bölgesindedir.",
                    "answer": f"Evet, {symbol} hissesi aşırı alım bölgesinde. RSI değeri {rsi:.1f} ile 70'in üzerinde."
                })
            elif rsi < 30:
                qa_pairs.append({
                    "question": f"{symbol} hissesi alım fırsatı veriyor mu?",
                    "context": f"{symbol} hissesi {date} tarihinde ₺{close:.2f}, RSI {rsi:.1f}. RSI 30'un altında olduğu için aşırı satım bölgesinde ve potansiel alım fırsatı sunuyor.",
                    "answer": f"Evet, {symbol} RSI {rsi:.1f} ile aşırı satım bölgesinde, alım fırsatı sunabilir."
                })
            
            # MACD based questions
            macd_signal = "pozitif" if macd > 0 else "negatif"
            qa_pairs.append({
                "question": f"{symbol} MACD sinyali nasıl?",
                "context": f"{symbol} hissesi {date} tarihinde MACD değeri {macd:.4f}. Bu {macd_signal} sinyal veriyor.",
                "answer": f"{symbol} MACD sinyali {macd_signal}. Değer: {macd:.4f}"
            })
            
            # Bollinger Bands questions
            if close > bol_upper:
                qa_pairs.append({
                    "question": f"{symbol} Bollinger Bandlarında nasıl konumlanmış?",
                    "context": f"{symbol} fiyatı ₺{close:.2f}, Bollinger üst bandı ₺{bol_upper:.2f}. Fiyat üst bandın üzerinde.",
                    "answer": f"{symbol} Bollinger üst bandının üzerinde. Fiyat: ₺{close:.2f}, Üst band: ₺{bol_upper:.2f}"
                })
            elif close < bol_lower:
                qa_pairs.append({
                    "question": f"{symbol} hangi Bollinger band seviyesinde?",
                    "context": f"{symbol} fiyatı ₺{close:.2f}, Bollinger alt bandı ₺{bol_lower:.2f}. Fiyat alt bandın altında.",
                    "answer": f"{symbol} Bollinger alt bandının altında. Fiyat: ₺{close:.2f}, Alt band: ₺{bol_lower:.2f}"
                })
            
            # Ichimoku questions
            if tenkan > kijun:
                qa_pairs.append({
                    "question": f"{symbol} Ichimoku trendi nasıl?",
                    "context": f"{symbol} Ichimoku analizi: Tenkan-sen ₺{tenkan:.2f}, Kijun-sen ₺{kijun:.2f}. Tenkan-sen üstte olduğu için yükseliş trendi.",
                    "answer": f"{symbol} Ichimoku yükseliş trendinde. Tenkan: ₺{tenkan:.2f} > Kijun: ₺{kijun:.2f}"
                })
            
            # Volume analysis
            if volume > 1000000:
                qa_pairs.append({
                    "question": f"{symbol} hacim durumu nasıl?",
                    "context": f"{symbol} {date} tarihinde {volume:,} adet işlem hacmi. Bu yüksek bir hacim seviyesi.",
                    "answer": f"{symbol} yüksek hacimle işlem görüyor: {volume:,} adet"
                })
            
            # Volatility questions
            volatility_level = "yüksek" if atr > 2.0 else "orta" if atr > 1.0 else "düşük"
            qa_pairs.append({
                "question": f"{symbol} volatilitesi nasıl?",
                "context": f"{symbol} ATR değeri {atr:.2f}. Bu {volatility_level} volatilite seviyesi anlamına gelir.",
                "answer": f"{symbol} volatilitesi {volatility_level}. ATR: {atr:.2f}"
            })
            
        logger.info(f"✅ {len(qa_pairs)} Turkish Q&A pairs generated")
        return qa_pairs[:3000]  # Limit for training efficiency
    
    def prepare_technical_analysis_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare technical analysis dataset for DP-LSTM training"""
        logger.info("📈 Preparing technical analysis dataset...")
        
        # Get sequential data for each symbol
        symbols = [row[0] for row in self.conn.execute("SELECT DISTINCT symbol FROM enhanced_stock_data").fetchall()]
        
        X_data = []
        y_data = []
        
        for symbol in symbols[:10]:  # Limit to first 10 symbols for demo
            # Get sequential data for this symbol
            data = pd.read_sql_query("""
                SELECT date, close, open, high, low, volume,
                       rsi_14, macd_26_12, atr_14, adx_14, 
                       stochastic_k_5, bol_upper_20_2, bol_middle_20_2, bol_lower_20_2,
                       tenkan_sen, kijun_sen
                FROM enhanced_stock_data 
                WHERE symbol = ? AND timeframe = '60m'
                ORDER BY date, time
            """, self.conn, params=(symbol,))
            
            if len(data) < self.config.lookback_window + self.config.prediction_horizon:
                continue
            
            # Normalize data
            numeric_columns = ['close', 'open', 'high', 'low', 'volume', 'rsi_14', 'macd_26_12', 
                             'atr_14', 'adx_14', 'stochastic_k_5', 'bol_upper_20_2', 
                             'bol_middle_20_2', 'bol_lower_20_2', 'tenkan_sen', 'kijun_sen']
            
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = (data[col] - data[col].mean()) / (data[col].std() + 1e-8)
            
            # Create sequences
            for i in range(len(data) - self.config.lookback_window - self.config.prediction_horizon + 1):
                # Features: technical indicators for lookback window
                X_sequence = data.iloc[i:i+self.config.lookback_window][numeric_columns].values
                
                # Target: price movement in next prediction_horizon periods
                future_prices = data.iloc[i+self.config.lookback_window:i+self.config.lookback_window+self.config.prediction_horizon]['close'].values
                current_price = data.iloc[i+self.config.lookback_window-1]['close']
                
                # Predict price change percentage
                price_change = (future_prices[-1] - current_price) / (abs(current_price) + 1e-8)
                
                X_data.append(X_sequence)
                y_data.append(price_change)
        
        X_array = np.array(X_data)
        y_array = np.array(y_data)
        
        logger.info(f"✅ Technical analysis dataset: X{X_array.shape}, y{y_array.shape}")
        return X_array, y_array

class TurkishQADataset(Dataset):
    """PyTorch Dataset for Turkish Q&A training"""
    
    def __init__(self, qa_pairs: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        
        # Tokenize question and context
        encoding = self.tokenizer(
            qa_pair['question'],
            qa_pair['context'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Find answer positions in context
        answer_start = qa_pair['context'].find(qa_pair['answer'])
        if answer_start == -1:
            answer_start = 0
        
        answer_end = answer_start + len(qa_pair['answer'])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(answer_start, dtype=torch.long),
            'end_positions': torch.tensor(answer_end, dtype=torch.long)
        }

class ComprehensiveAITrainer:
    """Main trainer for all AI models"""
    
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.data_processor = BISTDataProcessor(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def train_turkish_qa_model(self):
        """Train enhanced Turkish Q&A model"""
        logger.info("🤖 Starting Turkish Q&A model training...")
        
        # Prepare data
        qa_pairs = self.data_processor.prepare_turkish_qa_dataset()
        
        # Split data
        train_size = int(len(qa_pairs) * self.config.training_data_ratio)
        val_size = int(len(qa_pairs) * self.config.validation_ratio)
        
        train_data = qa_pairs[:train_size]
        val_data = qa_pairs[train_size:train_size+val_size]
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.turkish_bert_model)
        model = AutoModelForQuestionAnswering.from_pretrained(self.config.turkish_bert_model)
        
        # Create datasets
        train_dataset = TurkishQADataset(train_data, tokenizer, self.config.max_sequence_length)
        val_dataset = TurkishQADataset(val_data, tokenizer, self.config.max_sequence_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/turkish_qa",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=500,
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=True,
            hub_model_id=self.config.turkish_qa_model_name,
            hub_token=self.config.huggingface_token,
            report_to="wandb"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(f"{self.config.output_dir}/turkish_qa")
        
        logger.info("✅ Turkish Q&A model training completed")
        
    def train_dp_lstm_model(self):
        """Train enhanced DP-LSTM model for price prediction"""
        logger.info("📊 Starting DP-LSTM model training...")
        
        # Prepare technical analysis dataset
        X_data, y_data = self.data_processor.prepare_technical_analysis_dataset()
        
        # Split data
        train_size = int(len(X_data) * self.config.training_data_ratio)
        val_size = int(len(X_data) * self.config.validation_ratio)
        
        X_train = X_data[:train_size]
        X_val = X_data[train_size:train_size+val_size]
        X_test = X_data[train_size+val_size:]
        
        y_train = y_data[:train_size]
        y_val = y_data[train_size:train_size+val_size]
        y_test = y_data[train_size+val_size:]
        
        # Build DP-LSTM model (simplified version)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Save model
        model.save(f"{self.config.output_dir}/dp_lstm_enhanced.h5")
        
        # Evaluate on test set
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"✅ DP-LSTM training completed. Test Loss: {test_loss[0]:.4f}")
        
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        stats = self.data_processor.get_training_statistics()
        
        report = {
            'training_date': datetime.now().isoformat(),
            'data_statistics': stats,
            'models_trained': [
                {
                    'name': 'Turkish Q&A Enhanced',
                    'type': 'BERT-based Question Answering',
                    'base_model': self.config.turkish_bert_model,
                    'training_samples': '~3,000 Q&A pairs',
                    'output_path': f"{self.config.output_dir}/turkish_qa",
                    'huggingface_model': self.config.turkish_qa_model_name
                },
                {
                    'name': 'DP-LSTM Enhanced',
                    'type': 'Long Short-Term Memory for Price Prediction',
                    'features': '15 technical indicators',
                    'sequence_length': self.config.lookback_window,
                    'prediction_horizon': self.config.prediction_horizon,
                    'output_path': f"{self.config.output_dir}/dp_lstm_enhanced.h5"
                }
            ],
            'technical_indicators_used': [
                'RSI (14)', 'MACD (26,12)', 'ATR (14)', 'ADX (14)',
                'Stochastic (5,3)', 'Bollinger Bands (20,2)',
                'Ichimoku Cloud (Tenkan, Kijun)', 'Volume Analysis'
            ],
            'training_configuration': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.num_epochs,
                'sequence_length': self.config.max_sequence_length
            }
        }
        
        return report

def main():
    """Main training pipeline"""
    config = TrainingConfiguration()
    trainer = ComprehensiveAITrainer(config)
    
    logger.info("🚀 COMPREHENSIVE AI TRAINING PIPELINE STARTING...")
    logger.info("="*60)
    
    # Get training statistics
    stats = trainer.data_processor.get_training_statistics()
    
    print(f"""
    📊 TRAINING DATA OVERVIEW
    ========================
    📈 Total records: {stats['total_records']:,}
    🏢 Symbols: {stats['symbols']}
    📅 Date range: {stats['date_range'][0]} → {stats['date_range'][1]}
    ⏰ Timeframes: {', '.join([f"{k}({v:,})" for k, v in stats['timeframes'].items()])}
    
    🔧 Technical Indicator Completeness:
    • RSI: {stats['technical_completeness']['rsi']}%
    • MACD: {stats['technical_completeness']['macd']}%
    • Bollinger: {stats['technical_completeness']['bollinger']}%
    • Ichimoku: {stats['technical_completeness']['ichimoku']}%
    """)
    
    # Training menu
    print("\n🤖 AI TRAINING OPTIONS:")
    print("1. Train Turkish Q&A Model (Enhanced)")
    print("2. Train DP-LSTM Model (Enhanced)")
    print("3. Train Both Models")
    print("4. Generate Training Report Only")
    
    choice = input("\nSelect training option (1-4): ").strip()
    
    if choice == "1":
        trainer.train_turkish_qa_model()
    elif choice == "2":
        trainer.train_dp_lstm_model()
    elif choice == "3":
        trainer.train_turkish_qa_model()
        trainer.train_dp_lstm_model()
    elif choice == "4":
        report = trainer.generate_training_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Generate final report
    final_report = trainer.generate_training_report()
    
    # Save report
    with open(f"{config.output_dir}/training_report.json", 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Training completed! Report saved to {config.output_dir}/training_report.json")

if __name__ == "__main__":
    main()
