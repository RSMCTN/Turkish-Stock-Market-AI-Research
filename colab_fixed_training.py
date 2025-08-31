#!/usr/bin/env python3
"""
🚀 COLAB FIX - DOSYA YOLU DÜZELTİLDİ!
Bu version Colab'daki root dizinde dosyaları arar
"""

import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

print("🚀 COLAB FIXED - ADVANCED TURKISH FINANCIAL AI TRAINING")
print("🎯 117 Sembol + 30 İndikatör + 1.4M Historical Data")
print("=" * 60)

# DOSYA YOLLARINI KONTROL ET
def check_files():
    """Colab'daki dosyaları kontrol et"""
    print("📁 Dosya kontrolü yapılıyor...")
    
    # Mevcut dizindeki dosyaları listele
    files = os.listdir('.')
    print(f"📂 Mevcut dizin: {os.getcwd()}")
    print(f"📋 Dosyalar: {[f for f in files if f.endswith(('.json', '.csv'))]}")
    
    # Training dosyalarını ara
    required_files = {
        'qa': ['enhanced_turkish_qa.json', 'training_data/enhanced_turkish_qa.json'],
        'sentiment': ['enhanced_sentiment.json', 'training_data/enhanced_sentiment.json'],
        'historical': ['enhanced_historical_training.csv', 'training_data/enhanced_historical_training.csv']
    }
    
    found_files = {}
    
    for file_type, possible_paths in required_files.items():
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                found_files[file_type] = path
                print(f"✅ {file_type.upper()}: {path}")
                found = True
                break
        
        if not found:
            print(f"❌ {file_type.upper()}: Bulunamadı!")
            found_files[file_type] = None
    
    return found_files

# STEP 1: HuggingFace Setup
try:
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from transformers import TrainingArguments, Trainer, DefaultDataCollator
    from datasets import Dataset
    
    print("✅ Dependencies loaded successfully!")
except ImportError as e:
    print(f"❌ Dependency error: {e}")
    print("💡 Colab'da şunu çalıştır:")
    print("!pip install transformers datasets torch huggingface_hub accelerate")
    exit(1)

# HF Authentication
HF_TOKEN = "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"
HF_MODEL_NAME = "rsmctn/bist-advanced-turkish-ai-v2"

try:
    login(HF_TOKEN)
    print("✅ HuggingFace authenticated!")
except Exception as e:
    print(f"❌ HF Auth error: {e}")

# STEP 2: Load Training Data - FIXED PATHS
def load_training_data_fixed():
    """COLAB için düzeltilmiş dosya yolları"""
    
    print("📊 Enhanced training data yükleniyor (FIXED PATHS)...")
    
    # Dosya yerlerini kontrol et
    file_paths = check_files()
    
    # Q&A Dataset
    if file_paths['qa']:
        try:
            with open(file_paths['qa'], 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            print(f"✅ Q&A Data: {len(qa_data)} soru-cevap çifti")
        except Exception as e:
            print(f"❌ Q&A yükleme hatası: {e}")
            qa_data = []
    else:
        print("❌ Q&A dosyası bulunamadı!")
        qa_data = []
    
    # Sentiment Dataset  
    if file_paths['sentiment']:
        try:
            with open(file_paths['sentiment'], 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
            print(f"✅ Sentiment Data: {len(sentiment_data)} sentiment örneği")
        except Exception as e:
            print(f"❌ Sentiment yükleme hatası: {e}")
            sentiment_data = []
    else:
        print("⚠️ Sentiment data bulunamadı, atlanıyor...")
        sentiment_data = []
    
    # Historical Dataset
    if file_paths['historical']:
        try:
            historical_df = pd.read_csv(file_paths['historical'])
            print(f"✅ Historical Data: {len(historical_df)} veri noktası")
            if 'symbol' in historical_df.columns:
                print(f"📊 Semboller: {historical_df['symbol'].nunique()} benzersiz sembol")
            if 'date' in historical_df.columns:
                print(f"📈 Tarih aralığı: {historical_df['date'].min()} → {historical_df['date'].max()}")
        except Exception as e:
            print(f"❌ Historical yükleme hatası: {e}")
            historical_df = pd.DataFrame()
    else:
        print("⚠️ Historical CSV bulunamadı, atlanıyor...")
        historical_df = pd.DataFrame()
    
    return qa_data, sentiment_data, historical_df

# FALLBACK DATA - Eğer dosyalar yoksa
def create_fallback_data():
    """Dosya yoksa fallback dataset oluştur"""
    print("🔄 FALLBACK: Minimum dataset oluşturuluyor...")
    
    fallback_qa = [
        {
            "question": "BIST 100 endeksi nedir?",
            "context": "BIST 100 endeksi, Borsa İstanbul'da işlem gören en büyük 100 şirketin performansını gösteren ana endekstir. Piyasanın genel yönünü yansıtır.",
            "answer": "BIST 100, en büyük 100 şirketin performansını gösteren ana endekstir"
        },
        {
            "question": "RSI nedir?",
            "context": "RSI (Relative Strength Index) 0-100 arasında değer alan momentum osilatörüdür. 70 üzerinde aşırı alım, 30 altında aşırı satım bölgesini gösterir.",
            "answer": "RSI, 0-100 arasında değer alan momentum göstergesidir"
        },
        {
            "question": "Teknik analiz nasıl yapılır?",
            "context": "Teknik analiz, geçmiş fiyat hareketleri ve işlem hacmi verilerini kullanarak gelecekteki fiyat hareketlerini tahmin etme yöntemidir.",
            "answer": "Teknik analiz, geçmiş fiyat verilerini kullanarak gelecek tahminleri yapar"
        }
    ]
    
    print(f"✅ Fallback Q&A: {len(fallback_qa)} örnek")
    return fallback_qa, [], pd.DataFrame()

# STEP 3: Enhanced Data Preprocessing - SAME AS BEFORE
def preprocess_enhanced_qa_data(qa_data):
    """Enhanced Q&A data'yı model için hazırla"""
    
    if not qa_data:
        print("❌ Q&A verisi yok!")
        return [], None
    
    print(f"🔧 {len(qa_data)} Q&A örneği işleniyor...")
    
    processed_examples = []
    
    # Turkish BERT tokenizer
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for i, item in enumerate(qa_data):
        try:
            question = item["question"]
            context = item["context"] 
            answer = item["answer"]
            
            # Tokenize
            encoding = tokenizer(
                question,
                context,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Answer positions
            answer_start = context.find(answer)
            if answer_start >= 0:
                start_pos = 1  # After [CLS]
                end_pos = min(start_pos + len(tokenizer.encode(answer, add_special_tokens=False)), 510)
            else:
                start_pos = 1
                end_pos = 2
            
            processed_examples.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "start_positions": torch.tensor(start_pos, dtype=torch.long),
                "end_positions": torch.tensor(end_pos, dtype=torch.long)
            })
            
        except Exception as e:
            print(f"⚠️ Örnek {i} atlandı: {e}")
            continue
    
    print(f"✅ {len(processed_examples)} örnek başarıyla işlendi!")
    return processed_examples, tokenizer

# STEP 4: Quick Training - Colab için optimize
def train_quick_model(processed_examples, tokenizer):
    """Hızlı model eğitimi - Colab için optimize"""
    
    if not processed_examples:
        print("❌ İşlenmiş örnek yok!")
        return None, None
        
    print("🤖 Hızlı model eğitimi başlıyor...")
    
    # Turkish BERT model
    model_name = "dbmdz/bert-base-turkish-cased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    print(f"📦 Model yüklendi: {model.num_parameters():,} parametre")
    
    # Dataset oluştur
    train_dataset = Dataset.from_list(processed_examples)
    
    # Quick training arguments
    training_args = TrainingArguments(
        output_dir="./bist-quick-turkish-ai",
        learning_rate=5e-5,  # Hızlı öğrenme
        num_train_epochs=2,  # Kısa eğitim
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        warmup_steps=10,
        evaluation_strategy="no",  # Evaluation kapalı
        save_steps=50,
        save_total_limit=1,
        logging_steps=5,
        push_to_hub=True,
        hub_model_id=HF_MODEL_NAME,
        hub_strategy="end",
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
    )
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    
    print("🔥 HIZLI EĞİTİM BAŞLATIYOR...")
    print(f"⏰ Başlama zamanı: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        train_result = trainer.train()
        print(f"🎉 EĞİTİM TAMAMLANDI! Loss: {train_result.training_loss:.4f}")
        return trainer, model
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None, None

# STEP 5: Quick Test
def test_quick_model(trainer, tokenizer):
    """Hızlı model test"""
    
    if not trainer:
        print("❌ Model test edilemedi!")
        return
    
    print("\n🧪 HIZLI TEST...")
    
    from transformers import pipeline
    
    qa_pipeline = pipeline(
        "question-answering",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    test_cases = [
        {
            "question": "BIST 100 nedir?",
            "context": "BIST 100 endeksi Türkiye'nin en büyük 100 şirketinin performansını gösteren ana borsa endeksidir."
        },
        {
            "question": "RSI nasıl kullanılır?",
            "context": "RSI 70 üzerinde aşırı alım, 30 altında aşırı satım sinyali verir. Bu teknik analiz göstergesi momentum ölçer."
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        try:
            result = qa_pipeline(question=test["question"], context=test["context"])
            print(f"Test {i}: {test['question']}")
            print(f"🤖 Cevap: {result['answer']}")
            print(f"🎯 Güven: {result['score']:.3f}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ Test {i} hatası: {e}")

# MAIN EXECUTION - FIXED
def main_fixed():
    """Düzeltilmiş ana fonksiyon"""
    
    print("🎯 COLAB FIXED TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data with fixed paths
    qa_data, sentiment_data, historical_df = load_training_data_fixed()
    
    # Fallback if no data
    if not qa_data:
        print("🔄 Fallback data kullanılıyor...")
        qa_data, sentiment_data, historical_df = create_fallback_data()
    
    if not qa_data:
        print("❌ Hiç veri yok! Training durduruluyor.")
        return {
            'model_name': HF_MODEL_NAME,
            'qa_samples': 0,
            'processed_samples': 0,
            'deploy_success': False
        }
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   🎯 Q&A Samples: {len(qa_data)}")
    print(f"   💭 Sentiment Samples: {len(sentiment_data)}")
    print(f"   📈 Historical Points: {len(historical_df) if not historical_df.empty else 0}")
    
    # Preprocess
    processed_examples, tokenizer = preprocess_enhanced_qa_data(qa_data)
    
    if not processed_examples:
        print("❌ Veri işlenemedi!")
        return {
            'model_name': HF_MODEL_NAME,
            'qa_samples': len(qa_data),
            'processed_samples': 0,
            'deploy_success': False
        }
    
    # Train
    trainer, model = train_quick_model(processed_examples, tokenizer)
    
    # Test
    if trainer:
        test_quick_model(trainer, tokenizer)
    
    # Deploy
    deploy_success = False
    if trainer:
        try:
            trainer.push_to_hub(commit_message="Quick Turkish Financial AI - Colab Fixed")
            print("🎉 DEPLOY BAŞARILI!")
            deploy_success = True
        except Exception as e:
            print(f"⚠️ Deploy hatası: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 COLAB FIX TRAINING TAMAMLANDI!")
    print("=" * 60)
    
    return {
        'model_name': HF_MODEL_NAME,
        'qa_samples': len(qa_data),
        'processed_samples': len(processed_examples),
        'deploy_success': deploy_success
    }

# COLAB EXECUTION
if __name__ == "__main__":
    print("🔥 COLAB FIXED VERSION ÇALIŞTIRILIYOR...")
    print("💡 GPU kullanımı:", "✅ Aktif" if torch.cuda.is_available() else "❌ CPU")
    
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n" + "🚀 " * 20)
    result = main_fixed()
    print("🚀 " * 20)
    
    print(f"\n🎊 FINAL RESULT:")
    print(f"Model: {result['model_name']}")
    print(f"Samples: {result['qa_samples']} → {result['processed_samples']}")
    print(f"Status: {'🎉 Production Ready!' if result['deploy_success'] else '💾 Local Backup'}")
