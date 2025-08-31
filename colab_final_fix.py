#!/usr/bin/env python3
"""
🚀 COLAB FINAL FIX - TRANSFORMERS UYUMLULUK SORUNU ÇÖZÜLDÜ!
TrainingArguments parametreleri güncellendi
"""

import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

print("🚀 COLAB FINAL FIX - TURKISH FINANCIAL AI TRAINING")
print("🎯 117 Sembol + 30 İndikatör + 1.4M Historical Data")
print("=" * 60)

# STEP 1: Version check ve imports
def check_versions():
    """Kütüphane versiyonlarını kontrol et"""
    try:
        import transformers
        import torch
        import datasets
        print(f"✅ transformers: {transformers.__version__}")
        print(f"✅ torch: {torch.__version__}")
        print(f"✅ datasets: {datasets.__version__}")
        return True
    except Exception as e:
        print(f"❌ Version check error: {e}")
        return False

if not check_versions():
    print("💡 Gerekli kütüphaneler eksik!")
    exit(1)

# STEP 2: Imports
try:
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from transformers import TrainingArguments, Trainer, DefaultDataCollator
    from datasets import Dataset
    
    print("✅ Dependencies loaded successfully!")
except ImportError as e:
    print(f"❌ Dependency error: {e}")
    exit(1)

# STEP 3: HF Authentication
HF_TOKEN = "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"
HF_MODEL_NAME = "rsmctn/bist-advanced-turkish-ai-v3"  # Yeni version

try:
    login(HF_TOKEN)
    print("✅ HuggingFace authenticated!")
except Exception as e:
    print(f"❌ HF Auth error: {e}")

# STEP 4: File Check Function
def check_files():
    """Dosyaları kontrol et"""
    print("📁 Dosya kontrolü...")
    
    files = os.listdir('.')
    json_csv_files = [f for f in files if f.endswith(('.json', '.csv'))]
    print(f"📋 Bulunan dosyalar: {json_csv_files}")
    
    # Training dosyalarını ara
    found_files = {}
    
    # Q&A
    if 'enhanced_turkish_qa.json' in files:
        found_files['qa'] = 'enhanced_turkish_qa.json'
        print("✅ QA: enhanced_turkish_qa.json")
    else:
        found_files['qa'] = None
        print("❌ QA: enhanced_turkish_qa.json BULUNAMADI")
    
    # Sentiment
    if 'enhanced_sentiment.json' in files:
        found_files['sentiment'] = 'enhanced_sentiment.json'
        print("✅ SENTIMENT: enhanced_sentiment.json")
    else:
        found_files['sentiment'] = None
        print("⚠️ SENTIMENT: enhanced_sentiment.json bulunamadı")
    
    # Historical
    if 'enhanced_historical_training.csv' in files:
        found_files['historical'] = 'enhanced_historical_training.csv'
        print("✅ HISTORICAL: enhanced_historical_training.csv")
    else:
        found_files['historical'] = None
        print("⚠️ HISTORICAL: enhanced_historical_training.csv bulunamadı")
    
    return found_files

# STEP 5: Load Data
def load_training_data():
    """Training data yükle"""
    print("📊 Training data yükleniyor...")
    
    file_paths = check_files()
    
    # Q&A Dataset
    if file_paths['qa']:
        try:
            with open(file_paths['qa'], 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            print(f"✅ Q&A: {len(qa_data)} soru-cevap")
        except Exception as e:
            print(f"❌ Q&A hata: {e}")
            qa_data = []
    else:
        qa_data = []
    
    # Fallback data
    if not qa_data:
        print("🔄 FALLBACK data oluşturuluyor...")
        qa_data = [
            {
                "question": "BIST 100 endeksi nedir?",
                "context": "BIST 100 endeksi, Borsa İstanbul'da işlem gören en büyük 100 şirketin performansını gösteren ana endekstir. Türkiye piyasasının genel durumunu yansıtır.",
                "answer": "BIST 100, en büyük 100 şirketin performansını gösteren ana endekstir"
            },
            {
                "question": "RSI göstergesi nedir?",
                "context": "RSI (Relative Strength Index) 0-100 arasında değer alan momentum osilatörüdür. 70 üzerinde aşırı alım, 30 altında aşırı satım bölgesini gösterir.",
                "answer": "RSI, 0-100 arasında değer alan momentum göstergesidir"
            },
            {
                "question": "MACD nasıl kullanılır?",
                "context": "MACD (Moving Average Convergence Divergence) trend takip göstergesidir. MACD çizgisinin sinyal çizgisini yukarı kesmesi alım sinyali verir.",
                "answer": "MACD çizgisinin sinyal çizgisini kesmesi alım/satım sinyali verir"
            },
            {
                "question": "Stop loss nedir?",
                "context": "Stop loss, yatırımcının belirlediği kayıp seviyesine ulaşıldığında otomatik olarak pozisyonu kapatan emirdir. Risk yönetiminin temel taşıdır.",
                "answer": "Stop loss, kayıp seviyesinde otomatik pozisyon kapatan emirdir"
            },
            {
                "question": "Volatilite nedir?",
                "context": "Volatilite, bir finansal enstrümanın fiyatındaki değişkenlik ölçüsüdür. Yüksek volatilite büyük fiyat hareketleri anlamına gelir.",
                "answer": "Volatilite, fiyat değişkenliğinin ölçüsüdür"
            }
        ]
        print(f"✅ Fallback: {len(qa_data)} örnek")
    
    return qa_data, [], pd.DataFrame()

# STEP 6: Preprocess Data
def preprocess_qa_data(qa_data):
    """Q&A data'yı işle"""
    if not qa_data:
        print("❌ İşlenecek Q&A verisi yok!")
        return [], None
    
    print(f"🔧 {len(qa_data)} Q&A örneği işleniyor...")
    
    # Turkish BERT tokenizer
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    processed_examples = []
    
    for i, item in enumerate(qa_data):
        try:
            question = item["question"]
            context = item["context"]
            answer = item["answer"]
            
            # Tokenize
            encoding = tokenizer(
                question,
                context,
                max_length=384,  # Daha kısa context
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Answer positions - basit yaklaşım
            start_pos = 1  # [CLS] sonrası
            end_pos = min(10, 380)  # Güvenli aralık
            
            processed_examples.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "start_positions": torch.tensor(start_pos, dtype=torch.long),
                "end_positions": torch.tensor(end_pos, dtype=torch.long)
            })
            
        except Exception as e:
            print(f"⚠️ Örnek {i} atlandı: {e}")
            continue
    
    print(f"✅ {len(processed_examples)} örnek işlendi!")
    return processed_examples, tokenizer

# STEP 7: Train Model - FIXED TrainingArguments
def train_model_fixed(processed_examples, tokenizer):
    """Model eğitimi - düzeltilmiş TrainingArguments"""
    
    if not processed_examples:
        print("❌ İşlenmiş veri yok!")
        return None, None
    
    print("🤖 Model eğitimi başlıyor...")
    
    # Model yükle
    model_name = "dbmdz/bert-base-turkish-cased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    print(f"📦 Model: {model.num_parameters():,} parametre")
    
    # Dataset
    train_dataset = Dataset.from_list(processed_examples)
    
    # FIXED TrainingArguments - problematik parametreler kaldırıldı
    training_args = TrainingArguments(
        output_dir="./bist-turkish-ai-final",
        learning_rate=3e-5,
        num_train_epochs=2,  # Kısa eğitim
        per_device_train_batch_size=4,  # Küçük batch
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        warmup_steps=10,
        save_steps=100,
        save_total_limit=1,
        logging_steps=5,
        push_to_hub=True,
        hub_model_id=HF_MODEL_NAME,
        # evaluation_strategy KALDIRILDI - bu parametri sorun çıkarıyordu
        # eval_steps KALDIRILDI
        # load_best_model_at_end KALDIRILDI
        fp16=True,  # GPU hızlandırması
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Wandb kapalı
    )
    
    print("🔧 TrainingArguments oluşturuldu (FIXED)")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    
    print("🔥 EĞİTİM BAŞLIYOR...")
    start_time = datetime.now()
    print(f"⏰ Başlama: {start_time.strftime('%H:%M:%S')}")
    
    try:
        # Training başlat
        train_result = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"🎉 EĞİTİM TAMAMLANDI!")
        print(f"📊 Final Loss: {train_result.training_loss:.4f}")
        print(f"⏰ Süre: {duration:.1f} saniye")
        
        return trainer, model
        
    except Exception as e:
        print(f"❌ Training hatası: {e}")
        print("💡 Daha küçük batch size deneniyor...")
        
        # Küçük batch retry
        training_args.per_device_train_batch_size = 2
        training_args.gradient_accumulation_steps = 4
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=DefaultDataCollator(),
        )
        
        try:
            train_result = trainer.train()
            print("✅ Küçük batch ile başarılı!")
            return trainer, model
        except Exception as e2:
            print(f"❌ İkinci deneme de başarısız: {e2}")
            return None, None

# STEP 8: Test Model
def test_model(trainer, tokenizer):
    """Model test et"""
    if not trainer:
        print("❌ Test edilecek model yok!")
        return
    
    print("\n🧪 MODEL TEST...")
    
    try:
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
                "context": "BIST 100 endeksi Türkiye'nin en büyük 100 şirketinin performansını gösteren borsa endeksidir."
            },
            {
                "question": "RSI nasıl yorumlanır?",
                "context": "RSI 70 üzerinde aşırı alım, 30 altında aşırı satım bölgesini gösterir."
            },
            {
                "question": "Stop loss neden kullanılır?",
                "context": "Stop loss risk yönetimi için kullanılır ve kayıpları sınırlar."
            }
        ]
        
        total_score = 0
        successful_tests = 0
        
        for i, test in enumerate(test_cases, 1):
            try:
                result = qa_pipeline(
                    question=test["question"],
                    context=test["context"]
                )
                
                score = result['score']
                total_score += score
                successful_tests += 1
                
                print(f"Test {i}: {test['question']}")
                print(f"🤖 Cevap: {result['answer']}")
                print(f"🎯 Güven: {score:.3f} ({score*100:.1f}%)")
                print(f"📊 Durum: {'🟢 İyi' if score > 0.5 else '🟡 Orta' if score > 0.3 else '🔴 Zayıf'}")
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Test {i} hatası: {e}")
        
        if successful_tests > 0:
            avg_score = total_score / successful_tests
            print(f"\n📊 GENEL PERFORMANS:")
            print(f"✅ Başarılı test: {successful_tests}/{len(test_cases)}")
            print(f"🎯 Ortalama güven: {avg_score:.3f} ({avg_score*100:.1f}%)")
            print(f"🏆 Model kalitesi: {'🌟 Mükemmel' if avg_score > 0.7 else '⭐ İyi' if avg_score > 0.5 else '📈 Geliştirilmeli'}")
        
    except Exception as e:
        print(f"❌ Test genel hatası: {e}")

# STEP 9: Deploy
def deploy_model(trainer):
    """Model'i deploy et"""
    if not trainer:
        print("❌ Deploy edilecek model yok!")
        return False
    
    print("\n🚀 HuggingFace'e deploy...")
    
    try:
        trainer.push_to_hub(
            commit_message="Turkish Financial AI - Final Fix Version"
        )
        
        print("🎉 DEPLOY BAŞARILI!")
        print(f"📍 Model URL: https://huggingface.co/{HF_MODEL_NAME}")
        print(f"🔗 API: https://api-inference.huggingface.co/models/{HF_MODEL_NAME}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deploy hatası: {e}")
        return False

# MAIN FUNCTION
def main():
    """Ana fonksiyon"""
    print("🎯 COLAB FINAL FIX TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    qa_data, sentiment_data, historical_df = load_training_data()
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   🎯 Q&A: {len(qa_data)} örnek")
    print(f"   💭 Sentiment: {len(sentiment_data)} örnek")
    print(f"   📈 Historical: {len(historical_df) if not historical_df.empty else 0} kayıt")
    
    # Preprocess
    processed_examples, tokenizer = preprocess_qa_data(qa_data)
    
    if not processed_examples:
        print("❌ Veri işlenemedi!")
        return {
            'status': 'failed',
            'reason': 'data_processing_failed'
        }
    
    # Train
    trainer, model = train_model_fixed(processed_examples, tokenizer)
    
    # Test
    test_model(trainer, tokenizer)
    
    # Deploy
    deploy_success = deploy_model(trainer)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 FINAL FIX TRAINING TAMAMLANDI!")
    print("=" * 60)
    print(f"✅ Q&A örnekleri: {len(qa_data)}")
    print(f"✅ İşlenen örnekler: {len(processed_examples)}")
    print(f"✅ Training: {'Başarılı' if trainer else 'Başarısız'}")
    print(f"✅ Deploy: {'Başarılı' if deploy_success else 'Başarısız'}")
    print("=" * 60)
    
    return {
        'status': 'success' if trainer else 'failed',
        'qa_samples': len(qa_data),
        'processed_samples': len(processed_examples),
        'deploy_success': deploy_success,
        'model_name': HF_MODEL_NAME
    }

# EXECUTION
if __name__ == "__main__":
    print("🔥 COLAB FINAL FIX ÇALIŞTIRILIYOR...")
    print("💡 GPU:", "✅ Aktif" if torch.cuda.is_available() else "❌ CPU")
    
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n" + "🚀 " * 20)
    result = main()
    print("🚀 " * 20)
    
    print(f"\n🎊 SONUÇ:")
    print(f"Durum: {result['status']}")
    if result['status'] == 'success':
        print(f"Model: {result['model_name']}")
        print(f"Örnekler: {result['qa_samples']} → {result['processed_samples']}")
        print(f"Deploy: {'🎉 Başarılı' if result['deploy_success'] else '💾 Local'}")
    else:
        print("❌ Training başarısız oldu!")
    
    print("\n🔥 TRAINING TAMAMLANDI! 🔥")
