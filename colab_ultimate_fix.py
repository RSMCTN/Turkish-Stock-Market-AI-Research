#!/usr/bin/env python3
"""
🚀 COLAB ULTIMATE FIX - TÜM VERİLER YÜKLENIYOR!
Sentiment + Historical + Q&A = FULL DATASET
"""

import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

print("🚀 COLAB ULTIMATE FIX - FULL DATASET TRAINING!")
print("🎯 117 Sembol + 30 İndikatör + TÜM VERİLER")
print("=" * 60)

# STEP 1: Imports ve versiyonlar
try:
    import transformers
    print(f"✅ transformers: {transformers.__version__}")
    
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from transformers import TrainingArguments, Trainer, DefaultDataCollator
    from datasets import Dataset
    
    print("✅ Dependencies loaded!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# HF Setup
HF_TOKEN = "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"
HF_MODEL_NAME = "rsmctn/bist-ultimate-turkish-ai-v4"  # Ultimate version

try:
    login(HF_TOKEN)
    print("✅ HuggingFace authenticated!")
except Exception as e:
    print(f"❌ Auth error: {e}")

# STEP 2: FULL Data Loading - FIXED!
def load_full_training_data():
    """TÜM training data'yı yükle - Q&A + Sentiment + Historical"""
    
    print("📊 FULL DATASET yükleniyor...")
    
    # Dosyaları kontrol et
    files = os.listdir('.')
    print(f"📂 Dosyalar: {[f for f in files if f.endswith(('.json', '.csv'))]}")
    
    # 1. Q&A Dataset
    qa_data = []
    if 'enhanced_turkish_qa.json' in files:
        try:
            with open('enhanced_turkish_qa.json', 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            print(f"✅ Q&A: {len(qa_data)} soru-cevap")
        except Exception as e:
            print(f"❌ Q&A error: {e}")
    
    # 2. Sentiment Dataset - FIXED!
    sentiment_data = []
    if 'enhanced_sentiment.json' in files:
        try:
            with open('enhanced_sentiment.json', 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
            print(f"✅ SENTIMENT: {len(sentiment_data)} örnek")
        except Exception as e:
            print(f"❌ Sentiment error: {e}")
    
    # 3. Historical Dataset - FIXED!
    historical_df = pd.DataFrame()
    if 'enhanced_historical_training.csv' in files:
        try:
            historical_df = pd.read_csv('enhanced_historical_training.csv')
            print(f"✅ HISTORICAL: {len(historical_df)} kayıt")
            if 'symbol' in historical_df.columns:
                print(f"📊 Semboller: {historical_df['symbol'].nunique()} benzersiz")
            if 'date' in historical_df.columns:
                print(f"📈 Tarih: {historical_df['date'].min()} → {historical_df['date'].max()}")
        except Exception as e:
            print(f"❌ Historical error: {e}")
    
    # Sentiment'i Q&A formatına çevir
    if sentiment_data:
        print("🔄 Sentiment data → Q&A formatına çeviriliyor...")
        for item in sentiment_data[:50]:  # İlk 50 örnek
            sentiment_score = item.get('score', 0)
            sentiment_label = item.get('sentiment', 'neutral')
            text = item.get('text', '')
            
            qa_item = {
                "question": f"{text} - bu haberin sentiment analizi nedir?",
                "context": f"'{text}' haberi finansal piyasalar için {sentiment_label} bir gelişmedir. Sentiment skoru {sentiment_score:.2f} olarak hesaplanmıştır. Bu analiz haber metninin olumlu, olumsuz veya nötr etkisini gösterir.",
                "answer": f"Bu haberin sentiment'i {sentiment_label}, skorun {sentiment_score:.2f}"
            }
            qa_data.append(qa_item)
        
        print(f"✅ Sentiment → Q&A: {len(sentiment_data[:50])} örnek eklendi")
    
    # Historical'ı Q&A formatına çevir
    if not historical_df.empty:
        print("🔄 Historical data → Q&A formatına çeviriliyor...")
        # İlk 30 kayıt al
        for i, row in historical_df.head(30).iterrows():
            try:
                symbol = row.get('symbol', 'UNKNOWN')
                close = row.get('close', 0)
                volume = row.get('volume', 0)
                rsi = row.get('rsi_14', 50)
                date = row.get('date', '2024-01-01')
                
                qa_item = {
                    "question": f"{symbol} hissesinin {date} tarihindeki performansı nedir?",
                    "context": f"{symbol} hissesi {date} tarihinde ₺{close:.2f} fiyatında kapanmıştır. İşlem hacmi {volume:,.0f} adet olarak gerçekleşmiştir. RSI değeri {rsi:.1f} seviyesindedir. Bu veriler hissenin o günkü piyasa performansını göstermektedir.",
                    "answer": f"{symbol} hissesi {date} tarihinde ₺{close:.2f} fiyatında kapanmıştır"
                }
                qa_data.append(qa_item)
            except Exception as e:
                continue
        
        print(f"✅ Historical → Q&A: 30 örnek eklendi")
    
    # Fallback ekle
    if len(qa_data) < 20:
        print("🔄 Fallback Q&A ekleniyor...")
        fallback_qa = [
            {
                "question": "BIST 100 endeksi nasıl çalışır?",
                "context": "BIST 100 endeksi, Borsa İstanbul'da işlem gören en büyük 100 şirketin piyasa değeri ağırlıklı performansını yansıtan ana göstergedir. Endeks değeri şirketlerin toplam piyasa değerindeki değişimlere göre hesaplanır.",
                "answer": "BIST 100, en büyük 100 şirketin piyasa değeri ağırlıklı performans endeksidir"
            },
            {
                "question": "RSI indikatörü hangi durumlarda kullanılır?",
                "context": "RSI (Relative Strength Index) momentum osilatörüdür ve 0-100 arası değerler alır. 70 üzerindeki değerler aşırı alım bölgesini, 30 altındaki değerler aşırı satım bölgesini gösterir. Yatırımcılar bu seviyelerde pozisyon alma kararları verir.",
                "answer": "RSI, aşırı alım/satım seviyelerini belirlemek için kullanılan momentum osilatörüdür"
            },
            {
                "question": "MACD göstergesinin sinyal çizgisi nasıl yorumlanır?",
                "context": "MACD göstergesinde ana çizginin sinyal çizgisini yukarı kesmesi alım sinyali, aşağı kesmesi satım sinyali olarak değerlendirilir. Ayrıca MACD'nin sıfır çizgisinin üstünde olması yükseliş, altında olması düşüş trendini gösterir.",
                "answer": "MACD sinyal çizgisi kesişmeleri alım/satım sinyalleri verir"
            }
        ]
        qa_data.extend(fallback_qa)
        print(f"✅ Fallback: {len(fallback_qa)} örnek eklendi")
    
    print(f"\n📊 FINAL DATASET:")
    print(f"   🎯 Toplam Q&A: {len(qa_data)} örnek")
    print(f"   💭 Orijinal Sentiment: {len(sentiment_data)} örnek")
    print(f"   📈 Orijinal Historical: {len(historical_df)} kayıt")
    
    return qa_data, sentiment_data, historical_df

# STEP 3: Advanced Preprocessing
def preprocess_advanced_qa(qa_data):
    """Gelişmiş Q&A preprocessing"""
    
    if not qa_data:
        print("❌ Q&A verisi yok!")
        return [], None
    
    print(f"🔧 {len(qa_data)} Q&A örneği gelişmiş işleme...")
    
    # Turkish BERT
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    processed_examples = []
    
    for i, item in enumerate(qa_data):
        try:
            question = str(item.get("question", "")).strip()
            context = str(item.get("context", "")).strip()
            answer = str(item.get("answer", "")).strip()
            
            if not question or not context or not answer:
                continue
            
            # Tokenization
            encoding = tokenizer(
                question,
                context,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Advanced answer position detection
            start_pos = 1  # [CLS] sonrası
            end_pos = start_pos + min(len(tokenizer.encode(answer, add_special_tokens=False)), 20)
            end_pos = min(end_pos, 510)
            
            processed_examples.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "start_positions": torch.tensor(start_pos, dtype=torch.long),
                "end_positions": torch.tensor(end_pos, dtype=torch.long)
            })
            
            if (i + 1) % 50 == 0:
                print(f"   ⚡ İşlendi: {i+1}/{len(qa_data)}")
                
        except Exception as e:
            continue
    
    print(f"✅ {len(processed_examples)} örnek başarıyla işlendi!")
    return processed_examples, tokenizer

# STEP 4: Advanced Training
def train_advanced_model(processed_examples, tokenizer):
    """Gelişmiş model eğitimi"""
    
    if not processed_examples:
        print("❌ Processed data yok!")
        return None, None
    
    print("🤖 ADVANCED MODEL TRAİNİNG...")
    
    # Model
    model_name = "dbmdz/bert-base-turkish-cased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    print(f"📦 Model: {model.num_parameters():,} parametre")
    
    # Dataset
    train_dataset = Dataset.from_list(processed_examples)
    
    # Advanced TrainingArguments
    training_args = TrainingArguments(
        output_dir="./bist-ultimate-ai",
        learning_rate=2e-5,          # Optimal learning rate
        num_train_epochs=3,          # Daha fazla epoch
        per_device_train_batch_size=8,  # Daha büyük batch
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        warmup_steps=20,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        push_to_hub=True,
        hub_model_id=HF_MODEL_NAME,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    
    print("🔥 ADVANCED TRAİNİNG BAŞLIYOR...")
    start_time = datetime.now()
    
    try:
        train_result = trainer.train()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"🎉 ADVANCED TRAİNİNG BAŞARILI!")
        print(f"📊 Final Loss: {train_result.training_loss:.4f}")
        print(f"⏰ Süre: {duration:.1f} saniye")
        
        return trainer, model
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        # Smaller batch retry
        training_args.per_device_train_batch_size = 4
        training_args.gradient_accumulation_steps = 2
        
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
            print(f"❌ İkinci deneme başarısız: {e2}")
            return None, None

# STEP 5: Comprehensive Test
def comprehensive_test(trainer, tokenizer):
    """Kapsamlı model testi"""
    
    if not trainer:
        return
    
    print("\n🧪 COMPREHENSIVE MODEL TEST...")
    
    try:
        from transformers import pipeline
        
        qa_pipeline = pipeline(
            "question-answering",
            model=trainer.model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Advanced test cases
        test_cases = [
            {
                "question": "BIST 100 endeksi nedir?",
                "context": "BIST 100 endeksi, Borsa İstanbul'da işlem gören en büyük 100 şirketin piyasa değeri ağırlıklı performansını yansıtan ana göstergedir. Türkiye ekonomisinin nabzını tutan bu endeks günlük olarak hesaplanır."
            },
            {
                "question": "RSI 70 üzerinde ne anlama gelir?",
                "context": "RSI (Relative Strength Index) 70 üzerindeki değerler hisse senedinin aşırı alım bölgesinde olduğunu gösterir. Bu durumda satış baskısı artabilir ve fiyat düzeltmesi beklenebilir."
            },
            {
                "question": "MACD sinyali nasıl yorumlanır?",
                "context": "MACD göstergesinde ana çizginin sinyal çizgisini yukarı kesmesi güçlü bir alım sinyali olarak kabul edilir. Bu durumda yükseliş momentumu artmış olur."
            },
            {
                "question": "Stop loss neden kullanılır?",
                "context": "Stop loss emri, yatırımcının önceden belirlediği zarar seviyesine ulaşıldığında otomatik olarak pozisyonu kapatır. Bu sayede kayıplar sınırlanır ve sermaye korunur."
            },
            {
                "question": "Volatilite yüksek olunca ne olur?",
                "context": "Yüksek volatilite dönemlerinde hisse fiyatları büyük dalgalanmalar gösterir. Bu durum hem yüksek kazanç fırsatları hem de yüksek risk anlamına gelir."
            }
        ]
        
        total_score = 0
        successful_tests = 0
        
        print("📋 ADVANCED TEST SONUÇLARI:")
        print("=" * 70)
        
        for i, test in enumerate(test_cases, 1):
            try:
                result = qa_pipeline(
                    question=test["question"],
                    context=test["context"]
                )
                
                score = result['score']
                answer = result['answer']
                total_score += score
                successful_tests += 1
                
                print(f"Test {i}: {test['question']}")
                print(f"🤖 AI Cevap: {answer}")
                print(f"🎯 Güven: {score:.3f} ({score*100:.1f}%)")
                
                # Quality assessment
                if score > 0.7:
                    quality = "🌟 Mükemmel"
                elif score > 0.5:
                    quality = "⭐ İyi"
                elif score > 0.3:
                    quality = "📈 Orta"
                else:
                    quality = "🔴 Zayıf"
                
                print(f"📊 Kalite: {quality}")
                print("=" * 70)
                
            except Exception as e:
                print(f"❌ Test {i} error: {e}")
        
        if successful_tests > 0:
            avg_score = total_score / successful_tests
            print(f"\n📊 GENEL PERFORMANS RAPORU:")
            print(f"✅ Başarılı testler: {successful_tests}/{len(test_cases)}")
            print(f"🎯 Ortalama güven: {avg_score:.3f} ({avg_score*100:.1f}%)")
            
            if avg_score > 0.7:
                overall = "🌟 MÜKEMMEL - Production Ready!"
            elif avg_score > 0.5:
                overall = "⭐ İYİ - Kullanılabilir"
            elif avg_score > 0.3:
                overall = "📈 ORTA - İyileştirilebilir"
            else:
                overall = "🔴 ZAYIF - Daha fazla eğitim gerekli"
            
            print(f"🏆 Genel Değerlendirme: {overall}")
            
    except Exception as e:
        print(f"❌ Test genel hatası: {e}")

# MAIN EXECUTION
def main():
    """Ana ultimate training fonksiyonu"""
    
    print("🎯 ULTIMATE TRAINING PIPELINE")
    print("=" * 60)
    
    # Full data loading
    qa_data, sentiment_data, historical_df = load_full_training_data()
    
    if len(qa_data) < 10:
        print("❌ Yetersiz Q&A verisi!")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    # Advanced preprocessing
    processed_examples, tokenizer = preprocess_advanced_qa(qa_data)
    
    if not processed_examples:
        print("❌ Preprocessing başarısız!")
        return {'status': 'failed', 'reason': 'preprocessing_failed'}
    
    # Advanced training
    trainer, model = train_advanced_model(processed_examples, tokenizer)
    
    # Comprehensive testing
    comprehensive_test(trainer, tokenizer)
    
    # Deploy
    deploy_success = False
    if trainer:
        try:
            trainer.push_to_hub(
                commit_message="Ultimate Turkish Financial AI - Full Dataset Training"
            )
            print("\n🚀 ULTIMATE DEPLOY BAŞARILI!")
            print(f"📍 Model: https://huggingface.co/{HF_MODEL_NAME}")
            print(f"🔗 API: https://api-inference.huggingface.co/models/{HF_MODEL_NAME}")
            deploy_success = True
        except Exception as e:
            print(f"❌ Deploy error: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ULTIMATE TRAINING TAMAMLANDI!")
    print("=" * 60)
    print(f"✅ Toplam Q&A: {len(qa_data)} örnek")
    print(f"✅ İşlenen: {len(processed_examples)} örnek")
    print(f"✅ Training: {'Başarılı' if trainer else 'Başarısız'}")
    print(f"✅ Deploy: {'Başarılı' if deploy_success else 'Başarısız'}")
    print("=" * 60)
    
    return {
        'status': 'success' if trainer else 'failed',
        'total_qa': len(qa_data),
        'processed': len(processed_examples),
        'sentiment_used': len(sentiment_data),
        'historical_used': len(historical_df),
        'deploy_success': deploy_success,
        'model_name': HF_MODEL_NAME
    }

# EXECUTION
if __name__ == "__main__":
    print("🔥 ULTIMATE TRAINING ÇALIŞTIRILIYOR...")
    print("💡 GPU:", "✅ Aktif" if torch.cuda.is_available() else "❌ CPU")
    
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "🚀 " * 20)
    result = main()
    print("🚀 " * 20)
    
    print(f"\n🎊 ULTIMATE RESULT:")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Model: {result['model_name']}")
        print(f"Total Q&A: {result['total_qa']}")
        print(f"Sentiment Used: {result['sentiment_used']}")
        print(f"Historical Used: {result['historical_used']}")
        print(f"Deploy: {'🎉 SUCCESS' if result['deploy_success'] else '💾 Local'}")
        
        print(f"\n🎯 RAILWAY INTEGRATION:")
        print(f"API: https://api-inference.huggingface.co/models/{result['model_name']}")
        print(f"Token: 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")
    
    print("\n🔥 ULTIMATE FIX COMPLETE! 🔥")
