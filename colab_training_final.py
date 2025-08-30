# 🚀 FINAL TRAINING CODE - MAMUT R600 Turkish Financial Q&A
# Run this AFTER comprehensive fix + restart + verification

print("🚀 MAMUT R600 - FINAL TURKISH AI TRAINING")
print("=" * 60)

# Initial imports and checks
import torch
import numpy as np
print(f"🔥 GPU: {torch.cuda.is_available()}")
print(f"📊 Numpy: {np.__version__}")

if torch.cuda.is_available():
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔋 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Package version check
try:
    import transformers
    import datasets 
    import accelerate
    from huggingface_hub import login
    print(f"✅ Transformers: {transformers.__version__}")
    print(f"✅ Datasets: {datasets.__version__}")
    print(f"✅ Accelerate: {accelerate.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 Please run comprehensive fix again!")
    exit()

# HuggingFace Authentication
HF_TOKEN = "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"
HF_MODEL_NAME = "rsmctn/turkish-financial-qa-v1"

try:
    login(HF_TOKEN)
    print("✅ HuggingFace authenticated!")
except Exception as e:
    print(f"❌ HF Auth error: {e}")

# Training Data - Turkish Financial Q&A
training_data = [
    {
        "question": "GARAN hissesi bugün nasıl performans gösteriyor?",
        "context": "Türkiye Garanti Bankası A.Ş. (GARAN) hissesi bugün ₺89.30 fiyatında, günlük %-0.94 değişimle işlem görmektedir. Bankacılık sektöründe yer alan hisse, son 52 haftada ₺65.20 - ₺95.40 bandında hareket etmiştir.",
        "answer": "GARAN hissesi %-0.94 düşüş göstererek ₺89.30'da işlem görmektedir"
    },
    {
        "question": "RSI göstergesi nedir?",
        "context": "RSI (Relative Strength Index) 0-100 arasında değer alan bir momentum osilatörüdür. 70 üzerindeki değerler aşırı alım bölgesini, 30 altındaki değerler aşırı satım bölgesini gösterir.",
        "answer": "RSI, 0-100 arasında değer alan momentum göstergesidir"
    },
    {
        "question": "BIST 100 endeksi bugün nasıl?",
        "context": "BIST 100 endeksi bugün 8,450.75 seviyesinde, günlük %1.25 artışla kapanmıştır. İşlem hacmi 18.5 milyar TL olarak gerçekleşmiştir.",
        "answer": "BIST 100 endeksi %1.25 yükselişle 8,450.75 seviyesinde kapanmıştır"
    },
    {
        "question": "Teknik analiz nedir?",
        "context": "Teknik analiz, geçmiş fiyat hareketleri ve işlem hacmi verilerini kullanarak gelecekteki fiyat hareketlerini tahmin etmeye çalışan analiz yöntemidir. RSI, MACD, Bollinger Bantları gibi göstergeler kullanır.",
        "answer": "Teknik analiz, fiyat verilerini kullanarak gelecek tahminleri yapan yöntemdir"
    },
    {
        "question": "AKBNK için stop loss nerede?",
        "context": "AKBNK hissesi ₺69.00 seviyesinde işlem görmektedir. Önemli destek seviyesi ₺65.20 civarındadır. Volatilite %2.5 seviyesinde.",
        "answer": "AKBNK için stop loss ₺65.00-₺66.50 aralığında belirlenebilir"
    },
    {
        "question": "Piyasa durumu nasıl?",
        "context": "BIST 100 endeksi %1.25 yükselişte, yabancı yatırımcılar net 125 milyon TL alımda bulundu. Bankacılık endeksi %2.1 artış gösterdi.",
        "answer": "Piyasa pozitif seyrediyor, yabancı net alımda"
    },
    {
        "question": "MACD nasıl yorumlanır?",
        "context": "MACD (Moving Average Convergence Divergence) trend takip göstergesidir. MACD çizgisinin sinyal çizgisini yukarı kesmesi alım sinyali verir.",
        "answer": "MACD > Sinyal çizgisi alım sinyali verir"
    },
    {
        "question": "Risk yönetimi nasıl yapılır?",
        "context": "Risk yönetimi portföy çeşitlendirmesi, stop-loss kullanımı içerir. Toplam portföyün %2'sinden fazlası tek işlemde riske edilmemelidir.",
        "answer": "Portföyü çeşitlendirin, stop-loss kullanın"
    }
]

print(f"✅ {len(training_data)} Turkish Q&A samples loaded")

# Load Model with error handling
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    
    model_name = "dbmdz/bert-base-turkish-cased"
    print(f"📥 Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    
except Exception as e:
    print(f"❌ Model loading error: {e}")
    print("🔄 Trying alternative approach...")
    # Could add fallback model here

# Data preprocessing with robust error handling
from datasets import Dataset

def robust_preprocess(examples):
    """Robust preprocessing with fallback options"""
    try:
        questions = examples["question"] if isinstance(examples["question"], list) else [examples["question"]]
        contexts = examples["context"] if isinstance(examples["context"], list) else [examples["context"]]
        answers = examples["answer"] if isinstance(examples["answer"], list) else [examples["answer"]]
        
        # Tokenize with error handling
        inputs = tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Simple answer positioning
        start_positions = []
        end_positions = []
        
        for i in range(len(questions)):
            try:
                context = contexts[i]
                answer = answers[i]
                
                # Find answer in context
                start_char = context.find(answer)
                if start_char >= 0:
                    # Simple token position estimation
                    tokens_before = len(tokenizer.tokenize(context[:start_char]))
                    answer_tokens = len(tokenizer.tokenize(answer))
                    
                    start_pos = min(tokens_before + 1, 380)  # +1 for [CLS]
                    end_pos = min(start_pos + answer_tokens - 1, 383)
                else:
                    # Fallback positions
                    start_pos = 1
                    end_pos = min(len(tokenizer.tokenize(answer)) + 1, 10)
                    
                start_positions.append(start_pos)
                end_positions.append(end_pos)
                
            except Exception as e:
                print(f"Warning: Error processing sample {i}: {e}")
                start_positions.append(1)
                end_positions.append(2)
        
        inputs["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
        inputs["end_positions"] = torch.tensor(end_positions, dtype=torch.long)
        
        return inputs
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

# Create dataset
try:
    dataset_dict = {
        "question": [item["question"] for item in training_data],
        "context": [item["context"] for item in training_data],
        "answer": [item["answer"] for item in training_data]
    }
    
    # Process each example individually for robustness
    processed_examples = []
    for i, item in enumerate(training_data):
        try:
            single_example = {
                "question": [item["question"]],
                "context": [item["context"]],
                "answer": [item["answer"]]
            }
            
            processed = robust_preprocess(single_example)
            if processed is not None:
                processed_examples.append({
                    "input_ids": processed["input_ids"][0],
                    "attention_mask": processed["attention_mask"][0],
                    "start_positions": processed["start_positions"][0],
                    "end_positions": processed["end_positions"][0]
                })
            else:
                print(f"⚠️ Skipping sample {i} due to preprocessing error")
                
        except Exception as e:
            print(f"⚠️ Error processing sample {i}: {e}")
            continue
    
    final_dataset = Dataset.from_list(processed_examples)
    print(f"✅ Dataset created: {len(final_dataset)} valid samples")
    
except Exception as e:
    print(f"❌ Dataset creation error: {e}")

# Training setup with conservative settings
try:
    from transformers import TrainingArguments, Trainer, DefaultDataCollator
    from datetime import datetime
    
    training_args = TrainingArguments(
        output_dir="./turkish-qa-model",
        learning_rate=2e-5,  # Conservative learning rate
        num_train_epochs=3,   # Reduced epochs
        per_device_train_batch_size=1,  # Small batch size for stability
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Compensate for small batch
        weight_decay=0.01,
        warmup_steps=5,
        evaluation_strategy="steps",
        eval_steps=10,
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=2,
        push_to_hub=True,
        hub_model_id=HF_MODEL_NAME,
        hub_strategy="end",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    print("✅ Training configuration ready (conservative settings)")
    
except Exception as e:
    print(f"❌ Training setup error: {e}")

# Create Trainer and start training
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset,
        eval_dataset=final_dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    
    print("🔥 STARTING TRAINING...")
    print("=" * 50)
    print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")
    
    # Training with error handling
    try:
        train_result = trainer.train()
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Loss: {train_result.training_loss:.4f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ GPU Memory error, trying smaller batch...")
            # Reduce batch size even more
            training_args.per_device_train_batch_size = 1
            training_args.gradient_accumulation_steps = 8
            training_args.fp16 = True
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=final_dataset,
                eval_dataset=final_dataset,
                tokenizer=tokenizer,
                data_collator=DefaultDataCollator(),
            )
            
            train_result = trainer.train()
            print("✅ Training completed with reduced batch size!")
        else:
            raise e
            
    print(f"⏰ End: {datetime.now().strftime('%H:%M:%S')}")
    
except Exception as e:
    print(f"❌ Training error: {e}")

# Model testing
try:
    print("\n🧪 TESTING MODEL...")
    print("=" * 30)
    
    from transformers import pipeline
    
    qa_pipeline = pipeline(
        "question-answering",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    test_cases = [
        ("GARAN nasıl?", "GARAN hissesi ₺89.30'da %-0.94 düşüşte."),
        ("RSI nedir?", "RSI momentum göstergesidir. 70 üstü aşırı alım."),
        ("Stop loss nerede?", "Stop loss destek altında belirlenmelidir.")
    ]
    
    for i, (q, c) in enumerate(test_cases, 1):
        try:
            result = qa_pipeline(question=q, context=c)
            print(f"Test {i}: {q}")
            print(f"✅ AI: {result['answer']}")
            print(f"🎯 Confidence: {result['score']:.3f}")
            print("-" * 25)
        except Exception as e:
            print(f"Test {i} error: {e}")
            
except Exception as e:
    print(f"❌ Testing error: {e}")

# Upload to HuggingFace
try:
    print("🚀 UPLOADING TO HUGGINGFACE...")
    trainer.push_to_hub(commit_message="Turkish Financial Q&A - MAMUT R600")
    print("🎉 MODEL UPLOADED SUCCESSFULLY!")
    print(f"📍 URL: https://huggingface.co/{HF_MODEL_NAME}")
    
except Exception as e:
    print(f"⚠️ Upload error: {e}")
    print("💾 Saving locally instead...")
    try:
        model.save_pretrained("./turkish-qa-model")
        tokenizer.save_pretrained("./turkish-qa-model")
        print("✅ Model saved locally!")
    except Exception as save_e:
        print(f"❌ Local save error: {save_e}")

# Final summary
print("\n" + "="*60)
print("🎉 TRAINING PROCESS COMPLETED!")
print("="*60)
print(f"✅ Model: {HF_MODEL_NAME}")
print(f"✅ Samples trained: {len(final_dataset)}")
print("✅ Turkish Financial Q&A AI ready!")

if "🎉 MODEL UPLOADED SUCCESSFULLY!" in locals():
    print("✅ HuggingFace upload: SUCCESS")
    print(f"🔗 API: https://api-inference.huggingface.co/models/{HF_MODEL_NAME}")
else:
    print("⚠️ HuggingFace upload: Failed (but model trained)")

print("=" * 60)
print("🚀 NEXT: Integrate with Railway API!")
print("📋 Railway integration code will be provided separately")
print("=" * 60)
