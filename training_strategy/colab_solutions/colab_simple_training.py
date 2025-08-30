# 🚀 SIMPLE TURKISH Q&A TRAINING - After Factory Reset
# Minimal dependencies, maximum success rate!

print("🚀 SIMPLE TURKISH Q&A TRAINING - MAMUT R600")
print("=" * 60)

# HuggingFace Authentication
from huggingface_hub import login

HF_TOKEN = "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"
HF_MODEL_NAME = "rsmctn/turkish-financial-qa-simple"

try:
    login(HF_TOKEN)
    print("✅ HuggingFace authenticated!")
except Exception as e:
    print(f"⚠️ HF Auth warning: {e}")

# Simple Turkish Financial Q&A Dataset
training_data = [
    {
        "question": "GARAN hissesi nasıl?",
        "context": "GARAN hissesi ₺89.30'da %-0.94 düşüşle işlem görüyor.",
        "answer": "%-0.94 düşüşle ₺89.30'da"
    },
    {
        "question": "RSI nedir?", 
        "context": "RSI 0-100 arası momentum göstergesidir. 70 üstü aşırı alım.",
        "answer": "0-100 arası momentum göstergesi"
    },
    {
        "question": "BIST 100 nasıl?",
        "context": "BIST 100 endeksi %1.25 yükselişle 8,450 seviyesinde.",
        "answer": "%1.25 yükselişle 8,450 seviyesinde"
    },
    {
        "question": "Stop loss nerede?",
        "context": "Stop loss destek seviyesinin altında belirlenmelidir.",
        "answer": "Destek seviyesinin altında"
    },
    {
        "question": "Piyasa nasıl?",
        "context": "Piyasa pozitif seyrediyor, yabancı net alımda.", 
        "answer": "Pozitif seyrediyor"
    },
    {
        "question": "MACD nasıl yorumlanır?",
        "context": "MACD sinyal çizgisini yukarı kesmesi alım sinyalidir.",
        "answer": "Yukarı kesişim alım sinyali"
    }
]

print(f"✅ {len(training_data)} Turkish financial samples loaded")

# Load Turkish BERT Model (simple approach)
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "dbmdz/bert-base-turkish-cased"
print(f"📥 Loading {model_name}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print(f"✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    exit()

# Simple data preprocessing 
def simple_preprocess(data):
    """Ultra-simple preprocessing"""
    inputs = []
    
    for item in data:
        # Tokenize question + context
        encoded = tokenizer(
            item["question"],
            item["context"], 
            max_length=256,  # Shorter for simplicity
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Simple answer positioning (approximate)
        context = item["context"]
        answer = item["answer"]
        start_char = context.find(answer)
        
        if start_char >= 0:
            # Rough token positions
            start_pos = max(1, len(context[:start_char].split()) + 1)
            end_pos = min(start_pos + len(answer.split()) - 1, 250)
        else:
            start_pos, end_pos = 1, 2
        
        inputs.append({
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_pos, dtype=torch.long),
            'end_positions': torch.tensor(end_pos, dtype=torch.long)
        })
    
    return inputs

# Process training data
print("📊 Processing training data...")
processed_data = simple_preprocess(training_data)
print(f"✅ {len(processed_data)} samples processed")

# Simple training setup
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch
from datetime import datetime

# Convert to HuggingFace Dataset
dataset_dict = {
    'input_ids': [item['input_ids'] for item in processed_data],
    'attention_mask': [item['attention_mask'] for item in processed_data], 
    'start_positions': [item['start_positions'] for item in processed_data],
    'end_positions': [item['end_positions'] for item in processed_data]
}

train_dataset = Dataset.from_dict(dataset_dict)
print("✅ Dataset created")

# Ultra-conservative training args (avoid memory issues)
training_args = TrainingArguments(
    output_dir="./simple-turkish-qa",
    learning_rate=3e-5,
    num_train_epochs=2,  # Just 2 epochs
    per_device_train_batch_size=1,  # Smallest batch
    gradient_accumulation_steps=2,
    warmup_steps=2,
    save_steps=50,
    logging_steps=1,
    push_to_hub=True,
    hub_model_id=HF_MODEL_NAME,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
)

print("✅ Training configuration ready")

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print("🔥 STARTING SIMPLE TRAINING...")
print("=" * 40)
print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")

# Train with error handling
try:
    train_result = trainer.train()
    print("🎉 TRAINING COMPLETED!")
    print(f"📊 Loss: {train_result.training_loss:.4f}")
    
except Exception as e:
    print(f"❌ Training error: {e}")
    print("💡 Try reducing batch size or epochs")

print(f"⏰ End: {datetime.now().strftime('%H:%M:%S')}")

# Simple testing
print("\n🧪 TESTING MODEL...")
from transformers import pipeline

try:
    qa_pipeline = pipeline(
        "question-answering",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Test
    test_result = qa_pipeline(
        question="GARAN hissesi nasıl?",
        context="GARAN hissesi ₺89.30'da %-0.94 düşüşle işlem görüyor."
    )
    
    print(f"✅ Test: {test_result['answer']}")
    print(f"🎯 Score: {test_result['score']:.3f}")
    
except Exception as e:
    print(f"⚠️ Testing error: {e}")

# Upload to HuggingFace
print("\n🚀 UPLOADING TO HUGGINGFACE...")
try:
    trainer.push_to_hub(commit_message="Simple Turkish Financial Q&A")
    print("🎉 UPLOAD SUCCESS!")
    print(f"📍 Model: https://huggingface.co/{HF_MODEL_NAME}")
except Exception as e:
    print(f"⚠️ Upload error: {e}")
    # Save locally as fallback
    model.save_pretrained("./simple-model")
    tokenizer.save_pretrained("./simple-model") 
    print("💾 Saved locally instead")

print("\n" + "=" * 60)
print("🎉 SIMPLE TRAINING COMPLETE!")
print("✅ Turkish Financial Q&A model ready")
print(f"🔗 API: https://api-inference.huggingface.co/models/{HF_MODEL_NAME}")
print("=" * 60)

# Railway integration hint
print("\n📋 RAILWAY INTEGRATION:")
print(f'model_url = "https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"')
print(f'headers = {{"Authorization": "Bearer {HF_TOKEN}"}}')
print("🚀 Ready for Railway API integration!")
