# 🚀 ALTERNATIVE FIX APPROACH - Different working combination!
# İlk approach çalışmadı, alternatif tested combination dene

print("🚀 ALTERNATIVE FIX APPROACH - MAMUT R600")
print("=" * 60)

print("🎯 STRATEGY: Try different known working combination")
print("Previous attempt failed with transformers.utils.generic error")
print("Trying: transformers 4.30.0 + huggingface_hub 0.16.4 + tokenizers 0.13.3")
print()

# STEP 1: Current state check
print("1️⃣ CURRENT STATE CHECK")
print("=" * 30)

import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except:
    print("PyTorch: Not available")

# STEP 2: Complete clean slate (aggressive approach)
print("\n2️⃣ COMPLETE CLEAN SLATE (AGGRESSIVE)")
print("=" * 30)

print("🔧 Uninstalling ALL potentially conflicting packages...")
conflicting_packages = [
    'transformers', 'huggingface_hub', 'accelerate', 'datasets', 
    'tokenizers', 'peft', 'sentence-transformers', 'diffusers'
]

for package in conflicting_packages:
    print(f"  Uninstalling {package}...")
    !pip uninstall -y {package} -q

print("✅ All conflicting packages removed!")

# STEP 3: Install working combination (older, more stable)
print("\n3️⃣ INSTALLING STABLE OLDER COMBINATION")
print("=" * 30)

# This combination is from early 2023, very stable
print("🔧 Installing proven stable versions...")

# Install in very specific order
print("  1. Installing tokenizers 0.13.3...")
!pip install tokenizers==0.13.3 --no-deps -q

print("  2. Installing huggingface_hub 0.16.4...")  
!pip install huggingface_hub==0.16.4 --no-deps -q

print("  3. Installing safetensors...")
!pip install safetensors==0.3.1 -q

print("  4. Installing transformers 4.30.0...")
!pip install transformers==4.30.0 --no-deps -q

print("  5. Installing accelerate 0.20.3...")
!pip install accelerate==0.20.3 --no-deps -q

print("  6. Installing datasets 2.12.0...")
!pip install datasets==2.12.0 --no-deps -q

print("✅ Stable combination installed!")

# STEP 4: Install critical dependencies manually
print("\n4️⃣ MANUAL DEPENDENCY INSTALLATION")
print("=" * 30)

dependencies = [
    'regex', 'requests', 'tqdm', 'numpy', 'packaging',
    'filelock', 'typing-extensions', 'pyyaml', 'fsspec',
    'pyarrow>=8.0.0', 'dill>=0.3.0', 'pandas', 'xxhash'
]

for dep in dependencies:
    print(f"  Installing {dep}...")
    !pip install {dep} -q

print("✅ Dependencies installed!")

# STEP 5: Validation test
print("\n5️⃣ VALIDATION TEST")
print("=" * 30)

validation_results = {}

# Test each package individually
packages_to_test = [
    ('tokenizers', 'tokenizers'),
    ('huggingface_hub', 'huggingface_hub'), 
    ('transformers', 'transformers'),
    ('accelerate', 'accelerate'),
    ('datasets', 'datasets')
]

for name, module in packages_to_test:
    try:
        imported = __import__(module)
        version = getattr(imported, '__version__', 'unknown')
        print(f"✅ {name}: {version}")
        validation_results[name] = 'SUCCESS'
    except Exception as e:
        print(f"❌ {name}: {str(e)[:80]}...")
        validation_results[name] = 'FAILED'

# STEP 6: Transformers specific tests
if validation_results.get('transformers') == 'SUCCESS':
    print("\n6️⃣ TRANSFORMERS FUNCTIONALITY TEST")
    print("=" * 30)
    
    try:
        print("Testing AutoTokenizer import...")
        from transformers import AutoTokenizer
        print("✅ AutoTokenizer import: SUCCESS")
        
        print("Testing AutoModel import...")  
        from transformers import AutoModelForQuestionAnswering
        print("✅ AutoModel import: SUCCESS")
        
        print("Testing pipeline import...")
        from transformers import pipeline
        print("✅ Pipeline import: SUCCESS")
        
        validation_results['transformers_functionality'] = 'SUCCESS'
        
    except Exception as e:
        print(f"❌ Transformers functionality: {e}")
        validation_results['transformers_functionality'] = 'FAILED'

# STEP 7: Turkish model loading test
if validation_results.get('transformers_functionality') == 'SUCCESS':
    print("\n7️⃣ TURKISH MODEL LOADING TEST")
    print("=" * 30)
    
    try:
        print("Loading Turkish BERT tokenizer...")
        model_name = "dbmdz/bert-base-turkish-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Turkish tokenizer loaded!")
        
        print("Loading Turkish BERT model...")
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        print("✅ Turkish model loaded!")
        
        print("Testing basic tokenization...")
        test_text = "GARAN hissesi bugün nasıl?"
        tokens = tokenizer.encode(test_text)
        print(f"✅ Tokenization: {len(tokens)} tokens")
        
        validation_results['turkish_model'] = 'SUCCESS'
        
    except Exception as e:
        print(f"❌ Turkish model loading: {e}")
        validation_results['turkish_model'] = 'FAILED'

# STEP 8: HuggingFace authentication test
print("\n8️⃣ HUGGINGFACE AUTHENTICATION TEST")
print("=" * 30)

try:
    from huggingface_hub import login
    
    HF_TOKEN = "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"
    login(HF_TOKEN)
    print("✅ HuggingFace authentication: SUCCESS")
    validation_results['hf_auth'] = 'SUCCESS'
    
except Exception as e:
    print(f"⚠️ HF Auth warning: {e}")
    validation_results['hf_auth'] = 'WARNING'

# STEP 9: Final assessment
print("\n9️⃣ FINAL ASSESSMENT")
print("=" * 30)

critical_tests = ['transformers', 'transformers_functionality', 'turkish_model']
all_critical_passed = all(validation_results.get(test) == 'SUCCESS' for test in critical_tests)

if all_critical_passed:
    print("🎉 ALL CRITICAL TESTS PASSED!")
    print("✅ Ready for Turkish Q&A training")
    
    # STEP 10: Quick training test
    print("\n🔟 QUICK TRAINING TEST")
    print("=" * 30)
    
    try:
        # Minimal training data
        training_data = [
            {
                "question": "GARAN hissesi nasıl?",
                "context": "GARAN hissesi ₺89.30'da düşüş gösteriyor.",
                "answer": "düşüş gösteriyor"
            },
            {
                "question": "RSI nedir?",
                "context": "RSI momentum göstergesidir.",  
                "answer": "momentum göstergesi"
            }
        ]
        
        print("Processing training data...")
        processed = []
        
        for item in training_data:
            encoded = tokenizer(
                item["question"],
                item["context"],
                max_length=128,  # Very short for speed
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Simple answer positioning
            context = item["context"]
            answer = item["answer"] 
            start_char = context.find(answer)
            
            if start_char >= 0:
                start_pos = max(1, len(context[:start_char].split()))
                end_pos = min(start_pos + len(answer.split()), 120)
            else:
                start_pos, end_pos = 1, 2
            
            processed.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'start_positions': torch.tensor(start_pos, dtype=torch.long),
                'end_positions': torch.tensor(end_pos, dtype=torch.long)
            })
        
        print(f"✅ {len(processed)} samples processed")
        
        # Create minimal dataset
        from datasets import Dataset
        dataset_dict = {
            'input_ids': [item['input_ids'] for item in processed],
            'attention_mask': [item['attention_mask'] for item in processed],
            'start_positions': [item['start_positions'] for item in processed],
            'end_positions': [item['end_positions'] for item in processed]
        }
        
        train_dataset = Dataset.from_dict(dataset_dict)
        print("✅ Dataset created")
        
        # Minimal training setup
        from transformers import TrainingArguments, Trainer, DefaultDataCollator
        
        training_args = TrainingArguments(
            output_dir="./test-turkish-qa",
            learning_rate=3e-5,
            num_train_epochs=1,  # Just 1 epoch for test
            per_device_train_batch_size=1,
            save_steps=10,
            logging_steps=1,
            push_to_hub=False,  # Don't upload test
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=DefaultDataCollator(),
        )
        
        print("🚀 Starting test training (1 epoch)...")
        train_result = trainer.train()
        print("✅ Test training completed!")
        
        # Test the trained model
        qa_pipeline = pipeline(
            "question-answering",
            model=trainer.model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        result = qa_pipeline(
            question="GARAN hissesi nasıl?",
            context="GARAN hissesi düşüş gösteriyor."
        )
        
        print(f"🧪 Test result: {result['answer']}")
        print(f"🎯 Score: {result['score']:.3f}")
        
        print("\n🎉 FULL TRAINING PIPELINE WORKING!")
        validation_results['training_test'] = 'SUCCESS'
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        validation_results['training_test'] = 'FAILED'
        
else:
    print("❌ Critical tests failed, cannot proceed to training")

# FINAL SUMMARY
print("\n" + "=" * 60)
print("🎯 FINAL SUMMARY")
print("=" * 60)

if all_critical_passed and validation_results.get('training_test') == 'SUCCESS':
    print("🎉 ALTERNATIVE FIX SUCCESS!")
    print("✅ All imports working")
    print("✅ Turkish model loading")
    print("✅ Training pipeline tested")
    print("✅ Ready for full production training")
    
    print("\n📋 WORKING VERSION COMBINATION:")
    print("transformers==4.30.0")
    print("huggingface_hub==0.16.4") 
    print("tokenizers==0.13.3")
    print("accelerate==0.20.3")
    print("datasets==2.12.0")
    print("🎯 This is a PROVEN stable combination!")
    
else:
    print("⚠️ ALTERNATIVE FIX PARTIALLY SUCCESSFUL")
    print("Some components working, others need adjustment")
    
    print("\n📋 RESULTS:")
    for test, result in validation_results.items():
        status = "✅" if result == 'SUCCESS' else "⚠️" if result == 'WARNING' else "❌"
        print(f"{status} {test}: {result}")

print("=" * 60)
