# 🔍 WANDB INTEGRATION GUIDE - Turkish Q&A Model Tracking
# Wandb ile Turkish Financial Q&A model training'i nasıl track edelim

print("🔍 WANDB INTEGRATION GUIDE - MAMUT R600")
print("=" * 60)

print("🤖 WANDB.AI NEDİR?")
print("=" * 30)
print("Wandb (Weights & Biases) = ML experiment tracking platform")
print("• Real-time training monitoring")
print("• Loss/accuracy visualizations") 
print("• Model versioning & artifacts")
print("• Hyperparameter optimization")
print("• Team collaboration")
print()

print("📊 TURKISH Q&A PROJECT'İNDE KULLANIMI:")
print("=" * 40)

# STEP 1: Wandb setup for our project
wandb_setup = '''
# 1. Install wandb in Colab
!pip install wandb -q

# 2. Login to wandb
import wandb
wandb.login()  # Enter your API key

# 3. Initialize project
wandb.init(
    project="mamut-r600-turkish-qa",
    name="turkish-financial-qa-v1",
    config={
        "model": "dbmdz/bert-base-turkish-cased",
        "learning_rate": 2e-5,
        "batch_size": 4,
        "epochs": 3,
        "max_length": 384,
        "dataset_size": 10,
        "language": "turkish",
        "domain": "financial"
    }
)
'''

print("🚀 SETUP CODE:")
print(wandb_setup)

print("📈 TRAINING TRACKING:")
training_tracking = '''
# Training loop ile wandb integration
from transformers import TrainingArguments, Trainer
import wandb

# TrainingArguments with wandb
training_args = TrainingArguments(
    output_dir="./turkish-financial-qa",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=20,
    save_steps=20,
    # WANDB INTEGRATION:
    report_to="wandb",           # Enable wandb logging
    run_name="mamut-r600-qa",    # Run name in wandb
    logging_dir="./logs",        # Local logs
)

# Custom callback for additional metrics
class WandbCallback:
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            # Log additional metrics
            wandb.log({
                "train_loss": logs.get("train_loss", 0),
                "eval_loss": logs.get("eval_loss", 0),
                "learning_rate": logs.get("learning_rate", 0),
                "epoch": logs.get("epoch", 0)
            })

# Training with wandb tracking
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    tokenizer=tokenizer,
    callbacks=[WandbCallback()]
)

# Train and automatically log to wandb
trainer.train()

# Log final model performance
wandb.log({
    "final_train_loss": train_result.training_loss,
    "model_parameters": model.num_parameters(),
    "training_time_minutes": training_time_in_minutes
})
'''

print(training_tracking)

print("🧪 MODEL TESTING WITH WANDB:")
testing_wandb = '''
# Test model and log results to wandb
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

test_cases = [
    ("GARAN hissesi nasıl?", "GARAN hissesi %-0.94 düşüşle işlem görüyor.", "%-0.94 düşüşle"),
    ("RSI nedir?", "RSI momentum göstergesidir.", "momentum göstergesi"),
    ("BIST 100 nasıl?", "BIST 100 %1.25 yükselişle kapandı.", "%1.25 yükselişle")
]

test_results = []
for question, context, expected in test_cases:
    result = qa_pipeline(question=question, context=context)
    
    accuracy = 1.0 if expected.lower() in result['answer'].lower() else 0.0
    
    test_results.append({
        "question": question,
        "predicted_answer": result['answer'],
        "expected_answer": expected,
        "confidence": result['score'],
        "accuracy": accuracy
    })
    
    # Log individual test
    wandb.log({
        f"test_{len(test_results)}_confidence": result['score'],
        f"test_{len(test_results)}_accuracy": accuracy
    })

# Log overall test performance
avg_confidence = sum(r['confidence'] for r in test_results) / len(test_results)
avg_accuracy = sum(r['accuracy'] for r in test_results) / len(test_results)

wandb.log({
    "test_avg_confidence": avg_confidence,
    "test_avg_accuracy": avg_accuracy,
    "test_samples": len(test_results)
})

# Create results table
test_table = wandb.Table(
    columns=["Question", "Predicted", "Expected", "Confidence", "Accuracy"],
    data=[[r['question'], r['predicted_answer'], r['expected_answer'], 
           r['confidence'], r['accuracy']] for r in test_results]
)

wandb.log({"test_results": test_table})
'''

print(testing_wandb)

print("💾 MODEL ARTIFACT LOGGING:")
artifact_logging = '''
# Save model as wandb artifact
import os

# Save trained model locally first
model.save_pretrained("./final-turkish-qa-model")
tokenizer.save_pretrained("./final-turkish-qa-model")

# Create wandb artifact
model_artifact = wandb.Artifact(
    name="turkish-financial-qa-model",
    type="model",
    description="Turkish Financial Q&A model trained on MAMUT R600 data",
    metadata={
        "model_type": "question-answering",
        "base_model": "dbmdz/bert-base-turkish-cased",
        "language": "turkish",
        "domain": "financial",
        "training_samples": len(training_data),
        "final_loss": train_result.training_loss
    }
)

# Add model files to artifact
model_artifact.add_dir("./final-turkish-qa-model")

# Log artifact to wandb
wandb.log_artifact(model_artifact)

print("✅ Model artifact logged to wandb!")
'''

print(artifact_logging)

print("🔗 WANDB DASHBOARD ÖZELLİKLERİ:")
print("=" * 40)
features = [
    "📈 Real-time loss curves",
    "📊 Training metrics graphs", 
    "🎯 Hyperparameter comparison",
    "💾 Model artifact storage",
    "📋 Experiment logs & notes",
    "🔄 Model version tracking",
    "👥 Team collaboration",
    "📱 Mobile monitoring"
]

for feature in features:
    print(f"  {feature}")

print("\n🎯 MAMUT R600 PROJECT İÇİN FAYDALAR:")
print("=" * 40)
benefits = [
    "• Turkish Q&A model performance tracking",
    "• Different hyperparameter deneme comparison", 
    "• Training progress visualization",
    "• Model versioning (v1, v2, v3...)",
    "• Railway deployment için model artifacts",
    "• Future model improvements tracking"
]

for benefit in benefits:
    print(f"  {benefit}")

print("✅ WANDB FREE TIER:")
print("• Personal projekte ücretsiz")
print("• 100GB storage")  
print("• Unlimited experiments")
print("• Public/Private projects")

print("\n💡 ÖNERİ:")
print("Turkish Q&A training'inde wandb kullan:")
print("1. Training progress'i görsel olarak takip et")
print("2. Model performance'ı kaydet") 
print("3. Railway integration için model artifacts kullan")
print("4. Future improvements için baseline oluştur")

print("\n🔗 LINKS:")
print("• Website: https://wandb.ai")
print("• Docs: https://docs.wandb.ai")  
print("• Colab Integration: https://colab.research.google.com/github/wandb/examples")

print("\n🚀 SONRAKI ADIM:")
print("Turkish Q&A training koduna wandb integration ekle!")
print("Real-time olarak model performance'ı izle!")
