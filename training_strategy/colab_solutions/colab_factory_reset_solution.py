# 🔄 COLAB FACTORY RESET SOLUTION - En temiz çözüm!
# Karmaşık dependency conflicts için factory reset yaklaşımı

print("🔄 FACTORY RESET SOLUTION - MAMUT R600")
print("=" * 60)

# STEP 0: Factory Reset Instructions
print("🚨 FACTORY RESET NEEDED!")
print("=" * 40)
print("1️⃣ Runtime → Disconnect and delete runtime")
print("2️⃣ Runtime → Connect (yeni temiz runtime)")
print("3️⃣ Bu kodu çalıştır (minimal kurulum)")
print("=" * 40)
print()

# Check if this is a fresh runtime
import sys
print(f"🔍 Python: {sys.version}")

try:
    import numpy as np
    print(f"⚠️ Numpy already installed: {np.__version__}")
    print("🔄 Please do factory reset first!")
except ImportError:
    print("✅ Clean runtime - ready for minimal install!")

# STEP 1: Minimal GPU Check
import torch
print(f"🔥 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# STEP 2: Minimal package installation (avoid conflicts)
print("\n📦 MINIMAL PACKAGE INSTALLATION...")
print("=" * 40)

# Only install what we absolutely need
!pip install numpy==1.24.3 -q  # Older but stable
!pip install torch==2.0.1 -q   # Stable version
!pip install transformers==4.30.0 -q  # Older stable
!pip install datasets==2.12.0 -q
!pip install accelerate==0.20.0 -q
!pip install huggingface-hub==0.15.1 -q
!pip install scikit-learn -q

print("✅ Minimal packages installed!")

# STEP 3: Verify installation
print("\n🔍 VERIFICATION...")
try:
    import numpy as np
    import torch
    import transformers
    import datasets
    import accelerate
    from huggingface_hub import login
    
    print(f"✅ Numpy: {np.__version__}")
    print(f"✅ PyTorch: {torch.__version__}")  
    print(f"✅ Transformers: {transformers.__version__}")
    print(f"✅ GPU: {torch.cuda.is_available()}")
    
    # Critical test
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    print("✅ Model imports: SUCCESS!")
    
except Exception as e:
    print(f"❌ Import error: {e}")

print("\n" + "=" * 60)
print("🎯 FACTORY RESET SOLUTION COMPLETE!")
print("✅ Clean runtime with minimal stable packages")
print("🚀 Ready for simple Turkish Q&A training!")
print("=" * 60)
