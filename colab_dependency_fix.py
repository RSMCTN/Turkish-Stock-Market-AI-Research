# 🔧 COLAB DEPENDENCY FIX - Bu kodu önce çalıştırın!
# After running this, restart runtime, then run main training code

print("🔧 FIXING COLAB DEPENDENCY CONFLICTS...")
print("=" * 50)

# STEP 1: Uninstall conflicting packages
print("1️⃣ Uninstalling conflicting packages...")
!pip uninstall -y transformers datasets accelerate huggingface-hub peft sentence-transformers

# STEP 2: Clear cache
print("2️⃣ Clearing pip cache...")
!pip cache purge

# STEP 3: Install fresh compatible versions (forced)
print("3️⃣ Installing fresh compatible packages...")
!pip install --no-cache-dir --force-reinstall huggingface-hub==0.25.1
!pip install --no-cache-dir --force-reinstall transformers==4.45.0
!pip install --no-cache-dir --force-reinstall datasets==3.0.0
!pip install --no-cache-dir --force-reinstall accelerate==1.0.0
!pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install --no-cache-dir scikit-learn numpy pandas

print("✅ Package installation complete!")
print("🔄 CRITICAL: Runtime → Restart Runtime (Ctrl+M .) ZORUNLU!")
print("📋 After restart, run the main training code.")
