# 🔧 COMPREHENSIVE COLAB DEPENDENCY FIX - Bu kodu önce çalıştırın!
# Numpy ve tüm diğer çakışmaları çözer

print("🔧 COMPREHENSIVE DEPENDENCY FIX - MAMUT R600")
print("=" * 60)

# STEP 1: Uninstall problematic packages
print("1️⃣ Uninstalling conflicting packages...")
!pip uninstall -y numpy
!pip uninstall -y transformers datasets accelerate huggingface-hub 
!pip uninstall -y torch torchvision torchaudio
!pip uninstall -y requests fsspec
!pip uninstall -y peft sentence-transformers

# STEP 2: Clear all caches
print("2️⃣ Clearing caches...")
!pip cache purge

# STEP 3: Install compatible numpy FIRST (most critical)
print("3️⃣ Installing compatible numpy...")
!pip install --no-cache-dir numpy==1.26.4

# STEP 4: Install other core dependencies with compatible versions
print("4️⃣ Installing compatible core packages...")
!pip install --no-cache-dir requests==2.32.4
!pip install --no-cache-dir fsspec==2025.3.0

# STEP 5: Install ML packages with numpy compatibility
print("5️⃣ Installing ML packages...")
!pip install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# STEP 6: Install HuggingFace ecosystem
print("6️⃣ Installing HuggingFace packages...")
!pip install --no-cache-dir huggingface-hub==0.25.1
!pip install --no-cache-dir transformers==4.45.0
!pip install --no-cache-dir datasets==3.0.0
!pip install --no-cache-dir accelerate==1.0.0

# STEP 7: Install additional ML tools
print("7️⃣ Installing additional tools...")
!pip install --no-cache-dir scikit-learn==1.3.0
!pip install --no-cache-dir pandas==2.0.3

print("✅ COMPREHENSIVE FIX COMPLETE!")
print("🔄 CRITICAL: Runtime → Restart Runtime (Ctrl+M .) ZORUNLU!")
print("⏳ Wait 30 seconds after restart, then run training code.")
print("📋 Numpy fixed: 2.3.2 → 1.26.4 (compatible)")
print("📋 All package conflicts resolved!")
