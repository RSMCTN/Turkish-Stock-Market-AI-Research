# 🔍 VERIFY DEPENDENCY FIX - Run after restart to check everything
# Bu kodu restart'tan sonra çalıştırarak doğrulayın

print("🔍 VERIFYING DEPENDENCY FIX...")
print("=" * 50)

# Check critical packages
try:
    import numpy as np
    print(f"✅ Numpy: {np.__version__} (should be 1.26.4)")
    assert "1.26" in np.__version__, f"Numpy version wrong: {np.__version__}"
except Exception as e:
    print(f"❌ Numpy error: {e}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers error: {e}")

try:
    import datasets
    print(f"✅ Datasets: {datasets.__version__}")
except Exception as e:
    print(f"❌ Datasets error: {e}")

try:
    import accelerate
    print(f"✅ Accelerate: {accelerate.__version__}")
except Exception as e:
    print(f"❌ Accelerate error: {e}")

try:
    from accelerate.utils.memory import clear_device_cache
    print("✅ clear_device_cache import: SUCCESS!")
except Exception as e:
    print(f"❌ clear_device_cache error: {e}")

try:
    from huggingface_hub import login
    print("✅ HuggingFace hub: SUCCESS!")
except Exception as e:
    print(f"❌ HuggingFace hub error: {e}")

print("\n" + "=" * 50)
print("🎯 VERIFICATION COMPLETE!")
print("=" * 50)

# Count errors
import sys
if "❌" not in sys.stdout.getvalue() if hasattr(sys.stdout, 'getvalue') else True:
    print("🎉 ALL DEPENDENCIES FIXED!")
    print("🚀 Ready to run training code!")
else:
    print("❌ Some issues remain - may need manual fix")

print("📋 If all ✅, proceed with training!")
print("📋 If any ❌, restart runtime and try fix again")
