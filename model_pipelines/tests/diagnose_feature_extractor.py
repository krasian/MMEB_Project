# diagnose_feature_extractor.py
import sys
import os
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pipelines.config import AudioConfig
from transformers import AutoFeatureExtractor, AutoModel

cfg = AudioConfig()

print("="*70)
print("Diagnosing Bird-MAE Feature Extractor")
print("="*70)

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "DBD-research-group/Bird-MAE-Base",
    trust_remote_code=True
)

print(f"\nFeature extractor type: {type(feature_extractor)}")
print(f"Feature extractor class: {feature_extractor.__class__.__name__}")

# Check the __call__ method signature
import inspect
print("\n" + "="*60)
print("__call__ method signature:")
print("="*60)
try:
    sig = inspect.signature(feature_extractor.__call__)
    print(f"  {sig}")
except Exception as e:
    print(f"  Could not get signature: {e}")

# Try different calling conventions
print("\n" + "="*60)
print("Testing different calling conventions:")
print("="*60)


test_audio = np.zeros(32000 * 5, dtype=np.float32)  # 5 seconds of silence

# Test 1: No sampling_rate
print("\n1. Testing: feature_extractor(audio, return_tensors='pt')")
try:
    result = feature_extractor(test_audio, return_tensors="pt")
    print(f"   ✓ SUCCESS! Result keys: {result.keys()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 2: With sampling_rate
print("\n2. Testing: feature_extractor(audio, sampling_rate=32000, return_tensors='pt')")
try:
    result = feature_extractor(test_audio, sampling_rate=32000, return_tensors="pt")
    print(f"   ✓ SUCCESS! Result keys: {result.keys()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 3: With sample_rate (singular)
print("\n3. Testing: feature_extractor(audio, sample_rate=32000, return_tensors='pt')")
try:
    result = feature_extractor(test_audio, sample_rate=32000, return_tensors="pt")
    print(f"   ✓ SUCCESS! Result keys: {result.keys()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test contiguous vs non-contiguous
print("\n" + "="*60)
print("Testing contiguous vs non-contiguous arrays:")
print("="*60)

long_audio = np.random.randn(32000 * 10).astype(np.float32)
sliced_audio = long_audio[16000:48000]  # This is a view
copied_audio = sliced_audio.copy()       # This is a copy

print(f"  Sliced audio (view) - contiguous: {sliced_audio.flags['C_CONTIGUOUS']}")
print(f"  Copied audio (copy) - contiguous: {copied_audio.flags['C_CONTIGUOUS']}")

print("\n4. Testing with non-contiguous (sliced) array:")
try:
    result = feature_extractor(sliced_audio, sampling_rate=32000, return_tensors="pt")
    print(f"   ✓ SUCCESS! (non-contiguous works)")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n5. Testing with contiguous (copied) array:")
try:
    result = feature_extractor(copied_audio, sampling_rate=32000, return_tensors="pt")
    print(f"   ✓ SUCCESS!")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n" + "="*70)
print("Diagnosis complete!")