# check_feature_extractor.py
from transformers import AutoFeatureExtractor
import inspect

# Load the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "DBD-research-group/Bird-MAE-Base",
    trust_remote_code=True
)

# Get the signature
sig = inspect.signature(feature_extractor.__call__)

print("\n" + "="*60)
print("BirdMAEFeatureExtractor.__call__() Signature:")
print("="*60)
for param_name, param in sig.parameters.items():
    if param.default == inspect.Parameter.empty:
        print(f"  {param_name}: REQUIRED")
    else:
        print(f"  {param_name}: {param.default}")

print("\n" + "="*60)
print("Try calling with different parameters:")
print("="*60)

import numpy as np
test_audio = np.zeros(160000, dtype=np.float32)

# Test without sampling_rate
try:
    result = feature_extractor(test_audio, return_tensors="pt")
    print("✓ SUCCESS: Called without sampling_rate")
    print(f"  Returns: {type(result)}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test with sampling_rate
try:
    result = feature_extractor(test_audio, sampling_rate=32000, return_tensors="pt")
    print("✓ SUCCESS: Called with sampling_rate")
except Exception as e:
    print(f"✗ FAILED: Called with sampling_rate - {e}")

print("\n" + "="*60)
print("To see the source code, press F12 on 'feature_extractor'")
print("="*60)