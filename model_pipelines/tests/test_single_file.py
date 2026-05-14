# test_single_file.py
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from model_pipelines.config import AudioConfig
from model_pipelines.data.audio_dataset import build_audio_splits_with_windows
from model_pipelines.models.audio_encoder import precompute_window_features, create_window_dataloader, BirdMAEModel

def test_single_file():
    print("="*70)
    print("TESTING SINGLE FILE PIPELINE")
    print("="*70)
    
    cfg = AudioConfig()
    
    # Quick test mode - use just 1 epoch and small batch
    cfg.epochs = 1
    cfg.batch_size = 2
    
    print(f"\nConfiguration:")
    print(f"  Data root: {cfg.data_root}")
    print(f"  Native species: {cfg.get_active_native_folders()}")
    
    # Load data splits
    print("\n1. Loading data splits...")
    train_samples, val_samples, test_known, test_outlier = build_audio_splits_with_windows(cfg)
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  Test known: {len(test_known)}")
    print(f"  Test outlier: {len(test_outlier)}")
    
    if len(train_samples) == 0:
        print("\n✗ No training samples found! Check your data path.")
        return False
    
    # Take just 2 files for testing
    test_samples = train_samples[:2] + val_samples[:2]
    print(f"\n2. Testing with {len(test_samples)} files...")
    
    # Precompute features
    print("\n3. Precomputing window features...")
    try:
        window_features, window_to_label = precompute_window_features(test_samples, cfg)
        print(f"  Generated {len(window_features)} windows")
    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
        return False
    
    if len(window_features) == 0:
        print("\n✗ No windows generated! Check your audio files.")
        return False
    
    # Test dataloader
    print("\n4. Testing DataLoader...")
    try:
        loader = create_window_dataloader(test_samples, window_features, window_to_label, cfg, shuffle=True)
        print(f"  DataLoader created with {len(loader)} batches")
    except Exception as e:
        print(f"  ✗ DataLoader creation failed: {e}")
        return False
    
    # Test model creation
    print("\n5. Testing model creation...")
    try:
        model = BirdMAEModel(cfg, num_classes=cfg.number_of_classes)
        print(f"  Model created with {cfg.number_of_classes} classes")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False
    
    # Test forward pass
    print("\n6. Testing forward pass...")
    device = cfg.device()
    model = model.to(device)
    model.eval()
    
    try:
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            with torch.no_grad():
                logits = model(batch_features)
            print(f"  Forward pass successful!")
            print(f"    - Features shape: {batch_features.shape}")
            print(f"    - Logits shape: {logits.shape}")
            print(f"    - Labels: {batch_labels.tolist()}")
            break
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED! Pipeline is ready.")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_single_file()
    sys.exit(0 if success else 1)