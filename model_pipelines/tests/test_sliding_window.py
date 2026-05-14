#!/usr/bin/env python3
"""
Test script for sliding window functionality.

Run this script to verify that:
1. The sliding window correctly splits long audio files
2. Feature extraction works for each window
3. The prototypical probe processes windows correctly
4. The complete pipeline runs without errors

USAGE:
    python -m model_pipelines.tests.test_sliding_window
"""

import os
import sys
import numpy as np
import torch
import librosa
import soundfile as sf

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_pipelines.config import AudioConfig
from model_pipelines.models.audio_encoder import (
    BirdMAEExtractor,
    BirdMAEModel,
    precompute_window_features,
    create_window_dataloader
)
from model_pipelines.data.audio_dataset import build_audio_splits_with_windows


def create_test_audio_files(output_dir="test_audio", duration=30):
    """
    Create synthetic test audio files for testing.
    
    Creates:
    - A 30-second test file with synthetic bird-like calls at specific times
    - A 5-second test file (already correct length)
    - A 2-second test file (shorter than window)
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = 32000
    files_created = []
    
    # Test 1: Long file (30 seconds) with synthetic calls at 8-10s and 20-22s
    duration_long = 30
    samples_long = duration_long * sample_rate
    audio_long = np.zeros(samples_long, dtype=np.float32)
    
    # Add synthetic bird-like call at 8-10 seconds (240k-320k samples)
    call_duration = 2
    call_samples = call_duration * sample_rate
    t = np.linspace(0, call_duration, call_samples)
    # Create a chirp (rising frequency) - mimics bird call
    chirp = np.sin(2 * np.pi * (2000 + 1000 * t) * t)
    # Add harmonics
    chirp += 0.5 * np.sin(2 * np.pi * (4000 + 2000 * t) * t)
    # Apply envelope
    envelope = np.exp(-3 * t)  # Decay envelope
    chirp = chirp * envelope * 0.5
    
    # Place at 8 seconds
    start_8s = 8 * sample_rate
    audio_long[start_8s:start_8s + call_samples] = chirp
    
    # Place another at 20 seconds (different pattern)
    t2 = np.linspace(0, call_duration, call_samples)
    chirp2 = np.sin(2 * np.pi * (1500 + 800 * t2) * t2)
    chirp2 += 0.3 * np.sin(2 * np.pi * (3000 + 1600 * t2) * t2)
    envelope2 = np.exp(-2 * t2)
    chirp2 = chirp2 * envelope2 * 0.5
    
    start_20s = 20 * sample_rate
    audio_long[start_20s:start_20s + call_samples] = chirp2
    
    # Save
    long_path = os.path.join(output_dir, "test_long_30s.wav")
    sf.write(long_path, audio_long, sample_rate)
    files_created.append(('long', long_path, 30))
    
    # Test 2: Perfect 5-second file
    samples_5s = 5 * sample_rate
    audio_5s = np.random.randn(samples_5s).astype(np.float32) * 0.1
    # Add chirp in the middle
    chirp_5s = np.sin(2 * np.pi * 2500 * np.linspace(0, 5, samples_5s))
    chirp_5s *= np.exp(-3 * np.linspace(0, 5, samples_5s))
    audio_5s += chirp_5s * 0.5
    five_sec_path = os.path.join(output_dir, "test_5s.wav")
    sf.write(five_sec_path, audio_5s, sample_rate)
    files_created.append(('perfect', five_sec_path, 5))
    
    # Test 3: Short file (2 seconds) - will be padded
    samples_2s = 2 * sample_rate
    audio_2s = np.random.randn(samples_2s).astype(np.float32) * 0.1
    short_path = os.path.join(output_dir, "test_2s.wav")
    sf.write(short_path, audio_2s, sample_rate)
    files_created.append(('short', short_path, 2))
    
    print(f"  Created {len(files_created)} test audio files in {output_dir}")
    return files_created


def test_extractor():
    """Test that BirdMAEExtractor loads correctly."""
    print("\n" + "="*70)
    print("TEST 1: BirdMAEExtractor Loading")
    print("="*70)
    
    cfg = AudioConfig()
    try:
        extractor = BirdMAEExtractor(cfg)
        print("  ✓ BirdMAEExtractor loaded successfully")
        print(f"  ✓ Device: {extractor.device}")
        print(f"  ✓ Feature dimension: {extractor.embed_dim}")
        return extractor
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return None


def test_sliding_window_on_long_file(extractor, audio_path):
    """Test sliding window on a long audio file."""
    print("\n" + "="*70)
    print("TEST 2: Sliding Window on Long Audio File")
    print("="*70)
    
    print(f"  Audio file: {audio_path}")
    
    # Get audio info
    info = librosa.get_duration(path=audio_path)
    print(f"  Original duration: {info:.2f} seconds")
    
    # Process with sliding window
    window_duration = 5.0
    step_duration = 2.5
    
    result = extractor.process_with_sliding_window(
        audio_path,
        window_duration=window_duration,
        step_duration=step_duration,
        return_features=False  # Return audio windows for verification
    )
    
    print(f"\n  Results:")
    print(f"    - Total duration: {result['total_duration']:.2f}s")
    print(f"    - Number of windows: {result['num_windows']}")
    print(f"    - Window duration: {window_duration}s")
    print(f"    - Step duration: {step_duration}s")
    
    # Verify overlap
    if len(result['timestamps']) > 1:
        overlap = result['timestamps'][0]['end'] - result['timestamps'][1]['start']
        expected_overlap = window_duration - step_duration
        print(f"    - Overlap between windows: {overlap:.2f}s (expected: {expected_overlap:.2f}s)")
    
    # Verify timestamps
    print(f"\n  Window timestamps:")
    for i, ts in enumerate(result['timestamps'][:5]):  # Show first 5
        print(f"    Window {i}: {ts['start']:.2f}s - {ts['end']:.2f}s")
    if result['num_windows'] > 5:
        print(f"    ... and {result['num_windows'] - 5} more windows")
    
    # Check if we captured the synthetic calls (at ~8s and ~20s)
    call_windows = []
    for ts in result['timestamps']:
        if 8 < ts['start'] < 10 or 20 < ts['start'] < 22:
            call_windows.append(ts)
    
    if call_windows:
        print(f"\n  ✓ Synthetic calls captured in windows:")
        for ts in call_windows:
            print(f"    Window at {ts['start']:.2f}-{ts['end']:.2f}s")
    else:
        print(f"\n  ✗ Warning: Synthetic calls not captured in any window")
    
    return result


def test_window_feature_extraction(extractor, audio_path):
    """Test feature extraction for each window."""
    print("\n" + "="*70)
    print("TEST 3: Window Feature Extraction")
    print("="*70)
    
    result = extractor.precompute_window_features_for_file(
        audio_path,
        window_duration=5.0,
        step_duration=2.5
    )
    
    print(f"  Number of windows: {len(result)}")
    
    # Check feature shapes
    for window_key, data in list(result.items())[:3]:  # Check first 3
        features = data['features']
        print(f"    {window_key}: shape {features.shape}")
        
        # Verify shape matches expected [8, 32, 768]
        expected_shape = (8, 32, 768)
        if features.shape == expected_shape:
            print(f"      ✓ Correct shape")
        else:
            print(f"      ✗ Wrong shape! Expected {expected_shape}")
    
    return result


def test_dataloader_with_windows(samples, cfg):
    """Test that the window dataloader works."""
    print("\n" + "="*70)
    print("TEST 4: Window DataLoader")
    print("="*70)
    
    # Precompute window features
    print("  Precomputing window features...")
    window_features, window_to_label = precompute_window_features(samples, cfg)
    
    print(f"  Total windows: {len(window_features)}")
    print(f"  Unique labels: {set(window_to_label.values())}")
    
    # Create dataloader
    loader = create_window_dataloader(
        samples, 
        window_features, 
        window_to_label, 
        cfg, 
        shuffle=True
    )
    
    print(f"\n  DataLoader created:")
    print(f"    - Batch size: {cfg.batch_size}")
    print(f"    - Total batches: {len(loader)}")
    
    # Check a batch
    for batch_features, batch_labels in loader:
        print(f"\n  Sample batch:")
        print(f"    - Features shape: {batch_features.shape}")
        print(f"    - Labels shape: {batch_labels.shape}")
        print(f"    - Labels: {batch_labels.tolist()}")
        break
    
    return loader

def test_get_spatial_features_directly(extractor, audio_path):
    """Test get_spatial_features directly."""
    print("\n" + "="*70)
    print("TEST: Direct get_spatial_features call")
    print("="*70)
    
    try:
        features = extractor.get_spatial_features(audio_path)
        print(f"  ✓ get_spatial_features succeeded!")
        print(f"  ✓ Features shape: {features.shape}")
        return True
    except Exception as e:
        print(f"  ✗ get_spatial_features failed: {e}")
        return False


def test_end_to_end_pipeline(cfg, audio_path, label=0):
    """Test end-to-end pipeline with a single file."""
    print("\n" + "="*70)
    print("TEST 5: End-to-End Pipeline (Single File)")
    print("="*70)
    
    # Create dummy samples
    samples = [(audio_path, label)]
    
    # Precompute features
    print("  Precomputing window features...")
    window_features, window_to_label = precompute_window_features(samples, cfg)
    
    print(f"  Generated {len(window_features)} windows")
    
    # Create dataloader
    loader = create_window_dataloader(samples, window_features, window_to_label, cfg, shuffle=True)
    
    # Create model
    num_classes = cfg.number_of_classes
    model = BirdMAEModel(cfg, num_classes=num_classes)
    model.eval()
    
    device = cfg.device()
    model = model.to(device)
    
    print(f"\n  Model: {model.__class__.__name__}")
    print(f"    - Number of classes: {num_classes}")
    print(f"    - Prototypes per class: {cfg.num_prototypes}")
    
    # Process each window
    all_probs = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    print(f"\n  Results:")
    print(f"    - Windows processed: {len(all_probs)}")
    print(f"    - Predictions per window: {all_probs.shape}")
    print(f"    - Mean confidence: {np.mean(np.max(all_probs, axis=1)):.4f}")
    print(f"    - Max confidence: {np.max(all_probs):.4f}")
    
    return all_probs


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("SLIDING WINDOW TEST SUITE")
    print("="*70)
    
    cfg = AudioConfig()
    
    # Test 1: Create test audio files
    print("\n" + "="*70)
    print("Setting up test environment")
    print("="*70)
    test_files = create_test_audio_files()
    
    # Test 2: Load extractor
    extractor = test_extractor()
    if extractor is None:
        print("\n  ✗ Extractor failed to load. Aborting tests.")
        return
    
    # Test 3: Test sliding window on long file
    long_file = None
    for file_type, path, duration in test_files:
        if file_type == 'long':
            long_file = path
            break

    if long_file:
        test_get_spatial_features_directly(extractor, long_file)
    
    if long_file:
        test_sliding_window_on_long_file(extractor, long_file)
        test_window_feature_extraction(extractor, long_file)
    else:
        print("\n  ✗ Long test file not found")
    
    # Test 4: Test dataloader with samples
    samples = [(path, i) for i, (_, path, _) in enumerate(test_files)]
    test_dataloader_with_windows(samples, cfg)
    
    # Test 5: End-to-end test
    if long_file:
        all_probs = test_end_to_end_pipeline(cfg, long_file, label=0)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    # Check if we can import all required modules
    print("\n  Module imports:")
    modules = [
        ("BirdMAEExtractor", "from model_pipelines.models.audio_encoder import BirdMAEExtractor"),
        ("BirdMAEModel", "from model_pipelines.models.audio_encoder import BirdMAEModel"),
        ("precompute_window_features", "from model_pipelines.models.audio_encoder import precompute_window_features"),
        ("create_window_dataloader", "from model_pipelines.models.audio_encoder import create_window_dataloader"),
        ("build_audio_splits_with_windows", "from model_pipelines.data.audio_dataset import build_audio_splits_with_windows"),
    ]
    
    for name, import_stmt in modules:
        try:
            exec(import_stmt)
            print(f"    ✓ {name}")
        except Exception as e:
            print(f"    ✗ {name}: {e}")
    
    print("\n" + "="*70)
    print("To run the full pipeline with your real data:")
    print("="*70)
    print("\n  python -m model_pipelines.pipelines.run_audio")
    print("\nOr with skip training:")
    print("  python -m model_pipelines.pipelines.run_audio --skip-training")
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick verification without Bird-MAE")
    args = parser.parse_args()
    
    run_all_tests()