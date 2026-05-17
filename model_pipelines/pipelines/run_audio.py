#!/usr/bin/env python3
"""
Complete audio-only pipeline runner with prototypical probing (Bird-MAE paper).

This script implements a complete end-to-end pipeline for bird species classification
and novel species (outlier) detection using prototypical probing as described in 
the Bird-MAE paper (Rauch et al., 2024).

WHAT THIS SCRIPT DOES:
======================

1. DATA LOADING & SPLITTING
   - Reads metadata.json files from Xeno-Canto downloads for each species
   - Splits files into train (70%), validation (15%), and test (15%) sets
   - Known species (native birds) get positive labels (0, 1, 2, ...)
   - Outlier species (novel birds) get negative labels (-1, -2, ...) and go only to test set
   - Uses sliding window splitting to handle long audio files (creates overlapping 5s windows)

2. FEATURE EXTRACTION (One-time cost)
   - Loads frozen Bird-MAE model (85M parameters, pre-trained on 10,000+ bird species)
   - Converts each 5-second audio window into a spatial feature map [8, 32, 768]
   - Preserves spatial structure (8×32 grid of 768-dim patches) for prototypical matching
   - Caches features to disk for fast reuse across runs

3. PROTOTYPICAL PROBE TRAINING
   - Trains ONLY the lightweight prototypical probe (~153k parameters, not the frozen Bird-MAE)
   - Learns 20 prototype vectors per species (J=20 from paper ablation)
   - Each prototype learns to recognize a specific acoustic pattern in spectrogram patches
   - Uses BCE loss for classification + orthogonality loss to keep prototypes diverse
   - Saves best model checkpoint based on validation accuracy

4. OUTLIER DETECTION THRESHOLD
   - Computes confidence scores (max probability) on validation set
   - Takes percentile (default 95%) of correct predictions as outlier threshold
   - Files with confidence < threshold are flagged as "unknown species"

5. EVALUATION
   - Tests on known species (should have high confidence, > threshold)
   - Tests on outlier species (should have low confidence, < threshold)
   - Reports AUC-ROC, AUC-PR, F1, Accuracy, Precision, Recall
   - Per-class accuracy breakdown

6. OUTPUTS
   - Model checkpoint: {checkpoint_dir}/best_prototypical_probe.pt
   - Precomputed features: {checkpoint_dir}/window_features.npz
   - Threshold: {checkpoint_dir}/threshold.npy
   - Class mapping: {checkpoint_dir}/classes.json
   - Results: {results_dir}/results.json

ARCHITECTURE OVERVIEW:
======================

    Long Audio File (e.g., 30s)
        ↓
    Sliding Window (5s windows, 2.5s step)
        ↓
    Bird-MAE (Frozen, 85M params) → Spatial feature map [8, 32, 768]
        ↓
    Prototypical Probe (Learnable, 153k params)
        ↓
        For each species c, 20 prototypes p_{c,j} in ℝ⁷⁶⁸
        Cosine similarity at all 256 spatial locations
        max-pool across space: s_{c,j}
        logit_c = Σⱼ (w_{c,j} × s_{c,j}) + b_c
        ↓
    [Blackbird, BlueTit, GreatTit, ..., Outlier?]
        ↓
    If max confidence > threshold → Known species
    If max confidence < threshold → Novel species (outlier)

EXPECTED RUNTIME:
=================

    - First run: 1-2 hours (precomputing features + training)
    - Subsequent runs: 10-30 minutes (loading cached features + training)
    - Inference on single file: < 1 second

USAGE EXAMPLES:
===============

    # Run full pipeline (train + evaluate)
    python -m model_pipelines.pipelines.run_audio

    # Skip training (use existing checkpoint)
    python -m model_pipelines.pipelines.run_audio --skip-training

    # Override data root directory
    python -m model_pipelines.pipelines.run_audio --data-root /custom/path

"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_pipelines.config import AudioConfig
from model_pipelines.data.audio_dataset import build_audio_splits_with_windows 
from model_pipelines.models.audio_encoder import (
    BirdMAEModel,
    precompute_window_features,  
    create_window_dataloader      
)
from model_pipelines.training.train_audio import train_audio_model
from model_pipelines.outlier.evaluate import evaluate_outlier_detector_prototypical


def extract_probabilities_from_loader(model, loader, device):
    """Extract probabilities from the prototypical probe using window loader."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
            
            logits = model(features)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)
    
    return np.vstack(all_probs), np.concatenate(all_labels)


def compute_confidence_scores(probabilities):
    """Compute confidence scores for outlier detection (max probability)."""
    return np.max(probabilities, axis=1)


def compute_threshold_from_validation(val_probs, val_labels, percentile=95):
    """Compute confidence threshold for outlier detection."""
    val_confidences = compute_confidence_scores(val_probs)
    predictions = np.argmax(val_probs, axis=1)
    correct_mask = (predictions == val_labels)
    correct_confidences = val_confidences[correct_mask]
    
    if len(correct_confidences) > 0:
        threshold = np.percentile(correct_confidences, percentile)
    else:
        threshold = np.percentile(val_confidences, percentile)
    
    return threshold


def run_audio_pipeline(skip_training: bool = False, data_root: str = None):
    """Run the complete audio pipeline with prototypical probing."""
    cfg = AudioConfig()
    
    if data_root:
        cfg.root_dir = data_root
    
    os.makedirs(cfg.checkpoint_directory, exist_ok=True)
    os.makedirs(cfg.results_directory, exist_ok=True)
    
    print("="*70)
    print("           AUDIO PIPELINE - Prototypical Probing (Bird-MAE)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data root:            {cfg.data_root}")
    print(f"  Native species:       {len(cfg.get_active_native_folders())}")
    print(f"  Outlier species:      {len(cfg.get_active_outlier_folders())}")
    print(f"  Checkpoint dir:       {cfg.checkpoint_directory}")
    from model_pipelines.utils.device_utils import device_summary
    print(f"  Device:               {device_summary()}")
    print(f"  Window duration:      {getattr(cfg, 'window_duration', 5.0)}s")
    print(f"  Step duration:        {getattr(cfg, 'step_duration', 2.5)}s")
    
    # Step 1: Load and split data (using window-aware splitting)
    train_samples, val_samples, test_known, test_outlier = build_audio_splits_with_windows(cfg)
    
    # Save class mapping
    class_mapping = {
        i: cfg.folder_to_species.get(folder, folder)
        for i, folder in enumerate(cfg.native_folders)
        if os.path.exists(os.path.join(cfg.data_root, folder))
    }
    
    with open(os.path.join(cfg.checkpoint_directory, "classes.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\n  Native classes: {len(class_mapping)}")
    
    # Step 2: Precompute window features
    all_samples = train_samples + val_samples + test_known + test_outlier
    
    window_features, window_to_label, window_to_path = precompute_window_features(
    all_samples,
    cfg,
    )

    
    # Step 3: Train or load model
    checkpoint_path = os.path.join(cfg.checkpoint_directory, "best_prototypical_probe.pt")
    
    if not skip_training:
        # Pass window features to training function
        model = train_audio_model(cfg, train_samples, val_samples, window_features)
    else:
        device = cfg.device()
        model = BirdMAEModel(cfg, num_classes=cfg.number_of_classes)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            print(f"\nLoaded existing model from {checkpoint_path}")
        else:
            print(f"\n No checkpoint found at {checkpoint_path}")
            return

        model.eval()
    
    # Step 4: Extract validation probabilities using window loader
    print("\n" + "="*70)
    print("Computing Validation Probabilities")
    print("="*70)
    
    val_loader = create_window_dataloader(val_samples, window_features, window_to_label, window_to_path, cfg, shuffle=False)
    val_probs, val_labels = extract_probabilities_from_loader(model, val_loader, cfg.device())
    
    # Step 5: Compute threshold
    print("\n" + "="*70)
    print("Computing Outlier Detection Threshold")
    print("="*70)
    
    threshold = compute_threshold_from_validation(
        val_probs, val_labels, 
        percentile=getattr(cfg, 'percentile_of_threshold', 95)
    )
    
    np.save(os.path.join(cfg.checkpoint_directory, "threshold.npy"), threshold)
    print(f"  Confidence threshold: {threshold:.4f}")
    
    # Step 6: Evaluate using window loaders
    print("\n" + "="*70)
    print("Evaluating Outlier Detection")
    print("="*70)
    
    known_loader = create_window_dataloader(test_known, window_features, window_to_label, window_to_path, cfg, shuffle=False)
    known_probs, known_labels = extract_probabilities_from_loader(model, known_loader, cfg.device())
    known_confidences = compute_confidence_scores(known_probs)
    
    outlier_loader = create_window_dataloader(test_outlier, window_features, window_to_label, window_to_path, cfg, shuffle=False, include_outliers=True)
    outlier_probs, _ = extract_probabilities_from_loader(model, outlier_loader, cfg.device())
    outlier_confidences = compute_confidence_scores(outlier_probs)
    
    results = evaluate_outlier_detector_prototypical(
        known_confidences=known_confidences,
        outlier_confidences=outlier_confidences,
        threshold=threshold,
        known_labels=known_labels,
        class_mapping=class_mapping,
        results_directory=cfg.results_directory
    )
    
    # Save results
    results_path = os.path.join(cfg.results_directory, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Summary
    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)
    print(f"\n  AUC-ROC:    {results.get('auc_roc', 'N/A'):.4f}")
    print(f"  AUC-PR:     {results.get('auc_pr', 'N/A'):.4f}")
    print(f"  F1 Score:   {results.get('f1', 'N/A'):.4f}")
    print(f"  Accuracy:   {results.get('accuracy', 'N/A'):.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Audio Pipeline for Bird Anomaly Detection")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--data-root", type=str, default=None, help="Override data_root")
    
    args = parser.parse_args()
    run_audio_pipeline(skip_training=args.skip_training, data_root=args.data_root)


if __name__ == "__main__":
    main()