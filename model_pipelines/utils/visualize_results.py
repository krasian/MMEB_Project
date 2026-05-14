#!/usr/bin/env python3
"""
Generate visualizations from saved results.

USAGE:
    # Use default checkpoint directory (audio_checkpoints/)
    python -m model_pipelines.pipelines.visualize_results
    
    # Specify custom checkpoint directory
    python -m model_pipelines.pipelines.visualize_results --checkpoint-dir my_experiment/checkpoints
    
    # Specify specific checkpoint file
    python -m model_pipelines.pipelines.visualize_results --checkpoint-file audio_checkpoints/best_prototypical_probe.pt
    
    # Specify specific results JSON
    python -m model_pipelines.pipelines.visualize_results --results-file audio_results/results.json
    
    # Combine options
    python -m model_pipelines.pipelines.visualize_results --checkpoint-dir experiment_1 --results-dir experiment_1/results
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_pipelines.config import AudioConfig
from model_pipelines.data.audio_dataset import build_audio_splits
from model_pipelines.models.audio_encoder import (
    BirdMAEModel,
    create_dataloader,
    precompute_spatial_features
)
from model_pipelines.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_tsne,
    plot_training_curves
)


def extract_probabilities_from_loader(model, loader, device):
    """Extract probabilities from dataloader."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_probs), np.concatenate(all_labels)


def generate_visualizations(
    checkpoint_dir: str = None,
    checkpoint_file: str = None,
    results_dir: str = None,
    results_file: str = None,
    output_dir: str = None
):
    """
    Generate all visualizations from saved model and data.
    
    ARGUMENTS:
        checkpoint_dir: Directory containing model checkpoint and features
        checkpoint_file: Full path to specific checkpoint file
        results_dir: Directory containing results JSON
        results_file: Full path to specific results JSON
        output_dir: Directory to save visualizations (default: results_dir/visualizations)
    """
    cfg = AudioConfig()
    device = cfg.device()
    
    # Determine paths
    if checkpoint_file:
        checkpoint_path = checkpoint_file
        checkpoint_dir = os.path.dirname(checkpoint_file)
    elif checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "best_prototypical_probe.pt")
        checkpoint_dir = checkpoint_dir
    else:
        # Use default
        checkpoint_dir = cfg.checkpoint_directory
        checkpoint_path = os.path.join(checkpoint_dir, "best_prototypical_probe.pt")
    
    if results_file:
        results_path = results_file
        results_dir = os.path.dirname(results_file)
    elif results_dir:
        results_path = os.path.join(results_dir, "results.json")
        results_dir = results_dir
    else:
        # Use default
        results_dir = cfg.results_directory
        results_path = os.path.join(results_dir, "results.json")
    
    # Output directory
    if output_dir:
        vis_dir = output_dir
    else:
        vis_dir = os.path.join(results_dir, "visualizations")
    
    print("="*70)
    print("Generating Visualizations")
    print("="*70)
    print(f"\n  Checkpoint dir: {checkpoint_dir}")
    print(f"  Checkpoint file: {checkpoint_path}")
    print(f"  Results file: {results_path}")
    print(f"  Output dir: {vis_dir}")
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("   Run training first or specify correct path with --checkpoint-dir")
        return
    
    if not os.path.exists(results_path):
        print(f"\n⚠️ Warning: Results JSON not found at {results_path}")
        print("   Some visualizations may not be available")
    
    # Load class mapping
    classes_path = os.path.join(checkpoint_dir, "classes.json")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            class_mapping = json.load(f)
        class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
        print(f"\n  Loaded {len(class_names)} classes")
    else:
        print(f"\n⚠️ Warning: classes.json not found at {classes_path}")
        # Try to load from results
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            if 'per_class_accuracy' in results:
                class_names = list(results['per_class_accuracy'].keys())
                print(f"  Using class names from results: {len(class_names)} classes")
            else:
                class_names = [f"Class_{i}" for i in range(cfg.number_of_classes)]
        else:
            class_names = [f"Class_{i}" for i in range(cfg.number_of_classes)]
    
    # Load or precompute features
    features_path = os.path.join(checkpoint_dir, "spatial_features.npz")
    if os.path.exists(features_path):
        print(f"\n  Loading precomputed features from {features_path}")
        loaded = np.load(features_path, allow_pickle=True)
        spatial_features = loaded['features'].item()
        print(f"  Loaded {len(spatial_features)} feature maps")
    else:
        print(f"\n  Precomputing features (this may take a while)...")
        train_samples, val_samples, test_known, test_outlier = build_audio_splits(cfg)
        all_samples = train_samples + val_samples + test_known + test_outlier
        spatial_features = precompute_spatial_features(all_samples, cfg)
        # Save for future use
        np.savez(features_path, features=spatial_features)
        print(f"  Saved features to {features_path}")
    
    # Load model
    print(f"\n  Loading model from {checkpoint_path}")
    model = BirdMAEModel(cfg, num_classes=cfg.number_of_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'val_accuracy' in checkpoint:
            print(f"  Model validation accuracy: {checkpoint['val_accuracy']:.3f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load data splits
    print("\n  Loading data splits...")
    train_samples, val_samples, test_known, test_outlier = build_audio_splits(cfg)
    
    # Create loaders
    val_loader = create_dataloader(val_samples, spatial_features, cfg, shuffle=False)
    known_loader = create_dataloader(test_known, spatial_features, cfg, shuffle=False)
    outlier_loader = create_dataloader(test_outlier, spatial_features, cfg, shuffle=False)
    
    # Extract probabilities
    print("\n  Extracting probabilities from validation set...")
    val_probs, val_labels = extract_probabilities_from_loader(model, val_loader, device)
    
    print("  Extracting probabilities from known test set...")
    known_probs, known_labels = extract_probabilities_from_loader(model, known_loader, device)
    
    print("  Extracting probabilities from outlier test set...")
    outlier_probs, _ = extract_probabilities_from_loader(model, outlier_loader, device)
    
    # Create visualization directory
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    # 1. ROC Curve
    print("\n1. Generating ROC curve...")
    known_confidences = np.max(known_probs, axis=1)
    outlier_confidences = np.max(outlier_probs, axis=1)
    
    plot_roc_curve(
        scores=np.concatenate([known_confidences, outlier_confidences]),
        labels=np.concatenate([np.ones_like(known_confidences), np.zeros_like(outlier_confidences)]),
        save_path=os.path.join(vis_dir, "roc_curve.png")
    )
    
    # 2. Confusion Matrix
    print("\n2. Generating confusion matrix...")
    val_preds = np.argmax(val_probs, axis=1)
    plot_confusion_matrix(
        y_true=val_labels,
        y_pred=val_preds,
        class_names=class_names,
        save_path=os.path.join(vis_dir, "confusion_matrix.png")
    )
    
    # 3. t-SNE of validation probabilities
    print("\n3. Generating t-SNE visualization...")
    plot_tsne(
        features=val_probs,
        labels=val_labels,
        class_names=class_names,
        save_path=os.path.join(vis_dir, "tsne_val_probs.png")
    )
    
    # 4. t-SNE of known vs outlier confidences
    print("\n4. Generating known vs outlier t-SNE...")
    from model_pipelines.utils.visualization import set_plot_style
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    
    set_plot_style()
    
    # Create feature matrix for t-SNE (using confidence scores as 1D features)
    # Need at least 2D for t-SNE, so create a 2D feature
    known_features = np.column_stack([known_confidences, known_confidences * np.random.randn(len(known_confidences)) * 0.01])
    outlier_features = np.column_stack([outlier_confidences, outlier_confidences * np.random.randn(len(outlier_confidences)) * 0.01])
    
    all_features = np.vstack([known_features, outlier_features])
    all_labels_vis = np.concatenate([np.ones(len(known_confidences)), np.zeros(len(outlier_confidences))])
    
    # t-SNE
    perplexity = min(30, len(all_features) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    projections = tsne.fit_transform(all_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    known_mask = all_labels_vis == 1
    outlier_mask = all_labels_vis == 0
    
    ax.scatter(projections[known_mask, 0], projections[known_mask, 1], 
               c='blue', label='Known Species', alpha=0.7, s=50)
    ax.scatter(projections[outlier_mask, 0], projections[outlier_mask, 1], 
               c='red', label='Outlier Species', alpha=0.7, s=50)
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('t-SNE: Known vs Outlier Species (Confidence Space)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "tsne_known_vs_outlier.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Load and display results from JSON if available
    if os.path.exists(results_path):
        print("\n5. Loading saved results...")
        with open(results_path, "r") as f:
            results = json.load(f)
        
        print(f"\n  Results from {results_path}:")
        print(f"    AUC-ROC: {results.get('auc_roc', 'N/A')}")
        print(f"    AUC-PR: {results.get('auc_pr', 'N/A')}")
        print(f"    F1 Score: {results.get('f1', 'N/A')}")
        print(f"    Accuracy: {results.get('accuracy', 'N/A')}")
    
    print(f"\n✅ All visualizations saved to {vis_dir}")
    print(f"\n  Files generated:")
    print(f"    - roc_curve.png")
    print(f"    - confusion_matrix.png")
    print(f"    - tsne_val_probs.png")
    print(f"    - tsne_known_vs_outlier.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Use default paths
    python visualize_results.py
    
    # Specify custom checkpoint directory
    python visualize_results.py --checkpoint-dir my_experiment/audio_checkpoints
    
    # Specify specific checkpoint file
    python visualize_results.py --checkpoint-file best_model_epoch_20.pt
    
    # Specify results JSON
    python visualize_results.py --results-file my_experiment/results.json
    
    # Specify everything
    python visualize_results.py --checkpoint-dir exp1 --results-dir exp1/results --output-dir exp1/plots
        """
    )
    
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory containing model checkpoint and features (default: config checkpoint_directory)"
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default=None,
        help="Full path to specific checkpoint file (overrides --checkpoint-dir)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing results.json (default: config results_directory)"
    )
    parser.add_argument(
        "--results-file", type=str, default=None,
        help="Full path to specific results JSON file (overrides --results-dir)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save visualizations (default: results_dir/visualizations)"
    )
    
    args = parser.parse_args()
    
    generate_visualizations(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        results_dir=args.results_dir,
        results_file=args.results_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()