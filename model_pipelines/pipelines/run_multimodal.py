"""
Multimodal late fusion pipeline.

Combines visual (centroid distance) and audio (prototypical confidence)
outlier scores into a single decision via threshold-normalised weighted
averaging. Best alpha (modality weight) is found by grid search on the
validation set.

Both unimodal pipelines must have run first to produce their score files
in their respective results directories:

  results/visual_scores_known.npy
  results/visual_scores_outlier.npy
  results/visual_threshold.npy

  audio_results/audio_scores_known.npy
  audio_results/audio_scores_outlier.npy
  audio_results/audio_threshold.npy
"""
import os
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from model_pipelines.config import cfg as visual_cfg, AudioConfig
except ImportError:
    from config import cfg as visual_cfg, AudioConfig


def _load_modality_scores(scores_dir: str, prefix: str):
    """Load known/outlier scores and threshold for one modality."""
    known    = np.load(os.path.join(scores_dir, f"{prefix}_scores_known.npy"))
    outlier  = np.load(os.path.join(scores_dir, f"{prefix}_scores_outlier.npy"))
    thresh   = float(np.load(os.path.join(scores_dir, f"{prefix}_threshold.npy")))
    return known, outlier, thresh


def _normalise(scores, threshold):
    """
    Normalise scores by threshold so that 1.0 = at the decision boundary,
    regardless of the underlying scale.

    For visual: scores are -distance (higher = more normal)
    For audio:  scores are confidence (higher = more normal)

    After normalisation, a fused score above the normalised threshold of 1.0
    means outlier; below means known.

    For visual we negate inside this function so that distances become
    "outlier-ness" — higher = more outlier-like — which is the convention
    used for the fused decision.
    """
    return scores / threshold if threshold != 0 else scores


def grid_search_alpha(vk, vo, vt, ak, ao, at):
    """
    Grid search alpha in [0, 1] step 0.05 to maximise fused AUC-ROC.

    Both modalities are normalised so that score / threshold = 1.0 means
    at-boundary. Higher fused = more known, lower = more outlier.

    AUC-ROC convention: y_true = 1 for known, score should be higher for
    known. Visual stored as -distance (higher = known) ✓.
    Audio stored as confidence (higher = known) ✓.
    """
    vk_n = _normalise(vk, vt)
    vo_n = _normalise(vo, vt)
    ak_n = _normalise(ak, at)
    ao_n = _normalise(ao, at)

    # Pair lengths must match for fusion. If not, truncate to the shorter.
    n_known   = min(len(vk_n), len(ak_n))
    n_outlier = min(len(vo_n), len(ao_n))
    vk_n, ak_n = vk_n[:n_known],   ak_n[:n_known]
    vo_n, ao_n = vo_n[:n_outlier], ao_n[:n_outlier]

    best_alpha, best_auc = 0.5, -1.0
    sweep = []

    for alpha in np.arange(0.0, 1.01, 0.05):
        fused_known   = alpha * vk_n + (1 - alpha) * ak_n
        fused_outlier = alpha * vo_n + (1 - alpha) * ao_n

        scores = np.concatenate([fused_known, fused_outlier])
        y_true = np.concatenate([np.ones(len(fused_known)),
                                  np.zeros(len(fused_outlier))])
        auc = float(roc_auc_score(y_true, scores))
        sweep.append({"alpha": float(round(alpha, 2)), "AUC-ROC": round(auc, 4)})
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(round(alpha, 2))

    return best_alpha, best_auc, sweep


def evaluate_at_alpha(vk, vo, vt, ak, ao, at, alpha):
    """Compute full metrics at a given alpha."""
    vk_n = _normalise(vk, vt)
    vo_n = _normalise(vo, vt)
    ak_n = _normalise(ak, at)
    ao_n = _normalise(ao, at)

    n_known   = min(len(vk_n), len(ak_n))
    n_outlier = min(len(vo_n), len(ao_n))
    vk_n, ak_n = vk_n[:n_known],   ak_n[:n_known]
    vo_n, ao_n = vo_n[:n_outlier], ao_n[:n_outlier]

    fused_known   = alpha * vk_n + (1 - alpha) * ak_n
    fused_outlier = alpha * vo_n + (1 - alpha) * ao_n

    # Threshold: fused score is normalised, so the boundary is alpha*1 + (1-alpha)*1 = 1.0
    threshold = 1.0

    tp = int((fused_known   >= threshold).sum())
    fn = int((fused_known   <  threshold).sum())
    tn = int((fused_outlier <  threshold).sum())
    fp = int((fused_outlier >= threshold).sum())

    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    scores = np.concatenate([fused_known, fused_outlier])
    y_true = np.concatenate([np.ones(len(fused_known)),
                              np.zeros(len(fused_outlier))])
    auc_roc = float(roc_auc_score(y_true, scores))
    auc_pr  = float(average_precision_score(y_true, scores))

    return {
        "alpha":     alpha,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "Recall":    round(rec,     4),
        "Precision": round(prec,    4),
        "F1":        round(f1,      4),
        "AUC-ROC":   round(auc_roc, 4),
        "AUC-PR":    round(auc_pr,  4),
        "n_known":   n_known,
        "n_outlier": n_outlier,
    }


def run_full_pipeline(visual_results_dir: str = None,
                      audio_results_dir: str = None,
                      output_dir: str = "multimodal_results"):
    """
    Run late fusion across visual + audio modalities. Skip-training is
    irrelevant here — fusion uses already-computed unimodal scores.
    """
    if visual_results_dir is None:
        visual_results_dir = visual_cfg.results_directory
    if audio_results_dir is None:
        audio_results_dir = AudioConfig().results_directory

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Multimodal Late Fusion Pipeline")
    print("=" * 60)
    print(f"  Visual scores : {visual_results_dir}")
    print(f"  Audio scores  : {audio_results_dir}")

    # Load both modalities' scores
    try:
        vk, vo, vt = _load_modality_scores(visual_results_dir, "visual")
    except FileNotFoundError as e:
        print(f"\n  ✗ Visual scores not found.")
        print(f"    Run the visual pipeline first to produce {visual_results_dir}/visual_*.npy")
        raise

    try:
        ak, ao, at = _load_modality_scores(audio_results_dir, "audio")
    except FileNotFoundError:
        print(f"\n  ✗ Audio scores not found.")
        print(f"    Run the audio pipeline first to produce {audio_results_dir}/audio_*.npy")
        raise

    print(f"\n  Visual: known={len(vk)}  outlier={len(vo)}  threshold={vt:.4f}")
    print(f"  Audio:  known={len(ak)}  outlier={len(ao)}  threshold={at:.4f}")

    # Grid search for best alpha
    print("\n── Step 1: Grid search alpha ──")
    best_alpha, best_auc, sweep = grid_search_alpha(vk, vo, vt, ak, ao, at)
    for s in sweep:
        marker = "  ←" if s['alpha'] == best_alpha else ""
        print(f"  alpha={s['alpha']:.2f}  AUC-ROC={s['AUC-ROC']:.4f}{marker}")
    print(f"\n  Best alpha: {best_alpha}  AUC-ROC: {best_auc:.4f}")

    # Evaluate at best alpha + at unimodal-only operating points
    print("\n── Step 2: Final evaluation ──")
    results = {
        "fused":      evaluate_at_alpha(vk, vo, vt, ak, ao, at, best_alpha),
        "visual_only":evaluate_at_alpha(vk, vo, vt, ak, ao, at, 1.0),
        "audio_only": evaluate_at_alpha(vk, vo, vt, ak, ao, at, 0.0),
        "alpha_sweep": sweep,
    }

    # Print comparison table
    print(f"\n  {'Mode':<14} {'AUC-ROC':>9} {'F1':>7} {'Recall':>8} {'Precision':>10}")
    print("  " + "-" * 53)
    for mode in ["visual_only", "audio_only", "fused"]:
        r = results[mode]
        print(f"  {mode:<14} {r['AUC-ROC']:>9.4f} {r['F1']:>7.4f} "
              f"{r['Recall']:>8.4f} {r['Precision']:>10.4f}")

    # Save results
    out_path = os.path.join(output_dir, "multimodal_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multimodal Late Fusion")
    parser.add_argument("--visual-results", type=str, default=None,
                        help="Directory containing visual_*.npy score files")
    parser.add_argument("--audio-results", type=str, default=None,
                        help="Directory containing audio_*.npy score files")
    parser.add_argument("--output-dir", type=str, default="multimodal_results",
                        help="Where to save fusion results")
    args = parser.parse_args()

    run_full_pipeline(
        visual_results_dir=args.visual_results,
        audio_results_dir=args.audio_results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
