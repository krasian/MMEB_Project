#!/usr/bin/env python3
"""
Unified CLI for the bird novelty detection project.

Usage:
    python main.py train     visual [--skip-training] [--data-root PATH]
    python main.py train     audio  [--skip-training] [--data-root PATH]
    python main.py train     multimodal

    python main.py evaluate  visual
    python main.py evaluate  audio
    python main.py evaluate  multimodal

    python main.py predict   visual <path> [--save]
    python main.py predict   audio  <path> [--save]
    python main.py predict   multimodal --image <path> --audio <path>

For visual: trains EfficientNetB0 + ArcFace + centroid/Mahalanobis OOD detection.
For audio:  trains prototypical probe on Bird-MAE features + confidence threshold.
For multimodal: late fusion of saved visual + audio scores via grid-searched alpha.

Both visual and audio support DirectML (AMD GPU on Windows), CUDA (NVIDIA),
and CPU automatically via utils/device_utils.py — no extra config needed.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure both `model_pipelines.X` and `X` (top-level) imports work regardless
# of how the script is invoked. This lets visual code and audio code coexist
# even though they were written with different import styles.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd


def _configure_console_output():
    """Keep decorative CLI output from crashing on legacy Windows encodings."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


# ─────────────────────────────────────────────
# COMMON HELPERS
# ─────────────────────────────────────────────

def _image_extensions():
    """Read supported image extensions from cfg, with a safe fallback."""
    try:
        from config import cfg
        return getattr(cfg, "image_extensions",
                       {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    except ImportError:
        return {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _audio_extensions():
    return {".wav", ".mp3", ".flac", ".ogg"}


# ─────────────────────────────────────────────
# VISUAL HANDLERS
# ─────────────────────────────────────────────

def _predict_single_image(detector, image_path: str):
    result = detector.predict(image_path)
    if "error" in result:
        print(f"  ✗ Error: {result['error']}")
        return

    status = "OUTLIER ⚠" if result["is_outlier"] else "KNOWN   ✓"
    margin = result["distance"] - result["threshold"]
    print(f"\n  File     : {Path(image_path).name}")
    print(f"  Result   : {status}")
    print(f"  Predicted: {result['predicted_class']}")
    print(f"  Distance : {result['distance']:.4f}  "
          f"(threshold: {result['threshold']:.4f}, margin: {margin:+.4f})")


def _predict_image_folder(detector, folder_path: str, save_csv: bool = False):
    folder = Path(folder_path)
    exts = _image_extensions()
    images = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not images:
        print(f"  No images found in {folder_path}")
        return

    from config import cfg
    csv_name = getattr(cfg, "csv_out", "predictions.csv")
    print(f"\n  Scoring {len(images)} images in {folder_path}\n")
    print(f"  {'File':<40} {'Result':<12} {'Predicted Class':<25} "
          f"{'Distance':>10} {'Threshold':>10}")
    print("  " + "-" * 103)

    rows, n_known, n_outlier = [], 0, 0
    for img_path in images:
        result = detector.predict(str(img_path))
        if "error" in result:
            print(f"  {'ERROR':<40} {result['error']}")
            continue
        status = "OUTLIER ⚠" if result["is_outlier"] else "KNOWN   ✓"
        if result["is_outlier"]:
            n_outlier += 1
        else:
            n_known += 1
        print(f"  {img_path.name:<40} {status:<12} "
              f"{result['predicted_class']:<25} "
              f"{result['distance']:>10.4f} {result['threshold']:>10.4f}")
        rows.append({
            "file": str(img_path),
            "is_outlier": result["is_outlier"],
            "predicted_class": result["predicted_class"],
            "distance": result["distance"],
            "threshold": result["threshold"],
        })

    total = n_known + n_outlier
    print(f"\n  Summary: {n_known}/{total} known  |  {n_outlier}/{total} outliers")

    if save_csv and rows:
        out_path = folder / csv_name
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  ✓ Results saved to {out_path}")


def _cmd_train_visual(args):
    from pipelines.run_visual import run_full_pipeline
    run_full_pipeline(skip_training=args.skip_training)


def _cmd_evaluate_visual(args):
    from pipelines.run_visual import run_evaluation
    run_evaluation()


def _cmd_predict_visual(args):
    from inference.visual_detector import VisualAnomalyDetector
    print("── Loading visual model ──")
    detector = VisualAnomalyDetector(checkpoint_dir=args.checkpoint_dir)
    print(f"  Classes   : {detector.classes}")
    print(f"  Threshold : {detector.centroid_threshold:.4f}")

    target = Path(args.path)
    if target.is_file():
        _predict_single_image(detector, str(target))
    elif target.is_dir():
        _predict_image_folder(detector, str(target), save_csv=args.save)
    else:
        print(f"  ✗ Path not found: {args.path}")
        sys.exit(1)


# ─────────────────────────────────────────────
# AUDIO HANDLERS
# ─────────────────────────────────────────────

def _predict_single_audio(detector, audio_path: str):
    result = detector.predict(audio_path)
    if "error" in result:
        print(f"  ✗ Error: {result['error']}")
        return

    status = "OUTLIER ⚠" if result["is_outlier"] else "KNOWN   ✓"
    print(f"\n  File       : {Path(audio_path).name}")
    print(f"  Result     : {status}")
    print(f"  Predicted  : {result['predicted_class']}")
    print(f"  Confidence : {result['confidence']:.4f}  "
          f"(threshold: {result['threshold']:.4f})")
    print(f"  Windows    : {result['n_windows']}")


def _predict_audio_folder(detector, folder_path: str, save_csv: bool = False):
    folder = Path(folder_path)
    exts = _audio_extensions()
    files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not files:
        print(f"  No audio files found in {folder_path}")
        return

    print(f"\n  Scoring {len(files)} audio files in {folder_path}\n")
    print(f"  {'File':<40} {'Result':<12} {'Predicted Class':<25} "
          f"{'Confidence':>10} {'Threshold':>10}")
    print("  " + "-" * 103)

    rows, n_known, n_outlier = [], 0, 0
    for audio_path in files:
        result = detector.predict(str(audio_path))
        if "error" in result:
            print(f"  {'ERROR':<40} {result['error']}")
            continue
        status = "OUTLIER ⚠" if result["is_outlier"] else "KNOWN   ✓"
        if result["is_outlier"]:
            n_outlier += 1
        else:
            n_known += 1
        print(f"  {audio_path.name:<40} {status:<12} "
              f"{result['predicted_class']:<25} "
              f"{result['confidence']:>10.4f} {result['threshold']:>10.4f}")
        rows.append({
            "file": str(audio_path),
            "is_outlier": result["is_outlier"],
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
        })

    total = n_known + n_outlier
    print(f"\n  Summary: {n_known}/{total} known  |  {n_outlier}/{total} outliers")

    if save_csv and rows:
        out_path = folder / "audio_predictions.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  ✓ Results saved to {out_path}")


def _cmd_train_audio(args):
    try:
        from pipelines.run_audio import run_audio_pipeline
    except ImportError:
        from model_pipelines.pipelines.run_audio import run_audio_pipeline
    run_audio_pipeline(skip_training=args.skip_training, data_root=args.data_root)


def _cmd_evaluate_audio(args):
    # Audio's run_audio_pipeline does both training and evaluation; with
    # --skip-training it just re-evaluates from saved checkpoints.
    try:
        from pipelines.run_audio import run_audio_pipeline
    except ImportError:
        from model_pipelines.pipelines.run_audio import run_audio_pipeline
    run_audio_pipeline(skip_training=True, data_root=None)


def _cmd_predict_audio(args):
    try:
        from inference.audio_detector import AudioAnomalyDetector
    except ImportError:
        from model_pipelines.inference.audio_detector import AudioAnomalyDetector

    print("── Loading audio model ──")
    detector = AudioAnomalyDetector(checkpoint_dir=args.checkpoint_dir)
    print(f"  Classes   : {detector.classes}")
    print(f"  Threshold : {detector.threshold:.4f}")

    target = Path(args.path)
    if target.is_file():
        _predict_single_audio(detector, str(target))
    elif target.is_dir():
        _predict_audio_folder(detector, str(target), save_csv=args.save)
    else:
        print(f"  ✗ Path not found: {args.path}")
        sys.exit(1)


# ─────────────────────────────────────────────
# MULTIMODAL HANDLERS
# ─────────────────────────────────────────────

def _cmd_train_multimodal(args):
    """Multimodal has no training — it fuses already-trained unimodal scores."""
    print("  Multimodal does not require its own training.")
    print("  Train the visual and audio pipelines first, then run:")
    print("    python main.py evaluate multimodal")
    sys.exit(0)


def _cmd_evaluate_multimodal(args):
    try:
        from pipelines.run_multimodal import run_full_pipeline
    except ImportError:
        from model_pipelines.pipelines.run_multimodal import run_full_pipeline
    run_full_pipeline()


def _cmd_predict_multimodal(args):
    """Score one image+audio pair using late fusion."""
    try:
        from inference.visual_detector import VisualAnomalyDetector
        from inference.audio_detector  import AudioAnomalyDetector
    except ImportError:
        from model_pipelines.inference.visual_detector import VisualAnomalyDetector
        from model_pipelines.inference.audio_detector  import AudioAnomalyDetector

    print("── Loading both modalities ──")
    visual = VisualAnomalyDetector(checkpoint_dir=args.checkpoint_dir)
    audio  = AudioAnomalyDetector(checkpoint_dir=args.audio_checkpoint_dir)

    print(f"  Visual threshold: {visual.centroid_threshold:.4f}")
    print(f"  Audio threshold : {audio.threshold:.4f}")

    v_result = visual.predict(args.image)
    a_result = audio.predict(args.audio)

    if "error" in v_result or "error" in a_result:
        print(f"  Visual: {v_result}")
        print(f"  Audio:  {a_result}")
        sys.exit(1)

    # Normalise by threshold
    v_norm = -v_result["distance"]   / visual.centroid_threshold  # higher = known
    a_norm = a_result["confidence"]  / audio.threshold            # higher = known

    # Default alpha 0.5 — for a single sample we can't grid search
    alpha = args.alpha
    fused = alpha * v_norm + (1 - alpha) * a_norm
    is_outlier = fused < 1.0

    print(f"\n  Visual prediction : {v_result['predicted_class']} "
          f"(distance={v_result['distance']:.3f}, "
          f"normalised={v_norm:.3f})")
    print(f"  Audio prediction  : {a_result['predicted_class']} "
          f"(confidence={a_result['confidence']:.3f}, "
          f"normalised={a_norm:.3f})")
    print(f"\n  Fused score (alpha={alpha}): {fused:.3f}  "
          f"({'OUTLIER ⚠' if is_outlier else 'KNOWN ✓'})")


# ─────────────────────────────────────────────
# ARGPARSE
# ─────────────────────────────────────────────

def _add_common(parser):
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (visual only).")
    parser.add_argument("--data-root", default=None,
                        help="Override data root directory.")


def _add_train(parser):
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, recompute centroids/threshold only.")


def _add_predict_unimodal(parser):
    parser.add_argument("path", help="Path to a single file or folder.")
    parser.add_argument("--save", action="store_true",
                        help="Save folder results to CSV (folder mode only).")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint directory.")


def _add_predict_multimodal(parser):
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Visual modality weight (0..1). Default 0.5.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Visual checkpoint directory.")
    parser.add_argument("--audio-checkpoint-dir", default=None,
                        help="Audio checkpoint directory.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bird novelty detection — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    top = parser.add_subparsers(dest="cmd", required=True)

    # ── train ──
    p_train = top.add_parser("train", help="Train a model.")
    train_sub = p_train.add_subparsers(dest="modality", required=True)

    p_tv = train_sub.add_parser("visual", help="Train visual pipeline.")
    _add_train(p_tv); _add_common(p_tv)
    p_tv.set_defaults(func=_cmd_train_visual)

    p_ta = train_sub.add_parser("audio", help="Train audio pipeline.")
    _add_train(p_ta); _add_common(p_ta)
    p_ta.set_defaults(func=_cmd_train_audio)

    p_tm = train_sub.add_parser("multimodal", help="(no-op) Multimodal has no training.")
    p_tm.set_defaults(func=_cmd_train_multimodal)

    # ── evaluate ──
    p_eval = top.add_parser("evaluate", help="Evaluate an existing checkpoint.")
    eval_sub = p_eval.add_subparsers(dest="modality", required=True)

    p_ev = eval_sub.add_parser("visual", help="Evaluate visual pipeline.")
    _add_common(p_ev)
    p_ev.set_defaults(func=_cmd_evaluate_visual)

    p_ea = eval_sub.add_parser("audio", help="Evaluate audio pipeline.")
    _add_common(p_ea)
    p_ea.set_defaults(func=_cmd_evaluate_audio)

    p_em = eval_sub.add_parser("multimodal", help="Run late fusion across modalities.")
    _add_common(p_em)
    p_em.set_defaults(func=_cmd_evaluate_multimodal)

    # ── predict ──
    p_pred = top.add_parser("predict", help="Score one input or a folder.")
    pred_sub = p_pred.add_subparsers(dest="modality", required=True)

    p_pv = pred_sub.add_parser("visual", help="Score one image or folder.")
    _add_predict_unimodal(p_pv); _add_common(p_pv)
    p_pv.set_defaults(func=_cmd_predict_visual)

    p_pa = pred_sub.add_parser("audio", help="Score one audio file or folder.")
    _add_predict_unimodal(p_pa); _add_common(p_pa)
    p_pa.set_defaults(func=_cmd_predict_audio)

    p_pm = pred_sub.add_parser("multimodal", help="Score using both image and audio.")
    _add_predict_multimodal(p_pm); _add_common(p_pm)
    p_pm.set_defaults(func=_cmd_predict_multimodal)

    return parser


def main():
    _configure_console_output()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    # Visual config can be reloaded from yaml; audio config is hard-coded.
    if args.modality in ("visual", "multimodal") and getattr(args, "config", None):
        try:
            from load_config import apply_yaml_config
        except ImportError:
            from model_pipelines.load_config import apply_yaml_config
        apply_yaml_config(args.config)

    # Apply --data-root override (visual only).
    if getattr(args, "data_root", None):
        try:
            from config import cfg
            cfg.data_root = args.data_root
        except ImportError:
            pass

    args.func(args)


if __name__ == "__main__":
    main()
