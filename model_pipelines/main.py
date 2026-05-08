"""
Unified CLI entry point for the bird novelty detection project.

Usage shape:  python main.py <command> <modality> [options]

Commands × modalities:
  train      visual | audio | multimodal
  evaluate   visual | audio | multimodal
  predict    visual | audio | multimodal

Examples:
  python main.py train visual
  python main.py train visual --skip-training
  python main.py evaluate visual
  python main.py predict visual path/to/image.jpg
  python main.py predict visual path/to/folder --save


"""
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

from config import cfg
from load_config import apply_yaml_config


def _image_extensions():
    """Read supported image extensions from cfg, with a safe fallback."""
    return getattr(cfg, "image_extensions",
                   {".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _predict_single_image(detector, image_path: str):
    """Score one image and print a formatted result."""
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
          f"(threshold: {result['threshold']:.4f},  "
          f"margin: {margin:+.4f})")
    if result["is_outlier"]:
        print(f"  Further above threshold = more confident outlier")
    else:
        print(f"  Further below threshold = more confident known species")


def _predict_image_folder(detector, folder_path: str, save_csv: bool = False):
    """Score all images in a folder and print a summary table."""
    folder = Path(folder_path)
    exts = _image_extensions()
    images = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

    if not images:
        print(f"  No images found in {folder_path}")
        return

    csv_name = getattr(cfg, "csv_out", "predictions.csv")

    print(f"\n  Scoring {len(images)} images in {folder_path}\n")
    print(f"  {'File':<40} {'Result':<12} {'Predicted Class':<25} "
          f"{'Distance':>10} {'Threshold':>10}")
    print("  " + "-" * 103)

    rows = []
    n_known, n_outlier = 0, 0

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
              f"{result['distance']:>10.4f} "
              f"{result['threshold']:>10.4f}")

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


def _cmd_train_audio(args):
    # TODO: audio teammate to wire this up.
    # from pipelines.run_audio import run_full_pipeline as run_audio
    # run_audio(skip_training=args.skip_training)
    print("  ✗ Audio training pipeline is not implemented yet.")
    print("    Implement pipelines/run_audio.py and uncomment the import in main.py.")
    sys.exit(2)


def _cmd_train_multimodal(args):
    # TODO: design and implement.
    # from pipelines.run_multimodal import run_full_pipeline as run_multi
    # run_multi(skip_training=args.skip_training)
    print("  ✗ Multimodal training pipeline is not implemented yet.")
    print("    Implement pipelines/run_multimodal.py and uncomment the import in main.py.")
    sys.exit(2)


def _cmd_evaluate_visual(args):
    from pipelines.run_visual import run_evaluation
    run_evaluation()


def _cmd_evaluate_audio(args):
    # TODO: audio teammate to wire this up.
    # from pipelines.run_audio import run_evaluation as run_eval
    # run_eval()
    print("  ✗ Audio evaluation pipeline is not implemented yet.")
    sys.exit(2)


def _cmd_evaluate_multimodal(args):
    # TODO: design and implement.
    # from pipelines.run_multimodal import run_evaluation as run_eval
    # run_eval()
    print("  ✗ Multimodal evaluation pipeline is not implemented yet.")
    sys.exit(2)


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


def _cmd_predict_audio(args):
    # TODO: audio teammate to wire this up.
    # Expected interface (mirrors VisualAnomalyDetector):
    #     from inference.audio_detector import AudioAnomalyDetector
    #     detector = AudioAnomalyDetector(checkpoint_dir=args.checkpoint_dir)
    #     result   = detector.predict(args.path)   # single file
    #     # or detector.predict_folder(args.path, save_csv=args.save) for folders
    print("  ✗ Audio prediction is not implemented yet.")
    print("    Implement inference/audio_detector.py and uncomment the import in main.py.")
    sys.exit(2)


def _cmd_predict_multimodal(args):
    # TODO: design and implement.
    # Expected interface:
    #     from inference.visual_detector import VisualAnomalyDetector
    #     from inference.audio_detector  import AudioAnomalyDetector
    #     # ... combine scores (late fusion / averaged distances / etc.)
    print("  ✗ Multimodal prediction is not implemented yet.")
    print(f"    Would have used image: {args.image}")
    print(f"    Would have used audio: {args.audio}")
    sys.exit(2)



def _add_common_args(parser):
    """Args shared by every leaf subcommand (config + data-root override)."""
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: ./config.yaml).")
    parser.add_argument("--data-root", default=None,
                        help="Override data.data_root in config.yaml.")


def _add_train_args(parser):
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and recompute centroids + threshold only.")


def _add_predict_unimodal_args(parser):
    """Args for visual / audio predict subcommands (single path positional)."""
    parser.add_argument("path",
                        help="Path to a single file or a folder.")
    parser.add_argument("--save", action="store_true",
                        help="Save folder results to CSV (folder mode only).")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint directory (default: from config.yaml).")


def _add_predict_multimodal_args(parser):
    """Args for the multimodal predict subcommand (image + audio together)."""
    parser.add_argument("--image", required=True,
                        help="Path to the image (or folder of images).")
    parser.add_argument("--audio", required=True,
                        help="Path to the audio file (or folder of clips).")
    parser.add_argument("--save", action="store_true",
                        help="Save folder results to CSV (folder mode only).")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint directory (default: from config.yaml).")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bird novelty detection — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    top = parser.add_subparsers(dest="cmd", required=True)

    # ── train ─────────────────────────────────────────────────────────────
    p_train = top.add_parser("train", help="Train a model.")
    train_sub = p_train.add_subparsers(dest="modality", required=True)

    p_tv = train_sub.add_parser("visual", help="Train the visual pipeline.")
    _add_train_args(p_tv); _add_common_args(p_tv)
    p_tv.set_defaults(func=_cmd_train_visual)

    p_ta = train_sub.add_parser("audio", help="Train the audio pipeline.")
    _add_train_args(p_ta); _add_common_args(p_ta)
    p_ta.set_defaults(func=_cmd_train_audio)

    p_tm = train_sub.add_parser("multimodal", help="Train the multimodal pipeline.")
    _add_train_args(p_tm); _add_common_args(p_tm)
    p_tm.set_defaults(func=_cmd_train_multimodal)

    # ── evaluate ──────────────────────────────────────────────────────────
    p_eval = top.add_parser("evaluate", help="Evaluate an existing checkpoint.")
    eval_sub = p_eval.add_subparsers(dest="modality", required=True)

    p_ev = eval_sub.add_parser("visual", help="Evaluate the visual pipeline.")
    _add_common_args(p_ev)
    p_ev.set_defaults(func=_cmd_evaluate_visual)

    p_ea = eval_sub.add_parser("audio", help="Evaluate the audio pipeline.")
    _add_common_args(p_ea)
    p_ea.set_defaults(func=_cmd_evaluate_audio)

    p_em = eval_sub.add_parser("multimodal", help="Evaluate the multimodal pipeline.")
    _add_common_args(p_em)
    p_em.set_defaults(func=_cmd_evaluate_multimodal)

    # ── predict ───────────────────────────────────────────────────────────
    p_pred = top.add_parser("predict", help="Score one input or a folder.")
    pred_sub = p_pred.add_subparsers(dest="modality", required=True)

    p_pv = pred_sub.add_parser("visual",
                               help="Score one image or a folder of images.")
    _add_predict_unimodal_args(p_pv); _add_common_args(p_pv)
    p_pv.set_defaults(func=_cmd_predict_visual)

    p_pa = pred_sub.add_parser("audio",
                               help="Score one audio file or a folder of clips.")
    _add_predict_unimodal_args(p_pa); _add_common_args(p_pa)
    p_pa.set_defaults(func=_cmd_predict_audio)

    p_pm = pred_sub.add_parser("multimodal",
                               help="Score using both image and audio.")
    _add_predict_multimodal_args(p_pm); _add_common_args(p_pm)
    p_pm.set_defaults(func=_cmd_predict_multimodal)

    return parser


def main():
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    # Load config (yaml → cfg) before anything else runs.
    apply_yaml_config(args.config)

    # CLI --data-root takes final priority over config.yaml.
    if args.data_root:
        cfg.data_root = args.data_root

    args.func(args)


if __name__ == "__main__":
    main()
