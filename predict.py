import sys
import argparse
import pandas as pd
from pathlib import Path
from pipeline import BirdAnomalyDetector, cfg


def _image_extensions():
    """Read supported extensions from cfg, with a safe fallback."""
    return getattr(cfg, "IMAGE_EXTENSIONS_PREDICT",
                   {".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def predict_single(detector: BirdAnomalyDetector, image_path: str):
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
    print(f"  {'Further above threshold = more confident outlier' if result['is_outlier'] else 'Further below threshold = more confident known species'}")


def predict_folder(detector: BirdAnomalyDetector, folder_path: str,
                   save_csv: bool = False):
    """Score all images in a folder and print a summary table."""
    folder = Path(folder_path)
    exts   = _image_extensions()
    images = sorted([
        p for p in folder.rglob("*")
        if p.suffix.lower() in exts
    ])

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
            "file":            str(img_path),
            "is_outlier":      result["is_outlier"],
            "predicted_class": result["predicted_class"],
            "distance":        result["distance"],
            "threshold":       result["threshold"],
        })

    total = n_known + n_outlier
    print(f"\n  Summary: {n_known}/{total} known  |  {n_outlier}/{total} outliers")

    if save_csv and rows:
        out_path = folder / csv_name
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  ✓ Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Score bird images with the trained anomaly detector")
    parser.add_argument("path",
        help="Path to a single image file or a folder of images")
    parser.add_argument("--save", action="store_true",
        help="Save folder results to CSV (folder mode only)")
    parser.add_argument("--checkpoint-dir", default=None,
        help=f"Override checkpoint directory (default: from config.yaml)")
    parser.add_argument("--config", default=None,
        help="Path to config.yaml (default: ./config.yaml).")
    parser.add_argument("--data-root", default=None,
        help="Override data.data_root in config.yaml.")
    args = parser.parse_args()

    from load_config import apply_yaml_config
    apply_yaml_config(args.config)

    if args.data_root:
        cfg.data_root = args.data_root

    print("── Loading model ──")
    detector = BirdAnomalyDetector(checkpoint_dir=args.checkpoint_dir)
    print(f"  Classes   : {detector.classes}")
    print(f"  Threshold : {detector.centroid_threshold:.4f}")

    target = Path(args.path)

    if target.is_file():
        predict_single(detector, str(target))
    elif target.is_dir():
        predict_folder(detector, str(target), save_csv=args.save)
    else:
        print(f"  ✗ Path not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
