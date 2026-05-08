"""
Read config.yaml, expand ${ENV_VAR} references, and apply values to
the module-level `cfg` object in config.py — so every file that
imports cfg automatically sees the updated values.
"""
import os
import re
import yaml
from pathlib import Path
import multiprocessing as mp


def _resolve_env_vars(text: str) -> str:
    """Replace every ${VAR_NAME} in a YAML string with os.environ[VAR_NAME]."""
    def _replace(match):
        var = match.group(1)
        value = os.environ.get(var)
        if value is None:
            raise EnvironmentError(
                f"config.yaml references ${{{var}}} but the environment "
                f"variable '{var}' is not set.\n"
                f"  Windows : set {var}=D:\\path\\to\\images\n"
                f"  Linux   : export {var}=/path/to/images"
            )
        return value
    return re.sub(r'\$\{(\w+)\}', _replace, text)


def load_yaml(config_path: str = None) -> dict:
    """Read config.yaml, expand ${ENV_VAR} references, and return a dict."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at '{config_path}'. "
            "Make sure the file exists before calling apply_yaml_config()."
        )

    raw = config_path.read_text(encoding="utf-8")
    raw = _resolve_env_vars(raw)
    return yaml.safe_load(raw)


def apply_yaml_config(config_path: str = None, cfg=None) -> None:
    """
    Load config.yaml and apply every value to config.cfg in-place.

    Mutates the module-level `cfg` object so all files that import it
    automatically see the updated values.
    """
    import config as _config
    if cfg is None:
        cfg = _config.cfg

    y = load_yaml(config_path)
    c = cfg

    #Data
    data = y.get("data", {})
    c.data_root = data["data_root"]
    c.known_csv = dict(data["known_csvs"])
    c.outlier_csv = dict(data["outlier_csvs"])
    c.train_split_ratio = float(data["train_ratio"])
    c.validation_split_ratio = float(data["val_ratio"])
    c.number_of_classes = len(c.known_csv)

    # Update module-level name lists in config.py
    _config.names = list(c.known_csv.keys())
    _config.class_to_id = {n: i for i, n in enumerate(_config.names)}
    _config.outlier_names = list(c.outlier_csv.keys())

    #Model
    model = y.get("model", {})
    c.embedding_dim = int(model["embedding_dim"])
    c.image_size = int(model["img_size"])

    #Training
    training = y.get("training", {})
    c.batch = int(training["batch_size"])
    c.epoches = int(training["num_epochs"])
    c.learning_rate = float(training["learning_rate"])
    c.weight_decay = float(training["weight_decay"])
    c.arcface_scaler = float(training["arcface_s"])
    c.arcface_margin = float(training["arcface_m"])

    #Threshold
    c.percentile_of_threshold = int(y["threshold"]["percentile"])

    #Distance metric
    distance = y.get("distance", {})
    metric = distance.get("metric", "mahalanobis").lower()
    if metric not in ("mahalanobis", "euclidean"):
        raise ValueError(
            f"config.yaml: distance.metric must be 'mahalanobis' or 'euclidean', "
            f"got '{metric}'"
        )
    c.distance_metric = metric

    #Paths
    paths = y.get("paths", {})
    c.checkpoint_directory = paths["checkpoint_dir"]
    c.results_directory = paths["results_dir"]

    #Evaluation
    evaluation = y.get("evaluation", {})
    c.embeding_visulize_method = evaluation.get("embedding_viz_method", "tsne")
    c.result_dpi = int(evaluation.get("plot_dpi", 300))

    #Predict
    predict = y.get("predict", {})
    c.image_extensions = set(
        predict.get("image_extensions", [".jpg", ".jpeg", ".png", ".webp", ".bmp"]))
    c.csv_out = predict.get("csv_output_name", "predictions.csv")

    if mp.current_process().name == "MainProcess":
        print(f"  ✓ Config loaded from '{Path(config_path or 'config.yaml').resolve()}'")
        print(f"    DATA_ROOT      : {c.data_root}")
        print(f"    Known species  : {list(c.known_csv.keys())}")
        print(f"    Outlier species: {list(c.outlier_csv.keys())}")
        print(f"    Embedding dim  : {c.embedding_dim}  |  Epochs: {c.epoches}  "
              f"|  Batch: {c.batch}")
        print(f"    Threshold pct  : {c.percentile_of_threshold}  |  "
              f"ArcFace s={c.arcface_scaler} m={c.arcface_margin}")
