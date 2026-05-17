"""
Read config.yaml, expand ${ENV_VAR} references, and apply values to
the module-level `cfg` object in config.py -- so every file that
imports cfg automatically sees the updated values.

DUAL-IMPORT NOTE:
    main.py puts BOTH the project root AND the model_pipelines/ folder
    on sys.path. As a result, config.py can get loaded as TWO separate
    modules: 'config' and 'model_pipelines.config'. Each gets its own
    cfg object, names list, class_to_id dict, and outlier_names list --
    they are NOT shared by reference.

    This function therefore has to update ALL of them, in place, every
    time it's called. That way, no matter which import path any other
    file took (or whether it captured a local binding via `from-import`),
    every reference sees the updated values.
"""
import os
import re
import sys
import yaml
from pathlib import Path
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        # Try a few sensible locations.
        candidates = [
            Path(__file__).parent / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for cand in candidates:
            if cand.exists():
                config_path = cand
                break
        else:
            raise FileNotFoundError(
                "config.yaml not found in any of: "
                + ", ".join(str(c) for c in candidates)
            )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at '{config_path}'. "
            "Make sure the file exists before calling apply_yaml_config()."
        )

    raw = config_path.read_text(encoding="utf-8")
    raw = _resolve_env_vars(raw)
    return yaml.safe_load(raw)


def _all_config_modules():
    """Find every loaded module that represents config.py."""
    modules = []
    for name in ("config", "model_pipelines.config"):
        mod = sys.modules.get(name)
        if mod is not None:
            modules.append(mod)
    return modules


def _project_path(value: str) -> str:
    """Resolve relative project config paths against the repository root."""
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _apply_to_single_cfg(c, y: dict):
    """Write all YAML-driven settings onto a single cfg object."""
    # ── Data ─────────────────────────────────────────────────────
    data = y.get("data", {})
    c.data_root = _project_path(data["data_root"])

    new_known = dict(data["known_csvs"])
    new_outlier = dict(data["outlier_csvs"])

    # Mutate the existing dict objects in place when possible, so any
    # `from config import cfg` callers see the change through cfg.known_csv.
    # (cfg itself is shared by reference within one module, but its
    # nested dicts may be referenced separately by some downstream code.)
    if hasattr(c, "known_csv") and isinstance(c.known_csv, dict):
        c.known_csv.clear()
        c.known_csv.update(new_known)
    else:
        c.known_csv = new_known

    if hasattr(c, "outlier_csv") and isinstance(c.outlier_csv, dict):
        c.outlier_csv.clear()
        c.outlier_csv.update(new_outlier)
    else:
        c.outlier_csv = new_outlier

    c.train_split_ratio = float(data["train_ratio"])
    c.validation_split_ratio = float(data["val_ratio"])
    c.number_of_classes = len(c.known_csv)

    # ── Model ────────────────────────────────────────────────────
    model = y.get("model", {})
    c.embedding_dim = int(model["embedding_dim"])
    c.image_size = int(model["img_size"])

    # ── Training ─────────────────────────────────────────────────
    training = y.get("training", {})
    c.batch = int(training["batch_size"])
    c.epoches = int(training["num_epochs"])
    c.learning_rate = float(training["learning_rate"])
    c.weight_decay = float(training["weight_decay"])
    c.arcface_scaler = float(training["arcface_s"])
    c.arcface_margin = float(training["arcface_m"])

    # ── Threshold ────────────────────────────────────────────────
    c.percentile_of_threshold = int(y["threshold"]["percentile"])

    # ── Distance metric ──────────────────────────────────────────
    distance = y.get("distance", {})
    metric = distance.get("metric", "mahalanobis").lower()
    if metric not in ("mahalanobis", "euclidean"):
        raise ValueError(
            f"config.yaml: distance.metric must be 'mahalanobis' or 'euclidean', "
            f"got '{metric}'"
        )
    c.distance_metric = metric

    # ── Paths ────────────────────────────────────────────────────
    paths = y.get("paths", {})
    c.checkpoint_directory = _project_path(paths["checkpoint_dir"])
    c.results_directory = _project_path(paths["results_dir"])

    # ── Evaluation ───────────────────────────────────────────────
    evaluation = y.get("evaluation", {})
    c.embeding_visulize_method = evaluation.get("embedding_viz_method", "tsne")
    c.result_dpi = int(evaluation.get("plot_dpi", 300))

    # ── Predict ──────────────────────────────────────────────────
    predict = y.get("predict", {})
    c.image_extensions = set(
        predict.get("image_extensions", [".jpg", ".jpeg", ".png", ".webp", ".bmp"]))
    c.csv_out = predict.get("csv_output_name", "predictions.csv")


def _apply_module_level(mod, known_csv: dict, outlier_csv: dict):
    """
    Mutate the module-level names/class_to_id/outlier_names IN PLACE on a
    single config module, so callers that did `from config import names`
    (or class_to_id, or outlier_names) keep seeing the right data.
    """
    new_names = list(known_csv.keys())
    new_class_to_id = {n: i for i, n in enumerate(new_names)}
    new_outlier_names = list(outlier_csv.keys())

    if hasattr(mod, "names") and isinstance(mod.names, list):
        mod.names.clear()
        mod.names.extend(new_names)
    else:
        mod.names = list(new_names)

    if hasattr(mod, "class_to_id") and isinstance(mod.class_to_id, dict):
        mod.class_to_id.clear()
        mod.class_to_id.update(new_class_to_id)
    else:
        mod.class_to_id = dict(new_class_to_id)

    if hasattr(mod, "outlier_names") and isinstance(mod.outlier_names, list):
        mod.outlier_names.clear()
        mod.outlier_names.extend(new_outlier_names)
    else:
        mod.outlier_names = list(new_outlier_names)


def apply_yaml_config(config_path: str = None, cfg=None, verbose: bool = True) -> None:
    """
    Load config.yaml and patch every loaded copy of config.py in place.

    - If `cfg` is passed, also patch that object explicitly (useful when
      a caller has stashed a reference to a specific cfg instance).
    - Every cfg object on every loaded config module is patched, so
      both `from config import cfg` and
      `from model_pipelines.config import cfg` callers stay correct.
    - The names / class_to_id / outlier_names lists/dicts on each loaded
      config module are mutated in place, so callers that captured them
      via `from config import names` (a local binding) also see the
      update.
    """
    y = load_yaml(config_path)

    # Collect every distinct cfg object we need to update.
    cfgs_to_update = []
    seen_ids = set()

    def _add(candidate):
        if candidate is None:
            return
        if id(candidate) in seen_ids:
            return
        seen_ids.add(id(candidate))
        cfgs_to_update.append(candidate)

    # Explicit cfg passed in (if any).
    _add(cfg)

    # cfg on each loaded config module.
    modules = _all_config_modules()
    for mod in modules:
        _add(getattr(mod, "cfg", None))

    # If we still have nothing, force-load one.
    if not cfgs_to_update:
        try:
            import config as _config
        except ImportError:
            from model_pipelines import config as _config
        modules = _all_config_modules()
        _add(getattr(_config, "cfg", None))

    if not cfgs_to_update:
        raise RuntimeError(
            "apply_yaml_config: could not find a cfg object to update.")

    # Patch every distinct cfg with the YAML values.
    for c in cfgs_to_update:
        _apply_to_single_cfg(c, y)

    # Use the (now-patched) primary cfg to update module-level names lists.
    primary = cfgs_to_update[0]
    for mod in modules:
        _apply_module_level(mod, primary.known_csv, primary.outlier_csv)

    if verbose and mp.current_process().name == "MainProcess":
        print(f"  Config loaded from '{Path(config_path or 'config.yaml').resolve()}'")
        print(f"    DATA_ROOT      : {primary.data_root}")
        print(f"    Known species  : {list(primary.known_csv.keys())}")
        print(f"    Outlier species: {list(primary.outlier_csv.keys())}")
        print(f"    Embedding dim  : {primary.embedding_dim}  |  "
              f"Epochs: {primary.epoches}  |  Batch: {primary.batch}")
        print(f"    Threshold pct  : {primary.percentile_of_threshold}  |  "
              f"ArcFace s={primary.arcface_scaler} m={primary.arcface_margin}")
