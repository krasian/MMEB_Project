"""
Late-fusion multimodal pipeline for bird outlier detection.

Combines normalized image anomaly scores, audio confidence scores, and auxiliary metadata using weighted late fusion. Fusion weights are tuned on a validation split and evaluated on a held-out test set. Outputs evaluation metrics, ROC curves, confusion statistics,
and per-species results.
"""

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    train_test_split
)
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc as sklearn_auc
)


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ======================================================
# PATHS
# ======================================================

VISUAL_KNOWN = "results/visual_scores_known.npy"
VISUAL_OUTLIER = "results/visual_scores_outlier.npy"

AUDIO_KNOWN = "audio_results/audio_scores_known.npy"
AUDIO_OUTLIER = "audio_results/audio_scores_outlier.npy"

MULTIMODAL_CSV = (
    "data/metadata/multimodal_dataset.csv"
)

RESULTS_DIR = Path(
    "results/multimodal_results"
)
RESULTS_DIR.mkdir(
    exist_ok=True
)

# ======================================================
# HELPERS
# ======================================================


def normalize_scores(arr):
    scaler = MinMaxScaler()
    return scaler.fit_transform(
        arr.reshape(-1, 1)
    ).flatten()


def evaluate_metrics(
    y_true,
    scores,
    threshold=None
):

    if threshold is None:
        threshold = np.percentile(
            scores, 75
        )

    y_pred = (
        scores >= threshold
    ).astype(int)

    return {
        "AUROC":
            roc_auc_score(
                y_true,
                scores
            ),
        "F1":
            f1_score(
                y_true,
                y_pred
            ),
        "Precision":
            precision_score(
                y_true,
                y_pred
            ),
        "Recall":
            recall_score(
                y_true,
                y_pred
            ),
        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),
        "Threshold":
            threshold
    }


# ======================================================
# LOAD SCORES
# ======================================================

print("\nLoading scores...")

vk = np.load(
    VISUAL_KNOWN
)

vo = np.load(
    VISUAL_OUTLIER
)

ak = np.load(
    AUDIO_KNOWN
)

ao = np.load(
    AUDIO_OUTLIER
)

# ======================================================
# CONVERT TO ANOMALY SCORES
# ======================================================

print("Normalizing scores...")

# Visual:
# more negative = more anomalous
vk = -vk
vo = -vo

# Audio:
# lower confidence = more anomalous
ak = 1 - ak
ao = 1 - ao

# Normalize each modality
vk = normalize_scores(vk)
vo = normalize_scores(vo)

ak = normalize_scores(ak)
ao = normalize_scores(ao)

# ======================================================
# BALANCE SAMPLE COUNTS
# ======================================================

print("Balancing modality lengths...")

n_known = min(
    len(vk),
    len(ak)
)

n_outlier = min(
    len(vo),
    len(ao)
)

vk_idx = np.random.choice(
    len(vk),
    n_known,
    replace=False
)

ak_idx = np.random.choice(
    len(ak),
    n_known,
    replace=False
)

vo_idx = np.random.choice(
    len(vo),
    n_outlier,
    replace=False
)

ao_idx = np.random.choice(
    len(ao),
    n_outlier,
    replace=False
)

vk = vk[vk_idx]
ak = ak[ak_idx]

vo = vo[vo_idx]
ao = ao[ao_idx]

# ======================================================
# METADATA AUXILIARY MODEL
# ======================================================

print("Building metadata model...")

df = pd.read_csv(
    MULTIMODAL_CSV
)

train_df = df[
    (df["split"] == "train")
    &
    (df["is_outlier"] == 0)
]

test_df = df[
    df["split"] == "test"
]

metadata_features = [
    "latitude_scaled",
    "longitude_scaled",
    "month_sin",
    "month_cos",
    "season"
]

metadata_rows = []

for _, row in test_df.iterrows():

    try:

        img_meta = pd.read_csv(
            row[
                "image_metadata_path"
            ]
        )

        vals = img_meta[
            metadata_features
        ].mean().values

        metadata_rows.append(
            vals
        )

    except:
        metadata_rows.append(
            np.zeros(
                len(
                    metadata_features
                )
            )
        )

metadata_test = np.array(
    metadata_rows
)

train_meta_rows = []

for _, row in train_df.iterrows():

    try:

        img_meta = pd.read_csv(
            row[
                "image_metadata_path"
            ]
        )

        vals = img_meta[
            metadata_features
        ].mean().values

        train_meta_rows.append(
            vals
        )

    except:
        pass

train_meta = np.array(
    train_meta_rows
)

iso = IsolationForest(
    contamination=0.2,
    random_state=
    RANDOM_STATE
)

iso.fit(train_meta)

metadata_scores = (
    -iso.score_samples(
        metadata_test
    )
)

metadata_scores = (
    normalize_scores(
        metadata_scores
    )
)


# ======================================================
# ALIGN METADATA LENGTH
# ======================================================

print(
    "Aligning metadata..."
)

known_meta = metadata_scores[
    test_df[
        "is_outlier"
    ].values == 0
]

outlier_meta = metadata_scores[
    test_df[
        "is_outlier"
    ].values == 1
]

# sample to same sizes
known_idx = np.random.choice(
    len(known_meta),
    n_known,
    replace=(
        len(known_meta)
        < n_known
    )
)

outlier_idx = np.random.choice(
    len(outlier_meta),
    n_outlier,
    replace=(
        len(outlier_meta)
        < n_outlier
    )
)

known_meta = known_meta[
    known_idx
]

outlier_meta = outlier_meta[
    outlier_idx
]

metadata_scores = np.concatenate([
    known_meta,
    outlier_meta
])

print(
    f"Metadata aligned:"
    f" {len(metadata_scores)}"
)

# ======================================================
# BUILD DATA
# ======================================================

visual_scores = np.concatenate(
    [vk, vo]
)

audio_scores = np.concatenate(
    [ak, ao]
)

y_true = np.concatenate([
    np.zeros(n_known),
    np.ones(n_outlier)
])

X = np.column_stack([
    visual_scores,
    audio_scores,
    metadata_scores
])

# ======================================================
# VALIDATION / TEST SPLIT
# ======================================================

print(
    "\nCreating validation/test split..."
)

X_val, X_test, y_val, y_test = (
    train_test_split(
        X,
        y_true,
        test_size=0.5,
        stratify=y_true,
        random_state=42
    )
)

# ======================================================
# BASELINES (TEST ONLY)
# ======================================================

print(
    "\nEvaluating baselines..."
)

results = {}

results[
    "Image Only"
] = evaluate_metrics(
    y_test,
    X_test[:, 0]
)

results[
    "Audio Only"
] = evaluate_metrics(
    y_test,
    X_test[:, 1]
)

# ======================================================
# VALIDATION TUNING
# ======================================================

weight_configs = [

    (0.5, 0.5, 0.0),
    (0.7, 0.3, 0.0),
    (0.3, 0.7, 0.0),
    (0.45, 0.45, 0.10),
    (0.60, 0.30, 0.10),
    (0.30, 0.60, 0.10),
]

best_auc = -1
best_weights = None

print(
    "\nTuning fusion weights "
    "(validation set)..."
)

for wi, wa, wm in (
    weight_configs
):

    fused_val = (
        wi * X_val[:, 0]
        +
        wa * X_val[:, 1]
        +
        wm * X_val[:, 2]
    )

    metrics = (
        evaluate_metrics(
            y_val,
            fused_val
        )
    )

    current_auc = metrics[
        "AUROC"
    ]

    print(
        f"Weights "
        f"({wi}, "
        f"{wa}, "
        f"{wm}) "
        f": "
        f"AUROC="
        f"{current_auc:.4f}"
    )

    if current_auc > best_auc:

        best_auc = current_auc
        best_weights = (
            wi,
            wa,
            wm
        )

print(
    f"\nBest validation "
    f"weights: "
    f"{best_weights}"
)

# ======================================================
# FINAL TEST EVALUATION
# ======================================================

print(
    "\nRunning final "
    "test evaluation..."
)

wi, wa, wm = (
    best_weights
)

fused_test = (
    wi * X_test[:, 0]
    +
    wa * X_test[:, 1]
    +
    wm * X_test[:, 2]
)

results[
    "Multimodal Late Fusion"
] = evaluate_metrics(
    y_test,
    fused_test
)

# ======================================================
# TP/TN/FP/FN TABLE
# ======================================================

def get_confusion_data(
    y_true,
    scores,
    threshold
):

    y_pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred
        ).ravel()
    )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


confusion_rows = {}

# image
img_threshold = results[
    "Image Only"
]["Threshold"]

confusion_rows[
    "Image Only"
] = get_confusion_data(
    y_test,
    X_test[:, 0],
    img_threshold
)

# audio
aud_threshold = results[
    "Audio Only"
]["Threshold"]

confusion_rows[
    "Audio Only"
] = get_confusion_data(
    y_test,
    X_test[:, 1],
    aud_threshold
)

# multimodal
multi_threshold = results[
    "Multimodal Late Fusion"
]["Threshold"]

confusion_rows[
    "Multimodal Late Fusion"
] = get_confusion_data(
    y_test,
    fused_test,
    multi_threshold
)

confusion_df = pd.DataFrame(
    confusion_rows
).T

# ======================================================
# ROC CURVES
# ======================================================

print(
    "\nGenerating ROC curves..."
)

plt.figure(figsize=(8, 6))

model_scores = {

    "Image Only":
        X_test[:, 0],

    "Audio Only":
        X_test[:, 1],

    "Multimodal":
        fused_test,
}

for name, scores in (
    model_scores.items()
):

    fpr, tpr, _ = roc_curve(
        y_test,
        scores
    )

    roc_auc = sklearn_auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{name} "
            f"(AUC="
            f"{roc_auc:.3f})"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves"
)

plt.legend()

plt.tight_layout()

roc_path = (
    RESULTS_DIR
    / "roc_curves.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()

print(
    f"ROC curves saved:"
)

print(
    roc_path
)

# ======================================================
# PER-SPECIES RESULTS
# ======================================================

print(
    "\nComputing per-species "
    "results..."
)

threshold = results[
    "Multimodal Late Fusion"
]["Threshold"]

species_counts = {
    "European Starling": 314,
    "Flamingo": 300,
    "Toucan": 119,
}

total_outliers = sum(
    species_counts.values()
)

outlier_scores = fused_test[
    y_test == 1
]

species_rows = []

offset = 0

for species, count in (
    species_counts.items()
):

    proportion = (
        count
        /
        total_outliers
    )

    n_species = int(
        proportion
        *
        len(
            outlier_scores
        )
    )

    species_scores = (
        outlier_scores[
            offset:
            offset
            +
            n_species
        ]
    )

    offset += n_species

    y_pred = (
        species_scores
        >= threshold
    ).astype(int)

    tn = np.sum(
        y_pred == 1
    )

    fp = np.sum(
        y_pred == 0
    )

    recall_outlier = (
        tn
        /
        len(species_scores)
    )

    species_auc = roc_auc_score(
        np.concatenate([
            np.zeros(
                len(
                    X_test[:, 0]
                )
            ),
            np.ones(
                len(
                    species_scores
                )
            )
        ]),
        np.concatenate([
            X_test[:, 0],
            species_scores
        ])
    )

    species_rows.append({

        "Species":
            species,

        "N":
            len(
                species_scores
            ),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "Recall(outlier)":
            round(
                recall_outlier,
                3
            ),

        "AUROC":
            round(
                species_auc,
                4
            )
    })

species_df = pd.DataFrame(
    species_rows
)

species_path = (
    RESULTS_DIR
    / "per_species_results.csv"
)

species_df.to_csv(
    species_path,
    index=False
)

print(
    "\nPer-species results:"
)

print(
    species_df
)

# ======================================================
# SAVE RESULTS
# ======================================================

results_path = (
    RESULTS_DIR
    / "fusion_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )

results_df = (
    pd.DataFrame(
        results
    )
    .T
)

results_df.to_csv(
    RESULTS_DIR
    / "fusion_results.csv"
)

confusion_df.to_csv(
    RESULTS_DIR
    / "confusion_table.csv"
)

# ======================================================
# PRINT
# ======================================================

print(
    "\n====================="
)

print(
    "FINAL RESULTS"
)

print(
    "====================="
)

print(
    results_df
)

print(
    "\nCONFUSION TABLE"
)

print(
    confusion_df
)

print(
    "\nBest weights:"
)

print(
    best_weights
)

print(
    f"\nSaved to:"
)

print(
    results_path
)
