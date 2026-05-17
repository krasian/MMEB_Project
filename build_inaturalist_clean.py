"""
Preprocesses iNaturalist metadata for multimodal learning.

Cleans image metadata, removes invalid rows, encodes temporal information using cyclic month features, normalizes geographic coordinates, and prepares auxiliary metadata for the multimodal model.
"""
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/metadata/inaturalist")
OUTPUT_DIR = INPUT_DIR

MAX_SAMPLES = 2300

QUALITY_MAP = {
    "casual": 0,
    "needs_id": 1,
    "research": 2
}

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "observed_on",
    "quality_grade",
    "positional_accuracy"
]


def month_to_cyclic(month):
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return month_sin, month_cos


def month_to_season(month):

    if month in [12, 1, 2]:
        return 0  # winter

    elif month in [3, 4, 5]:
        return 1  # spring

    elif month in [6, 7, 8]:
        return 2  # summer

    return 3  # autumn


for csv_file in INPUT_DIR.glob("*.csv"):

    if "clean" in csv_file.name:
        continue

    print(f"Processing {csv_file.name}")

    df = pd.read_csv(csv_file)

    df = df.dropna(subset=REQUIRED_COLUMNS)

    df["observed_on"] = pd.to_datetime(
        df["observed_on"],
        errors="coerce"
    )

    df = df.dropna(subset=["observed_on"])

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["positional_accuracy"] = pd.to_numeric(
        df["positional_accuracy"],
        errors="coerce"
    )

    df = df.dropna(subset=[
        "latitude",
        "longitude",
        "positional_accuracy"
    ])

    # normalize coordinates to 0-1
    df["latitude_scaled"] = (
        df["latitude"] + 90
    ) / 180

    df["longitude_scaled"] = (
        df["longitude"] + 180
    ) / 360

    # month
    df["month"] = (
        df["observed_on"]
        .dt.month
    )

    # cyclic month encoding
    (
        df["month_sin"],
        df["month_cos"]
    ) = zip(
        *df["month"]
        .apply(month_to_cyclic)
    )

    # season
    df["season"] = (
        df["month"]
        .apply(month_to_season)
    )

    # normalize season
    df["season"] = (
        df["season"] / 3
    )

    # encode quality
    df["quality_grade_encoded"] = (
        df["quality_grade"]
        .map(QUALITY_MAP)
    )

    df = df.dropna(
        subset=["quality_grade_encoded"]
    )

    # normalize quality (0-1)
    df["quality_grade_encoded"] = (
        df["quality_grade_encoded"] / 2
    )

    # normalize positional accuracy
    max_accuracy = (
        df["positional_accuracy"]
        .max()
    )

    if max_accuracy > 0:
        df["positional_accuracy"] = (
            df["positional_accuracy"]
            / max_accuracy
        )

    df = df.sample(
        frac=1,
        random_state=42
    )

    df = df.head(MAX_SAMPLES)

    clean_df = df[[
        "id",
        "scientific_name",
        "image_url",
        "latitude_scaled",
        "longitude_scaled",
        "month_sin",
        "month_cos",
        "season",
        "quality_grade_encoded",
        "positional_accuracy"
    ]]

    output_file = (
        OUTPUT_DIR
        / f"{csv_file.stem}_clean.csv"
    )

    clean_df.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")