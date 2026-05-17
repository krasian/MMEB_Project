"""
Preprocesses Xeno-Canto metadata for multimodal learning. Cleans audio metadata, converts recording duration to seconds, encodes temporal information using cyclic month features, maps recording quality scores, and prepares auxiliary metadata
for the multimodal model.
"""
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path

INPUT_DIR = Path(
    "data/metadata/xeno_canto"
)

OUTPUT_DIR = INPUT_DIR

QUALITY_MAP = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
    "E": 0
}


def month_to_cyclic(month):
    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    return month_sin, month_cos


def month_to_season(month):

    if month in [12, 1, 2]:
        return 0

    elif month in [3, 4, 5]:
        return 1

    elif month in [6, 7, 8]:
        return 2

    return 3


def duration_to_seconds(duration):

    try:
        minutes, seconds = (
            duration.split(":")
        )

        return (
            int(minutes) * 60
            + int(seconds)
        )

    except:
        return np.nan


def clean_altitude(value):

    if pd.isna(value):
        return np.nan

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    return (
        float(numbers[0])
        if numbers
        else np.nan
    )


for json_file in INPUT_DIR.glob("*.json"):

    if "clean" in json_file.name:
        continue

    print(
        f"Processing {json_file.name}"
    )

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    rows = []

    for item in data:

        date = pd.to_datetime(
            item.get("date"),
            errors="coerce"
        )

        month = (
            date.month
            if pd.notnull(date)
            else np.nan
        )

        if pd.notnull(month):

            (
                month_sin,
                month_cos
            ) = month_to_cyclic(
                month
            )

            season = (
                month_to_season(
                    month
                )
            )

        else:

            month_sin = np.nan
            month_cos = np.nan
            season = np.nan

        lat = pd.to_numeric(
            item.get("lat"),
            errors="coerce"
        )

        lon = pd.to_numeric(
            item.get("lon"),
            errors="coerce"
        )

        row = {

            "id":
                item.get("id"),

            "scientific_name":
                f"{item.get('gen')} "
                f"{item.get('sp')}",

            "audio_url":
                item.get("file"),

            "latitude_scaled":
                (
                    lat + 90
                ) / 180
                if pd.notnull(lat)
                else np.nan,

            "longitude_scaled":
                (
                    lon + 180
                ) / 360
                if pd.notnull(lon)
                else np.nan,

            "month_sin":
                month_sin,

            "month_cos":
                month_cos,

            "season":
                (
                    season / 3
                )
                if pd.notnull(season)
                else np.nan,

            "altitude":
                clean_altitude(
                    item.get("alt")
                ),

            "quality_encoded":
                QUALITY_MAP.get(
                    item.get("q"),
                    np.nan
                ),

            "length_seconds":
                duration_to_seconds(
                    item.get(
                        "length",
                        ""
                    )
                )
        }

        rows.append(row)

    clean_df = pd.DataFrame(rows)

    # missing values
    clean_df = clean_df.fillna(
        clean_df.median(
            numeric_only=True
        )
    )

    # mormalize quality
    clean_df[
        "quality_encoded"
    ] /= 4

    # normalize altitude
    if (
        clean_df[
            "altitude"
        ].max() > 0
    ):
        clean_df[
            "altitude"
        ] /= (
            clean_df[
                "altitude"
            ].max()
        )

    # normalize recording length
    clean_df["length_seconds"] = (
        clean_df["length_seconds"]
        .clip(upper=300)
    / 300
)

    output_file = (
        OUTPUT_DIR
        / f"{json_file.stem}_clean.csv"
    )

    clean_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )
