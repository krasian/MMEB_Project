"""
Builds the multimodal dataset index. Creates a pseudo-paired multimodal dataset by linking image and audio samples from the same species, assigning train/test splits, and generating metadata references for multimodal late-fusion experiments.
"""
import pandas as pd
from pathlib import Path
import random

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

IMAGE_ROOT = Path(
    "data/images"
)

AUDIO_ROOT = Path(
    "data/audio"
)

IMAGE_METADATA_ROOT = Path(
    "data/metadata/inaturalist"
)

AUDIO_METADATA_ROOT = Path(
    "data/metadata/xeno_canto"
)

OUTPUT_PATH = Path(
    "data/metadata/multimodal_dataset.csv"
)

TRAIN_SPECIES = [
    "blackbird",
    "blue_tit",
    "Great_Tit",
    "sparrow",
    "Mallard",
    "Crow",
    "Robin",
    "Thrush"
]

OUTLIER_SPECIES = [
    "starling",
    "Flamingo",
    "Toucan"
]

ALL_SPECIES = (
    TRAIN_SPECIES
    + OUTLIER_SPECIES
)

MAX_SAMPLES_PER_SPECIES = 500
TEST_RATIO = 0.2

rows = []

for species in ALL_SPECIES:

    image_folder = (
        IMAGE_ROOT / species
    )

    audio_folder = (
        AUDIO_ROOT / species
    )

    if (
        not image_folder.exists()
        or not audio_folder.exists()
    ):
        print(
            f"Skipping {species}"
        )
        continue

    image_files = list(
        image_folder.glob("*.jpg")
    )

    audio_files = list(
        audio_folder.glob("*.mp3")
    )

    if (
        len(image_files) == 0
        or len(audio_files) == 0
    ):
        continue

    sample_size = min(
        len(image_files),
        len(audio_files),
        MAX_SAMPLES_PER_SPECIES
    )

    selected_images = random.sample(
        image_files,
        sample_size
    )

    selected_audio = random.sample(
        audio_files,
        sample_size
    )

    for i in range(sample_size):

        image_path = (
            selected_images[i]
        )

        audio_path = (
            selected_audio[i]
        )

        split = (
            "test"
            if (
                species
                in OUTLIER_SPECIES
            )
            else (
                "test"
                if random.random()
                < TEST_RATIO
                else "train"
            )
        )

        rows.append({

            "pair_id":
                f"{species}_{i}",

            "species":
                species,

            "split":
                split,

            "is_outlier":
                int(
                    species
                    in OUTLIER_SPECIES
                ),

            "image_path":
                str(image_path),

            "audio_path":
                str(audio_path),

            "image_metadata_path":
                str(
                    IMAGE_METADATA_ROOT
                    /
                    f"{species}_clean.csv"
                ),

            "audio_metadata_path":
                str(
                    AUDIO_METADATA_ROOT
                    /
                    f"{species}_clean.csv"
                )
        })

df = pd.DataFrame(rows)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"Saved: {OUTPUT_PATH}"
)

print(
    f"Total samples: {len(df)}"
)

print(
    df["split"]
    .value_counts()
)

print(
    df["species"]
    .value_counts()
)
