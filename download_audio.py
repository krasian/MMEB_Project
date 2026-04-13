import os

# config
API_KEY = "346c4d05b60516e9707d6fccf84cbcc3067acca0"
BASE_URL = "https://xeno-canto.org/api/3/recordings"

SPECIES_LIST = [
    "Turdus merula",
    "Cyanistes caeruleus",
    "Sturnus vulgaris",
    "Parus major",
    "Passer domesticus",
    "Corvus corone",
    "Phoenicopterus roseus",
    "Ramphastos sulfuratus",
    "Turdus philomelos",
    "Erithacus rubecula",
]

OUTPUT_DIR = "xenocanto_data"
MIN_PER_SPECIES = 1
MAX_PER_SPECIES = 1
QUALITY_GRADES = ["A", "B", "C"]


def main():
    print("Setup complete")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    main()