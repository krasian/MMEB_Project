import os
import requests
import time

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


def fetch_recordings_by_quality(species_name, quality):
    page = 1
    all_recordings = []
    genus, sp = species_name.split(" ", 1)
    query = f'gen:{genus} sp:{sp} cnt:Netherlands q:{quality}'

    while True:
        params = {"query": query, "key": API_KEY, "page": page, "per_page": 500}

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Request failed on page {page}: {e}")
            break

        recordings = data.get("recordings", [])
        if not recordings:
            break

        all_recordings.extend(recordings)

        if page >= int(data.get("numPages", 1)):
            break

        page += 1
        time.sleep(0.5)

    return all_recordings


def collect_recordings(species_name):
    collected = []
    seen_ids = set()

    for quality in QUALITY_GRADES:
        if len(collected) >= MAX_PER_SPECIES:
            break

        recs = fetch_recordings_by_quality(species_name, quality)

        new_recs = [r for r in recs if r["id"] not in seen_ids]
        for r in new_recs:
            seen_ids.add(r["id"])

        slots_left = MAX_PER_SPECIES - len(collected)
        collected.extend(new_recs[:slots_left])

    return collected


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for species in SPECIES_LIST:
        print(f"Collecting {species}")
        recs = collect_recordings(species)
        print(f"Collected {len(recs)} recordings")


if __name__ == "__main__":
    main()