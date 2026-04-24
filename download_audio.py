import os
import requests
from tqdm import tqdm
import json
import time

# config
API_KEY = "346c4d05b60516e9707d6fccf84cbcc3067acca0"
BASE_URL = "https://xeno-canto.org/api/3/recordings"

SPECIES_LIST = [
    "Turdus merula",      # Eurasian Blackbird
    "Cyanistes caeruleus",    # Eurasian Blue Tit
    "Sturnus vulgaris",   # European Starling
    "Parus major",         # Great Tit
    "Passer domesticus",    # House Sparrow
    "Corvus corone",          # Carrion Crow
    "Phoenicopterus roseus",  # Greater Flamingo
    "Ramphastos sulfuratus",  # Keel-billed Toucan
    "Turdus philomelos",      # Song Thrush
    "Erithacus rubecula",     # European Robin
]

OUTPUT_DIR = "xenocanto_data"
MIN_PER_SPECIES = 300
MAX_PER_SPECIES = 300
QUALITY_GRADES = ["A", "B", "C"]


# functions
def fetch_recordings_by_quality(species_name, quality):
    page = 1
    all_recordings = []
    genus, sp = species_name.split(" ", 1)
    countries = ["Netherlands", "Germany", "France", "Belgium"]

    while True:
        query = f'gen:"{genus}" sp:"{sp}" q:"{quality}"'
        params = {
            "query": query,
            "key": API_KEY,
            "page": page,
            "per_page": 500,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Request failed on page {page}: {e}")
            break

        recordings = data.get("recordings", [])
        recordings = [r for r in recordings if r.get("cnt") in countries]

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

        print(f"  Fetching quality {quality}...", end=" ")
        recs = fetch_recordings_by_quality(species_name, quality)

        new_recs = [r for r in recs if r["id"] not in seen_ids]
        for r in new_recs:
            seen_ids.add(r["id"])

        slots_left = MAX_PER_SPECIES - len(collected)
        added = new_recs[:slots_left]
        collected.extend(added)

        print(f"found {len(recs)}, added {len(added)} "
              f"(total so far: {len(collected)})")

    return collected


def download_recordings(recordings, species_name):
    species_dir = os.path.join(OUTPUT_DIR, species_name.replace(" ", "_"))
    os.makedirs(species_dir, exist_ok=True)

    # Save combined metadata
    with open(os.path.join(species_dir, "metadata.json"), "w") as f:
        json.dump(recordings, f, indent=2)

    print(f"  Downloading {len(recordings)} files to {species_dir}")

    for rec in tqdm(recordings, desc=f"  {species_name}"):
        raw_url = rec.get("file", "")
        if not raw_url:
            continue
        file_url = "https:" + raw_url if raw_url.startswith("//") else raw_url
        file_path = os.path.join(species_dir, f"{rec['id']}_{rec.get('q', '?')}.mp3")

        if os.path.exists(file_path):
            continue

        try:
            r = requests.get(file_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            print(f"\n Error downloading {file_url}: {e}")


# main
def main():
    print(f" API key:  {API_KEY[:8]}...")
    print(f" Output:   {os.path.abspath(OUTPUT_DIR)}")
    print(f" Target:   {MIN_PER_SPECIES}–{MAX_PER_SPECIES} recordings per species")
    print(f" Quality:  {', '.join(QUALITY_GRADES)}")
    print(f" Species:  {len(SPECIES_LIST)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = []

    for i, species in enumerate(SPECIES_LIST, 1):
        print(f"[{i}/{len(SPECIES_LIST)}] {species}")

        recordings = collect_recordings(species)

        if not recordings:
            print("No recordings found, skipping.\n")
            summary.append({"species": species, "downloaded": 0, "status": "skipped"})
            continue

        if len(recordings) < MIN_PER_SPECIES:
            print(f"Only {len(recordings)} found "
                  f"(below target of {MIN_PER_SPECIES}), downloading anyway...")

        download_recordings(recordings, species)

        summary.append({
            "species": species,
            "downloaded": len(recordings),
            "status": "ok"
        })
        print(f"Saved {len(recordings)} recordings.\n")

    # Print final summary
    total = 0
    for s in summary:
        print(f"  {s['species']:<30} {s['downloaded']} recordings")
        total += s["downloaded"]

    print(f"Total recordings downloaded: {total}")
    print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()