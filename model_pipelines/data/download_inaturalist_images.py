"""
Download iNaturalist images from URLs in cleaned CSVs.

Reads each CSV in INPUT_CSV, downloads the image at `image_url` for each
row, saves it to a per-species folder, and writes out a new CSV where
`image_url` has been replaced with the local image path (renamed to
`image_path`).

Features:
- Prevents Windows from sleeping/hibernating while running.
- Shows a tqdm progress bar with live MB/s throughput.
- Writes failed downloads to a log file (download_failures.log).
- Press 'p' to pause/resume, 'q' to quit gracefully (Windows).
- Resumable: re-running skips images that already exist on disk.

Run directly:
    python -m data.download_inaturalist_images
"""
import os
import sys
import ctypes
import logging
import pandas as pd
import requests
from tqdm import tqdm

if sys.platform == "win32":
    import msvcrt


# ---- Configuration ---------------------------------------------------------
DATA_RAW_DIR = r"D:\MMEB-Project\data\raw"
DATA_PROCESSED_DIR = r"D:\MMEB-Project\data\processed"

INPUT_CSV = [
    "blackbird_clean.csv",
    "blue_tit_clean.csv",
    "crow_clean.csv",
    "flamingo_clean.csv",
    "great_tit_clean.csv",
    "mallard_clean.csv",
    "robin_clean.csv",
    "sparrow_clean.csv",
    "starling_clean.csv",
    "thrush_clean.csv",
    "toucan_clean.csv",
    "mallard-test.csv"
]

OUTPUT_CSV = [
    "updated_blackbird_data.csv",
    "updated_EurasianBlueTit_data.csv",
    "updated_crow_data.csv",
    "updated_flamingo_data.csv",
    "updated_GreatTit_data.csv",
    "updated_mallard_data.csv",
    "updated_robin_data.csv",
    "updated_HouseSparrow_data.csv",
    "updated_EuropeanStarling.csv",
    "updated_thrush_data.csv",
    "updated_toucan_data.csv",
    'updated_mallard_test_data.csv'
]

IMAGE_FOLDER = [
    "downloaded_Blackbird_images",
    "downloaded_Eurasian_images",
    "downloaded_crow_images",
    "downloaded_flamingo_images",
    "downloaded_GreatTit_images",
    "downloaded_mallard_images",
    "downloaded_robin_images",
    "downloaded_HouseSparrow_images",
    "downloaded_EuropeanStarling_images",
    "downloaded_thrush_images",
    "downloaded_toucan_images",
    "downloaded_mallard_test_images"
]

LOG_FILE = os.path.join(DATA_PROCESSED_DIR, "download_failures.log")


# ---- Logging ---------------------------------------------------------------
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

logger = logging.getLogger("downloader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
logger.propagate = False


# ---- Keep Windows awake ----------------------------------------------------
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def prevent_sleep():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )


def allow_sleep():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


# ---- Pause / quit handling -------------------------------------------------
def _read_key_blocking():
    ch = msvcrt.getch()
    return ch.decode("ascii", errors="ignore").lower()


def check_pause_quit():
    if sys.platform != "win32":
        return False
    if not msvcrt.kbhit():
        return False
    key = msvcrt.getch().decode("ascii", errors="ignore").lower()
    if key == "q":
        return True
    if key == "p":
        tqdm.write("Paused. Press 'p' to resume, 'q' to quit.")
        while True:
            k = _read_key_blocking()
            if k == "p":
                tqdm.write("Resumed.")
                return False
            if k == "q":
                return True
    return False


# ---- Download logic --------------------------------------------------------
def download_image(url: str, filename: str, row_id):
    """Download URL to filename. Returns (path, bytes_downloaded).
    bytes_downloaded is 0 for cached/skipped files and failures."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return filename, 0
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename, len(response.content)
        logger.warning(f"id={row_id} HTTP {response.status_code} url={url}")
        return None, 0
    except Exception as e:
        logger.warning(f"id={row_id} error={type(e).__name__}: {e} url={url}")
        return None, 0


# Bar format with the iteration rate stripped out — we show MB/s in postfix.
BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"


def main():
    prevent_sleep()
    try:
        for folder in IMAGE_FOLDER:
            os.makedirs(os.path.join(DATA_PROCESSED_DIR, folder), exist_ok=True)

        tqdm.write("Controls: press 'p' to pause/resume, 'q' to quit.")

        for n, csv_in in enumerate(INPUT_CSV):
            df = pd.read_csv(os.path.join(DATA_RAW_DIR, csv_in))
            new_image_paths = []
            output_csv_path = os.path.join(DATA_PROCESSED_DIR, OUTPUT_CSV[n])
            failures = 0
            total_bytes = 0
            quit_now = False

            logger.info(f"=== Starting {csv_in} ({len(df)} rows) ===")

            pbar = tqdm(
                df.iterrows(),
                total=len(df),
                desc=csv_in,
                bar_format=BAR_FORMAT,
            )
            for index, row in pbar:
                if check_pause_quit():
                    quit_now = True
                    break

                image_url = row["image_url"]
                if pd.isna(image_url):
                    new_image_paths.append(None)
                    continue

                filename = f"{row['id']}.jpg"
                filepath = os.path.join(DATA_PROCESSED_DIR, IMAGE_FOLDER[n], filename)

                saved_path, nbytes = download_image(image_url, filepath, row["id"])
                new_image_paths.append(saved_path)
                total_bytes += nbytes

                if saved_path is None:
                    failures += 1

                # Live MB/s based on bytes actually downloaded this CSV.
                elapsed = pbar.format_dict.get("elapsed", 0) or 0
                mbps = (total_bytes / 1e6 / elapsed) if elapsed > 0 else 0
                pbar.set_postfix({"MB/s": f"{mbps:.2f}", "failed": failures})
            pbar.close()

            if quit_now:
                msg = (
                    f"Quit requested during {csv_in}. CSV not saved. "
                    f"Image files are kept — rerun to resume."
                )
                logger.info(msg)
                tqdm.write(msg)
                return

            df["image_url"] = new_image_paths
            df = df.rename(columns={"image_url": "image_path"})
            df.to_csv(output_csv_path, index=False)

            summary = (
                f"Finished {csv_in}: {len(df) - failures} ok, {failures} failed, "
                f"{total_bytes / 1e6:.1f} MB downloaded. CSV -> {output_csv_path}"
            )
            logger.info(summary)
            tqdm.write(summary)
    except KeyboardInterrupt:
        tqdm.write("\nCtrl+C detected. Image files are kept; rerun to resume.")
        logger.info("Interrupted via Ctrl+C.")
    finally:
        allow_sleep()


if __name__ == "__main__":
    main()