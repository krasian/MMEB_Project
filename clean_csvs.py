import pandas as pd
import os

INPUT_FOLDER = "data/raw_csv"
OUTPUT_FOLDER = "data/clean_csv"

# seperate train and test species
TRAIN_SPECIES = ["blackbird", "blue_tit", "great_tit", "sparrow"]
TEST_SPECIES = ["robin", "crow", "starling", "thrush","flamingo","toucan"]

MAX_SAMPLES = 2300

# clean function
def clean_df(df, country=None):

    if "quality_grade" in df.columns:
        df = df[df["quality_grade"] == "research"]

    df = df.dropna(subset=["image_url"])
    df = df.drop_duplicates(subset=["image_url"])

    if country and "place_country_name" in df.columns:
        df = df[df["place_country_name"] == country]

    return df

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]

for file in files:

    path = os.path.join(INPUT_FOLDER, file)
    name = file.lower()

    print(f"\nProcessing: {file}")

    df = pd.read_csv(path)

    if any(sp in name for sp in TRAIN_SPECIES):
        print("TRAIN species (Netherlands filter)")
        df = clean_df(df, country="Netherlands")

    elif any(sp in name for sp in TEST_SPECIES):
        print("TEST species (no country filter)")
        df = clean_df(df)

    else:
        print("SKIPPED (unknown species)")
        continue

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=42)

    print("Final rows:", len(df))

    out_name = file.replace(".csv", "_clean.csv")
    out_path = os.path.join(OUTPUT_FOLDER, out_name)

    df.to_csv(out_path, index=False)
    print("Saved:", out_path)










