import pandas as pd
import os

INPUT_FOLDER = "data/raw_csv"

files = os.listdir(INPUT_FOLDER)

for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        print(df.head())