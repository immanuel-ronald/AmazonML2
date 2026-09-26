import pandas as pd

base = "dataset/train/"

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

for file in files:
    path = base + file
    df = pd.read_csv(path, sep="\t")

    print("\n" + "=" * 50)
    print(file)
    print("=" * 50)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())