from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_all_data  # noqa: E402


def main():
    df = load_all_data()  # uses iiit-5k/traindata.csv by default
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))

    # Verify that the first image path actually exists on disk
    first_path = REPO_ROOT / df.iloc[0]["image_path"]
    print("\nFirst resolved image_path:", first_path)
    print("Exists:", first_path.exists())


if __name__ == "__main__":
    main()
