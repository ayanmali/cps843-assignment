from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "iiit-5k" / "IIIT5K-Word_V3.0" / "IIIT5K"

TRAIN_MAT = DATA_ROOT / "traindata.mat"
TEST_MAT = DATA_ROOT / "testdata.mat"

OUT_TRAIN_CSV = REPO_ROOT / "iiit-5k" / "traindata.csv"
OUT_TEST_CSV = REPO_ROOT / "iiit-5k" / "testdata.csv"


def _to_str(x) -> str:
    """Convert MATLAB/NumPy string-ish values into a normal Python string."""
    if x is None:
        return ""
    # numpy arrays of chars
    if isinstance(x, np.ndarray):
        # sometimes it's an array of strings/chars
        if x.dtype.kind in {"U", "S"}:
            return "".join(x.tolist()).strip()
        # sometimes it's object array of chars
        try:
            flat = x.flatten().tolist()
            if flat and all(isinstance(c, (str, np.str_)) for c in flat):
                return "".join(flat).strip()
        except Exception:
            pass
    # bytes -> decode
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


def _find_records(mat: dict) -> np.ndarray:
    """
    Find the MATLAB variable containing the list of samples.
    In IIIT5K, it is commonly 'traindata' / 'testdata'.
    """
    candidates = [k for k in mat.keys() if not k.startswith("__")]
    # Prefer obvious names if present
    for key in ("traindata", "testdata"):
        if key in mat:
            return mat[key]
    if len(candidates) == 1:
        return mat[candidates[0]]
    # Fallback: pick the largest ndarray
    arrays = [(k, mat[k]) for k in candidates if isinstance(mat[k], np.ndarray)]
    if not arrays:
        raise ValueError(f"No array-like records found. Keys: {candidates}")
    arrays.sort(key=lambda kv: kv[1].size, reverse=True)
    return arrays[0][1]


def mat_to_df(mat_path: Path) -> pd.DataFrame:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    records = _find_records(mat)

    rows = []
    # records can be a single struct or an array of structs
    if isinstance(records, np.ndarray):
        iterable = records.flatten().tolist()
    else:
        iterable = [records]

    for r in iterable:
        # MATLAB structs become objects with attributes
        img = (
            getattr(r, "ImgName", None)
            or getattr(r, "imgName", None)
            or getattr(r, "image", None)
        )
        gt = (
            getattr(r, "GroundTruth", None)
            or getattr(r, "groundTruth", None)
            or getattr(r, "Word", None)
        )

        img_s = _to_str(img)
        gt_s = _to_str(gt)

        # Skip empty rows
        if not img_s:
            continue

        rows.append({"ImgName": img_s, "GroundTruth": gt_s})

    df = pd.DataFrame(rows)

    # Basic cleanup: remove any accidental NaNs/empties
    df["ImgName"] = df["ImgName"].astype(str).str.strip()
    df["GroundTruth"] = df["GroundTruth"].astype(str).str.strip()

    df = df[df["ImgName"] != ""].reset_index(drop=True)
    return df


def main() -> None:
    if not TRAIN_MAT.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_MAT}")
    if not TEST_MAT.exists():
        raise FileNotFoundError(f"Missing: {TEST_MAT}")

    train_df = mat_to_df(TRAIN_MAT)
    test_df = mat_to_df(TEST_MAT)

    OUT_TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_TRAIN_CSV, index=False)
    test_df.to_csv(OUT_TEST_CSV, index=False)

    print(f"Wrote: {OUT_TRAIN_CSV}  (rows={len(train_df)})")
    print(f"Wrote: {OUT_TEST_CSV}   (rows={len(test_df)})")
    print("\nSample rows (train):")
    print(train_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
