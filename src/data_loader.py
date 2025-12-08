"""
Data loading and K-fold cross validation for IIIT-5K dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import Tuple, Iterator
import os


def load_data(csv_path: str, base_dir: str = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K") -> pd.DataFrame:
    """
    Load data from CSV file and prepare image paths.
    
    Args:
        csv_path: Path to the CSV file (e.g., 'iiit-5k/traindata.csv')
        base_dir: Base directory where images are stored
    
    Returns:
        DataFrame with columns: ImgName, GroundTruth, image_path
    """
    df = pd.read_csv(csv_path)
    
    # Construct full image paths
    df['image_path'] = df['ImgName'].apply(
        lambda x: os.path.join(base_dir, x) if not os.path.isabs(x) else x
    )
    
    # Verify that images exist (optional, can be commented out for speed)
    # df = df[df['image_path'].apply(lambda x: os.path.exists(x))]
    
    return df


def create_kfold_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits.
    
    Args:
        data: DataFrame with image data
        n_splits: Number of folds (default: 5)
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
    
    Yields:
        Tuple of (train_indices, val_indices) for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for train_idx, val_idx in kf.split(data):
        yield train_idx, val_idx


def get_fold_data(
    data: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get training and validation data for a specific fold.
    
    Args:
        data: Full DataFrame
        train_indices: Indices for training set
        val_indices: Indices for validation set
    
    Returns:
        Tuple of (train_df, val_df)
    """
    train_df = data.iloc[train_indices].reset_index(drop=True)
    val_df = data.iloc[val_indices].reset_index(drop=True)
    
    return train_df, val_df


def load_all_data(
    train_csv: str = "iiit-5k/traindata.csv",
    base_dir: str = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K"
) -> pd.DataFrame:
    """
    Load training data for K-fold CV.
    
    Args:
        train_csv: Path to training CSV file
        base_dir: Base directory where images are stored
    
    Returns:
        Combined DataFrame with all data
    """
    train_df = load_data(train_csv, base_dir)

    return train_df

