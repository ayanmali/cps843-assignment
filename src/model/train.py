"""
training the model with K-fold cross validation
"""
import sys
import os
import pandas as pd

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import create_kfold_splits, get_fold_data

def train_with_kfold(
    n_splits: int = 5,
    train_csv: str = "iiit-5k/traindata.csv",
    test_csv: str = "iiit-5k/testdata.csv",
    shuffle: bool = True,
    random_state: int = 42
):
    """
    Train model using K-fold cross validation.
    
    Args:
        n_splits: Number of folds for cross validation (default: 5)
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        base_dir: Base directory where images are stored
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
    """
    # Load all data (combining train and test for K-fold CV)
    print(f"Loading data from {train_csv} and {test_csv}...")
    train_df = pd.read_csv(train_csv)
    print(f"Loaded {len(train_df)} total samples")
    
    # Create K-fold splits
    print(f"\nCreating {n_splits}-fold cross validation splits...")
    fold_generator = create_kfold_splits(train_df, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Iterate through each fold
    for fold_num, (train_indices, val_indices) in enumerate(fold_generator, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/{n_splits}")
        print(f"{'='*60}")
        
        # Get data for this fold
        train_df, val_df = get_fold_data(train_df, train_indices, val_indices)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        # TODO: Replace this with your actual training code
        # Example:
        # model = create_model()
        # train_model(model, train_df)
        # val_accuracy = evaluate_model(model, val_df)
        # fold_results.append(val_accuracy)
        
        # Placeholder for training
        print("Training model...")
        # Your training code here
        
        print("Evaluating on validation set...")
        # Your evaluation code here
        
        # Example: Store fold results
        # fold_results.append({
        #     'fold': fold_num,
        #     'train_size': len(train_df),
        #     'val_size': len(val_df),
        #     'val_accuracy': val_accuracy
        # })
    
    # Print summary
    print(f"\n{'='*60}")
    print("K-Fold Cross Validation Summary")
    print(f"{'='*60}")
    # if fold_results:
    #     accuracies = [r['val_accuracy'] for r in fold_results]
    #     print(f"Mean validation accuracy: {np.mean(accuracies):.4f}")
    #     print(f"Std validation accuracy: {np.std(accuracies):.4f}")
    #     print(f"\nPer-fold results:")
    #     for result in fold_results:
    #         print(f"  Fold {result['fold']}: {result['val_accuracy']:.4f}")


if __name__ == "__main__":
    # Example usage
    train_with_kfold(n_splits=5)
    
    # You can also customize parameters:
    # train_with_kfold(
    #     n_splits=10,
    #     shuffle=True,
    #     random_state=42
    # )
