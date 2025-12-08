"""
Example script showing how to use K-fold cross validation with the IIIT-5K dataset
"""
from src.data_loader import load_all_data, create_kfold_splits, get_fold_data

def main():
    # Load all data (combining train and test CSV files)
    print("Loading data...")
    all_data = load_all_data(
        train_csv="iiit-5k/traindata.csv",
        test_csv="iiit-5k/testdata.csv",
        base_dir="iiit-5k/IIIT5K-Word_V3.0/IIIT5K"
    )
    
    print(f"Total samples: {len(all_data)}")
    print(f"Columns: {list(all_data.columns)}")
    print("\nFirst few rows:")
    print(all_data[['ImgName', 'GroundTruth']].head())
    
    # Create 5-fold cross validation splits
    print("\n" + "="*60)
    print("Creating 5-fold cross validation splits...")
    print("="*60)
    
    n_splits = 5
    fold_generator = create_kfold_splits(
        all_data,
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
    
    # Iterate through each fold
    for fold_num, (train_indices, val_indices) in enumerate(fold_generator, 1):
        print(f"\nFold {fold_num}/{n_splits}:")
        
        # Get training and validation data for this fold
        train_df, val_df = get_fold_data(all_data, train_indices, val_indices)
        
        print(f"  Training samples: {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")
        print(f"  Sample training image: {train_df.iloc[0]['ImgName']}")
        print(f"  Sample validation image: {val_df.iloc[0]['ImgName']}")
        
        # Here you would:
        # 1. Load images using train_df['image_path'] and val_df['image_path']
        # 2. Train your model on train_df
        # 3. Evaluate on val_df
        # 4. Store results
        
        # Example: Access image paths and ground truth
        # for idx, row in train_df.iterrows():
        #     image_path = row['image_path']
        #     ground_truth = row['GroundTruth']
        #     # Load image and process...
    
    print("\n" + "="*60)
    print("K-fold cross validation setup complete!")
    print("="*60)
    print("\nTo use in your training code:")
    print("1. Import: from src.data_loader import load_all_data, create_kfold_splits, get_fold_data")
    print("2. Load data: all_data = load_all_data(...)")
    print("3. Create folds: fold_generator = create_kfold_splits(all_data, n_splits=5)")
    print("4. Iterate: for train_idx, val_idx in fold_generator:")
    print("5. Get data: train_df, val_df = get_fold_data(all_data, train_idx, val_idx)")


if __name__ == "__main__":
    main()

