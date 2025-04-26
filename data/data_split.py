import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(root_dir, target_dir, test_ratio=0.2, val_ratio=0.2, seed=42):
    """
    Split dataset into train/val/test sets with preserved directory structure
    Args:
        root_dir: Path to original dataset (e.g., r"EBH-HE-IDS\ColHis-IDS")
        target_dir: Path to save split datasets (e.g., r"EBHI-Split")
        test_ratio: Proportion for test set (default: 20%)
        val_ratio: Proportion for validation set (default: 20%)
        seed: Random seed
    """
    # Validate input ratios
    assert test_ratio + val_ratio < 1.0, "Sum of test_ratio and val_ratio must be less than 1"

    # Define ratios
    val_test_ratio = val_ratio + test_ratio
    val_ratio_adjusted = val_ratio / val_test_ratio if val_test_ratio != 0 else 0

    # Check root directory existence
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    # Create class labels from directory structure
    classes = [d for d in os.listdir(root_dir) 
              if os.path.isdir(os.path.join(root_dir, d))]
    
    if not classes:
        raise ValueError(f"No valid classes found in root directory: {root_dir}")

    for cls in classes:
        class_path = os.path.join(root_dir, cls)
        all_samples = []
        
        # Collect all 200x images
        print(f"\nProcessing class: {cls}")
        for root, _, files in os.walk(class_path):
            # Check for magnification directory using path components
            dir_components = root.split(os.sep)
            if "200" in dir_components[-1]:  # More flexible matching
                print(f"Found 200x directory: {root}")
                png_files = [os.path.join(root, f) for f in files 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Multiple formats
                print(f"Found {len(png_files)} images")
                all_samples.extend(png_files)

        # Check for empty class
        if not all_samples:
            print(f"⚠️ Warning: No valid images found for class {cls}")
            continue

        # Ensure minimum samples per split
        min_samples = 2  # At least 1 sample per split
        if len(all_samples) < min_samples:
            print(f"⚠️ Warning: Insufficient samples ({len(all_samples)}) for class {cls}")
            continue

        # Split dataset with stratification
        try:
            # First split: train vs (val + test)
            train, val_test = train_test_split(
                all_samples,
                test_size=val_test_ratio,
                random_state=seed,
                shuffle=True,
                stratify=[cls]*len(all_samples)  # Maintain class distribution
            )
            
            # Second split: val vs test
            val, test = train_test_split(
                val_test,
                test_size=test_ratio/val_test_ratio,
                random_state=seed,
                stratify=[cls]*len(val_test)
            )
        except ValueError as e:
            print(f"❌ Error splitting class {cls}: {e}")
            print(f"Total samples: {len(all_samples)}")
            print("Adjusting to simple random split...")
            train, val_test = train_test_split(all_samples, test_size=val_test_ratio, random_state=seed)
            val, test = train_test_split(val_test, test_size=test_ratio/val_test_ratio, random_state=seed)

        # Create target directories
        splits = {'train': train, 'val': val, 'test': test}
        for split_name, split_files in splits.items():
            for src_path in split_files:
                # Preserve original structure
                rel_path = os.path.relpath(src_path, root_dir)
                dst_path = os.path.join(target_dir, split_name, rel_path)
                
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)  # Preserve metadata

        print(f"✅ Class {cls} split complete:")
        print(f"   Train: {len(train)}\n   Val: {len(val)}\n   Test: {len(test)}")

if __name__ == "__main__":
    # Example usage
    split_dataset(
        root_dir=r"./11111_9444/EBH-HE-IDS/ColHis-IDS",
        target_dir=r"./11111_9444/EBH-HE-IDS/EBHI-Split",
        test_ratio=0.2,
        val_ratio=0.2,
        seed=42
    )