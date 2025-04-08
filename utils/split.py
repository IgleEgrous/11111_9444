import os
import shutil
import random
from pathlib import Path

# Set paths
source_dir = Path("D:/Productivity/github/11111_9444/ColHis-IDS_restructured")
target_dir = Path("ColHis-IDS_split")
magnifications = ["40", "100", "200", "400"]

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create target directories
for split in ['train', 'val', 'test']:
    for mag in magnifications:
        (target_dir / split / mag).mkdir(parents=True, exist_ok=True)

# Split data
for mag in magnifications:
    mag_path = source_dir / mag
    for cls in os.listdir(mag_path):
        cls_path = mag_path / cls
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        random.shuffle(images)
        
        train_cutoff = int(train_ratio * len(images))
        val_cutoff = int((train_ratio + val_ratio) * len(images))
        
        split_data = {
            'train': images[:train_cutoff],
            'val': images[train_cutoff:val_cutoff],
            'test': images[val_cutoff:]
        }

        for split, img_list in split_data.items():
            split_cls_dir = target_dir / split / mag / cls
            split_cls_dir.mkdir(parents=True, exist_ok=True)
            for img_name in img_list:
                src_path = cls_path / img_name
                dst_path = split_cls_dir / img_name
                shutil.copy(src_path, dst_path)

print("Dataset split complete.")
