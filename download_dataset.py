import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Download the dataset
def download_dataset():
    print("Downloading the dataset...")
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    print(f"Dataset downloaded to: {path}")
    return path

# Explore the dataset structure
def explore_dataset(dataset_path):
    print("\nExploring dataset structure:")
    
    # List all directories and files
    all_items = list(Path(dataset_path).rglob("*"))
    
    # Count files by type
    dirs = [item for item in all_items if item.is_dir()]
    files = [item for item in all_items if item.is_file()]
    
    print(f"Total directories: {len(dirs)}")
    print(f"Total files: {len(files)}")
    
    # Print directory structure recursively with limited depth
    def print_directory_structure(directory, depth=0, max_depth=3):
        if depth > max_depth:
            return
        
        items = list(directory.iterdir())
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        indent = "  " * depth
        print(f"{indent}- {directory.name}/")
        
        if len(files) > 0:
            file_count = len(files)
            print(f"{indent}  Contains {file_count} files")
            for file in files[:3]:  # Show only first 3 files
                print(f"{indent}  - {file.name}")
            if file_count > 3:
                print(f"{indent}  - ... and {file_count - 3} more files")
        
        for d in dirs:
            print_directory_structure(d, depth + 1, max_depth)
    
    # Get the main dataset directory
    dataset_dir = Path(dataset_path)
    
    # Search for the actual dataset directory structure
    if (dataset_dir / "Dataset").exists():
        dataset_dir = dataset_dir / "Dataset"
    
    print("\nDirectory structure:")
    print_directory_structure(dataset_dir)
    
    # Count images in train/test/validation splits
    train_dir = dataset_dir / "Train" if (dataset_dir / "Train").exists() else None
    test_dir = dataset_dir / "Test" if (dataset_dir / "Test").exists() else None
    val_dir = dataset_dir / "Validation" if (dataset_dir / "Validation").exists() else None
    
    splits = {
        "Train": train_dir,
        "Test": test_dir,
        "Validation": val_dir
    }
    
    # Check for different class directories
    class_counts = {}
    split_counts = {}
    
    for split_name, split_dir in splits.items():
        if split_dir and split_dir.exists():
            real_images = sum(1 for _ in split_dir.rglob("real*/*.jpg")) + sum(1 for _ in split_dir.rglob("real*/*.png"))
            fake_images = sum(1 for _ in split_dir.rglob("fake*/*.jpg")) + sum(1 for _ in split_dir.rglob("fake*/*.png"))
            
            class_counts[f"{split_name}_real"] = real_images
            class_counts[f"{split_name}_fake"] = fake_images
            split_counts[split_name] = real_images + fake_images
    
    print("\nImage counts by class and split:")
    for key, count in class_counts.items():
        print(f"{key}: {count} images")
    
    print("\nTotal images by split:")
    for split, count in split_counts.items():
        print(f"{split}: {count} images")
    
    # Plot distribution if we have data
    if class_counts:
        plt.figure(figsize=(12, 6))
        
        # Create bars for real and fake in each split
        splits = ["Train", "Test", "Validation"]
        real_counts = [class_counts.get(f"{split}_real", 0) for split in splits]
        fake_counts = [class_counts.get(f"{split}_fake", 0) for split in splits]
        
        x = range(len(splits))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], real_counts, width, label='Real')
        plt.bar([i + width/2 for i in x], fake_counts, width, label='Fake')
        
        plt.xlabel('Dataset Split')
        plt.ylabel('Number of Images')
        plt.title('Image Distribution by Class and Split')
        plt.xticks(x, splits)
        plt.legend()
        
        plt.savefig('dataset_distribution.png')
        print("\nDistribution plot saved as 'dataset_distribution.png'")

if __name__ == "__main__":
    dataset_path = download_dataset()
    explore_dataset(dataset_path) 