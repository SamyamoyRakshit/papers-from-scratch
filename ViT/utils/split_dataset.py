import os
import shutil
import random
from pathlib import Path

class DatasetSplitter:
    def __init__(self, source_dir="data", target_base="dataset", split_ratio=0.8, seed=42):
        self.source_dir = Path(source_dir)
        self.target_base = Path(target_base)
        self.train_dir = self.target_base / "train"
        self.test_dir = self.target_base / "test"
        self.split_ratio = split_ratio
        self.seed = seed

    def split(self):
        # Skip if target already exists and if it's not empty
        if self.target_base.exists() and any(self.target_base.iterdir()):
            print(f"Skipping split: '{self.target_base}' already exists and is not empty.")
            return

        # Set seed for reproducibility
        random.seed(self.seed)

        # Create output folders
        for folder in [self.train_dir, self.test_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        # Process each class folder
        for class_folder in self.source_dir.iterdir():
            if class_folder.is_dir():
                # Match all image extensions
                valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

                images = [
                img for img in class_folder.rglob("*")
                if img.is_file() and img.suffix.lower().strip().lstrip(".") in [e.lstrip(".") for e in valid_exts]
                ]
               
                if not images:
                    continue

                random.shuffle(images)
                split_point = int(len(images) * self.split_ratio)

                train_images = images[:split_point]
                test_images = images[split_point:]

                # Create class subfolders
                (self.train_dir / class_folder.name).mkdir(exist_ok=True)
                (self.test_dir / class_folder.name).mkdir(exist_ok=True)

                # Copy files
                for img_path in train_images:
                    shutil.copy(img_path, self.train_dir / class_folder.name / img_path.name)
                for img_path in test_images:
                    shutil.copy(img_path, self.test_dir / class_folder.name / img_path.name)

                # Print counts in target folders
                train_count = len(list((self.train_dir / class_folder.name).glob("*")))
                test_count = len(list((self.test_dir / class_folder.name).glob("*")))
                # print the length of images for "data" folder's images
                print(f"{class_folder.name}: Found {len(images)} valid images.")
                print(f"{class_folder.name}: train = {train_count}, test = {test_count}, total = {train_count + test_count}")
                print("="*50)


        print("Dataset split complete.")

