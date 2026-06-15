import os
import random
from PIL import Image

raw_original = "../data/raw/Original_images"
raw_augmented = "../data/raw/Augmented_images"
processed = "../data/processed"

train_dir = os.path.join(processed, "train")
val_dir = os.path.join(processed, "val")
test_dir = os.path.join(processed, "test")

img_size = (224, 224)

train_split = 0.7
val_split = 0.15
test_split = 0.15

seed = 42
random.seed(42)

types = sorted([
    d for d in os.listdir(raw_original)
    if os.path.isdir(os.path.join(raw_original, d))
])

print("Detected types: ", types)

for split in [train_dir, val_dir, test_dir]:
    for t in types:
        os.makedirs(os.path.join(split, t), exist_ok=True)

def process_and_copy(origin, destination, archive):
    img = Image.open(os.path.join(origin, archive)).convert("RGB")
    img = img.resize(img_size, Image.LANCZOS)
    img.save(os.path.join(destination, archive))

for t in types:
    route_type_orig = os.path.join(raw_original, t)
    archives = [f for f in os.listdir(route_type_orig)
                if f.lower().endswith((".png", ".jpg", ".jpge"))]
    random.shuffle(archives)

    n_total = len(archives)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_files = archives[:n_train]
    val_files = archives[n_train:n_train+n_val]
    test_files = archives[n_train+n_val:]

    print(f"\nType: {t} | Originals: {n_total} -> train:{len(train_files)} val: {len(val_files)} test: {len(test_files)}")

    for f in train_files:
        process_and_copy(route_type_orig, os.path.join(train_dir, t), f)
    for f in val_files:
        process_and_copy(route_type_orig, os.path.join(val_dir, t), f)
    for f in test_files:
        process_and_copy(route_type_orig, os.path.join(test_dir, t), f)

print("\n--- Adding augmented images to the train set ---")
for t in types:
    folder_augmented = "Augmented " + t
    route_aug = os.path.join(raw_augmented, folder_augmented)
    if not os.path.exists(route_aug):
        print(f" Not found {route_aug}, is omited")
        continue

    archives_aug = [f for f in os.listdir(route_aug)
                        if f.lower().endswith((".png", ".jpg", ".jpge"))]
    print(f" {t}: {len(archives_aug)} augmenteds -> train")

    for f in archives_aug:
        new_name = f"aug_{f}"
        img = Image.open(os.path.join(route_aug, f)).convert("RGB")
        img = img.resize(img_size, Image.LANCZOS)
        img.save(os.path.join(train_dir, t, new_name))

def count(folder):
    total = 0
    for t in types:
        route = os.path.join(folder, t)
        total += len(os.listdir(route)) if os.path.exists(route) else 0
    return total

print("\n========== FINAL SUMMARY ==========")
print(f"Train: {count(train_dir)} images")
print(f"Validation: {count(val_dir)} images")
print(f"Test: {count(test_dir)} images")
print("===================================")
