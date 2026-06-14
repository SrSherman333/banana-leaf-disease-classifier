import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw_dir = "../data/raw/Original_images"

types = os.listdir(raw_dir)

types = [t for t in types if os.path.isdir(os.path.join(raw_dir, t))]

print("=ORIGINAL IMAGES=")
print("Types of images")
for i, t in enumerate(types):
    print(f"{i+1}. {t}")

print("\nImages per type")
count = {}
for t in types:
    route = os.path.join(raw_dir, t)
    num_ims = len(os.listdir(route))
    count[t] = len(os.listdir(route))
    print(f"{t}: {num_ims} images")

print("\n=AUGMENTED IMAGES=")
raw_dir_aug = "../data/raw/Augmented_images"

types_aug = os.listdir(raw_dir_aug)
types_aug = [t for t in types_aug if os.path.isdir(os.path.join(raw_dir_aug, t))]

print("Types of images")
for i, t in enumerate(types_aug):
    print(f"{i+1}. {t}")

print("\nImages per type")
count_aug = {}
for t in types_aug:
    route = os.path.join(raw_dir_aug, t)
    num_ims = len(os.listdir(route))
    count_aug[t] = len(os.listdir(route))
    print(f"{t}: {num_ims} images")

plt.figure(figsize=(10, 5))
plt.bar(count.keys(), count.values())
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of images per type")
plt.tight_layout()
plt.savefig("../data/processed/distribution_types.png")

plt.figure(figsize=(10, 5))
plt.bar(count_aug.keys(), count_aug.values())
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of images per type")
plt.tight_layout()
plt.savefig("../data/processed/distribution_types_augmented.png")
print("\nGenerated graphs")

for t in types:
    route = os.path.join(raw_dir, t)
    images = [f for f in os.listdir(route) if f.lower().endswith(('.png','.jpg','.jpeg'))][:4]
    if not images:
        continue
    fig, axes = plt.subplots(1, len(images), figsize=(12,4))
    if len(images) == 1:
        axes = [axes]
    for ax, img_name in zip(axes, images):
        img = cv2.imread(os.path.join(route, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    fig.suptitle(t)
    plt.tight_layout()
    plt.savefig(f"../data/processed/sample_{t.replace(' ','_')}.png")
    plt.close()
print("\nSamples saved in data/processed/")
