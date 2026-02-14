import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = "dataset/plantvillage dataset/color"
output_dir = "dataset_clean"

classes = [
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy"
]

for cls in classes:
    class_path = os.path.join(source_dir, cls)

    if not os.path.exists(class_path):
        print(f"Class folder not found: {class_path}")
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"{cls} -> Found {len(images)} images")

    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for split, img_list in zip(["train", "test"], [train_imgs, test_imgs]):
        split_path = os.path.join(output_dir, split, cls)
        os.makedirs(split_path, exist_ok=True)

        for img in img_list:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(split_path, img)
            )

print("Dataset prepared successfully!")