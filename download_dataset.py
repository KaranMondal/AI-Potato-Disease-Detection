import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

print("Downloaded to:", path)

# Move dataset to project folder
destination = "dataset"

if not os.path.exists(destination):
    shutil.copytree(path, destination)

print("Dataset moved to project folder!")