import os

path = "dataset/plantvillage dataset/color"

for folder in os.listdir(path):
    if "Potato" in folder:
        print(folder)
