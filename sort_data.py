import os
import shutil

def sort_data(data_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(f"{output_root}/images", exist_ok=True)
    os.makedirs(f"{output_root}/labels", exist_ok=True)

    alll = os.listdir(data_root)
    for i in alll:
        if ".txt" in i:
            shutil.copy2(f"{data_root}/{i}", f"{output_root}/labels")
        elif ".jpg" in i:
            shutil.copy2(f"{data_root}/{i}", f"{output_root}/images")
        else:
            print(f"skip file: {i}")

sort_data(r"tmp/train", r"dataset/train")
sort_data(r"tmp/valid", r"dataset/val")
sort_data(r"tmp/test", r"dataset/test")