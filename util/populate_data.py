# Testing script to populate the trianing arrays
import cv2
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

"""
Each split folder
For each region
Iter through flood label dir
Store flood label in Y_Train/Test
Find VV and VH file with the same filename
Concatenate VV/VH
Any more processing like RGB
Store in X_Train/Test
"""

ds_root = Path("ETCI-2021-Flood-Detection_edited/data/")
if not os.path.exists(ds_root):
    raise FileNotFoundError("Dataset not in given directory")


def concatenate_SAR(vv_path, vh_path):
    vv_img = cv2.imread(vv_path, cv2.IMREAD_GRAYSCALE)
    vh_img = cv2.imread(vh_path, cv2.IMREAD_GRAYSCALE)
    if vv_img is  None:
        raise FileExistsError(f"Could not open vv file with path {vv_path}")
    if vh_img is None:
        raise FileExistsError(f"Could not open vh file with path {vh_path}")
    
    concatenated_SAR = np.stack((vv_img, vh_img), axis=-1)
    if concatenated_SAR.shape != (*vv_img.shape, 2):
        raise ValueError("Final concatenated image shape mismatch.")
    return concatenated_SAR

def find_vv(flood_label_path: Path) -> Path:
    file_name = flood_label_path.name
    vv_path = flood_label_path.parent.parent / "vv" / file_name
    vv_path = vv_path.with_name(f"{vv_path.stem}_vv{vv_path.suffix}")
    if vv_path.exists():
        return vv_path
    else:
        print(vv_path)
        raise FileNotFoundError(f"Couldn't find vv file for {flood_label_path}")
    
def find_vh(flood_label_path: Path) -> Path:
    file_name = flood_label_path.name
    vh_path = flood_label_path.parent.parent / "vh" / file_name
    vh_path = vh_path.with_name(f"{vh_path.stem}_vh{vh_path.suffix}")
    if vh_path.exists():
        return vh_path
    else:
        print(vh_path)
        raise FileNotFoundError(f"Couldn't find vh file for {flood_label_path}")
    
X_Train = []
X_Test = []
Y_Train = []
Y_Test = []
regions = [] #might need it
for split in ["train", "test"]:
    split_path = ds_root / split
    
    for region_dir in tqdm(split_path.iterdir()):
        regions.append(region_dir.name)
        flood_label_dir = region_dir / "tiles" / "flood_label"

        for flood_label in flood_label_dir.iterdir():
            if flood_label.suffix != ".png":
                continue
            fl_img = cv2.imread(flood_label, cv2.IMREAD_GRAYSCALE)
            vv_path = find_vv(flood_label)
            vh_path = find_vh(flood_label)
            SAR_img = concatenate_SAR(vv_path, vh_path)
            
            if split == "train":
                X_Train.append(SAR_img)
                Y_Train.append(fl_img)
            else:
                X_Test.append(SAR_img)
                Y_Test.append(fl_img)

print(len(X_Train), len(Y_Train))

# TESTING#
# ETCI-2021-Flood-Detection_edited\data\train\bangladesh_20170314t115609\tiles\vv\bangladesh_20170314t115609_x-0_y-31_vv.png
# test_region_dir = Path("ETCI-2021-Flood-Detection_edited/data/train/bangladesh_20170314t115609/tiles/flood_label")
# flood_labels = [flood_label for flood_label in test_region_dir.iterdir() if flood_label.suffix == ".png"]
# test_flood_label_path = flood_labels[0]
# if not test_flood_label_path.exists():
#     print("test path doesnt exist")
# vv_path = find_vv(test_flood_label_path)
# print(vv_path)