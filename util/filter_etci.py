
"""
Algorithm flow:
loop through train, test, test internal files
In each folder, loop through each region folder
If no flood label, del folder
Else: Calc true flood labels in flood map, if < threshold, delete itself and the corresponding VV VH files


Deletion scenarios:
Is a water_body_label folder
Sample has no flood labels
Flood label ratio Not flood:flood below the threshold
"""

import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm
from pathlib import Path
# import matplotlib.pyplot as plt # only needed for troubleshooting

LO_THRESHOLD = 0.4
HI_THRESHOLD = 0.95
DS_FILE = "ETCI2021_filtered_40pct/data/"

def check_pct_flood_pixels(sample_path, pct_lo_threshold, pct_hi_threshold) -> bool:
    if not sample_path.suffix.lower() == ".png":
        return True # is not an image
    img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if img is None: # sometimes it is a non PNG filetype, should probs delete those too
        return False
    flood_pixels = np.sum(img == 255)
    pct_flood = (flood_pixels / (img.shape[0] * img.shape[1]))
    return pct_flood < pct_lo_threshold or pct_flood > pct_hi_threshold


def construct_VV_path(flood_label_path: Path) -> Path:
    # Create path to VV file with same name as flood label
    new_parts = [p if p != "flood_label" else "vv" for p in flood_label_path.parts] # need to add _vv at the end, same for vh
    vv_path = Path(*new_parts)
    vv_path = vv_path.with_name(f"{vv_path.stem}_vv{vv_path.suffix}")
    if vv_path.is_file():
        return vv_path
    else:
        print("VV path doesn't exist")
        return None

def constuct_VH_path(flood_label_path: Path) -> Path:
    # Create path to VH file with same name as flood label
    new_parts = [p if p != "flood_label" else "vh" for p in flood_label_path.parts]
    vh_path = Path(*new_parts)
    vh_path = vh_path.with_name(f"{vh_path.stem}_vh{vh_path.suffix}")
    if os.path.exists(vh_path):
        return vh_path
    else:
        print("VH path doesn't exist")
        return None

def construct_WBL_path(flood_label_path: Path) -> Path:
    # Create path to Water Body Label file with same name as flood label
    # some files don't have a water body label bare in mind
    tile_dir = flood_label_path.parent # up one levels
    if os.path.exists(tile_dir / "water_body_label"):
        wbl_path = tile_dir / "water_body_label"
        return wbl_path
    else:
        print("Water body labels don't exist")
        return None

def main(ds_root: Path):
    folder_removed = 0
    files_removed = 0
    total_flood_labels = 0
    wbl_not_found = 0
    actually_delete = True # to actually delete the files, mostly for testing purposes
    if not ds_root.exists():
        raise FileExistsError(f"Path doesn't exist: {str(ds_root)}")
    
    print("Delete flag is ", actually_delete)
    if actually_delete:
        response = input("Do you wish to continue? Y/N ")
        if response not in ['Y', 'y', 'yes', 'Yes', 'YES']:
            return 0, 0, 0, 0


    # go through train/test folders
    for split_dir in tqdm(ds_root.iterdir(), desc="Data folders"):
        if not split_dir.is_dir():
            continue
            
        # region folders 
        for region_dir in split_dir.iterdir():
            if not region_dir.is_dir():
                continue
                
            tiles_path = region_dir / "tiles" #always a tiles dir
            
            if not tiles_path.exists():
                print("Tiles don't exist thats worrying")
                return

            # check if 'flood_label' folder exists inside the region samples
            flood_label_dir = tiles_path / "flood_label"
            
            if not flood_label_dir.exists(): # if no flood labels: delete folder
                folder_removed += 1
                sample_path = tiles_path.parent
                print(sample_path)
                if os.path.exists(sample_path) and sample_path.is_dir():
                    if actually_delete:
                        try:
                            shutil.rmtree(sample_path)
                            print(f"Deleted directory with no flood labels: {tiles_path}")
                        except OSError as e:
                            print(f"Error: {e.filename} - {e.strerror}")

            else: # check flood labels; if they don't meet the threshold: del label and corresponding files
                # delete water_body_label folder to save space, only need flood labels
                WBL_path = construct_WBL_path(flood_label_dir)
                if WBL_path == None:
                    wbl_not_found += 1
                else: 
                    folder_removed += 1
                    # print(WBL_path)
                    if actually_delete: 
                        if os.path.exists(WBL_path) and WBL_path.is_dir():
                            shutil.rmtree(WBL_path)

                for flood_label in flood_label_dir.iterdir():
                    if ".ipynb" in str(flood_label): #ignore the checkpoint ipynb
                        continue
                    flood_label_path = Path(flood_label)
                    total_flood_labels += 1

                    delete_file = check_pct_flood_pixels(flood_label_path, LO_THRESHOLD, HI_THRESHOLD)

                    if delete_file:
                        files_removed += 1
                        
                        if os.path.exists(flood_label_path):
                            VV_path = construct_VV_path(flood_label_path)
                            VH_path = constuct_VH_path(flood_label_path)

                            for path in [VV_path, VH_path, flood_label_path]:
                                if path == None:
                                    continue

                                if actually_delete:
                                    try:
                                        if os.path.exists(path):
                                            os.remove(path)
                                            # print(f"Deleted files with insufficient flood labels: {flood_label}")
                                    except OSError as e:
                                        print(f"Error: {e.filename} - {e.strerror}")


    return folder_removed, files_removed, total_flood_labels, wbl_not_found

print("Using dataset path: ", Path(DS_FILE).resolve())
folders, files, total_flood_labels, wbl_not_found = main(ds_root=Path(DS_FILE))

print(f"Removed {folders} folders")
print(f"Removed {files} individual flood labels out of {total_flood_labels} total")
print(f"Leaving {total_flood_labels - files} samples remaining")
print(f"Water body labels not found {wbl_not_found} times")