from rembg.bg import remove
import numpy as np
import io
from PIL import Image
import glob
import os 
from tqdm import tqdm 
import subprocess 

data_dir = './toDataset/oxygen_woval_train_cleaned/bounding_box_train'

dst_dir = 'removed_bg' + data_dir[data_dir.rfind('/'):]
os.makedirs(dst_dir, exist_ok=True)

for img_path in tqdm(glob.glob(f'{data_dir}/*')):
    target_path = img_path[img_path.rfind('/')+1:]
    subprocess.call(["rembg", "-o", f"{dst_dir}/{target_path}", f"{img_path}"])