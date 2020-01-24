import os, sys
import shutil
root_dir = os.getcwd()
def find_root_dir(filepath, basename):
    if os.path.basename(filepath) == basename:
        return filepath
    else:
        root_dir = find_root_dir(os.path.dirname(filepath), basename)
        return root_dir

root_dir = find_root_dir(root_dir, "HDImGeneration")
sys.path.append(root_dir)
from utils import *

## specify the stroma slides mat file path
maskfolder = "/home/wenyuan/Documents/Project/pixpixHD/datasets/cityscapes/train_label"
imfolder = "/home/wenyuan/Documents/Project/pixpixHD/datasets/cityscapes/train_img"
imnames = files_under_folder_with_suffix(imfolder, "png")
masknames = files_under_folder_with_suffix(maskfolder, "png")

maskfolder_dest = os.path.join(root_dir, "Datasets/ProstatePair/train_label")
imfolder_dest = os.path.join(root_dir, "Datasets/ProstatePair/train_img")

stroma_maskfolder_dest = os.path.join(root_dir, "Datasets/ProstatePair/stroma_slides/label")
stroma_imfolder_dest = os.path.join(root_dir, "Datasets/ProstatePair/stroma_slides/img")


stroma_slides = [3, 11, 15, 18, 22, 28, 40, 45, 47, 60, 61, 62, 67, 70, 75, 89, 94, 102, 119, 121, 131, 144, 147, 158,
                 159, 160, 164, 168, 171, 174, 181, 198, 200, 202, 203, 208, 210, 211, 212, 219, 222, 230, 231, 235, 236,
                 237, 252, 253, 260, 261, 262, 268, 269, 276, 281, 283, 285, 301, 302, 303, 309, 310, 324, 325, 332, 333,
                 337, 347, 364, 367, 370, 373, 375, 378, 384, 385, 388, 393, 400, 403, 407, 413, 418, 423, 440, 443, 460,
                 462, 467, 473, 475, 476, 478, 483, 488, 493, 499, 504, 508, 509]

for i in range(len(imnames)):
    if i in stroma_slides:
        shutil.move(os.path.join(imfolder, imnames[i]), os.path.join(stroma_imfolder_dest, imnames[i]))
        shutil.move(os.path.join(maskfolder, masknames[i]), os.path.join(stroma_maskfolder_dest, masknames[i]))
    else:
        shutil.move(os.path.join(imfolder, imnames[i]), os.path.join(imfolder_dest, imnames[i]))
        shutil.move(os.path.join(maskfolder, masknames[i]), os.path.join(maskfolder_dest, masknames[i]))