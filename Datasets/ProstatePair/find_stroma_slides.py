import os, sys
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
maskfolder = "/data/wenyuan/Path_R_CNN/Data_Pre_Processing/masks_sementic_mod-1200by1200"
filenames = files_under_folder_with_suffix(maskfolder, "mat")

stroma_slides = []

for i in range(len(filenames)):
    mask = scipy.io.loadmat(os.path.join(maskfolder, filenames[i]))['ATmask']
    if np.all(mask == 0):
        stroma_slides.append(filenames[i])

print(stroma_slides, len(stroma_slides))