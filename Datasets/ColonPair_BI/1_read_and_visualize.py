from PIL import Image
import csv
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils import *
image_folder = "Original"

coarse_label = {" benign": 1, " malignant": 2}
fine_label = {" healthy": 1, " adenomatous": 2, " moderately differentiated": 3, " moderately-to-poorly differentated": 4,
              " poorly differentiated": 5}
train_id = 0
with open(os.path.join(image_folder, "Grade.csv"), newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(filereader):
        if i == 0:
            continue

        if "train" in row[0]:
            im = Image.open(os.path.join(image_folder, row[0] + ".bmp"))
            im = np.array(im)
            label = Image.open(os.path.join(image_folder, row[0] + "_anno.bmp"))
            label = np.array(label)
            (o, p, _) = im.shape
            x_inc = [0, o - 512]
            y_inc = [0, p - 512]

            for x in x_inc:
                for y in y_inc:
                    crop_im = im[x : x + 512, y : y + 512, :].astype(np.uint8)
                    if crop_im.shape != (512, 512, 3):
                        continue
                    crop_label = label[x : x + 512, y : y + 512]
                    crop_im = Image.fromarray(crop_im)
                    name = "%s" % train_id
                    name = name.zfill(4)
                    crop_im.save("train_img/" + name + ".png")
                    ## store binary label:
                    crop_label_bi = np.where(crop_label != 0, coarse_label[row[2]], 0).astype(np.uint8)
                    crop_label_bi = Image.fromarray(crop_label_bi)
                    crop_label_bi.save("train_label/" + name + "_sementic.png")
                    ## store fine label:
                    crop_label_fine = np.where(crop_label != 0, fine_label[row[3]], 0).astype(np.uint8)
                    crop_label_fine = Image.fromarray(crop_label_fine)
                    crop_label_fine.save("train_label_fine/" + name + "_sementic.png")
                    train_id += 1

