import numpy as np
import random
from shutil import copyfile
from distutils.dir_util import copy_tree
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_dir)
from utils import *

def image_partition(prob, total_number):
    image_list = [i for i in range(total_number)]
    image_numberA = int(total_number * prob)
    image_listA = []
    while len(image_listA) < image_numberA:
        r = random.randint(0, total_number - 1)
        if r not in image_listA: image_listA.append(r)
    image_listB = list(set(image_list) - set(image_listA))
    image_listA = sorted(image_listA)
    return image_listA, image_listB

name = "ProstatePair"
image_folder = os.path.join(root_dir, "Datasets/" + name + "/train_img")
label_folder = os.path.join(root_dir, "Datasets/" + name + "/train_label")
test_folder = os.path.join(root_dir, "Datasets/" + name + "/test_img")
test_label_folder = os.path.join(root_dir, "Datasets/" + name + "/test_label")
filenames = files_under_folder_with_suffix(image_folder, ".png")
testnames = len(files_under_folder_with_suffix(test_folder, ".png"))
probs = [0.2, 0.4]
partition_dict = {}

for prob in probs:
    listA, listB = image_partition(prob, len(filenames))
    saved_folder = os.path.join(root_dir, "Datasets/" + name + "/train_img_%s"%str(int(prob*100)).zfill(2))
    saved_label_folder = os.path.join(root_dir, "Datasets/" + name + "/train_label_%s"%str(int(prob*100)).zfill(2))
    check_folder_exists(saved_folder)
    check_folder_exists(saved_label_folder)
    count = 0
    for i in listA:
        copyfile(os.path.join(image_folder, "%s"%str(i).zfill(4) + ".png"),
                 os.path.join(saved_folder, "%s"%str(count).zfill(4) + ".png"))
        copyfile(os.path.join(label_folder, "%s" % str(i).zfill(4) + "_sementic.png"),
                 os.path.join(saved_label_folder, "%s" % str(count).zfill(4) + "_sementic.png"))
        count += 1
    partition_dict[prob]=listA


    ## make transduction folder
    saved_folder_trans = os.path.join(root_dir, "Datasets/" + name + "/train_img_%s_trans" % str(int(prob * 100)).zfill(2))
    saved_label_folder_trans = os.path.join(root_dir, "Datasets/" + name + "/train_label_%s_trans" % str(int(prob * 100)).zfill(2))
    copy_tree(saved_folder, saved_folder_trans)
    copy_tree(saved_label_folder, saved_label_folder_trans)

    for i in range(testnames):
        copyfile(os.path.join(test_folder, "%s" % str(i).zfill(4) + ".png"),
                 os.path.join(saved_folder_trans, "%s" % str(count).zfill(4) + ".png"))
        copyfile(os.path.join(test_label_folder, "%s" % str(i).zfill(4) + "_sementic.png"),
                 os.path.join(saved_label_folder_trans, "%s" % str(count).zfill(4) + "_sementic.png"))
        count += 1


    saved_folder = os.path.join(root_dir, "Datasets/" + name + "/train_img_%s"%str(int((1- prob)*100)).zfill(2))
    saved_label_folder = os.path.join(root_dir, "Datasets/" + name + "/train_label_%s"%str(int((1-prob)*100)).zfill(2))
    check_folder_exists(saved_folder)
    check_folder_exists(saved_label_folder)
    count = 0
    for i in listB:
        copyfile(os.path.join(image_folder, "%s" % str(i).zfill(4) + ".png"),
                 os.path.join(saved_folder, "%s" % str(count).zfill(4) + ".png"))
        copyfile(os.path.join(label_folder, "%s" % str(i).zfill(4) + "_sementic.png"),
                 os.path.join(saved_label_folder, "%s" % str(count).zfill(4) + "_sementic.png"))
        count += 1
    partition_dict[1-prob]=listB

    ## make transduction folder
    saved_folder_trans = os.path.join(root_dir,
                                      "Datasets/" + name + "/train_img_%s_trans" % str(int((1-prob) * 100)).zfill(2))
    saved_label_folder_trans = os.path.join(root_dir,
                                            "Datasets/" + name + "/train_label_%s_trans" % str(int((1-prob) * 100)).zfill(
                                                2))
    copy_tree(saved_folder, saved_folder_trans)
    copy_tree(saved_label_folder, saved_label_folder_trans)

    for i in range(testnames):
        copyfile(os.path.join(test_folder, "%s" % str(i).zfill(4) + ".png"),
                 os.path.join(saved_folder_trans, "%s" % str(count).zfill(4) + ".png"))
        copyfile(os.path.join(test_label_folder, "%s" % str(i).zfill(4) + "_sementic.png"),
                 os.path.join(saved_label_folder_trans, "%s" % str(count).zfill(4) + "_sementic.png"))
        count += 1


np.save(os.path.join(root_dir, "Datasets/" + name, name + ".npy"), partition_dict)