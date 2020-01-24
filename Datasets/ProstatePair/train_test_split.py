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
image_folder = os.path.join(root_dir, "Datasets/" + name + "/whole_img")
label_folder = os.path.join(root_dir, "Datasets/" + name + "/whole_label")
# train_folder = os.path.join(root_dir, "Datasets/" + name + "/train_img")
# train_label_folder = os.path.join(root_dir, "Datasets/" + name + "/train_label")
# test_folder = os.path.join(root_dir, "Datasets/" + name + "/test_img")
# test_label_folder = os.path.join(root_dir, "Datasets/" + name + "/test_label")
filenames = files_under_folder_with_suffix(image_folder, ".png")

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

prob = 0.2

test_list, train_list = image_partition(prob, len(filenames))
train_test_split = {}

saved_folder = os.path.join(root_dir, "Datasets/" + name + "/train_img")
saved_label_folder = os.path.join(root_dir, "Datasets/" + name + "/train_label")
check_folder_exists(saved_folder)
check_folder_exists(saved_label_folder)
count = 0
for i in train_list:
    copyfile(os.path.join(image_folder, filenames[i]),
             os.path.join(saved_folder, "%s" % str(count).zfill(4) + ".png"))
    copyfile(os.path.join(label_folder, filenames[i].split(".")[0] + "_sementic.png"),
             os.path.join(saved_label_folder, "%s" % str(count).zfill(4) + "_sementic.png"))
    count += 1
train_test_split["train"] = train_list

saved_folder = os.path.join(root_dir, "Datasets/" + name + "/test_img")
saved_label_folder = os.path.join(root_dir, "Datasets/" + name + "/test_label")
check_folder_exists(saved_folder)
check_folder_exists(saved_label_folder)
count = 0
for i in test_list:
    copyfile(os.path.join(image_folder, filenames[i]),
             os.path.join(saved_folder, "%s" % str(count).zfill(4) + ".png"))
    copyfile(os.path.join(label_folder, filenames[i].split(".")[0] + "_sementic.png"),
             os.path.join(saved_label_folder, "%s" % str(count).zfill(4) + "_sementic.png"))
    count += 1
train_test_split["test"] = test_list

np.save(os.path.join(root_dir, "Datasets/" + name, "train_test_split.npy"), train_test_split)