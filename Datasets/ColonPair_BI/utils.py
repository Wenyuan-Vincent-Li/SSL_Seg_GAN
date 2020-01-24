"""
Other utilities
"""

from __future__ import division

from skimage import morphology as skmp
from skimage import color
import numpy as np
import os

def files_under_folder_with_suffix(dir_name, suffix = ''):
    """
    Return a filename list that under certain folder with suffix
    :param dir_name: folder specified
    :param suffix: suffix of the file name, eg '.jpg'
    :return: List of filenames in order
    """
    files = [f for f in os.listdir(dir_name) if (os.path.isfile(os.path.join(dir_name, f)) and f.endswith(suffix))]
    files.sort()
    return files

def standardize_brightness(I):
    """
    An image is a numpy array of size HxWx3 (RGB) with integer values in the range 0-255 (uint8).
    We standardize so 10% of the elements of this array take the value 255.
    For already bright images this makes little change. For dark images this makes a significant difference,
    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """
    Remove zeros in an image, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB)
    :param I:
    :return:
    """
    I = remove_zeros(I)  # we don't want to take the log of zero..
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB
    RGB = 255 * exp(-1*OD_RGB)
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize the rows of an array.
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def notwhite_mask(tile, thresh=None):
    tile_hsv = color.rgb2hsv(np.asarray(tile))
    roi1 = (tile_hsv[:, :, 0] >= 0.33) & (tile_hsv[:, :, 0] <= 0.67)
    roi1 = ~roi1

    skmp.remove_small_holes(roi1, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi1, min_size=500, connectivity=20, in_place=True)

    tile_gray = color.rgb2gray(np.asarray(tile))
    masked_sample = np.multiply(tile_gray, roi1)
    roi2 = (masked_sample <= 0.8) & (masked_sample >= 0.2)

    skmp.remove_small_holes(roi2, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi2, min_size=500, connectivity=20, in_place=True)

    return roi2


def sign(x):
    """
    Returns the sign of x.
    :param x:
    :return:
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def check_ihc_slide(slide):
    """
    check whether the current slide is IHC stained
    :param slide:
    :return:
    """
    sample = slide.read_region((0, 0), slide.level_count - 1,
                              (slide.level_dimensions[slide.level_count - 1][0],
                               slide.level_dimensions[slide.level_count - 1][1]))
    sample = sample.convert('RGB')
    sample_hsv = color.rgb2hsv(np.asarray(sample))
    # brownish stain
    roi_ihc = (sample_hsv[:, :, 0] >= 0.056) & (sample_hsv[:, :, 0] <= 0.34) & (sample_hsv[:, :, 2] > 0.2) & (
                sample_hsv[:, :, 1] > 0.04)
    skmp.remove_small_holes(roi_ihc, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi_ihc, min_size=500, connectivity=20, in_place=True)

    is_ihc = float(np.sum(roi_ihc)) / float((roi_ihc.shape[0] * roi_ihc.shape[1])) > 0.01

    return is_ihc


def generate_binary_mask(tile):
    """
    generate binary mask for a given tile
    :param tile:
    :return:
    """
    tile_hsv = color.rgb2hsv(np.asarray(tile))
    roi1 = (tile_hsv[:, :, 0] >= 0.33) & (tile_hsv[:, :, 0] <= 0.67)
    roi1 = ~roi1

    skmp.remove_small_holes(roi1, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi1, min_size=500, connectivity=20, in_place=True)

    tile_gray = color.rgb2gray(np.asarray(tile))
    masked_sample = np.multiply(tile_gray, roi1)
    roi2 = (masked_sample <= 0.8) & (masked_sample >= 0.2)

    skmp.remove_small_holes(roi2, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi2, min_size=500, connectivity=20, in_place=True)

    return tile_hsv, roi2