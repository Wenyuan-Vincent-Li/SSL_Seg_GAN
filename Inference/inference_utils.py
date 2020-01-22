import math
import numpy as np
import scipy.ndimage
import skimage.measure
import PIL

def postprocess(result):
    """Helper function postprocess inference_image result of GlaS challenge"""
    splitted = split_objects(result)
    labeled = skimage.measure.label(np.array(splitted))
    temp = remove_small_object(labeled, threshold=500)
    growed = grow_to_fill_borders(temp, result[1] > 0.5)
    hole_filled = hole_filling_per_object(growed)
    temp = remove_small_object(hole_filled, threshold=500)
    # final = resize_to_size(temp, image)
    final = temp
    return final



def resize_image(image, ratio=(775/512)):
    """Helper function to resize image with specific ration"""
    new_size = (int(round(image.size[0] / ratio)),
                int(round(image.size[1] / ratio)))

    image = image.resize(new_size)
    return image

def pad_image(image, size=(352, 512)):
    """Helper function to pad image to size (height, width)"""
    pad_h = max((size[0] - image.shape[0]) / 2, 0)
    pad_w = max((size[1] - image.shape[1]) / 2, 0)
    pad_h = (math.floor(pad_h), math.ceil(pad_h))
    pad_w = (math.floor(pad_w), math.ceil(pad_w))

    # pad to image size
    padded_image = np.pad(image, ((pad_h[0], pad_h[1]), (pad_w[0], pad_w[1]), (0, 0)), mode='reflect')
    return padded_image

def crop_result(result, image):
    """Helper function to pad image to size (height, width)"""
    pad_h = max((352 - image.shape[0]) / 2, 0)
    pad_w = max((512 - image.shape[1]) / 2, 0)
    pad_h = (math.floor(pad_h), math.ceil(pad_h))
    pad_w = (math.floor(pad_w), math.ceil(pad_w))

    result = result[:,
                    pad_h[0]:result.shape[1] - pad_h[1],
                    pad_w[0]:result.shape[2] - pad_w[1]]
    return result


def split_objects(image):
    """Helper function to threshold image and thereby split close glands"""
    return (image[0] > 0.55)

def remove_small_object(labeled_image, threshold=500):
    """Helper function to remove small objects"""
    regionprops = skimage.measure.regionprops(labeled_image)
    new_results = np.array(labeled_image).copy()
    for prop in regionprops:
        if prop.area < threshold:
            new_results[new_results == prop.label] = 0
    return new_results

def grow_to_fill_borders(eroded_result, full_result):
    """
    Helper function to use a maximum filter and grow all labeled regions
    constraint to the area of the full prediction.
    """
    for i in range(10):
        new_labeled = scipy.ndimage.maximum_filter(eroded_result, 3)
        eroded_result[full_result == 1] = new_labeled[full_result == 1]
    eroded_result[full_result == 0] = 0
    return eroded_result

def hole_filling_per_object(image):
    """Helper function to fill holes inside individual labeled regions"""
    grow_labeled = image
    for i in np.unique(grow_labeled):
        if i == 0: continue
        filled = scipy.ndimage.morphology.binary_fill_holes(grow_labeled == i)
        grow_labeled[grow_labeled == i] = 0
        grow_labeled[filled == 1] = i
    return grow_labeled

def resize_to_size(image, gt):
    """
    Helper function to resize np.array image (uint8) to size of gt
    image: [64,64]
    gt: [3, 64, 64]
    """
    new_results_img = PIL.Image.fromarray(image.squeeze().astype(np.uint8))
    new_results_img = new_results_img.resize(gt.size)
    new_results_img = np.array(new_results_img)
    return new_results_img