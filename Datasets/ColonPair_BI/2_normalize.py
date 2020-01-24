from __future__ import division
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2 as cv
import utils
from PIL import Image
import numpy as np


class ReinhardNormalizer:
    """
    Normalize a patch stain to the target image using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
    """

    def __init__(self):
        self.target_concentrations = np.array([[148.60, 41.56], [169.30, 9.01], [105.97, 6.67]])

    def fit(self, target=None, mask=None):
        """
        Fit to a target image
        :param target:
        :param mask: white background mask
        :return:
        """
        if target is not None:
            target = utils.standardize_brightness(target)
            means, stds = self.get_mean_std(target, mask)
            self.target_concentrations = np.array([[means[0], stds[0]], [means[1], stds[1]], [means[2], stds[2]]])

    def transform(self, I, mask=None):
        """
        Transform an image
        :param I:
        :param mask
        :return:
        """
        if mask is None:
            whitemask = ~utils.notwhite_mask(I)
        else:
            whitemask = ~mask
        imagelab = cv.cvtColor(I, cv.COLOR_RGB2LAB)

        imageL, imageA, imageB = cv.split(imagelab)
        # mask is valid when true
        imageLM = np.ma.MaskedArray(imageL, whitemask)
        imageAM = np.ma.MaskedArray(imageA, whitemask)
        imageBM = np.ma.MaskedArray(imageB, whitemask)
        # Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI

        epsilon = 1e-11
        imageLMean = imageLM.mean()
        imageLSTD = imageLM.std() + epsilon
        imageAMean = imageAM.mean()
        imageASTD = imageAM.std() + epsilon

        imageBMean = imageBM.mean()
        imageBSTD = imageBM.std() + epsilon

        # normalization in lab
        imageL = (imageL - imageLMean) / imageLSTD * self.target_concentrations[0][1] + self.target_concentrations[0][0]
        imageA = (imageA - imageAMean) / imageASTD * self.target_concentrations[1][1] + self.target_concentrations[1][0]
        imageB = (imageB - imageBMean) / imageBSTD * self.target_concentrations[2][1] + self.target_concentrations[2][0]

        imagelab = cv.merge((imageL, imageA, imageB))
        imagelab = np.clip(imagelab, 0, 255)
        imagelab = imagelab.astype(np.uint8)

        # Back to RGB space
        returnimage = cv.cvtColor(imagelab, cv.COLOR_LAB2RGB)
        # Replace white pixels
        returnimage[whitemask] = I[whitemask]

        return returnimage

    def get_mean_std(self, I, mask):
        """
        Get mean and standard deviation of each channel
        :param I: uint8
        :param mask: mask
        :return:
        """
        if mask is None:
            whitemask = ~utils.notwhite_mask(I)
        else:
            whitemask = ~mask
        imagelab = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        imageL, imageA, imageB = cv.split(imagelab)
        # mask is valid when true
        imageLM = np.ma.MaskedArray(imageL, whitemask)
        imageAM = np.ma.MaskedArray(imageA, whitemask)
        imageBM = np.ma.MaskedArray(imageB, whitemask)

        epsilon = 1e-11
        imageLMean = imageLM.mean()
        imageLSTD = imageLM.std() + epsilon
        imageAMean = imageAM.mean()
        imageASTD = imageAM.std() + epsilon

        imageBMean = imageBM.mean()
        imageBSTD = imageBM.std() + epsilon

        return [imageLMean, imageAMean, imageBMean], [imageLSTD, imageASTD, imageBSTD]

    def get_norm_method(self):
        return "reinhard"

if __name__ == "__main__":
    target_im = Image.open("train_img/0111.png")
    target_im = np.array(target_im)
    normalizer = ReinhardNormalizer()
    normalizer.fit(target_im)

    filenames = utils.files_under_folder_with_suffix("train_img", ".png")

    for i in filenames:
        im = Image.open("train_img/" + i)
        im = np.array(im)
        im = normalizer.transform(im)
        im = Image.fromarray(im)
        im.save("train_img/" + i)