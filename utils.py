import os, sys
root_dir = os.getcwd()
def find_root_dir(filepath, basename):
    if os.path.basename(filepath) == basename:
        return filepath
    else:
        root_dir = find_root_dir(os.path.dirname(filepath), basename)
        return root_dir
root_dir = find_root_dir(root_dir, "SSL_Seg_GAN")
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.misc
import random
import colorsys

import numpy as np

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

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[c, :, :] = np.where(mask == 1,
                                  image[c, :, :] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[c, :, :])
    return image

def display_sementic(image, segmentation_mask, num_classes=int(6),
                     title="", figsize=(16, 16), ax=None):
    label_colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1)]

    ## red, green, blue, yellow, cyan, white

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Show area outside image boundaries.
    height, width = image.shape[1:3]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint32).copy()

    for label in range(num_classes):
        mask = np.zeros_like(segmentation_mask)
        mask[np.where(segmentation_mask == label)] = 1
        masked_image = apply_mask(masked_image, mask, label_colours[label])

    masked_image = np.moveaxis(masked_image, 0, 2)
    ax.imshow(masked_image.astype(np.uint8))
    plt.show()


def image_w_mask_overlay(image, segmentation_mask, num_classes = int(4)):

    label_colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    ## 0/stroma : red; 1/benign: green; 2/low-grade: blue 3/high-grade: yellow

    if image.ndim == 1:
        masked_image = image.astype(np.uint8).copy()
        for label in range(num_classes):
            mask = np.zeros_like(segmentation_mask)
            mask[np.where(segmentation_mask == label)] = 1
            masked_image = apply_mask(masked_image, mask, label_colours[label])
    else:
        n = image.shape[0]
        masked_image = []
        for i in range(n):
            mask_image = image[i, ...].astype(np.uint8).copy()
            for label in range(num_classes):
                mask = np.zeros_like(segmentation_mask[i, ...])
                mask[np.where(segmentation_mask[i, ...] == label)] = 1
                mask_image = apply_mask(mask_image, mask, label_colours[label])
            masked_image.append(np.expand_dims(mask_image, axis = 0))
        masked_image = np.concatenate(masked_image, axis = 0)
    return masked_image


def plot(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(Nc, Nh))
    plt.clf()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(samples.shape[0]):
        sample = samples[i, :]
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            immin=(image[:,:]).min()
            immax=(image[:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image,cmap ='gray')
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
            immin=(image[:,:,:]).min()
            immax=(image[:,:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image)
    plt.savefig("fig.png")
    return fig

def check_folder_exists(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            return False
        except:
            check_folder_exists(os.path.dirname(path))
            os.mkdir(path)
    else:
        return True


def variable_name_string_specified(variables):
    name_string = ''
    for v in variables:
        name_string += v.name + '\n'
    return name_string


def save_dict_as_txt(Dict, dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    for key in Dict.keys():
        np.savetxt(dir_name + key + '.txt', np.asarray(Dict[key]))


def convert_list_2_nparray(varlist):
    var_np = np.empty((0,))
    for i in range(len(varlist)):
        var_np = np.concatenate((var_np, varlist[i]))
    return var_np

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images + 1.)/2.


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def image_w_instance_overlay(images, masks, colors=None):
    """
    masks: [N, 1, height, width]
    colors: (optional) An array or colors to use with each object
    """
    # Number of slides
    N = images.shape[0]
    masked_images = []
    for i in range(N):
        # Number of instances in the image
        n = np.max(masks[i])
        # Generate random colors
        colors = random_colors(n)
        mask_image = images[i].astype(np.uint32).copy()
        for j in range(n):
            color = colors[j]
            # Mask
            mask = np.zeros_like(masks[i, ...])
            mask[np.where(masks[i, ...] == j)] = 1
            mask_image = apply_mask(mask_image, mask, color)

        masked_images.append(np.expand_dims(mask_image, axis=0))
    masked_image = np.concatenate(masked_images, axis=0)
    return masked_image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors



###########################
## Test code
###########################

if __name__ == "__main__":
    exit()