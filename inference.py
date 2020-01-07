import matplotlib.pyplot as plt
import numpy as np
import os
from Testing.manipulate import *
from options.train_options import TrainOptions
from Training import functions
from InputPipeline.DataLoader import CreateDataset
from Training.imresize import imresize

opt = TrainOptions().parse()
opt.alpha = 0.1 ## Not to use reconstruction loss
opt.name = "colon_fine"
opt.dataroot = './Datasets/ColonPair_Fine/'
opt.label_nc = 6
opt.scale_factor = 2.00
save_every_scale = True
signature = 100
signature = str(signature).zfill(4)
label_manipulate = False

Gs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
opt.reals = reals
opt.batchSize = 2
opt.num_samples = 4
opt.mode = 'train'
opt.scale_num = len(Gs) - 1

opt.out = functions.generate_dir2save(opt)  # TrainedModels/path_01/scale_factor=0.750000,alpha=10
dir2save = '%s/RandomSamples/%s/gen_start_scale=%d_%s' % (opt.out, opt.name, 0, signature)
try:
    os.makedirs(dir2save)
except OSError:
    pass


random = True
dataset = CreateDataset(opt, fixed=True)
N = len(dataset.A_paths)

if random:
    n = np.random.randint(N)
else:
    n = 8

data = dataset[n]
_, im_x, im_y = data['label'].shape
data['label'] = data['label'].expand(opt.num_samples, 1, im_x, im_y)
data['image'] = data['image'].expand(opt.num_samples, 3, im_x, im_y)
masks = []
for idx, val in enumerate(data['down_scale_label']):
    _, x, y = val.shape
    masks += [val.expand(opt.num_samples, 1, x, y)]
masks += [data['label']]

if label_manipulate:
    def label_man(label, scale_label, source_label, target_label):
        mask = label == source_label
        label[mask] = target_label

        man_scale_label = []
        for cur_label in scale_label:
            mask = cur_label == source_label
            cur_label[mask] = target_label
            man_scale_label.append(cur_label)

        return label, man_scale_label

    data['label'], masks = label_man(data['label'], masks, source_label = 4, target_label = 3)


print(data['path'])

im_gen = SinGAN_generate(Gs, reals, masks, NoiseAmp, opt, num_samples=opt.num_samples)

## Save mask, org image and masks for every scale
if save_every_scale:
    _, _, im_x, im_y = data['image'].shape
    for idx, mask in enumerate(masks):
        _, _, x, y = mask.shape
        img = imresize(data['image'], x/im_x, opt)
        plt.imsave('%s/scale_%d_org_img.png' % (dir2save, idx),
                   functions.convert_image_np(img))
        plt.imsave('%s/scale_%d_mask.png' % (dir2save, idx),
                    functions.convert_mask_np(mask, num_classes=opt.label_nc))
        plt.imsave('%s/scale_%d_gen_img.png' % (dir2save, idx),
                   functions.convert_image_np(im_gen[idx]))
else:
    mask = masks[-1]
    _,_, im_x, im_y = im_gen[-1].shape
    _,_, org_x, org_y = data['image'].shape
    img = imresize(data['image'], im_x / org_x, opt)
    plt.imsave('%s/org_img.png' % (dir2save),
               functions.convert_image_np(data['image']))
    plt.imsave('%s/scale_mask.png' % (dir2save),
               functions.convert_mask_np(masks[-1], num_classes=opt.label_nc))
    gen_img = im_gen[-1]
    N, _, _, _ = gen_img.shape
    for i in range(N):
        plt.imsave('%s/gen_img_%d.png' % (dir2save, i),
                   functions.convert_image_np(gen_img[i:i+1, ...]))