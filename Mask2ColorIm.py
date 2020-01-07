from Testing.manipulate import *
from options.train_options import TrainOptions
from Training import functions
from InputPipeline.DataLoader import CreateDataLoader, CreateDataset

import matplotlib.pyplot as plt
from Training.imresize import imresize

opt = TrainOptions().parse()
opt.alpha = 0.1 ## Not to use reconstruction loss
opt.name = "prostateHD"
opt.scale_factor = 0.87
save_every_scale = False
signature = 20
signature = str(signature).zfill(4)

Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
Gs = Gs[:-1]
opt.reals = reals
opt.batchSize = 1
opt.num_samples = 4
opt.mode = 'train'
opt.scale_num = len(Gs) - 1

dir2save = 'Datasets/ProstatePair/masks'
try:
    os.makedirs(dir2save)
except OSError:
    pass

data_loader = CreateDataLoader(opt, fixed=True)
dataset = data_loader.load_data()

for idx, data in enumerate(dataset):
    filename = data['path'][0].split('/')[-1]
    plt.imsave('%s/%s' % (dir2save, filename),
               functions.convert_mask_np(data['label'], num_classes=4))