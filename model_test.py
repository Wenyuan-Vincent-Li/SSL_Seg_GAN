import torch
from options.train_options import TrainOptions
from Models.init_models import *

opt = TrainOptions().parse()
netD, netG, netS = init_models(opt)
w = h = 64
x = torch.rand(4, 3, w, h).to(opt.device)
y = torch.rand(4, 3, w, h).to(opt.device)
mask = torch.rand(4, 4, w, h).to(opt.device)

outputG = netG(x, y, mask) ## should be a image of size (N, 3, w, h)
print(outputG.shape)

outputS = netS(x) ## should be a segmentation mask of size (N, 4, w, h)
print(outputS.shape)

outputD1 = netD(outputG, mask) ## should be a list of feature map
print(len(outputD1), len(outputD1[0]))

outputD2 = netD(x, outputS)
print(len(outputD2), len(outputD2[0]))
