import torch
from options.train_options import TrainOptions
from Models.init_models import *

opt = TrainOptions().parse()
netD, netG, netS = init_models(opt)
w = h = 64
x = torch.rand(1, 3, w, h).to(opt.device)
y = torch.rand(1, 3, w, h).to(opt.device)
mask = torch.rand(1, 4, w, h).to(opt.device)
_, label = torch.max(mask, dim = 1, keepdim=False)
criterion = nn.CrossEntropyLoss()

outputG = netG(x, y, mask) ## should be a image of size (N, 3, w, h)
print(outputG.shape)

outputS_logit, outputS_prob, outputS_mask = netS(x, mask) ## should be a segmentation mask of size (N, 4, w, h)
print(outputS_logit.shape, outputS_mask.shape)

loss = criterion(mask, label)
print(loss)
exit()
outputD1 = netD(outputG, mask) ## should be a list of feature map
print(len(outputD1), len(outputD1[0]))

outputD2 = netD(x, outputS_mask)
print(len(outputD2), len(outputD2[0]))
