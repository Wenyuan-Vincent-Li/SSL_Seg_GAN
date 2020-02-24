from options.train_options import TrainOptions
from Training.train import train

opt = TrainOptions().parse()
opt.name = 'prostateHD'  ## change #1
# opt.name = 'colon_Fine_100'
# opt.dataroot = './Datasets/ColonPair_Fine/'
opt.dataroot = './Datasets/ProstatePair/' ## change #2
opt.label_nc = 4 ## change #3
opt.contour = False ## change #4
Gs = []
Ss = []
NoiseAmp = []
NoiseAmpS = []

# opt.reals = [[64, 64], [128, 128], [192, 192], [256, 256], [320, 320], [384, 384], [448, 448], [512, 512]]
opt.reals = [[64,64],[128,128],[256,256]]
opt.erod = [1, 3, 6, 13]
opt.alpha = 0.1
opt.scale_factor = 1.20
opt.noise_amp = 1
opt.stop_scale = len(opt.reals)
opt.phase = "train"

train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, opt.reals)
