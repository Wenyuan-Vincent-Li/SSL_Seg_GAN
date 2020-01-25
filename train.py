from options.train_options import TrainOptions
from Training.train import train

opt = TrainOptions().parse()
# opt.name = 'prostateHD'
opt.name = 'colon_BI_80'
opt.dataroot = './Datasets/ColonPair_BI/'
opt.label_nc = 2
opt.contour = True
Gs = []
Ss = []
NoiseAmp = []
NoiseAmpS = []

# opt.reals = [[64, 64], [128, 128], [192, 192], [256, 256], [320, 320], [384, 384], [448, 448], [512, 512]]
opt.reals = [[64,64],[128,128],[256,256],[512,512]]
opt.erod = [1, 3, 6, 13]
opt.alpha = 0.1
opt.scale_factor = 0.20
opt.noise_amp = 1
opt.stop_scale = len(opt.reals)
opt.phase = "train_80"

train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, opt.reals)
