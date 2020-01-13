from options.train_options import TrainOptions
from Training.train import train

opt = TrainOptions().parse()
# opt.name = 'prostateHD'
opt.name = 'colon'
opt.dataroot = './Datasets/ColonPair_Fine/'
opt.label_nc = 6
Gs = []
Ss = []
NoiseAmp = []
NoiseAmpS = []

# opt.reals = [[64, 64], [128, 128], [192, 192], [256, 256], [320, 320], [384, 384], [448, 448], [512, 512]]
opt.reals = [[64,64],[128,128],[256,256],[512,512]]
opt.alpha = 0.1
opt.scale_factor = 0.50
opt.noise_amp = 1
opt.stop_scale = len(opt.reals)

train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, opt.reals)
