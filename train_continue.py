from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0.1
opt.name = 'prostateHD'

# opt.name = 'colon'
# opt.dataroot = './Datasets/ColonPair_Fine/'
# opt.label_nc = 6


opt.scale_factor = 0.10
opt.niter = 200
opt.noise_amp = 1
Gs, Ss, reals, NoiseAmp, NoiseAmpS = functions.load_trained_pyramid(opt)
opt.reals = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]
reals = opt.reals
opt.stop_scale = len(reals)
train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, reals)