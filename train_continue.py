from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0.1
# opt.name = 'prostateHD'

opt.name = 'colon_BI'
opt.dataroot = './Datasets/ColonPair_BI/'
opt.label_nc = 2


opt.scale_factor = 0.10
opt.niter = 200
opt.noise_amp = 1
Gs, Ss, reals, NoiseAmp, NoiseAmpS = functions.load_trained_pyramid(opt)
Gs = Gs[:1]
Ss = Ss[:1]
opt.reals = [[64, 64], [128, 128], [256, 256], [512, 512]]
reals = opt.reals
opt.stop_scale = len(reals)
train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, reals)