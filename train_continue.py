from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0.1
opt.name = 'prostateHD'
opt.scale_factor = 0.10
opt.niter = 2
opt.noise_amp = 1
Gs, Ss, reals, NoiseAmp, NoiseAmpS = functions.load_trained_pyramid(opt)
opt.reals = [[32, 32], [64, 64], [128, 128]]
reals = opt.reals
opt.stop_scale = len(reals)
train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, reals)