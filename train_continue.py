from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0.1
# opt.name = 'prostateHD'

opt.name = 'colon_BI_80'
opt.dataroot = './Datasets/ColonPair_BI/'
opt.label_nc = 2
opt.contour = True


opt.scale_factor = 0.30
opt.noise_amp = 1
Gs, Ss, reals, NoiseAmp, NoiseAmpS = functions.load_trained_pyramid(opt)
Gs = Gs[:]
Ss = Ss[:]
opt.erod = [1, 3, 6, 13]
opt.reals = [[64, 64], [128, 128], [256, 256], [512, 512]]
reals = opt.reals
opt.phase = "train_80"
opt.stop_scale = len(reals)
train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, reals)