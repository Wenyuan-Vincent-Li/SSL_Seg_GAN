from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0.1
opt.name = 'prostateHD'
# opt.name = 'colon_BI_80'
opt.dataroot = './Datasets/ProstatePair/'
opt.label_nc = 4
opt.contour = False


opt.scale_factor = 1.20
opt.noise_amp = 1
Gs, Ss, reals, NoiseAmp, NoiseAmpS = functions.load_trained_pyramid(opt)
Gs = Gs[:2]
Ss = Ss[:2]
opt.erod = [1, 3, 6, 13]
opt.reals = [[64, 64], [128, 128], [192,192]]
reals = opt.reals
opt.phase = "train"
opt.stop_scale = len(reals)
train(opt, Gs, Ss, NoiseAmp, NoiseAmpS, reals)