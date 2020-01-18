from Models.pix2pixHD2 import *
from Models.DenseNet import *

def init_models(opt):
    norm_layer = get_norm_layer(norm_type=opt.norm)
    #generator initialization:
    netG = GeneratorConcatSkip2CleanAdd(opt.input_nc + opt.label_nc, opt.input_nc, opt).to(opt.device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = WDiscriminator(opt.input_nc + opt.label_nc, opt, norm_layer=norm_layer, getIntermFeat=not opt.no_ganFeat_loss,
                          use_sigmoid=opt.no_lsgan).to(opt.device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    netS = DenseNet(n_classes = opt.label_nc, n_channels_in = opt.input_nc + opt.label_nc, opt=opt).to(opt.device)
    if opt.netS != '':
        netS.load_state_dict(torch.load(opt.netS))
    print(netS)

    return netD, netG, netS