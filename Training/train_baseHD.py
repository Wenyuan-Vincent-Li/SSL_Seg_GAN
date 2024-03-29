import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from Training import functions
from Training.imresize import imresize
import matplotlib.pyplot as plt
from Models.pix2pixHD_base import GANLoss, VGGLoss
from Models.pix2pixHD2 import mask2onehot

class Losses():
    def __init__(self, opt):
        self.criterionGAN = GANLoss(not opt.no_lsgan)
        self.criterionFeat = nn.L1Loss()
        if opt.contour:
            self.crossEntropy = nn.BCEWithLogitsLoss()
        else:
            self.crossEntropy = nn.CrossEntropyLoss()
        if not opt.no_vgg_loss:
            self.criterionVGG = VGGLoss()



def train_single_scale(dataloader, netD, netG, netS, reals, Gs, Ss, in_s, in_s_S, NoiseAmp, NoiseAmpS, opt):
    '''
    :param netD: currD
    :param netG: currG
    :param netS: currS
    :param reals: a list of image pyramid ## TODO: you can just pass image shape here
    :param Gs: list of prev netG
    :param Ss: list of prev netS
    :param in_s: 0-> all zero [1, 3, 26, 26]
    :param NoiseAmp: [] -> [1]
    :param opt: config
    :return:
    '''
    loss = Losses(opt)
    real = reals[opt.scale_num]  # find the current level image xn
    opt.nzx = real[0]
    opt.nzy = real[1]
    # z_opt = 0 ## dummy z_opt

    alpha = opt.alpha

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    optimizerS = optim.Adam(netS.parameters(), lr=opt.lr_s, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[opt.niter * 0.8], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[opt.niter * 0.8], gamma=opt.gamma)
    schedulerS = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerS, milestones=[opt.niter * 0.8],
                                                      gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []

    for epoch in range(opt.niter):  # niter = 2000
        if Gs == [] and Ss == []:
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], opt.batchSize)  # [None, 1, 32, 32]
            noise_ = noise_.expand(opt.batchSize, 3, opt.nzx, opt.nzy)
            ## Noise_: for generated false samples through generator
        else:
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], opt.batchSize)

        for j, data in enumerate(dataloader):
            data['image'] = data['image'].to(opt.device)
            data['label'] = data['label'].long().to(opt.device)
            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################

            # train with real
            netD.zero_grad()
            pred_real = netD(data['image'], data['label'][:,0:1,...])
            loss_D_real = loss.criterionGAN(pred_real, True)

            D_x = loss_D_real.item()

            # train with fake
            if (j == 0) & (epoch == 0):  # first iteration training in this level
                if Gs == [] and Ss == []:
                    prev = torch.full([opt.batchSize, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = prev  # full of 0 [None, 3, 32, 32]
                    prev_S = torch.full([opt.batchSize, opt.label_nc, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s_S = prev_S # full of 0 [None, 4, 32, 32]
                    mask = data['label'][:,0:1,...]
                    opt.noise_amp = opt.noise_amp_init
                    opt.noise_amp_S = opt.noise_amp_init
                else:
                    prev = draw_concat(Gs, data['down_scale_label'], reals, NoiseAmp, in_s, 'generator', opt)
                    ## given a new noise, prev is a image generated by previous Generator with bilinear upsampling [1, 3, 33, 33]
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(data['image'], prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    prev_S = draw_concat(Ss, data['down_scale_image'], reals, NoiseAmpS, in_s_S, 'segment', opt) ## prob with [None, 4, 32, 32]
                    onehot_label = mask2onehot(data['label'][:,0:1,...], opt.label_nc)
                    RMSE_S = torch.sqrt(criterion(onehot_label, prev_S))
                    # RMSE_S = 0
                    opt.noise_amp_S = opt.noise_amp_init * RMSE_S
                    mask = data['label'][:,0:1,...]
            else:
                prev = draw_concat(Gs, data['down_scale_label'], reals, NoiseAmp, in_s, 'generator', opt)
                prev_S = draw_concat(Ss, data['down_scale_image'], reals, NoiseAmpS, in_s_S, 'segment', opt)
                mask = data['label'][:,0:1,...]

            if Gs == []:
                noise = noise_  ## Gausiaan noise for generating image [None, 3, 42, 42]
            else:
                noise = opt.noise_amp * noise_ + prev  ## [None, 3, 43, 43] new noise is equal to the prev generated image plus the gaussian noise.

            fake = netG(noise.detach(), prev, mask)  # [None, 3, 32, 32] the same size with the input image
            # detach() make sure that the gradients don't go to the noise.
            # prev:[None, 3, 42, 42] -> [None, 3, 43, 43] first step prev = 0, second step prev = a image generated by previous Generator with bilinaer upsampling
            pred_fake = netD(fake.detach(), data['label'][:,0:1,...])  # output shape [1, 1, 16, 16] -> [1, 1, 23, 23]
            # print(len(pred_fake), len(pred_fake[0]))
            loss_D_fake = loss.criterionGAN(pred_fake, False)
            D_G_z = loss_D_fake.item()
            # segment_logit, segment_mask = netS(data['image'], mask2onehot(prev_S, opt.label_nc))
            # print(data['image'].shape, onehot.shape)
            # print(epoch, j)
            segment_logit, segment_prob, segment_mask = netS(data['image'], prev_S.detach())
            pred_fake_S = netD(data['image'], segment_prob.detach())

            loss_D_fake_S = loss.criterionGAN(pred_fake_S, False)
            D_S_z = loss_D_fake_S.item()

            errD = (loss_D_real + 0.5 * loss_D_fake + 0.5 * loss_D_fake_S)  ## Todo: figure out a proper coefficient
            errD.backward()
            optimizerD.step()

            errD2plot.append(errD.detach())  ## errD for each iteration

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            netG.zero_grad()
            pred_fake = netD(fake, data['label'][:,0:1,...])
            loss_G_GAN = 0.5 * loss.criterionGAN(pred_fake, True)

            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            if not opt.no_ganFeat_loss:
                feat_weights = 4.0 / (opt.n_layers_D + 1)
                D_weights = 1.0 / opt.num_D
                for i in range(opt.num_D):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_GAN_Feat += D_weights * feat_weights * \
                                           loss.criterionFeat(pred_fake[i][j],
                                                              pred_real[i][j].detach()) * opt.lambda_feat

            # VGG feature matching loss
            loss_G_VGG = 0
            if not opt.no_vgg_loss:
                loss_G_VGG = loss.criterionVGG(fake, data['image']) * opt.lambda_feat

            ## reconstruction loss
            if alpha != 0:  ## alpha = 10 calculate the reconstruction loss
                Recloss = nn.MSELoss()
                rec_loss = alpha * Recloss(fake, data['image'])
            else:
                rec_loss = 0

            errG = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG + rec_loss
            errG.backward()
            optimizerG.step()

            ############################
            # (3) Update S network: maximize D(S(z))
            ###########################
            netS.zero_grad()
            pred_fake_S = netD(data['image'], segment_prob)
            loss_G_GAN_S = 0.03 * loss.criterionGAN(pred_fake_S, True)

            # Segmentation loss
            if opt.contour:
                loss_G_Seg = loss.crossEntropy(segment_logit, data['label'].float())
            else:
                loss_G_Seg = loss.crossEntropy(segment_prob, torch.squeeze(data['label'][:,0:1,...], dim =1))

            # GAN feature matching loss
            loss_G_GAN_Feat_S = 0
            if not opt.no_ganFeat_loss:
                feat_weights = 4.0 / (opt.n_layers_D + 1)
                D_weights = 1.0 / opt.num_D
                for i in range(opt.num_D):
                    for j in range(len(pred_fake_S[i]) - 1):
                        loss_G_GAN_Feat_S += D_weights * feat_weights * \
                                           loss.criterionFeat(pred_fake_S[i][j],
                                                              pred_real[i][j].detach()) * opt.lambda_feat

            errS = loss_G_GAN_S + loss_G_GAN_Feat_S + loss_G_Seg
            errS.backward()
            optimizerS.step()


        ## for every epoch, do the following:
        errG2plot.append(errG.detach())  ## ErrG for each iteration
        D_real2plot.append(D_x)  ##  discriminator loss on real
        D_fake2plot.append(D_G_z + D_S_z)  ## discriminator loss on fake

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (opt.scale_num, epoch, opt.niter))

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            plt.imsave('%s/fake_sample_%d.png' % (opt.outf, epoch),
                       functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample_real_%d.png' % (opt.outf, epoch),
                       functions.convert_image_np(data['image']), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample_mask_%d.png' % (opt.outf, epoch),
                       functions.convert_mask_np(data['label'][:,0:1,...], num_classes= opt.label_nc))
            plt.imsave('%s/segmentation_mask_%d.png' % (opt.outf, epoch),
                       functions.convert_mask_np(segment_mask.detach(), num_classes=opt.label_nc))

        schedulerD.step()
        schedulerG.step()
        schedulerS.step()

    functions.save_networks(netG, netD, netS, opt)  ## save netG, netD, z_opt, opt is used to parser output path
    return in_s, in_s_S, netG, netS


def draw_concat(Gs, masks, reals, NoiseAmp, in_s, mode, opt):
    '''
    :param Gs: [G0]
    :param mask: [down scaled _mask]
    :param reals: [image pyramid] only used to represent the image shape
    :param NoiseAmp: [1]
    :param in_s: all zeros [1, 3, 26, 26]
    :param mode: 'rand'
    :param opt:
    :return:
    '''
    G_z = in_s[:opt.batchSize, :, :, :]  # [None, 3, 26, 26] all zeros, image input for the corest level
    if len(Gs) > 0:
        if mode == 'generator':
            count = 0
            for G, mask, real_curr, real_next, noise_amp in zip(Gs, masks, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, real_curr[0], real_curr[1]],
                                                 opt.batchSize)
                    z = z.expand(opt.batchSize, G_z.shape[1], z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise(
                        [opt.nc_z, real_curr[0], real_curr[1]], opt.batchSize)
                G_z = G_z[:, :, 0:real_curr[0], 0:real_curr[1]]  ## G_z [None, 3, 32, 32]
                z_in = noise_amp * z + G_z
                G_z = G(z_in.detach(), G_z, mask)  ## [1, 3, 26, 26] output of previous generator
                G_z = imresize(G_z, real_next[1] / real_curr[1], opt)
                G_z = G_z[:, :, 0:real_next[0],
                      0:real_next[1]]  ## resize the image to be compatible with current G [1, 3, 33, 33]
                count += 1
        elif mode == 'segment':
            count = 0
            for G, mask, real_curr, real_next, noise_amp in zip(Gs, masks, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr[0], 0:real_curr[1]]  ## G_z [None, 3, 32, 32]
                _, G_z, _ = G(mask, G_z)  ## [1, 3, 26, 26] output of previous generator
                if opt.contour:
                    G_z = torch.cat((G_z, 1-G_z), 1)
                G_z = imresize(G_z, real_next[1] / real_curr[1], opt)
                G_z = G_z[:, :, 0:real_next[0],
                      0:real_next[1]]  ## resize the image to be compatible with current G [1, 3, 33, 33]
                count += 1
    return G_z
