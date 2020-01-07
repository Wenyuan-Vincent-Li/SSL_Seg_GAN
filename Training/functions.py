import torch
import torch.nn as nn
import numpy as np
import math
import os

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    if type == 'uniform+poisson':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise2 = np.random.poisson(10, [num_samp, int(size[0]), int(size[1]), int(size[2])])
        noise2 = torch.from_numpy(noise2).to(device)
        noise2 = noise2.type(torch.cuda.FloatTensor)
        noise = noise1+noise2
    if type == 'poisson':
        noise = np.random.poisson(0.1, [num_samp, int(size[0]), int(size[1]), int(size[2])])
        noise = torch.from_numpy(noise).to(device)
        noise = noise.type(torch.cuda.FloatTensor)
    return noise

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def create_reals_pyramid(real_shape, reals, opt):
    w, h = real_shape
    for i in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = [math.floor(w * scale), math.floor(h * scale)]
        reals.append(curr_real)
    return reals

def calc_gradient_penalty(netD, real_data, fake_data, mask, LAMBDA): # Notice that gradient penalty only works in D's loss function
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, mask)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def calc_gradient_penalty_mask(netD, real_data, fake_data, LAMBDA): # Notice that gradient penalty only works in D's loss function
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_dir2save(opt):
    dir2save = None
    if (opt.isTrain):
        dir2save = 'TrainedModels/%s/scale_factor=%.2f,alpha=%d' % (opt.name, opt.scale_factor,opt.alpha)
    # elif opt.mode == 'random_samples':
    #     dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    # elif opt.mode == 'random_samples_arbitrary_sizes':
    #     dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    return dir2save

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def convert_mask_np(segmentation_mask, num_classes = 4):
    label_colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), [0, 1, 1], [1, 1, 1]]
    def apply_mask(image, mask, color):
        """Apply the given mask to the image.
        """
        mask = mask.transpose((1,2,0))
        for c in range(3):
            image[:, :, c] = np.where(np.squeeze(mask == 1), color[c] * 255, image[:, :, c])
        return image
    segmentation_mask = segmentation_mask.cpu().numpy()
    segmentation_mask = segmentation_mask[-1, :, :, :]
    mask_image = np.zeros((segmentation_mask.shape[1], segmentation_mask.shape[2], 3))
    for label in range(num_classes):
        mask = np.zeros_like(segmentation_mask) ## [1, 1, 162, 162]
        mask[np.where(segmentation_mask == label)] = 1
        mask_image = apply_mask(mask_image, mask, label_colours[label])
    return mask_image.astype(np.uint8)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def save_networks(netG,netD,netS,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(netS.state_dict(), '%s/netS.pth' % (opt.outf))


def load_trained_pyramid(opt):
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Ss = torch.load('%s/Ss.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
        NoiseAmpS = torch.load('%s/NoiseAmpS.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    return Gs,Ss,reals,NoiseAmp,NoiseAmpS

def mask2onehot(label_map, label_nc):
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

if __name__ == "__main__":
    from utils import *
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    real_shape = [1024, 1024]
    reals = []
    reals = create_reals_pyramid(real_shape, reals, opt)
    print(reals)