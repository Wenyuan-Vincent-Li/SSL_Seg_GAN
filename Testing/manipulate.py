import torch
from Training import functions

def SinGAN_generate(Gs, reals, masks, NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,num_samples=10):
    if in_s is None:
        in_s = torch.full((num_samples, 3, reals[0][0], reals[0][1]), 0, device=opt.device)
    I_prev = None
    images_gen = []

    for G,real,noise_amp, mask in zip(Gs,opt.reals,NoiseAmp, masks):
        nzx = real[0]*scale_v
        nzy = real[1]*scale_h

        if n == 0:
            z_curr = functions.generate_noise([1, nzx, nzy], num_samples)
            z_curr = z_curr.expand(num_samples, 3, z_curr.shape[2], z_curr.shape[3])
        else:
            z_curr = functions.generate_noise([opt.nc_z, nzx, nzy])

        if I_prev is None:
            I_prev = in_s
        else:
            I_prev = I_prev[:, :, 0:round(scale_v * reals[n][0]), 0:round(scale_h * reals[n][1])]
            I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
            I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])

        z_in = noise_amp * z_curr + I_prev

        I_curr = G(z_in.detach(), I_prev, mask)

        images_gen.append(I_curr)
        I_prev = I_curr
        n += 1
    return images_gen