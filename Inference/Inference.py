from utils import *
from Training import functions
import torch
import torch.nn as nn
from Training.metrics import RunningConfusionMatrix


def Segmentation(Ss, reals, images, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0):
    """

    :param Ss: list of segmentation network
    :param reals: list of real image shape on each level
    :param images: list of images on each level
    :param NoiseAmp:
    :param opt:
    :param in_s:
    :param scale_v:
    :param scale_h:
    :param n: start segmentation level
    :return:
    """
    if in_s is None:
        in_s = torch.full((images[0].shape[0], opt.label_nc, reals[0][0], reals[0][1]), 0, device=opt.device)
    I_prev = None
    masks = []
    with torch.no_grad():
        for S,real,noise_amp, image in zip(Ss,opt.reals,NoiseAmp, images):
            S.eval()
            nzx = real[0]*scale_v
            nzy = real[1]*scale_h
            image = image.to(opt.device)
            if I_prev is None:
                I_prev = in_s
            else:
                if opt.contour:
                    I_prev = torch.cat((I_prev, 1 - I_prev), 1)
                I_prev = I_prev[:, :, 0:round(scale_v * reals[n][0]), 0:round(scale_h * reals[n][1])]
                I_prev = functions.upsampling(I_prev, nzx, nzy)

            _, I_curr, _ = S(image, I_prev)

            masks.append(I_curr)
            I_prev = I_curr
            n += 1
    return masks, masks[-1]

class Inference(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.data_loader = data_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.val_step = len(self.data_loader)
        if self.opt.label_nc > 2:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def infer(self, Ss, NoiseAmp):
        print("Start Inference!")
        if self.opt.label_nc == 1:
            mIOU = RunningConfusionMatrix([x for x in range(self.opt.label_nc + 1)])
        else:
            mIOU = RunningConfusionMatrix([x for x in range(self.opt.label_nc)])

        ## Validation after every training epoch
        gts = []
        predictions = []
        images = []
        val_loss = 0.0

        for j, val_data in enumerate(self.data_loader):
            if j >= self.val_step:
                break
            val_data["down_scale_image"] += [val_data["image"]]
            labels = val_data["label"][:,0:1,...].to(self.device)
            _, outputs = Segmentation(Ss, self.opt.reals, val_data["down_scale_image"], NoiseAmp, self.opt)
            if self.opt.label_nc > 2:
                labels = torch.squeeze(labels, dim=1).long()
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()

            ## TODO: MIOU evaluation
            if self.opt.label_nc > 2:
                _, preds = outputs.max(1)
            else:
                preds = (outputs[:, 0, :, :] > 0.5).float()

            if self.opt.contour:
                labels_x = labels[:,0, ...]
                mIOU.update_matrix(labels_x.reshape(-1).int().cpu().numpy(), preds.view(-1).cpu().numpy())
            else:
                mIOU.update_matrix(labels.view(-1).int().cpu().numpy(),
                                   preds.view(-1).cpu().numpy())

            predictions.append(outputs.cpu().numpy())
            images.append(val_data["image"].cpu().numpy())
            gts.append(labels.cpu().numpy())

        cur_IOU = mIOU.compute_current_mean_intersection_over_union()
        print('val_loss: %.3f, mIOU: %.3f' %
              (val_loss / (1e-12 + j + 1), cur_IOU))
        images = np.concatenate(images, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        gts = np.concatenate(gts, axis=0)
        return images, gts, predictions