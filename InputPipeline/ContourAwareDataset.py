import os.path
from InputPipeline.base_dataset import BaseDataset, get_params, get_transform, get_downscale_transform
from InputPipeline.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import scipy.ndimage
from Training.imresize import imresize_single

class ContourAwareDataset(BaseDataset):
    def initialize(self, opt, fixed=False):
        self.fixed = fixed
        self.opt = opt
        self.root = opt.dataroot

        ### input labels (label maps)
        dir_A = '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input image (real images)
        if opt.isTrain:
            dir_B = '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)

        # ## Do data augmentation here
        transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, fixed=self.fixed)
        transform_A_toTensor = transforms.ToTensor()
        A_temp = transform_A(A)
        A_tensor = transform_A_toTensor(A_temp) * 255.0
        A_tensor = self.create_eroded_mask(torch.squeeze(A_tensor))

        down_scale_label = []
        for i in range(self.opt.scale_num):
            down_scale_transform_A = get_downscale_transform(self.opt.reals[i][0], method=Image.NEAREST)
            A_curr = down_scale_transform_A(A_temp)
            if self.opt.contour:
                down_scale_label.append(((transform_A_toTensor(A_curr) * 255.0) > 0).float())
            else:
                down_scale_label.append((transform_A_toTensor(A_curr) * 255.0))


        B_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params, fixed=self.fixed)
            B_tensor = transform_B(B)
            B_temp = B_tensor

            down_scale_image = []
            for i in range(self.opt.scale_num):
                B_curr = imresize_single(B_temp, self.opt.reals[i][0] / self.opt.reals[self.opt.scale_num][0], self.opt)
                B_curr = B_curr[:, 0: self.opt.reals[i][0], 0: self.opt.reals[i][1]]
                down_scale_image.append(B_curr)

        input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path, 'down_scale_label': down_scale_label,
                      'down_scale_image': down_scale_image}
        return input_dict

    def create_eroded_mask(self, label):
        """Helper function to create a mask where every gland is eroded"""
        boundaries = torch.zeros(label.shape)
        for i in torch.unique(label):
            if i == 0: continue  # the first label is background
            gland_mask = (label == i).float()
            binarized_mask_border = scipy.ndimage.morphology.binary_erosion(gland_mask,
                                                                            structure=np.ones((4, 4)),
                                                                            border_value=1)

            binarized_mask_border = torch.from_numpy(binarized_mask_border.astype(np.float32))
            boundaries[label == i] = binarized_mask_border[label == i]

        label = (label > 0).float()
        label = torch.stack((boundaries, label))
        return label

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'ContoureAwareDataset'
