import argparse
import os
from utils import check_folder_exists
from InputPipeline.image_folder import make_dataset
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='prostateHD',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--cuda', action='store_true', help='enables cuda', default=1)
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')

        # input/output sizes
        # self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument('--randomScale', action='store_false', default=True, help='whether to do random scale before crop')
        self.parser.add_argument('--fineSize', type=int, default=448, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=4, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./Datasets/ProstatePair/')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        # self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
        # self.parser.add_argument('--tf_log', action='store_true',
        #                          help='if specified, use tensorboard logging. Requires tensorflow installed')

        # pyramid parameters:
        self.parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)

        # networks hyper parameters:
        self.parser.add_argument('--nfc', type=int, default=32)

        # for generator
        self.parser.add_argument('--netG', default='', help="path to netG (to continue training)")
        self.parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)

        # for discriminator
        self.parser.add_argument('--netD', default='', help="path to netD (to continue training)")

        # for segementor
        self.parser.add_argument('--netS', default='', help="path to netS (to continue training)")

        #
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        self.opt.device = torch.device("cuda:0" if self.opt.cuda else "cpu")
        self.opt.noise_amp_init = self.opt.noise_amp
        dir_A = '_label'
        self.opt.num_images = len(make_dataset(os.path.join(self.opt.dataroot, self.opt.phase + dir_A)))

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        check_folder_exists(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt