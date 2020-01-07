from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # optimization hyper parameters:
        self.parser.add_argument('--niter', type=int, default=200, help='number of epochs to train per scale')
        self.parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
        self.parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
        self.parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
        self.parser.add_argument('--lr_s', type=float, default=0.0005, help='learning rate, default=0.0005')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        # self.parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
        self.parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=0.1)


        # for discriminators
        self.parser.add_argument('--num_D', type=int, default=1, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')

        self.isTrain = True