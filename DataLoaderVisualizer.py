if __name__ == "__main__":
    from utils import *
    from options.train_options import TrainOptions
    from InputPipeline.DataLoader import CreateDataLoader
    from Training import functions
    from matplotlib import pyplot as plt

    opt = TrainOptions().parse()
    reals = []
    opt.reals = functions.create_reals_pyramid([opt.fineSize, opt.fineSize], reals, opt)
    opt.scale_num = 5
    opt.dataroot = './Datasets/ColonPair_Fine/'
    print(opt.reals)
    opt.num_images = 5
    fixed_data_loader = CreateDataLoader(opt, batchSize=opt.num_images, shuffle=True, fixed=True)
    dataset = fixed_data_loader.load_data()

    data = next(iter(dataset))
    print(data['label'].shape, data['image'].shape)
    print(len(data['down_scale_label']), data['down_scale_label'][0].shape, data['down_scale_label'][1].shape,
          data['down_scale_label'][2].shape)
    opt.num_images = data['image'].shape[0]
    print(opt.num_images)
    print(data['path'])
    print(np.unique(data['label']))
    display_sementic((data['image'][0, ...].numpy() + 1) / 2 * 255, data['label'][0, 0, ...].numpy())
