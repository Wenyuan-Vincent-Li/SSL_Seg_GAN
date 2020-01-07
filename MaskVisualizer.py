if __name__ == "__main__":
    from utils import *
    from options.train_options import TrainOptions
    from InputPipeline.DataLoader import CreateDataLoader
    from Training import functions
    from matplotlib import pyplot as plt

    opt = TrainOptions().parse()
    reals = []
    opt.reals = functions.create_reals_pyramid([opt.fineSize, opt.fineSize], reals, opt)
    opt.scale_num = 3
    opt.name = "mask"
    print(opt.reals)
    opt.num_images = 5
    fixed_data_loader = CreateDataLoader(opt, batchSize=opt.num_images, shuffle=False, fixed=False)
    dataset = fixed_data_loader.load_data()

    data = next(iter(dataset))
    print(data['label'].shape)
    print(len(data['down_scale_label']), data['down_scale_label'][0].shape, data['down_scale_label'][1].shape,
          data['down_scale_label'][2].shape)
    print(np.unique(data['label']))
    plt.imsave("test1.png", functions.convert_mask_np(data['label']))