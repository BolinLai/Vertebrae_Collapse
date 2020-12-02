# coding:utf-8
import os
import warnings


class Config(object):
    env = 'Vertebrae_Collapse'

    root = '/DB/rhome/bllai/PyTorchProjects/Vertebrae_Collapse'

    # train_paths = [os.path.join(root, 'dataset/train_VBOri.csv'),
    #                os.path.join(root, 'dataset/val_VBOri.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_VBOri.csv')]

    train_paths = [os.path.join(root, 'dataset/train_VB.csv'),
                   os.path.join(root, 'dataset/val_VB.csv')]
    test_paths = [os.path.join(root, 'dataset/test_VB.csv')]

    save_model_dir = None
    save_model_name = None
    load_model_path = None
    result_file = None

    data_balance = 'upsample'
    padding = True
    useRGB = True
    usetrans = True
    num_classes = 3

    batch_size = 32
    num_workers = 8
    print_freq = 100
    max_epoch = 100
    max_iter = 10000
    lr = 0.0001
    lr_decay = 0.95
    weight_decay = 1e-5

    use_gpu = True
    parallel = False
    num_of_gpu = 2

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f'Warning: config has no attribute {k}')
            setattr(self, k, v)

        print('Use config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


config = Config()
