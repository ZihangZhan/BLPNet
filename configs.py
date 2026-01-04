import os
import torch
import ml_collections
from Network.model import *
import datetime

# 获取当前时间
current_time = datetime.datetime.now()
time = current_time.strftime('%m%d-%H-%M-%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs = ml_collections.ConfigDict()

data_class = 'OCTA_6mm'
if data_class == 'OCTA_6mm':
    configs.data_dir = '/root/autodl-tmp/my_oct/date/OCTA_6mm'
    configs.imgsize = 400
elif data_class == 'OCTA_3mm':
    configs.data_dir = '/root/autodl-tmp/my_oct/date/OCTA_3mm'
    configs.imgsize = 304
elif data_class == 'rose':
    configs.data_dir = '/root/autodl-tmp/my_oct/date/rose'
    configs.imgsize = 304
else:
    print("Error/....")
    exit(0)

configs.DataParallel = False
configs.ContinueTraining = False
configs.mode = 'TRAIN'   # TRAIN or TEST
# ---------------------Hyper Parameters---------------------#
configs.dataset_name = configs.data_dir.split('/')[-1]
configs.eval_kernel = (1, 1)
configs.kernel_size = 9
configs.extend_scope = 1.0
configs.channel = 3
configs.n_basic_layer = 16
configs.dim = 1
configs.n_classes = 1
configs.batch_size = 4
configs.block_size = [160,100,100]
configs.estop = 500
configs.if_offset = True
configs.if_aug = True
configs.patch_size = None
configs.rotate = None
configs.resize = None
configs.centercrop = None
configs.init_lr = 1e-3
configs.weight_decay = 1e-3
configs.power = 0.9
configs.epochs = 1
configs.epoch_min = 1
configs.epoch_max = configs.epochs
configs.alpha = 2
configs.threshold = 0.5
configs.optimizer = 'Adam'
configs.loss = 'bce_dice'         # select from dice / bce / mse / bce_dice

#----------------------Select Network-----------------------#
configs.model = BLPNet().to(device)


configs.models_save_dir = './{}/{}/{}/models'.format(
    configs.dataset_name, configs.model.__class__.__name__,time
)
if not os.path.exists(configs.models_save_dir):
    os.makedirs(configs.models_save_dir)

configs.results_save_dir = './{}/{}/{}/results'.format(
    configs.dataset_name, configs.model.__class__.__name__,time
)
if not os.path.exists(configs.results_save_dir):
    os.makedirs(configs.results_save_dir)

configs.log_file_path = './{}/{}/{}/{}.log'.format(
    configs.dataset_name, configs.model.__class__.__name__,time,time
)
if not os.path.exists(configs.results_save_dir):
    os.makedirs(configs.results_save_dir)

configs.path = './{}/{}/{}'.format(
    configs.dataset_name, configs.model.__class__.__name__,time
)
