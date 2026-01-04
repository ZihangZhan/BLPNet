import os
m_gpu = 0
import torch
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % m_gpu
torch.cuda.set_device(m_gpu)
torch.cuda.is_available()
torch.cuda.current_device()
import time
import shutil
import datetime
import matplotlib.pyplot as plt
import warnings
from torch import optim

from dataset import load_data
from train import train_common
from test import test_test
from evaluate import evaluate, test
from logger import get_logger, print_configs
from configs import configs

def if_stop(file_path, target_value):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if target_value in line:
                    return True
        return False
    except FileNotFoundError:
        return False

def main(configs, logger, model, trainSet, validSet):
    p_epoch = 0
    if configs.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=configs.init_lr, weight_decay=configs.weight_decay)
    elif configs.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=configs.init_lr, momentum=0.99, weight_decay=configs.weight_decay)

    best_v = {"epoch_faz": 0, "dice_faz": 0, "epoch_rv": 0, "dice_rv": 0}
    best_t = {"epoch_faz": 0, "dice_faz": 0, "epoch_rv": 0, "dice_rv": 0}

    logger.info("-------Training Started...-------\n")
    start = time.time()
    losses = []
    for epoch in range(configs.epochs):
        if not if_stop('./if_stop.txt','run'):
            break

        logger.info("[Epoch {}/{} on Network-{} and DataSet-{}]".format(
            epoch + 1, configs.epochs, configs.model.__class__.__name__, configs.dataset_name)
        )
        model = train_common(configs, logger, trainSet, model, optimizer, epoch + 1, losses)
        if (epoch + 1) % 5 == 0 or epoch == configs.epochs - 1:
            model, best_v = evaluate(configs, logger, best_v, validSet, model, epoch + 1, isSave=False)
        logger.info("\n")

        if (epoch + 1) % 5 == 0 or epoch == configs.epochs - 1:
            model, best_t, f = test(configs, logger, best_t, testSet, model, epoch + 1, isSave=False)
            if f == 1:
                p_epoch = epoch
            else:
                if (epoch - p_epoch) > configs.estop:
                    break
        logger.info("\n")

    end = time.time()
    logger.info("-------Training Completed!!! Time Cost: %d s-------\n" % (end - start))
    
    plt.plot(range(1, configs.epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(configs.path+'train_faz_loss_epoch.png')
    plt.show()


warnings.filterwarnings("ignore")
m_gpu = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % m_gpu
torch.cuda.set_device(m_gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

now = datetime.datetime.now()
logger = get_logger(configs.log_file_path)
print_configs(logger, configs, 'Joint Segmentation')

trainSet, validSet, testSet = load_data(configs)
with open('./if_stop.txt', 'w') as file:
    file.write('run')

if configs.mode == 'TRAIN':
    model = configs.model
    main(configs, logger, model, trainSet, validSet)
    
    model_faz = torch.load(configs.models_save_dir + '/model-best-faz.pth')
    model_rv = torch.load(configs.models_save_dir + '/model-best-rv.pth')
    logger.info("-------Evaluating on best model...-------")

    if os.path.exists(configs.results_save_dir + "/out"):
        shutil.rmtree(configs.results_save_dir + "/out")

    dice_dct_faz, dice_dct_rv = test_test(configs, logger, testSet, model_faz, model_rv, isSave=True)
    test_dice_faz = round(dice_dct_faz["out"].mean() + 1e-12, 4)
    test_dice_rv = round(dice_dct_rv["out"].mean() + 1e-12, 4)
    note = ""
    
    os.rename(configs.log_file_path, configs.log_file_path[:-4] + '-{}-{}-{}-{}.log'.format(
        str(test_dice_faz), str(test_dice_rv), configs.model.__class__.__name__, note
    ))
    os.rename(configs.models_save_dir + '/model-t-best-faz.pth',
              configs.models_save_dir + '/model-t-{}-faz.pth'.format(str(test_dice_faz)))
    os.rename(configs.models_save_dir + '/model-t-best-rv.pth',
              configs.models_save_dir + '/model-t-{}-rv.pth'.format(str(test_dice_rv)))

    os.rename(configs.path,
              configs.path + '-faz-{}-rv-{}-{}-epoch-{}'.format(str(test_dice_faz), str(test_dice_rv),note,str(configs.epochs)))
