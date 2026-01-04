import torch
import torch.nn as nn
from tqdm import tqdm
from loss import dice_loss
from utils import get_lr, adjust_lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_common(configs, logger, trainSet, model, optimizer, epoch, losses):
    print("Training...")
    BCE_Loss = nn.BCELoss()
    Dice_Loss = dice_loss()
    
    epoch_loss = 0
    faz_epoch_loss = 0
    rv_epoch_loss = 0
    
    for sample in tqdm(trainSet):
        img = sample[0].to(device)
        faz_gt = sample[1].to(device)
        rv_gt = sample[2].to(device)
        
        optimizer.zero_grad()
        
        pfaz, faz_pred, rv_pred = model(img, epoch)
        
        faz_loss = Dice_Loss(pfaz, faz_gt) * 0.2 + Dice_Loss(faz_pred, faz_gt) * 0.8
        rv_loss = Dice_Loss(rv_pred, rv_gt) * 0.5 + BCE_Loss(rv_pred, rv_gt) * 0.5
        
        weighted_loss = faz_loss * 0.5 + 0.5 * rv_loss
        
        weighted_loss.backward(retain_graph=False)
        optimizer.step()
        
        epoch_loss += weighted_loss.item()
        faz_epoch_loss += faz_loss.item()
        rv_epoch_loss += rv_loss.item()
        
    current_lr = get_lr(optimizer)
    
    losses.append(faz_epoch_loss/len(trainSet))
    
    logger.info("[Epoch %d/%d FAZ Loss: %0.4f, RV Loss: %0.4f]" % (
        epoch, configs.epochs, faz_epoch_loss / len(trainSet), rv_epoch_loss / len(trainSet)))

    logger.info("[Epoch %d/%d Loss: %0.4f, Learning Rate: %f]" % (epoch, configs.epochs, epoch_loss/len(trainSet), current_lr))
    adjust_lr(optimizer, configs.init_lr, epoch, configs.epochs, configs.power)

    return model
