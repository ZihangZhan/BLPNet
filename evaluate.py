import os
import torch
from test import test_train

def evaluate(configs, logger, best, validSet, model, epoch, isSave):
    model.eval()
    
    _, _, _, _, _, dice_dct_faz, _, _, _, _, _, _, dice_dct_rv, _ = test_train(configs, logger, validSet, model, isSave)
    
    dice_faz = round(dice_dct_faz["out"].mean() + 1e-12, 4)
    dice_rv = round(dice_dct_rv["out"].mean() + 1e-12, 4)
    
    if dice_faz >= best["dice_faz"]:
        best["epoch_faz"] = epoch
        best["dice_faz"] = dice_faz
        logger.info("Best FAZ Model Updated!")

    if dice_rv >= best["dice_rv"]:
        best["epoch_rv"] = epoch
        best["dice_rv"] = dice_rv
        logger.info("Best RV Model Updated!")
    
    logger.info("[evaluate][Best: FAZ - Epoch %d/%d - Dice: %.4f, RV - Epoch %d/%d - Dice: %.4f]" % (best["epoch_faz"], configs.epochs, best["dice_faz"], best["epoch_rv"], configs.epochs, best["dice_rv"]))
    
    model.train(mode=True)
    
    return model, best


def test(configs, logger, best, validSet, model, epoch, isSave):
    model.eval()

    _, _, _, _, _, dice_dct_faz, _, _, _, _, _, _, dice_dct_rv, _ = test_train(configs, logger, validSet, model,isSave)
    f = 0
    dice_faz = round(dice_dct_faz["out"].mean() + 1e-12, 4)
    dice_rv = round(dice_dct_rv["out"].mean() + 1e-12, 4)

    if dice_faz >= best["dice_faz"]:
        best["epoch_faz"] = epoch
        best["dice_faz"] = dice_faz
        f = 1
        torch.save(model, os.path.join(configs.models_save_dir, "model-t-best-faz.pth"))
        logger.info("Best FAZ Model Updated!")

    if dice_rv >= best["dice_rv"]:
        best["epoch_rv"] = epoch
        best["dice_rv"] = dice_rv
        f = 1
        torch.save(model, os.path.join(configs.models_save_dir, "model-t-best-rv.pth"))
        logger.info("Best RV Model Updated!")

    logger.info("[test][Best: FAZ - Epoch %d/%d - Dice: %.4f, RV - Epoch %d/%d - Dice: %.4f]" % (
    best["epoch_faz"], configs.epochs, best["dice_faz"], best["epoch_rv"], configs.epochs, best["dice_rv"]))

    model.train(mode=True)

    return model, best, f
