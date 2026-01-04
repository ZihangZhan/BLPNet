import os
import cv2
import torch
from tqdm import tqdm
from metric import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def get_results(configs, logger, auc_lst, acc_lst, sen_lst, spe_lst, iou_lst, dice_lst, jac_lst, dataloader,
                results_save_dir, pred, gt, isSave, type):
    pred_arr = pred.squeeze().cpu().numpy()
    gt_arr = gt.squeeze().cpu().numpy()

    pred_img = np.array(pred_arr * 255, np.uint8)
    gt_img = np.array(gt_arr * 255, np.uint8)
    
    _, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    auc_lst.append(calc_auc(thresh_pred_img / 255.0, gt_img / 255.0, mask_arr=None))
    acc_lst.append(calc_acc(thresh_pred_img / 255.0, gt_img / 255.0, configs.eval_kernel))
    sen_lst.append(calc_sen(thresh_pred_img / 255.0, gt_img / 255.0, configs.eval_kernel))
    spe_lst.append(calc_spe(thresh_pred_img / 255.0, gt_img / 255.0, configs.eval_kernel))
    iou_lst.append(calc_iou(thresh_pred_img / 255.0, gt_img / 255.0, configs.eval_kernel))
    dice_lst.append(calc_dice(thresh_pred_img / 255.0, gt_img / 255.0, configs.eval_kernel))
    jac_lst.append(calc_jac(thresh_pred_img / 255.0, gt_img / 255.0, configs.eval_kernel))

    # Save Results
    imgName = dataloader.dataset.getFileName()
    if isSave:
        dice_str = '%.4f' % (calc_dice(thresh_pred_img / 255.0, gt_img / 255.0, configs.eval_kernel))
        mkdir(results_save_dir + "/{}/".format(type))
        cv2.imwrite(results_save_dir + "/{}/".format(type) + dice_str + '_' + imgName[:-4] + '.png', pred_img)
        cv2.imwrite(results_save_dir + "/{}/".format(type) + dice_str + '_' + imgName[:-4] + '.png', thresh_pred_img)


def print_results(logger, auc_lst, acc_lst, sen_lst, spe_lst, iou_lst, dice_lst, jac_lst, type):
    auc_arr = np.array(auc_lst)
    acc_arr = np.array(acc_lst)
    sen_arr = np.array(sen_lst)
    spe_arr = np.array(spe_lst)
    iou_arr = np.array(iou_lst)
    dice_arr = np.array(dice_lst)
    jac_arr = np.array(jac_lst)

    logger.info(type)
    logger.info("AUC  - mean: " + str(auc_arr.mean()) + "\tstd: " + str(auc_arr.std()))
    logger.info("ACC  - mean: " + str(acc_arr.mean()) + "\tstd: " + str(acc_arr.std()))
    logger.info("SPE  - mean: " + str(spe_arr.mean()) + "\tstd: " + str(spe_arr.std()))
    logger.info("SEN  - mean: " + str(sen_arr.mean()) + "\tstd: " + str(sen_arr.std()))
    logger.info("Jac  - mean: " + str(jac_arr.mean()) + "\tstd: " + str(jac_arr.std()))
    logger.info("Dice - mean: " + str(dice_arr.mean()) + "\tstd: " + str(dice_arr.std()))

    return auc_arr, acc_arr, sen_arr, spe_arr, iou_arr, dice_arr, jac_arr


def test_train(configs, logger, validSet, model, isSave):
    auc_dct_faz = {"out": []}
    acc_dct_faz = {"out": []}
    sen_dct_faz = {"out": []}
    spe_dct_faz = {"out": []}
    iou_dct_faz = {"out": []}
    dice_dct_faz = {"out": []}
    jac_dct_faz = {"out": []}

    auc_dct_rv = {"out": []}
    acc_dct_rv = {"out": []}
    sen_dct_rv = {"out": []}
    spe_dct_rv = {"out": []}
    iou_dct_rv = {"out": []}
    dice_dct_rv = {"out": []}
    jac_dct_rv = {"out": []}

    with torch.no_grad():
        logger.info("Evaluating...")
        for sample in tqdm(validSet):
            img = sample[0].to(device)
            faz_gt = sample[1].to(device)
            rv_gt = sample[2].to(device)
            
            _, faz_pred, rv_pred = model(img, 10)

            get_results(configs, logger, auc_dct_faz["out"], acc_dct_faz["out"], sen_dct_faz["out"],
                        spe_dct_faz["out"], iou_dct_faz["out"],
                        dice_dct_faz["out"], jac_dct_faz["out"], validSet, configs.results_save_dir + "/out", faz_pred,
                        faz_gt, isSave, 'faz')
            get_results(configs, logger, auc_dct_rv["out"], acc_dct_rv["out"], sen_dct_rv["out"],
                        spe_dct_rv["out"], iou_dct_rv["out"],
                        dice_dct_rv["out"], jac_dct_rv["out"], validSet, configs.results_save_dir + "/out", rv_pred,
                        rv_gt, isSave, 'rv')

    auc_dct_faz["out"], acc_dct_faz["out"], sen_dct_faz["out"], spe_dct_faz["out"], iou_dct_faz["out"], dice_dct_faz[
        "out"], jac_dct_faz["out"] = \
        print_results(logger, auc_dct_faz["out"], acc_dct_faz["out"], sen_dct_faz["out"], spe_dct_faz["out"],
                      iou_dct_faz["out"], dice_dct_faz["out"], jac_dct_faz["out"], 'FAZ Metrics: ')
    auc_dct_rv["out"], acc_dct_rv["out"], sen_dct_rv["out"], spe_dct_rv["out"], iou_dct_rv["out"], dice_dct_rv["out"], \
    jac_dct_rv["out"] = \
        print_results(logger, auc_dct_rv["out"], acc_dct_rv["out"], sen_dct_rv["out"], spe_dct_rv["out"],
                      iou_dct_rv["out"], dice_dct_rv["out"], jac_dct_rv["out"], 'RV Metrics: ')

    return auc_dct_faz, acc_dct_faz, sen_dct_faz, spe_dct_faz, iou_dct_faz, dice_dct_faz, jac_dct_faz, auc_dct_rv, acc_dct_rv, sen_dct_rv, spe_dct_rv, iou_dct_rv, dice_dct_rv, jac_dct_rv


def test_test(configs, logger, validSet, model_faz, model_rv, isSave):
    auc_dct_faz = {"out": []}
    acc_dct_faz = {"out": []}
    sen_dct_faz = {"out": []}
    spe_dct_faz = {"out": []}
    iou_dct_faz = {"out": []}
    dice_dct_faz = {"out": []}
    jac_dct_faz = {"out": []}

    auc_dct_rv = {"out": []}
    acc_dct_rv = {"out": []}
    sen_dct_rv = {"out": []}
    spe_dct_rv = {"out": []}
    iou_dct_rv = {"out": []}
    dice_dct_rv = {"out": []}
    jac_dct_rv = {"out": []}

    with torch.no_grad():
        logger.info("Evaluating...")
        for sample in tqdm(validSet):
            img = sample[0].to(device)
            faz_gt = sample[1].to(device)
            rv_gt = sample[2].to(device)

            _, faz_pred, _ = model_faz(img, 0)
            _, _, rv_pred = model_rv(img, 0)

            get_results(configs, logger, auc_dct_faz["out"], acc_dct_faz["out"], sen_dct_faz["out"],
                        spe_dct_faz["out"], iou_dct_faz["out"],
                        dice_dct_faz["out"], jac_dct_faz["out"], validSet, configs.results_save_dir + "/out", faz_pred,
                        faz_gt, isSave, 'faz')
            get_results(configs, logger, auc_dct_rv["out"], acc_dct_rv["out"], sen_dct_rv["out"],
                        spe_dct_rv["out"], iou_dct_rv["out"],
                        dice_dct_rv["out"], jac_dct_rv["out"], validSet, configs.results_save_dir + "/out", rv_pred,
                        rv_gt, isSave, 'rv')

    auc_dct_faz["out"], acc_dct_faz["out"], sen_dct_faz["out"], spe_dct_faz["out"], iou_dct_faz["out"], dice_dct_faz[
        "out"], jac_dct_faz["out"] = \
        print_results(logger, auc_dct_faz["out"], acc_dct_faz["out"], sen_dct_faz["out"], spe_dct_faz["out"],
                      iou_dct_faz["out"], dice_dct_faz["out"], jac_dct_faz["out"], 'FAZ Metrics: ')
    auc_dct_rv["out"], acc_dct_rv["out"], sen_dct_rv["out"], spe_dct_rv["out"], iou_dct_rv["out"], dice_dct_rv["out"], \
    jac_dct_rv["out"] = \
        print_results(logger, auc_dct_rv["out"], acc_dct_rv["out"], sen_dct_rv["out"], spe_dct_rv["out"],
                      iou_dct_rv["out"], dice_dct_rv["out"], jac_dct_rv["out"], 'RV Metrics: ')

    return dice_dct_faz, dice_dct_rv
