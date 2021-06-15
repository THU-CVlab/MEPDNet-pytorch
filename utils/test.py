import os
import cv2
import glob
import torch
import shutil
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from os.path import split,splitext
from utils import setup_logger, is_image_file, SeqTestDataset, SingleTestDataset
from torch.utils.data import DataLoader

from timeit import default_timer as timer


def calDSC(predict,target):
    eps = 1e-4
    predict = predict.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.dot(predict.reshape(-1),target.reshape(-1))
    union = predict.sum() + target.sum() + eps
    dsc = (2 * intersection + eps) / union
    return dsc


def calRecall(predict, target):
    eps = 1e-4
    predict = predict.astype(np.int8).reshape(-1)
    target = target.astype(np.int8).reshape(-1)
    TP = (predict == 1) & (target == 1)
    FN = (predict == 0) & (target == 1)
    recall = TP.sum() / (TP.sum() + FN.sum() + eps)
    return recall


def calPrecision(predict, target):
    eps = 1e-4
    predict = predict.astype(np.int8).reshape(-1)
    target = target.astype(np.int8).reshape(-1)
    TP = (predict == 1) & (target == 1)
    FP = (predict == 1) & (target == 0)
    precision = TP.sum() / (TP.sum() + FP.sum() + eps)
    return precision


def test(cfg,net,model_path):
    mode = cfg['mode']
    device = cfg['device']
    class_num = cfg['class_num']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    test_img_dir = cfg['test_img_dir']
    test_mask_dir = cfg['test_mask_dir']

    model_name = net.__class__.__name__
    if len(cfg['gpu_ids']) > 1:
        model_name = net.module.__class__.__name__

    performance_dir = os.path.join('test_performance_add', model_name)
    if os.path.exists(performance_dir):
        shutil.rmtree(performance_dir)
    os.makedirs(performance_dir,exist_ok=True)

    performance_dir = os.path.join('test_performance_add', 'GT')
    if os.path.exists(performance_dir):
        shutil.rmtree(performance_dir)
    os.makedirs(performance_dir,exist_ok=True)

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log',mode,'{} {}.log'.
                    format(model_name,current_time.strftime('%Y%m%d %H:%M:%S')))
    logger = setup_logger(f'{model_name} {mode}',logger_file)
    
    if cfg['seq']:
        dataset = SeqTestDataset(test_img_dir, test_mask_dir, cfg['input_transform'], cfg['target_transform'], logger)
    else:
        dataset = SingleTestDataset(test_img_dir, test_mask_dir, cfg['input_transform'], cfg['target_transform'], logger)
    loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    net.load_state_dict(torch.load(model_path,map_location=device))

    dsc = []
    recall = []
    precision = []

    net.eval()
    for iter, item in enumerate(loader):
        if cfg['seq']:
            imgu1s, imgs, imgd1s, targets, filenames = item
            imgu1s, imgs, imgd1s = imgu1s.to(device), imgs.to(device), imgd1s.to(device)
        else:
            imgs, targets, filenames = item
            imgs = imgs.to(device)
    
        with torch.no_grad():
            if cfg['seq']:
                predict = net(imgu1s, imgs, imgd1s)
            else:
                predict = net(imgs)

        if class_num > 1:
            probs = F.softmax(predict,dim=1)
        else:
            probs = torch.sigmoid(predict)
            
        masks = torch.argmax(probs,dim=1).cpu().numpy()
            
        for i, file_name in enumerate(filenames):
            img = cv2.imread(file_name)
            mask = masks[i].astype(np.uint8)
            ground_truth = targets[i].numpy().astype(np.uint8)
            contours, hierarchy = cv2.findContours(ground_truth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [0, 0, 255]

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [255, 255, 0]
                
            dsc.append(calDSC(mask, ground_truth))
            recall.append(calRecall(mask, ground_truth))
            precision.append(calPrecision(mask, ground_truth))

            dst = os.path.join('test_performance_add', model_name, file_name.split('/')[-1]).replace('.jpg', '.png')
            colors = np.full(img.shape, [159, 255, 84], dtype=np.uint8) \
                * np.array([mask for i in range(3)]).transpose(1,2,0)
            cv2.imwrite(dst, cv2.addWeighted(img, 0.84, colors, 0.16, 0))

            dst = os.path.join('test_performance_add', 'GT', file_name.split('/')[-1]).replace('.jpg', '.png')
            colors = np.full(img.shape, [159, 255, 84], dtype=np.uint8) \
                * np.array([ground_truth for i in range(3)]).transpose(1,2,0)
            cv2.imwrite(dst, cv2.addWeighted(img, 0.84, colors, 0.16, 0))
        
    print('DSC of {}: {}'.format(model_name, np.mean(dsc)))
        