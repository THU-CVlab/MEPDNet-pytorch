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
from utils import setup_logger, is_image_file, SeqUseDataset, SingleUseDataset
from torch.utils.data import DataLoader


def use(cfg,net,model_path):
    mode = cfg['mode']
    device = cfg['device']
    class_num = cfg['class_num']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    use_img_dir = cfg['use_img_dir']

    model_name = net.__class__.__name__
    if len(cfg['gpu_ids']) > 1:
        model_name = net.module.__class__.__name__

    performance_dir = os.path.join('extra_vis')
    if os.path.exists(performance_dir):
        shutil.rmtree(performance_dir)
    os.makedirs(performance_dir,exist_ok=True)

    performance_dir = os.path.join('extra_mask')
    if os.path.exists(performance_dir):
        shutil.rmtree(performance_dir)
    os.makedirs(performance_dir,exist_ok=True)

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log',mode,'{} {}.log'.
                    format(model_name,current_time.strftime('%Y%m%d %H:%M:%S')))
    logger = setup_logger(f'{model_name} {mode}',logger_file)
    
    if cfg['seq']:
        dataset = SeqUseDataset(use_img_dir, cfg['input_transform'], logger)
    else:
        dataset = SingleUseDataset(use_img_dir, cfg['input_transform'], logger)
    loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    net.load_state_dict(torch.load(model_path,map_location=device))

    cnt = 0
    net.eval()
    with torch.no_grad():
        for iter, item in enumerate(loader):
            if cfg['seq']:
                imgu1s, imgs, imgd1s, filenames = item
                imgu1s, imgs, imgd1s = imgu1s.to(device), imgs.to(device), imgd1s.to(device)
            else:
                imgs, filenames = item
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
 
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    for point in contour:
                        x, y = point[0]
                        img[y, x, :] = [255, 255, 0]
                
                dst = os.path.join('extra_vis', file_name.split('/')[-1])
                cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

                dst = os.path.join('extra_mask', file_name.split('/')[-1]).replace('.jpg', '.npy')
                np.save(dst, mask)

                cnt += 1
                print('{} Done'.format(cnt))
        
        print('All Done.')
