import os
import ast
import json
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import MEPDNet

from utils import get_range_limited_float_type,train, test, use
from utils import timewrapper,setup_logger,ToLabel
from torchvision.transforms import Compose,Normalize,ToTensor,ToPILImage

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode',type=str,choices=['train', 'test', 'use'],required=True)
    parser.add_argument('--config-path',type=str,default='config/cfg.json')
    parser.add_argument('--gpu-ids',type=int,nargs='+',default=0,dest='gpu_ids')
    parser.add_argument('--state',type=int,default=1,dest='state')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-p', '--port', type=int,default=10001,
                        help='Visualization port', dest='port')
    parser.add_argument('-w', '--worker-num', type=int, default=1,
                        help='Dataloader worker number', dest='num_workers')
    parser.add_argument('-c', '--class-num', type=int, default=2,
                        help='class number', dest='class_num')
    parser.add_argument('-v', '--valid-percent', type=get_range_limited_float_type(0,100), default=10.0,
                        help='Percent of the data that is used as validation (0-100)', dest='valid_percent')
    parser.add_argument('-s', '--sequence', default=False, type=ast.literal_eval, choices=[True, False], help="sequence model", dest='seq')

    args = parser.parse_args()

    assert os.path.exists(args.config_path),'config json not exists'
    with open(args.config_path,'r') as f:
        config = json.load(f)

    for arg in vars(args):
        config[arg]=getattr(args,arg)
    
    if isinstance(config['gpu_ids'],int):
        config['gpu_ids'] = [config['gpu_ids']]
    config['gpu_ids'] = list(set(config['gpu_ids']))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['device'] == 'cuda':
        gpu_num = torch.cuda.device_count()
        assert len(config['gpu_ids'])!=0,'unexpected gpu number'
        for gpu_id in config['gpu_ids']:
            assert gpu_id>=0 and gpu_id<gpu_num,'invalid gpu id input'

    config['input_transform'] = Compose([
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]), 
    ])
    
    config['target_transform'] = Compose([
        ToLabel(),
    ])
            
    return config


def main(cfg):
    cfg['seq'] = False
    if cfg['model'] == 'mepdnet':
        net = MEPDNet(n_channels=3, n_classes=cfg['class_num'])
        cfg['seq'] = True
    else:
        raise Exception('model {} not available'.format(cfg['model']))

    if cfg['device']=='cuda':
        if len(cfg['gpu_ids'])==1:
            torch.cuda.set_device(cfg['gpu_ids'][0])
            net = net.cuda()
        else:
            net = net.cuda()
            net = nn.DataParallel(net,device_ids=cfg['gpu_ids'])

    torch.backends.cudnn.benchmark = True #

    torch.manual_seed(2020) # 
    torch.cuda.manual_seed(2020) #
    torch.cuda.manual_seed_all(2020) #

    if cfg['mode'] == 'train':
        train(cfg,net)
    elif cfg['mode'] == 'test':
        test(cfg, net, 'checkpoints/{}_{}.pth'.format(cfg['model'],cfg['state']))
    elif cfg['mode'] == 'use':
        use(cfg, net, 'checkpoints/{}_{}.pth'.format(cfg['model'],cfg['state']))
        

if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
