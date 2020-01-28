import time
import argparse
import collections

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

from lib.model import model
from lib.dataloader.dataloader import CocoDataset, VocDataset, Resizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from lib.util import coco_eval
from lib.util import voc_eval
from lib import config

def main(args=None):

    parser     = argparse.ArgumentParser(description='Simple testing script for testing a RetinaNet network.')

    parser.add_argument('--dataset',type=str)
    parser.add_argument('--test_epoch',type=int,default=0)

    parser = parser.parse_args(args)
    
    if parser.dataset=='coco':
        config.dataset=config.dataset_coco
    elif parser.dataset=='voc':
        config.dataset=config.dataset_voc
    
    set_name=[iset for iset in config.dataset['test_set'].split('+')]
    if config.dataset['dataset']=='coco': 
        dataset_val = CocoDataset(config.dataset['path'], set_name=set_name, transform=transforms.Compose([Normalizer(), Resizer()]))
    elif config.dataset['dataset']=='voc':
        dataset_val = VocDataset(config.dataset['path'], set_name=set_name, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Not implemented.')	
  
    if config.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    elif config.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
    else:
        raise ValueError('Not implemented.')	

    use_gpu=True
    if use_gpu:
        retinanet = retinanet.cuda()
	
    retinanet = torch.nn.DataParallel(module=retinanet,device_ids=[config.gpu_ids[0]]).cuda()

    retinanet.load_state_dict(torch.load('models/'+config.dataset['dataset']+'_retinanet_'+str(parser.test_epoch)+'.pt'))

    retinanet.training = False

    retinanet.eval()
    retinanet.module.freeze_bn()

    if config.dataset['dataset']=='coco':
        coco_eval.evaluate_coco(dataset_val, retinanet)
    elif config.dataset['dataset']=='voc':
        voc_eval.evaluate_voc(dataset_val, retinanet)
    else:
        raise ValueError('Not implemented.')

if __name__ == '__main__':
    with torch.cuda.device(config.gpu_ids[0]):
        main()
