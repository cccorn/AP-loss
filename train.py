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
from lib.dataloader.dataloader import CocoDataset, VocDataset, collater, AspectRatioBasedSampler, Augmentation
from torch.utils.data import Dataset, DataLoader

from lib import config


print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):

    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset',type=str)
    parser.add_argument('--resume',type=bool, default=False)
    parser.add_argument('--resume_epoch',type=int, default=-1)

    parser = parser.parse_args(args)
    
    if parser.dataset=='coco':
        config.dataset=config.dataset_coco
    elif parser.dataset=='voc':
        config.dataset=config.dataset_voc

    set_name=[iset for iset in config.dataset['train_set'].split('+')]
    # Create the data loaders
    if config.dataset['dataset'] == 'coco':
        dataset_train = CocoDataset(config.dataset['path'], set_name=set_name, transform=Augmentation())
    elif config.dataset['dataset'] == 'voc':
        dataset_train = VocDataset(config.dataset['path'], set_name=set_name, transform=Augmentation())
    else:
        raise ValueError('Not implemented.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=config.batch_size*len(config.gpu_ids))
    dataloader_train = DataLoader(dataset_train, num_workers=len(config.gpu_ids), collate_fn=collater, batch_sampler=sampler)

    # Create the model 
    if config.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif config.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Not implemented')

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()
	
    retinanet = torch.nn.DataParallel(module=retinanet,device_ids=config.gpu_ids).cuda()
    
    retinanet.training = True

    optimizer = optim.SGD(retinanet.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.dataset['lr_step'], gamma=0.1)

    warmup=config.warmup
    begin_epoch=0
    if parser.resume==True:
        retinanet.load_state_dict(torch.load('./models/'+config.dataset['dataset']+'_retinanet_'+str(parser.resume_epoch)+'.pt'))
        begin_epoch=parser.resume_epoch+1 
        for jj in range(begin_epoch):
            scheduler.step()

    cls_loss_hist = collections.deque(maxlen=300)
    reg_loss_hist = collections.deque(maxlen=300)
    tic_hist = collections.deque(maxlen=100)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(begin_epoch,config.dataset['epochs']):

        retinanet.train()
        retinanet.module.freeze_bn()

        tic=time.time()
        for iter_num, data in enumerate(dataloader_train):

            optimizer.zero_grad()

            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            if warmup and optimizer._step_count<=config.warmup_step:
                init_lr=config.lr
                warmup_lr=init_lr*config.warmup_factor + optimizer._step_count/float(config.warmup_step)*(init_lr*(1-config.warmup_factor))
                for ii_ in optimizer.param_groups:
                    ii_['lr']=warmup_lr 

            optimizer.step()
 
            tic_hist.append(time.time()-tic)
            tic=time.time()
            speed=(config.batch_size*len(config.gpu_ids)*len(tic_hist))/(np.sum(tic_hist))
            cls_loss_hist.append(float(classification_loss))
            reg_loss_hist.append(float(regression_loss))
            print('Epoch: {} | Iteration: {} | Classification loss: avg: {:1.5f}, cur: {:1.5f} | Regression loss: avg: {:1.5f}, cur: {:1.5f} | Speed: {:1.5f} images per second'.format(epoch_num, iter_num, np.mean(cls_loss_hist), float(classification_loss), np.mean(reg_loss_hist), float(regression_loss), speed))

            del classification_loss
            del regression_loss 

        scheduler.step()

        torch.save(retinanet.state_dict(), 'models/{}_retinanet_{}.pt'.format(config.dataset['dataset'], epoch_num))

    retinanet.eval()

    torch.save(retinanet.state_dict(), 'models/model_final.pt'.format(epoch_num))

if __name__ == '__main__':
    with torch.cuda.device(config.gpu_ids[0]):
        main()
