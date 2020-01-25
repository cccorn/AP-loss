from __future__ import print_function

import numpy as np
import json
import os

import torch

def evaluate_voc(dataset, model, threshold=0.01):
    
    model.eval()
    
    with torch.no_grad():

        all_boxes = [[[] for _ in xrange(len(dataset))] for _ in xrange(21)]

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = model([data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0),data['im_info'].cuda()])
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes[:,2]=boxes[:,2]-boxes[:,0]+1
            boxes[:,3]=boxes[:,3]-boxes[:,1]+1
            boxes /= scale
            boxes[:,2]=boxes[:,2]+boxes[:,0]-1
            boxes[:,3]=boxes[:,3]+boxes[:,1]-1 

            for j in range(1, 21):
                indexes=np.where(labels==j-1)[0]
                cls_scores=scores[indexes,np.newaxis]
                cls_boxes=boxes[indexes,:]
                cls_dets=np.hstack((cls_boxes,cls_scores))
                all_boxes[j][index] = cls_dets[:,:]

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        dataset.evaluate_detections(all_boxes)

        return
