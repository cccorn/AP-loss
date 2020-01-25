from __future__ import print_function

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os

import torch

def evaluate_coco(dataset, model, threshold=0.01):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

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

            if boxes.shape[0] > 0:
 
                # compute predicted labels and scores
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :] 

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index][1],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index][1])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r') 

        # write output
        json.dump(results, open('./results/{}_bbox_results.json'.format(dataset.set_name[0]), 'w'), indent=4)

        if 'test' in dataset.set_name[0]:
            return

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_true = coco_true[list(coco_true.keys())[0]]
        coco_pred = coco_true.loadRes('./results/{}_bbox_results.json'.format(dataset.set_name[0]))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return
