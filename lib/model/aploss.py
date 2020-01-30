import numpy as np
import torch
import torch.nn as nn
from .. import config
from ..util.calc_iou import calc_iou

class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, classifications, regressions, anchors, annotations):
 
        batch_size = classifications.shape[0]
        regression_losses = []

        regression_grads=torch.zeros(regressions.shape).cuda()
        p_num=torch.zeros(1).cuda()
        labels_b=[]

        anchor = anchors[0, :, :].type(torch.cuda.FloatTensor)

        anchor_widths  = anchor[:, 2] - anchor[:, 0]+1.0
        anchor_heights = anchor[:, 3] - anchor[:, 1]+1.0
        anchor_ctr_x   = anchor[:, 0] + 0.5 * (anchor_widths-1.0)
        anchor_ctr_y   = anchor[:, 1] + 0.5 * (anchor_heights-1.0)

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                labels_b.append(torch.zeros(classification.shape).cuda())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            ######
            gt_IoU_max, gt_IoU_argmax = torch.max(IoU, dim=0)
            gt_IoU_argmax=torch.where(IoU==gt_IoU_max)[0]
            positive_indices = torch.ge(torch.zeros(IoU_max.shape).cuda(),1)
            positive_indices[gt_IoU_argmax.long()] = True
            ######

            positive_indices = positive_indices | torch.ge(IoU_max, 0.5)
            negative_indices = torch.lt(IoU_max, 0.4)

            p_num+=positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[negative_indices, :] = 0
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            labels_b.append(targets)

            # compute the loss for regression
            if positive_indices.sum() > 0:

                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]+1.0
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]+1.0
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * (gt_widths-1.0)
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * (gt_heights-1.0)

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets2 = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets2 = targets2.t()

                targets2 = targets2/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                #negative_indices = ~ positive_indices

                regression_diff = regression[positive_indices, :]-targets2
                regression_diff_abs= torch.abs(regression_diff)

                regression_loss = torch.where(
                    torch.le(regression_diff_abs, 1.0 / 1.0),
                    0.5 * 1.0 * torch.pow(regression_diff_abs, 2),
                    regression_diff_abs - 0.5 / 1.0
                )
                regression_losses.append(regression_loss.sum())


                regression_grad=torch.where(
                    torch.le(regression_diff_abs,1.0/1.0),
                    1.0*regression_diff,
                    torch.sign(regression_diff))
                regression_grads[j,positive_indices,:]=regression_grad

            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        p_num=torch.clamp(p_num,min=1)
        regression_grads/=(4*p_num)

        ########################AP-LOSS##########################
        labels_b=torch.stack(labels_b)
        classification_grads,classification_losses=AP_loss(classifications,labels_b)
        #########################################################

        ctx.save_for_backward(classification_grads,regression_grads)
        return classification_losses, torch.stack(regression_losses).sum(dim=0, keepdim=True)/p_num

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1,g2=ctx.saved_tensors
        return g1*out_grad1,g2*out_grad2,None,None


def AP_loss(logits,targets):
    
    delta=1.0

    grad=torch.zeros(logits.shape).cuda()
    metric=torch.zeros(1).cuda()

    if torch.max(targets)<=0:
        return grad, metric
  
    labels_p=(targets==1)
    fg_logits=logits[labels_p]
    threshold_logit=torch.min(fg_logits)-delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n=((targets==0)&(logits>=threshold_logit))
    valid_bg_logits=logits[valid_labels_n] 
    valid_bg_grad=torch.zeros(len(valid_bg_logits)).cuda()
    ########

    fg_num=len(fg_logits)
    prec=torch.zeros(fg_num).cuda()
    order=torch.argsort(fg_logits)
    max_prec=0

    for ii in order:
        tmp1=fg_logits-fg_logits[ii] 
        tmp1=torch.clamp(tmp1/(2*delta)+0.5,min=0,max=1)
        tmp2=valid_bg_logits-fg_logits[ii]
        tmp2=torch.clamp(tmp2/(2*delta)+0.5,min=0,max=1)
        a=torch.sum(tmp1)+0.5
        b=torch.sum(tmp2)
        tmp2/=(a+b)
        current_prec=a/(a+b)
        if (max_prec<=current_prec):
            max_prec=current_prec
        else:
            tmp2*=((1-max_prec)/(1-current_prec))
        valid_bg_grad+=tmp2
        prec[ii]=max_prec 

    grad[valid_labels_n]=valid_bg_grad
    grad[labels_p]=-(1-prec) 

    fg_num=max(fg_num,1)

    grad /= (fg_num)
    
    metric=torch.sum(prec,dim=0,keepdim=True)/fg_num

    return grad, 1-metric
