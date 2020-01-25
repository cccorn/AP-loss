import torch.nn as nn
import torch
import math
import time
from ..util.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
from aploss import APLoss

from .. import config
import torchvision

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        #self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        #out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, conv1_bias):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=conv1_bias)
        
        for ii in self.conv1.parameters():
            ii.requires_grad=False
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        for ii in self.layer1.parameters():
            ii.requires_grad=False
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256, num_anchors=config.num_anchors)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes, num_anchors=config.num_anchors)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()
        
        self.aploss= APLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        
        #self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.weight.data.normal_(0,0.01)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        #self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.weight.data.normal_(0,0.001)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for ii in layer.parameters():
                    ii.requires_grad=False 

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch, im_info = inputs
 
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regressions = [self.regressionModel(feature) for feature in features]

        classifications = [self.classificationModel(feature) for feature in features]

        anchors = self.anchors(img_batch)

        if self.training:
            regressions=torch.cat(regressions, dim=1)
            classifications=torch.cat(classifications, dim=1)
            anchors=torch.cat(anchors, dim=1)
            return self.aploss.apply(classifications, regressions, anchors, annotations)
        else:
            box_all=[]
            cls_all=[]
            score_all=[]
            for ii, anchor in enumerate(anchors):

                classification=classifications[ii].view(-1)
                regression=regressions[ii]

                classification=torch.sigmoid(classification)

                ###filter
                num_topk=min(1000,classification.size(0))
                ordered_score,ordered_idx=classification.sort(descending=True)
                ordered_score=ordered_score[:num_topk]
                ordered_idx=ordered_idx[:num_topk]
 
                if ii<4:
                    score_th=0.01
                else:
                    score_th=0
                keep_idx=(ordered_score>score_th)
                ordered_score=ordered_score[keep_idx]
                ordered_idx=ordered_idx[keep_idx]

                anchor_idx = ordered_idx // self.num_classes
                cls_idx = ordered_idx % self.num_classes

                transformed_anchor = self.regressBoxes(anchor[:,anchor_idx,:], regression[:,anchor_idx,:])
                transformed_anchor = self.clipBoxes(transformed_anchor, im_info[0])
                
                box_all.append(transformed_anchor[0])
                cls_all.append(cls_idx)
                score_all.append(ordered_score)

            box_all=torch.cat(box_all,dim=0)
            cls_all=torch.cat(cls_all,dim=0)
            score_all=torch.cat(score_all,dim=0)

            keep=torchvision.ops.boxes.batched_nms(box_all,score_all,cls_all,0.5)
            keep=keep[:100]
            return [score_all[keep], cls_all[keep], box_all[keep, :]]
 
def resnet50(num_classes, pretrained=False):

    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], conv1_bias=True)
    if pretrained:
        model.load_state_dict(torch.load('models/resnet50-pytorch.pth'), strict=False)
    return model

def resnet101(num_classes, pretrained=False):
 
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], conv1_bias=False)
    if pretrained:
        model.load_state_dict(torch.load('models/resnet101-pytorch.pth'), strict=False)
    return model
