from __future__ import print_function, division
import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO
from ..util.pascal_voc_eval import voc_eval
import cv2
from .. import config

from augmentations import Augmentation

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name=['train2017'], transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco={}
        for set_name_ii in self.set_name:
            prefix_ii='instances' if 'test' not in set_name_ii else 'image_info'
            self.coco[set_name_ii] = COCO(os.path.join(self.root_dir, 'annotations', prefix_ii + '_' + set_name_ii + '.json'))

        self.image_ids = []
        for set_name_ii in self.set_name:
            self.image_ids.extend([[set_name_ii,ids] for ids in self.coco[set_name_ii].getImgIds()])

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        set_name_ii=self.set_name[0]
        categories = self.coco[set_name_ii].loadCats(self.coco[set_name_ii].getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info_info = self.image_ids[image_index]
        image_info = self.coco[image_info_info[0]].loadImgs(image_info_info[1])[0]
        path       = os.path.join(self.root_dir, 'images', image_info_info[0], image_info['file_name'])

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        return img.astype(np.float32)

    def load_annotations(self, image_index):

        image_info_info = self.image_ids[image_index]
        annotations_ids = self.coco[image_info_info[0]].getAnnIds(imgIds=image_info_info[1], iscrowd=False)

        loaded_img=self.coco[image_info_info[0]].loadImgs(image_info_info[1])

        width=loaded_img[0]['width']
        height=loaded_img[0]['height']

        valid_boxes=[]
        coco_annotations = self.coco[image_info_info[0]].loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations): 

            x1,y1=a['bbox'][0],a['bbox'][1]
            x2=x1+np.maximum(0.,a['bbox'][2]-1.)
            y2=y1+np.maximum(0.,a['bbox'][3]-1.)

            x1=np.minimum(width-1.,np.maximum(0.,x1))
            y1=np.minimum(height-1.,np.maximum(0.,y1))
            x2=np.minimum(width-1.,np.maximum(0.,x2))
            y2=np.minimum(height-1.,np.maximum(0.,y2)) 

            label=self.coco_label_to_label(a['category_id'])

            if a['area']>0 and x2>x1 and y2>y1:
                valid_boxes.append([x1,y1,x2,y2,label])
 
        gt_boxes=np.zeros((len(valid_boxes),5),dtype=np.float32)
        for ii,jj in enumerate(valid_boxes):
            gt_boxes[ii,:]=jj

        return gt_boxes

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image_info_info = self.image_ids[image_index]
        image = self.coco[image_info_info[0]].loadImgs(image_info_info[1])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

    def num_gt(self, image_index):
        gt_boxes=self.load_annotations(image_index)
        return len(gt_boxes)

class VocDataset(Dataset):
    """Voc dataset."""

    def __init__(self, root_dir, set_name=['2007_trainval'], transform=None):
        """
        Args:
            root_dir (string): VOC directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.set_name = set_name
        self.devkit_path = root_dir
        self.transform = transform

        self.classes = ['__background__',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']


        self.image_ids = []
        for set_name_ii in self.set_name:
            self.image_ids.extend([[set_name_ii,ids] for ids in self.load_image_set_index(set_name_ii)])

        self.num_images = len(self.image_ids)
        print('num_images:' , self.num_images)

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

        self.image_size=[]
        for ii in range(len(self.image_ids)):
            height,width,_=self.load_image(ii).shape
            self.image_size.append([height,width])
        
    def load_image_set_index(self, image_set):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        year,image_set=image_set.split('_')
        data_path=os.path.join(self.devkit_path,'VOC'+year)
        image_set_index_file = os.path.join(data_path, 'ImageSets', 'Main', image_set + '.txt')
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        if len(self.set_name)==1 and self.set_name[0]=='2012_test':
            annot=np.zeros((0,5),dtype=np.float32)
        else:
            annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):

        image_set, index = self.image_ids[image_index]
        year,image_set=image_set.split('_')
        data_path=os.path.join(self.devkit_path,'VOC'+year)
 
        image_path = os.path.join(data_path, 'JPEGImages', index + '.jpg')
        img = cv2.imread(image_path,cv2.IMREAD_COLOR)

        return img.astype(np.float32)
    
    def load_annotations(self, image_index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET 

        image_set, index = self.image_ids[image_index]
        year,image_set=image_set.split('_')
        data_path=os.path.join(self.devkit_path,'VOC'+year)
        
        height,width=self.image_size[image_index]

        filename = os.path.join(data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs 
        num_objs = len(objs)

        valid_boxes=[]
        class_to_index = dict(zip(self.classes, range(len(self.classes))))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            
            x1=np.minimum(width-1.,np.maximum(0.,x1))
            y1=np.minimum(height-1.,np.maximum(0.,y1))
            x2=np.minimum(width-1.,np.maximum(0.,x2))
            y2=np.minimum(height-1.,np.maximum(0.,y2))

            cls = class_to_index[obj.find('name').text.lower().strip()]
            if x2>x1 and y2>y1:
                valid_boxes.append([x1,y1,x2,y2,cls-1])

        gt_boxes=np.zeros((len(valid_boxes),5),dtype=np.float32)
        for ii,jj in enumerate(valid_boxes):
            gt_boxes[ii,:]=jj

        return gt_boxes

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        image_set=self.set_name[0]
        year,image_set=image_set.split('_')

        year_folder = os.path.join('results', 'VOC' + year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join('results', 'VOC' + year, 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        image_set=self.set_name[0]
        year,image_set=image_set.split('_')

        res_file_folder = os.path.join('results', 'VOC' + year, 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, set_index in enumerate(self.image_ids):
                    _,index=set_index
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: None
        """
        image_set=self.set_name[0]
        year,image_set=image_set.split('_')
        data_path=os.path.join(self.devkit_path,'VOC'+year)

        annopath = os.path.join(data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(data_path, 'ImageSets', 'Main', image_set + '.txt')

        aps1 = [] 
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps1 += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))

        print('Mean AP = {:.4f}'.format(np.mean(aps1)))

    def image_aspect_ratio(self, image_index):

        height,width=self.image_size[image_index]

        return float(width) / float(height)

    def num_classes(self):
        return 20

    def num_gt(self, image_index):
        gt_boxes=self.load_annotations(image_index)
        return len(gt_boxes)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        min_side=config.test_img_size[0]
        max_side=config.test_img_size[1]

        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = float(min_side) / float(smallest_side)

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = float(max_side) / float(largest_side)

        # resize the image with the computed scale
        image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        rows2, cols2, cns = image.shape

        pad_w = 32 - rows2%32
        pad_h = 32 - cols2%32

        if pad_w==32:
            pad_w=0
        if pad_h==32:
            pad_h=0

        new_image = np.zeros((rows2 + pad_w, cols2 + pad_h, cns)).astype(np.float32)
        new_image[:rows2, :cols2, :] = image.astype(np.float32)

        annots_wh = annots[:,2:4]-annots[:,0:2]+1.0
        annots[:,0:2] = annots[:,0:2]/np.array([cols,rows])*np.array([cols2,rows2])
        annots[:,2:4]=annots[:,0:2]+annots_wh/np.array([cols,rows])*np.array([cols2,rows2])-1.0

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'im_info': torch.tensor([[rows2,cols2]])}


class Normalizer(object):

    def __init__(self): 
        self.mean = np.array([[[102.9801, 115.9465, 122.7717]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)), 'annot': annots}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.groups = self.group_images()

    def __iter__(self):
        self.groups = self.group_images()
        for group in self.groups:
            yield group

    def __len__(self): 
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):

        img_filter=True
        if img_filter:
            valid_img=[]
            for ii in range(len(self.data_source)):
                if self.data_source.num_gt(ii)>0:
                    valid_img.append(ii)
        else:
            valid_img=range(len(self.data_source))

        print('Shuffle')
        print('images_num:'+str(len(valid_img)))

        aspect_ratio_grouping=False
        if aspect_ratio_grouping:
            aspect_ratios=[self.data_source.image_aspect_ratio(ii) for ii in valid_img]
            aspect_ratios=np.array(aspect_ratios)
            g1=(aspect_ratios>=1)
            g2=np.logical_not(g1)
            g1_inds=np.where(g1)[0]
            g2_inds=np.where(g2)[0]

            pad_g1=self.batch_size-len(g1_inds)%self.batch_size
            pad_g2=self.batch_size-len(g2_inds)%self.batch_size
            if pad_g1==self.batch_size:
                pad_g1=0
            if pad_g2==self.batch_size:
                pad_g2=0
            g1_inds=np.hstack([g1_inds,g1_inds[:pad_g1]])
            g2_inds=np.hstack([g2_inds,g2_inds[:pad_g2]])
            random.shuffle(g1_inds)
            random.shuffle(g2_inds)
            inds=np.hstack((g1_inds,g2_inds))

            inds=np.reshape(inds[:],(-1,self.batch_size))
            row_perm=np.arange(inds.shape[0])
            random.shuffle(row_perm)
            inds=np.reshape(inds[row_perm,:],(-1,))

        else:
            inds=np.arange(len(valid_img))
            random.shuffle(inds)
            pad=self.batch_size-len(inds)%self.batch_size
            if pad==self.batch_size:
                pad=0
            inds=np.hstack([inds,inds[:pad]])
            random.shuffle(inds)

        return [[valid_img[inds[x]] for x in range(i,i+self.batch_size)] for i in range(0,len(inds),self.batch_size)]
