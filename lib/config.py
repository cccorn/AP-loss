import numpy as np

gpu_ids=[0,1]
batch_size=8
epochs=100
lr=0.001
lr_step=[60,80]
warmup=True
warmup_step=500
warmup_factor=0.33333333

train_img_size=512
test_img_size=[500,833]

anchor_ratios=np.array([0.5,1.0,2.0])
anchor_scales=np.array([2**0,2**(1.0/2.0)])
num_anchors=len(anchor_ratios)*len(anchor_scales)

pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]])

dataset_coco={'dataset':'coco', 'path':'data/coco', 'train_set':'train2017', 'test_set':'val2017'}
dataset_voc={'dataset':'voc', 'path':'data/voc', 'train_set':'2007_trainval+2012_trainval', 'test_set':'2007_test'}

depth=101
