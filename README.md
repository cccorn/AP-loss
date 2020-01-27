# AP-loss
The implementation of “[Towards accurate one-stage object detection with AP-loss](https://arxiv.org/abs/1904.06373)” and its journal version.

### Requirements
- Python 2.7
- PyTorch 1.3+
- Cuda

### Installation
1. Clone this repo
```
git clone https://github.com/cccorn/AP-loss.git
cd AP-loss
```
2. Install the python packages:
```
pip install pycocotools
pip install opencv-python
```
3. Create directories:
```
mkdir data models results
```
4. Prepare Data. You can use
```
ln -s $YOUR_PATH_TO_coco data/coco
ln -s $YOUR_PATH_TO_VOCdevkit data/voc
```
The directories should be arranged like:
```
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   │   ├── test-dev2017
│   ├── voc
│   │   ├── VOC2007
│   │   ├── VOC2012
```
5. Prepare the pre-trained models and put them in `models` like:
```
├── models
│   ├── resnet50-pytorch.pth
|   ├── resnet101-pytorch.pth
```
We use the ResNet-50 and ResNet-101 pre-trained models which are converted from [here](https://github.com/KaimingHe/deep-residual-networks). We also provide the converted pre-trained models at [this link](https://1drv.ms/u/s!AgPNhBALXYVSa1pQCFJNNk6JgaA?e=PqhsWD).

### Training

```
bash train.sh
```
You can modify the configurations in `lib/config.py` to change the gpu_ids, network depth, image size, etc.

### Testing

```
bash test.sh
```

### Note

We release the AP-loss implementation in PyTorch instead of in MXNet due to an engineering [issue](https://github.com/apache/incubator-mxnet/issues/8884): the python custom operator in MXNet does not run in parrallel when using multi-gpus. It is more practical to implement AP-loss in PyTorch, for faster training speed. 

### Acknowledgements

- Many thanks to the pytorch implementation of RetinaNet at [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet).

### Citation

If you find this repository useful in your research, please consider citing:
```
@inproceedings{chen2019towards,
  title={Towards accurate one-stage object detection with AP-loss},
  author={Chen, Kean and Li, Jianguo and Lin, Weiyao and See, John and Wang, Ji and Duan, Lingyu and Chen, Zhibo and He, Changwei and Zou, Junni},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5119--5127},
  year={2019}
}
```
