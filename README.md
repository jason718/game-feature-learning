# game-feature-learning

<img src="https://jason718.github.io/project/cvpr18/files/archi.png" width="400"/>

[[Project]](https://https://jason718.github.io/project/cvpr18/main.html) [[Paper]](https://jason718.github.io/project/cvpr18/files/cvpr18_jason_camera_ready.pdf) 

If you feel this useful, please consider cite:
```bibtex
@inproceedings{ren-cvpr2018,
  title = {Cross-Domain Self-supervised Multi-task Feature Learning using Synthetic Imagery},
  author = {Ren, Zhongzheng and Lee, Yong Jae},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
Feel free to contact Jason Ren (zr5@illinois.edu) if you have any questions!

## Prerequisites
- Pytorch-0.4 (some evaluation code borrowed from other places requiring Caffe)
- Python2 (One evaluation code requiring Python3)
- NVIDIA GPU + CUDA CuDNN (Sorry no CPU version)

## Getting Started
### Installation
- Install PyTorch 0.4 and torchvision from http://pytorch.org
- All other dependencies can be installed by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/jason718/game-feature-learning
cd game-feature-learning
```

### Pre-trained models:
- Caffemodel(Caffenet): Coming soon
- Pytorch model: Coming soon

Since I greatly changed the code structure, I am retraining using the new code to reproduce

### Train/Test
- Dataset:
SUNCG

SceneNet

For Domain Adaptation:

Places-365

- Train a model:
```bash
sh ./scripts/train.sh
```
- Evaluate on feature learning

- Evaluate on three tasks


## Useful Resources
There are lots of awesome papers studying self-supervision for various tasks such as Image/Video Representation learning,
Reinforcement learning, and Robotics. I am maintaining a paper list [[awesome-self-supervised-learning]](https://github.com/jason718/awesome-self-supervised-learning) on Github. You are more than welcome to contribute and share :) 

Supervised Learning is awesome but limited. Un-/Self-supervised learning generalizes better and sometimes 
also works better (which is already true in some geometry tasks)!

## Acknowledgement
This work was supported in part by the National Science Foundation under Grant No. 1748387, the AWS Cloud Credits for Research Program, and GPUs donated by NVIDIA. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
