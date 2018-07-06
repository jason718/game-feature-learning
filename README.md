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

## Change Log
1. ~~First update: alexnet training code.~~[Done!]
2. Second update: trained model ~~and feature-learning evaluation module.~~
3. Third update: vgg training code and tasks evaluation module


## Prerequisites
- Pytorch-0.4 (some evaluation code borrowed from other places requiring Caffe)
- Python2 (One evaluation code requiring Python3)
- NVIDIA GPU + CUDA CuDNN (Sorry no CPU version)

## Getting Started
### Installation
- Install PyTorch 0.4 and torchvision from http://pytorch.org
- Python packages
    - torchvision
    - numpy
    - opencv
    - tensorflow (necesary for the use of tensorboard. Will change it to tensorboardX)
    - scikit-image
- Clone this repo:
```bash
git clone https://github.com/jason718/game-feature-learning
cd game-feature-learning
```
- Change the config files under configs.

### Pre-trained models:
- Caffemodel(Caffenet): Coming in second update
- Pytorch model: Coming in second update

Since I greatly changed the code structure, I am retraining using the new code to reproduce the paper results.

### Dataset:
   - SUNCG: Download the SUNCG images from [suncg website](http://suncg.cs.princeton.edu/).
        And make sure to put the files as the following structure:
        ```
            suncg_image
            ├── depth
               ├── room_id1
               ├── ...
            ├── normal
               ├── room_id1
               ├── ...
            ├── edge
               ├── room_id1
               ├── ...
            └── lab
                ├── room_id1
                ├── ...
        ```

   - SceneNet: Download the SceneNet images from [scenenet website](https://robotvault.bitbucket.io/scenenet-rgbd.html).
         And make sure to put the files as the following structure:
        ```
            scenenet_image
            └── train
               ├── 1
               ├── 2
               ├── ...
        ```
- Dataset For Domain Adaptation:
    - Places-365: Download the Places images from [places website](http://places2.csail.mit.edu/).
    - Or you can choose other dataset for DA such ImageNet...

### Train/Test
- Train a model:
```bash
sh ./scripts/train.sh
```
- Evaluate on feature learning
  - Read each README.md under folder "eval-3rd-party"
  
- Evaluate on three tasks
    - Coming in Third Update


## Useful Resources
There are lots of awesome papers studying self-supervision for various tasks such as Image/Video Representation learning,
Reinforcement learning, and Robotics. I am maintaining a paper list [[awesome-self-supervised-learning]](https://github.com/jason718/awesome-self-supervised-learning) on Github. You are more than welcome to contribute and share :) 

Supervised Learning is awesome but limited. Un-/Self-supervised learning generalizes better and sometimes 
also works better (which is already true in some geometry tasks)!

## Acknowledgement
This work was supported in part by the National Science Foundation under Grant No. 1748387, the AWS Cloud Credits for Research Program, and GPUs donated by NVIDIA. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
