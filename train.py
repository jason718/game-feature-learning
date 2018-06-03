import argparse

parser = argparse.ArgumentParser(description='Training game feature learner')
# Archi
parser.add_argument('--crop', type=int, default=224, help='The input of the network')
parser.add_argument('--arch', type=str, default='alexnet', help='Which network to use')
# Domain Adaptation
parser.add_argument('--poolsize', type=int, default=100, help='Discriminator pool')
parser.add_argument('--gan_weight', type=float, default=0.05, help='Weight for the gan loss')
parser.add_argument('--hasDA', action='store_true',  help='Use Domain Adaptation or not')
parser.add_argument('--feat_layer', type=str, default='conv5', help='out put features at this layer for Domain Adaptation')
# Training
parser.add_argument('--bs', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--vis_point', type=int, default=300, help='vis. the results every 20 iter. Default=20')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--log_path', type=str, default='./tmp/logs', help='Where to save log files')
parser.add_argument('--model_path', type=str, default='./model_vgg/vgg_gan_c4_netH_ep14', help='Where to save log files')
parser.add_argument('--outf', type=str, default='./tmp', help='output path')
# Dataset
parser.add_argument('--real_dataset', type=str, default='./dataset/places', help='real-world dataset path')
parser.add_argument('--suncg', type=str, default='/home/SSD5/jason-data/SUNCG/suncg_image256/', help='suncg path')
parser.add_argument('--scenenet_p', type=str, default=None, help='scenenet path')
# Optimization
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.0002')
parser.add_argument('--lrB', type=float, default=0.0002, help='Learning Rate. Default=0.0002')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum. Default=0.9')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay. Default=0.0005')
parser.add_argument('--stepsize', type=float, default=20, help='weight_decay. Default=0.0005')
# Others..
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--pretrained', action='store_true',  help='whether load pre-trained or not')


opt = parser.parse_args()
print(opt)

# Use the mean from ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])

# Setting up dataset for training
if opt.data_syn == "suncg":
    train_syn = GameDataset(txt_file='dataset/suncg_lst/suncg_train_lst.txt', suncg_dir=opt.suncg,
        transform=transforms.Compose([RandomCrop(opt.crop), ToTensor()]), norm = normalize)
elif opt.data_syn == "scenenet":
     train_syn = GameDataset(txt_file='dataset/suncg_lst/suncg_train_lst.txt', suncg_dir=opt.suncg,
        transform=transforms.Compose([RandomCrop(opt.crop), ToTensor()]), norm = normalize)
elif opt.data_syn == "suncg+scenenet-1.5m":
    train_syn = GameDataset(txt_file='dataset/train_1m_shuffle.txt',
        suncg_dir=opt.suncg, scenenet_dir=opt.scenenet_p,
        transform=transforms.Compose([RandomCrop(opt.crop),
        ToTensor()]), norm = normalize)
elif opt.data_syn == "suncg+scenent":
    train_syn = GameDataset(txt_file='dataset/train_1m_shuffle.txt',
        suncg_dir=opt.suncg, scenenet_dir=opt.scenenet_p,
        transform=transforms.Compose([RandomCrop(opt.crop),
        ToTensor()]), norm = normalize)
else:
    raise ValueError("Not Supported Synthetic dataset!")
print("==> Using %s dataset for training" % opt.data_syn)

# Seeting up real-world dataset for domain adaptation is there is any
if opt.data_real is None:
    hasDA = False
    print("==> Not using Domain Adaptation")
else:
    hasDA = True
    if opt.data_real == 'places':
        train_real = datasets.ImageFolder(opt.real_dataset,
                        transforms.Compose([transforms.Scale((256,256)),
                                        transforms.RandomCrop((opt.crop, opt.crop)),
                                        transforms.ToTensor(), normalize]))
    elif opt.data_real == "ImageNet":
        pass
    elif opt.data_real == "places+ImageNet":
        pass
    else:
        raise ValueError("Not Supported Real-world dataset!")
    print('==> Using %s dataset for Domain Adaptation' % opt.data_real)

# Begiin Training
if hasDA:
    train_DA(train_syn, train_real, opt)
else:
    train(train_syn, opt)

print("==> Training Done!")
