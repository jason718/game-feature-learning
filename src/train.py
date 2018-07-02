import pdb
import yaml
import argparse
import torch
import torchvision

from dataset import GameDataset
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/alexnet.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    with open(opts.config, 'r') as f_in:
        cfg = yaml.load(f_in)
    print(cfg)

    ## Get dataset
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
    data_syn = GameDataset(cfg, normalize)
    dataloader_syn = torch.utils.data.DataLoader(data_syn, num_workers=cfg['NUM_WORKER'],
                batch_size=cfg['BATCH_SIZE'], shuffle=True)
    dataiterator_syn = iter(dataloader_syn)
    print('==> Number of synthetic training images: %d.' % len(data_syn))

    if cfg['USE_DA']:
        data_real = torchvision.datasets.ImageFolder(cfg['REAL_DIR'],
                    torchvision.transforms.Compose([
                        torchvision.transforms.Resize((cfg['CROP_SIZE'],cfg['CROP_SIZE'])),
                        torchvision.transforms.ToTensor(),
                        normalize]))
        dataloader_real = torch.utils.data.DataLoader(data_real, num_workers=cfg['NUM_WORKER'],
                        batch_size=cfg['BATCH_SIZE'], shuffle=True)
        dataiterator_real = iter(dataloader_real)
        print('==> Number of real training images: %d.' % len(data_real))

    ## Get model
    model = Model()
    model.initialize(cfg)

    ## Training
    for epoch in range(cfg['N_EPOCH']):
        # Get input data, doing this since they of diff. size
        inputs = {}
        while True:
            try:
                inputs['syn'] = next(dataiterator_syn)
            except StopIteration:
                dataiterator_syn=iter(dataloader_syn)
                break
            if cfg['USE_DA']:
                try:
                    inputs['real'] = next(dataiterator_real)
                except StopIteration:
                    dataiterator_real=iter(dataloader_real)
                    inputs['real'] = next(dataiterator_real)

            ## update
            model.set_input(inputs)
            model.optimize()

            ## logging
            model.print_n_log_losses(epoch)
            model.visualize_pred(epoch)

        ## saving
        if (epoch+1) % cfg['SAVE_FREQ'] == 0:
            model.save_networks(epoch)

    print('==> Finished Training')
    del dataiterator_syn
    del dataiterator_real
