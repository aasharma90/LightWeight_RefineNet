"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# general libs
import argparse
import logging
import os
import random
import re
import sys
import time

# misc
import cv2
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

# custom libs
from miou_utils import compute_iu, fast_cm
from util import *
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set the below parameters for testing
#################################################################################
# DATASET PARAMETERS
DATA_NAME = 'cityscapes' 
RUN_NAME  = 'run_20190219'
TEST_DIR  = '/disk1/aashishsharma/Datasets/CityScapes_Dataset/'
TEST_LIST = './data/cityscapes/val.lst'
NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD
NUM_WORKERS  = 16
NUM_CLASSES  = 19
# ENCODER PARAMETERS
ENC = '101'            # Making it 101 by default (from 50 for NYU before)
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
SNAPSHOT_DIR = './ckpt/' + RUN_NAME
CKPT_PATH    = './ckpt/' + RUN_NAME + '/checkpoint.pth.tar'

# FOR VISUALS
SAVE_VISUALS = True    # Set this option to False for not saving visuals
SAVE_VISUALS_DIR = './outputs/' + RUN_NAME + '/' + DATA_NAME
#################################################################################
# Set the above parameters for testing

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Testing")

    # Dataset
    parser.add_argument("--data-name", type=str, default=DATA_NAME,
                        help="Name of the dataset to be used")
    parser.add_argument("--test-dir", type=str, default=TEST_DIR,
                        help="Path to the test set directory.")
    parser.add_argument("--test-list", type=str, default=TEST_LIST,
                        help="Path to the test set list.")
    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS,
                        help="Normalisation parameters [scale, mean, std],")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of output classes for each task.")

    # Encoder
    parser.add_argument("--enc", type=str, default=ENC,
                        help="Encoder net type.")
    parser.add_argument("--enc-pretrained", type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Path to directory for storing checkpoints.")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH,
                        help="Path to the checkpoint file.")

    # For visuals
    parser.add_argument("--save-visuals", type=bool, default=SAVE_VISUALS,
                        help="whether to save visuals (predicted labels?)")
    parser.add_argument("--save-visuals-dir", type=str, default=SAVE_VISUALS_DIR,
                        help="where the visuals to be saved?")

    # Optimisers
    return parser.parse_args()

def create_segmenter(
    net, pretrained, num_classes
    ):
    """Create Encoder; for now only ResNet [50,101,152]"""
    from models.resnet import rf_lw50, rf_lw101, rf_lw152
    if str(net) == '50':
        return rf_lw50(num_classes, imagenet=pretrained)
    elif str(net) == '101':
        return rf_lw101(num_classes, imagenet=pretrained)
    elif str(net) == '152':
        return rf_lw152(num_classes, imagenet=pretrained)
    else:
        raise ValueError("{} is not supported".format(str(net)))

def create_loaders(
    test_dir, test_list, normalise_params, num_workers
    ):
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    if args.data_name == 'nyu':
        from datasets import NYUDataset as Dataset
    elif args.data_name == 'cityscapes':
        from datasets import CSDataset as Dataset
    from datasets import ToTensor, Normalise

    composed_test = transforms.Compose([Normalise(*normalise_params),
                                    ToTensor()])
    ## Test Set ##
    testset = Dataset(data_file=test_list,
                     data_dir=test_dir,
                     transform_trn=None,
                     transform_val=composed_test)

    logger.info(" Created test set with {} examples".format(len(testset)))
    ## Test Loader ##
    test_loader = DataLoader(testset,
                             batch_size=1, 
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return test_loader

def load_ckpt(
    ckpt_path, ckpt_dict
    ):
    best_val = epoch_start = 0
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(ckpt_path)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k])
        best_val = ckpt.get('best_val', 0)
        epoch_start = ckpt.get('epoch_start', 0)
        logger.info(" Loaded checkpoint at {} with best_val of {:.4f} at epoch {}".
            format(
                ckpt_path, best_val, epoch_start
            ))
    return best_val, epoch_start

def test(
    segmenter, test_loader, num_classes=-1
    ):
    """test segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      test_loader (DataLoader) : test data iterator
      num_classes (int) : number of classes to consider

    Returns:
      Mean IoU (float)
    """
    test_loader.dataset.set_stage('val') # 'val' would suffice
    segmenter.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    logger.info(" Testing begins.")
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            start  = time.time()
            input  = sample['image']
            target = sample['mask']
            input_var = torch.autograd.Variable(input).float().cuda()
            # Compute output
            output = segmenter(input_var)
            output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                target.size()[1:][::-1],
                                interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
            # Compute IoU
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes # Ignore every class index larger than the number of classes
            cm += fast_cm(output[gt_idx], gt[gt_idx], num_classes)

            # logger.info('Testing: [{}/{}]\t'
            #             'Mean IoU: {:.3f}'.format(
            #              i, len(test_loader),
            #              compute_iu(cm).mean()
            #              ))

            if args.save_visuals and args.save_visuals_dir is not None:
                with open(args.test_list, 'rb') as f:
                    testlist = f.readlines()
                test_list = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split('\t'), testlist)]
                img_path  = os.path.join(args.test_dir, test_list[i][0])
                img_name  = img_path.split('/')[-1]
                if not os.path.exists(args.save_visuals_dir):
                    os.makedirs(args.save_visuals_dir)
                if args.data_name == 'cityscapes':
                    palette   = get_palette_cityscapes()
                    output_im = Image.fromarray(np.array(output))
                    output_im.putpalette(palette)
                    output_im.save(args.save_visuals_dir+'/'+img_name)
                else:
                    # Not implemented for other datasets for now
                    # Kindly set SAVE_VISUALS=False, or implement it for your dataset
                    raise NotImplementedError

    ious = compute_iu(cm)
    logger.info(" IoUs: {}".format(ious))
    miou = np.mean(ious)
    logger.info(' Mean IoU: {:.4f}'.format(miou))
    return miou

def main():
    global args, logger
    args = get_arguments()
    logger = logging.getLogger(__name__)
    ## Generate Segmenter ##
    segmenter = nn.DataParallel(
        create_segmenter(args.enc, args.enc_pretrained, args.num_classes)
        ).cuda()
    logger.info(" Loaded Segmenter {}, ImageNet-Pre-Trained={}"
                .format(args.enc, args.enc_pretrained))
    ## Load the checkpoint to test the model at ##
    best_val, epoch_best_val = load_ckpt(args.ckpt_path, {'segmenter' : segmenter})

    start = time.time()
    torch.cuda.empty_cache()
    ## Create test dataloaders ##
    test_loader = create_loaders(args.test_dir,
                                 args.test_list,
                                 args.normalise_params,
                                 args.num_workers)
    mIoU_score  = test(segmenter, test_loader, num_classes=args.num_classes)

    logger.info("Testing finished. Mean IoU score is {:.4f}".format(mIoU_score))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
