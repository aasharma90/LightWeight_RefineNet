import numpy as np

# DATASET PARAMETERS
DATA_NAME = 'cityscapes' # Set to 'cityscapes' (from 'nyu')
RUN_NAME  = 'run_20190219'
TRAIN_DIR = '/disk1/aashishsharma/Datasets/CityScapes_Dataset/'
VAL_DIR   = TRAIN_DIR
TRAIN_LIST = ['./data/cityscapes/train.lst'] * 3
VAL_LIST = ['./data/cityscapes/val.lst'] * 3
SHORTER_SIDE = [1024] * 3 # Set to 1024 for CityScapes (from 350 for NYU)
CROP_SIZE = [800] * 3     # Set to 800 for CityScapes (from 500 for NYU)
NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD
BATCH_SIZE = [6] * 3
NUM_WORKERS = 16
NUM_CLASSES = [19] * 3   # Set to 19 for CityScapes (from 40 for NYU)
LOW_SCALE = [0.5] * 3
HIGH_SCALE = [2.0] * 3
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '101'            # Making it 101 by default (from 50 for NYU before)
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 3
NUM_SEGM_EPOCHS = [100] * 3
PRINT_EVERY = 10
RANDOM_SEED = 42
SNAPSHOT_DIR = './ckpt/' + RUN_NAME
CKPT_PATH = './ckpt/' + RUN_NAME + '/checkpoint.pth.tar'
VAL_EVERY = [5] * 3 # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [5e-3, 2.5e-3, 1e-3]
MOM_ENC = [0.9] * 3 # TO FREEZE, PUT 0
MOM_DEC = [0.9] * 3
WD_ENC = [1e-5] * 3 # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 3
OPTIM_DEC = 'sgd'
