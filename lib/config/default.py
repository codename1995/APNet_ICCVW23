
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = -1
_C.SAVE_WEIGHTS = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'apnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.PPNET_PRETRAINED = ''
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 1
_C.MODEL.FUSION = "ADDITION"  # ADDITION, CONCAT, ATTENTION, DEEPFUSION
_C.MODEL.SRC_PTS = False

_C.MODEL.PARTIAL_CONV = False
_C.MODEL.IMG_BLUR = False

_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.K_N = 16
_C.MODEL.NUM_LAYERS = 4
_C.MODEL.SUB_SAMPLING_RATIO = [4, 4, 4, 4]
_C.MODEL.D_OUT = [16, 64, 128, 256]

_C.MODEL.OCR = CN()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [1]

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'sensaturban'
_C.DATASET.NUM_CLASSES = 13
_C.DATASET.TRAIN_SET = 'list/sensaturban/tra_list_V4_Set1.txt'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/sensaturban/val_list_V4_Set1.txt'
_C.DATASET.NUM_POINTS = 4096
_C.DATASET.RANDOM_POINTS = True
_C.DATASET.PIXEL_SIZE = 0.04
_C.DATASET.THRESHOLD_POSSIBILITY = 0.1 # Only used for patch-based random sampling
_C.DATASET.VISUALIZE_EACH_SAMPLE = False
_C.DATASET.SAMPLES_TRA = -1
_C.DATASET.SAMPLES_VAL = -1

# training
_C.TRAIN = CN()

_C.TRAIN.FREEZE_LAYERS = ''
_C.TRAIN.FREEZE_EPOCHS = -1
_C.TRAIN.NONBACKBONE_KEYWORDS = []
_C.TRAIN.NONBACKBONE_MULT = 10

_C.TRAIN.IMAGE_SIZE = [512, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048 # VJ: What is it?
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.LR_MIN_SCALAR = 0.0001
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0
_C.TRAIN.EPOCH_ITERS = 500
_C.TRAIN.SAMPLES_PER_EPOCH = 800

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 8
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0
_C.TEST.SAMPLES_PER_EPOCH = 750

_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]

_C.TEST.OUTPUT_INDEX = -1

# Tricks
_C.TRICK = CN()
_C.TRICK.POINT_ROTATION = ""
_C.TRICK.POINT_SCALE_ANISOTROPIC = False
_C.TRICK.POINT_SCALE = [1., 1.]
_C.TRICK.POINT_SYMMETRIES = [False, False, False]
_C.TRICK.POINT_NOISE = 0.
_C.TRICK.POINT_NOISE_MAX = 0.
_C.TRICK.CITY_WISE_SUB = False
_C.TRICK.COLOR_DROPPING = 0.
_C.TRICK.COLOR_AUTO_CONTRAST = 0.

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

