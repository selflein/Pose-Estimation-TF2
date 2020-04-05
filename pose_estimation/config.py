import os
import os.path as osp
from pathlib import Path

import numpy as np
import imgaug.augmenters as iaa


class Config:
    ## dataset
    dataset = 'COCO'  # 'COCO', 'PoseTrack', 'MPII'
    testset = 'val'  # train, test, val (there is no validation set for MPII)

    ## directory
    cur_dir = Path(osp.dirname(os.path.abspath(__file__)))
    root_dir = cur_dir.parent
    data_dir = root_dir / 'data'
    model_dump_dir = root_dir / 'models' / 'checkpoints'

    ## input, output
    input_shape = (256, 192, 3)  # (256,192), (384,288)
    output_shape = (input_shape[0] // 4, input_shape[1] // 4)
    if output_shape[0] == 64:
        sigma = 2
    elif output_shape[0] == 96:
        sigma = 3
    ## training config
    end_epoch = 140
    lr = 3e-3
    lr_dec_rate = 0.95
    optimizer = 'adam'
    weight_decay = 0.00004
    bn_train = True
    batch_size = 32

    # Scale augmentation (+- in percent)
    scale_factor = 0.25
    # Rotation augmentation (+- in degrees)
    rotation_factor = 30

    # Image augmentations
    img_augmentations = iaa.Sequential([
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.MotionBlur((3, 8)),
        iaa.ChannelShuffle(0.1),
        iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
    ], random_order=True)

    ## testing config
    useGTbbox = False
    flip_test = True
    oks_nms_thr = 0.9
    score_thr = 0.2
    test_batch_size = 64

    ## others
    multi_thread_enable = True
    num_thread = 8
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    display = 1

    ## helper functions
    def get_lr(self, epoch):
        return cfg.lr * (cfg.lr_dec_rate ** epoch)

    def normalize_input(self, img):
        return img / 127.5 - 1

    def denormalize_input(self, img):
        return (img + 1) * 127.5

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using /gpu:{}'.format(self.gpu_ids))


cfg = Config()

from pose_estimation.data_utils.coco import COCODataset

dbcfg = COCODataset
cfg.num_kps = dbcfg.num_kps
cfg.kps_names = dbcfg.kps_names
cfg.kps_lines = dbcfg.kps_lines
cfg.kps_symmetry = dbcfg.kps_symmetry
cfg.vis_keypoints = dbcfg.vis_keypoints

