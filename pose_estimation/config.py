import os
import os.path as osp
import sys
import numpy as np


class Config:
    ## dataset
    dataset = 'COCO'  # 'COCO', 'PoseTrack', 'MPII'
    testset = 'val'  # train, test, val (there is no validation set for MPII)

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dump_dir = osp.join(output_dir, 'model_dump', dataset)
    vis_dir = osp.join(output_dir, 'vis', dataset)
    log_dir = osp.join(output_dir, 'log', dataset)
    result_dir = osp.join(output_dir, 'result', dataset)

    ## model setting
    backbone = 'resnet50'  # 'resnet50', 'resnet101', 'resnet152'
    init_model = osp.join(data_dir, 'imagenet_weights',
                          'resnet_v1_' + backbone[6:] + '.ckpt')

    ## input, output
    input_shape = (256, 192)  # (256,192), (384,288)
    output_shape = (input_shape[0] // 4, input_shape[1] // 4)
    if output_shape[0] == 64:
        sigma = 2
    elif output_shape[0] == 96:
        sigma = 3
    pixel_means = np.array([[[123.68, 116.78, 103.94]]])

    ## training config
    lr_dec_epoch = [90, 120]
    end_epoch = 140
    lr = 5e-4
    lr_dec_factor = 10
    optimizer = 'adam'
    weight_decay = 1e-5
    bn_train = True
    batch_size = 32
    scale_factor = 0.3
    rotation_factor = 40

    ## testing config
    useGTbbox = False
    flip_test = True
    oks_nms_thr = 0.9
    score_thr = 0.2
    test_batch_size = 32

    ## others
    multi_thread_enable = True
    num_thread = 10
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    display = 1

    ## helper functions
    def get_lr(self, epoch):
        for e in self.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.lr_dec_epoch[-1]:
            i = self.lr_dec_epoch.index(e)
            return self.lr / (self.lr_dec_factor ** i)
        else:
            return self.lr / (self.lr_dec_factor ** len(self.lr_dec_epoch))

    def normalize_input(self, img):
        return img - self.pixel_means

    def denormalize_input(self, img):
        return img + self.pixel_means

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
