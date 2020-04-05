import random
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from pose_estimation.config import cfg
from pose_estimation.data_utils.coco import COCODataset
from pose_estimation.data_utils.mpii import MPIIDataset
from pose_estimation.data_utils.push_up_dataset import PushUpDataset

from pose_estimation.data_utils.transforms import (affine_transform,
                                                   get_affine_transform)


def generate_batch(image_path: tf.Tensor, bbox: tf.Tensor, joints: tf.Tensor, stage='train', scales=(1, 2, 4)):
    image_path = image_path.numpy().decode("utf-8")
    bbox = bbox.numpy()
    joints = joints.numpy()

    img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    x, y, w, h = bbox
    aspect_ratio = cfg.input_shape[1] / cfg.input_shape[0]
    center = np.array([x + w * 0.5, y + h * 0.5])
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h]) * 1.25
    rotation = 0

    if stage == 'train':
        # data augmentation
        scale *= np.clip(np.random.randn() * cfg.scale_factor + 1,
                         1 - cfg.scale_factor, 1 + cfg.scale_factor)

        rotation = np.clip(np.random.randn() * cfg.rotation_factor,
                           -cfg.rotation_factor * 2, cfg.rotation_factor * 2)

        # Flipping augmentation
        if random.random() <= 0.5:
            img = img[:, ::-1, :]
            center[0] = img.shape[1] - 1 - center[0]
            joints[:, 0] = img.shape[1] - 1 - joints[:, 0]
            for (q, w) in cfg.kps_symmetry:
                joints_q, joints_w = joints[q, :].copy(), joints[w, :].copy()
                joints[w, :], joints[q, :] = joints_q, joints_w

    trans = get_affine_transform(center, scale, rotation,
                                 (cfg.input_shape[1], cfg.input_shape[0]))
    cropped_img = cv2.warpAffine(img, trans,
                                 (cfg.input_shape[1], cfg.input_shape[0]),
                                 flags=cv2.INTER_LINEAR)

    if stage == 'train':
        if random.random() < 0.7:
            cropped_img = cfg.img_augmentations(image=cropped_img)

    for i in range(joints.shape[0]):
        if joints[i, 2] >= 1:
            joints[i, :2] = affine_transform(joints[i, :2], trans)
            joints[i, 2] *= ((joints[i, 0] >= 0)
                             & (joints[i, 0] < cfg.input_shape[1])
                             & (joints[i, 1] >= 0)
                             & (joints[i, 1] < cfg.input_shape[0]))

    # BGR -> RGB
    cropped_img = cropped_img[:, :, ::-1]
    cropped_img = cfg.normalize_input(cropped_img)

    if stage == 'test':
        # (x, y) coordinate of upper left and bottom right corner of crop with
        # origin in top left corner of image
        crop_info = np.asarray(
            [center[0] - scale[0] * 0.5, center[1] - scale[1] * 0.5,
             center[0] + scale[0] * 0.5, center[1] + scale[1] * 0.5])
        return cropped_img, crop_info

    target_coord = joints[:, :2]
    target_valid = (joints[:, 2] >= 1).astype(np.float32)

    target = render_gaussian_heatmap(target_coord, cfg.output_shape, cfg.sigma)
    target = target_valid * target
    targets = [target[::scale, ::scale, :] for scale in scales]
    return (cropped_img, *targets)


def render_gaussian_heatmap(coord, output_shape, sigma):
    x = tf.range(output_shape[1])
    y = tf.range(output_shape[0])
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.cast(xx, tf.float32), (*output_shape, 1))
    yy = tf.reshape(tf.cast(yy, tf.float32), (*output_shape, 1))

    x = tf.floor(coord[:, 0] / cfg.input_shape[1] * output_shape[1] + 0.5)
    y = tf.floor(coord[:, 1] / cfg.input_shape[0] * output_shape[0] + 0.5)

    heatmap = tf.exp(- (((xx - x) / tf.cast(sigma, tf.float32)) ** 2)
                     / tf.cast(2, tf.float32)
                     - (((yy - y) / tf.cast(sigma, tf.float32)) ** 2)
                     / tf.cast(2, tf.float32))
    return heatmap


def dataset_generator(samples):
    def dataset_gen():
        for s in samples:
            bbox = np.array(s['bbox'], dtype=np.float32)
            joints = np.array(s['joints'], dtype=np.float32).reshape(-1, 3)
            yield s['imgpath'], bbox, joints
    return dataset_gen


def get_dataloader(samples, batch_size, buffer, num_workers, split='train', scales=(1, 2, 4)):
    def map_func(image_path, bbox, joints):
        out = tf.py_function(
            generate_batch, [image_path, bbox, joints, split, scales],
            (tf.float32, *((tf.float32,) * len(scales)))
        )
        out[0].set_shape((256, 192, 3))
        for i, scale in enumerate(scales, 1):
            out[i].set_shape((64 // scale, 48 // scale, 17))
        return out[0], tuple(o for o in out[1:])

    dataset_gen = dataset_generator(samples)
    dataset = tf.data.Dataset.from_generator(
        dataset_gen,
        output_types=(tf.string, tf.float32, tf.float32),
        output_shapes=(None, (4,), (17, 3))
    )

    dataset = (dataset.map(map_func, num_workers)
                      .batch(batch_size)
                      .repeat()
                      .prefetch(buffer))
    return dataset


def get_dataloaders(dataset_name, batch_size, buffer, num_workers, scales=(1,)):
    name_to_dataset = {
        'COCO': COCODataset,
        'PushUp': PushUpDataset
    }
    dataset = name_to_dataset[dataset_name]()

    samples_train = dataset.load_train_data()
    dataset_train = get_dataloader(samples_train, batch_size, buffer, num_workers, scales=scales)

    samples_val = dataset.load_val_data_with_annot()
    dataset_val = get_dataloader(samples_val, batch_size, buffer, num_workers, scales=scales)

    return (dataset_train.shuffle(10),
            len(samples_train) // batch_size,
            dataset_val,
            len(samples_val) // batch_size)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ds = get_dataloaders('COCO', 4, 2, 1, scales=(1, 2, 4))[0]
    batch = next(iter(ds))
    print(batch[0].shape, batch[1][1].shape)
