import os
import random

import cv2
import numpy as np
import tensorflow as tf

from pose_estimation.config import cfg
from pose_estimation.data_utils.coco import COCODataset
from pose_estimation.data_utils.transforms import (affine_transform,
                                                   get_affine_transform)


def generate_batch(image_path: tf.Tensor, bbox: tf.Tensor, joints: tf.Tensor, stage='train'):
    image_path = image_path.numpy().decode("utf-8")
    img = cv2.imread(os.path.join(cfg.img_path, image_path),
                     cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        print('cannot read ' + os.path.join(cfg.img_path, image_path))
        assert 0

    x, y, w, h = bbox.numpy()
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
        scale = scale * np.clip(np.random.randn() * cfg.scale_factor + 1,
                                1 - cfg.scale_factor, 1 + cfg.scale_factor)
        rotation = np.clip(np.random.randn() * cfg.rotation_factor,
                           -cfg.rotation_factor * 2, cfg.rotation_factor * 2) \
            if random.random() <= 0.6 else 0
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
        # cropped_img = cropped_img[:,:, ::-1]
        cropped_img = cfg.normalize_input(cropped_img)

        for i in range(cfg.num_kps):
            if joints[i, 2] > 0:
                joints[i, :2] = affine_transform(joints[i, :2], trans)
                joints[i, 2] *= ((joints[i, 0] >= 0) & (
                        joints[i, 0] < cfg.input_shape[1]) & (
                                         joints[i, 1] >= 0) & (
                                         joints[i, 1] < cfg.input_shape[0]))
        target_coord = joints[:, :2]
        target_valid = joints[:, 2]

        target = render_gaussian_heatmap(target_coord, cfg.output_shape, cfg.sigma)
        return cropped_img, target

    else:
        trans = get_affine_transform(center, scale, rotation,
                                     (cfg.input_shape[1], cfg.input_shape[0]))
        cropped_img = cv2.warpAffine(img, trans,
                                     (cfg.input_shape[1], cfg.input_shape[0]),
                                     flags=cv2.INTER_LINEAR)
        # cropped_img = cropped_img[:,:, ::-1]
        cropped_img = cfg.normalize_input(cropped_img)

        crop_info = np.asarray(
            [center[0] - scale[0] * 0.5, center[1] - scale[1] * 0.5,
             center[0] + scale[0] * 0.5, center[1] + scale[1] * 0.5])

        return [cropped_img, crop_info]


def render_gaussian_heatmap(coord, output_shape, sigma):
    x = list(range(output_shape[1]))
    y = list(range(output_shape[0]))
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.cast(xx, tf.float32), (1, *output_shape, 1))
    yy = tf.reshape(tf.cast(yy, tf.float32), (1, *output_shape, 1))

    x = tf.floor(tf.reshape(coord[:, :, 0], [-1, 1, 1, cfg.num_kps])
                 / cfg.input_shape[1] * output_shape[1] + 0.5)
    y = tf.floor(tf.reshape(coord[:, :, 1], [-1, 1, 1, cfg.num_kps])
                 / cfg.input_shape[0] * output_shape[0] + 0.5)

    return tf.exp(-(((xx - x) / tf.cast(sigma, tf.float32)) ** 2)
                  / tf.cast(2, tf.float32)
                  - (((yy - y) / tf.cast(sigma, tf.float32)) ** 2)
                  / tf.cast(2, tf.float32))


def dataset_generator(samples):
    for s in samples:
        bbox = np.array(s['bbox'], dtype=np.float32)
        joints = np.array(s['joints'], dtype=np.float32).reshape(-1, 3)
        yield s['imgpath'], bbox, joints


def get_dataloaders(batch_size, buffer, num_workers):
    samples = COCODataset(cfg.data_dir).load_train_data()
    dataset_gen = dataset_generator(samples)
    dataset = tf.data.Dataset.from_generator(dataset_gen,
                                             output_types=(tf.string, tf.float32, tf.float32))

    map_func = lambda x: tf.py_function(generate_batch, x, (tf.float32, tf.float32))
    dataset = (dataset.map(map_func, num_workers)
                      .batch(batch_size)
                      .repeat()
                      .prefetch(buffer))

    return dataset


if __name__ == '__main__':
    ds = get_dataloaders(4, 2, 2)
    batch = next(iter(ds))
    print(batch)
