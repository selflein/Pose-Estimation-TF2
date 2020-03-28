import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pose_estimation.data_utils.coco import COCODataset
from pose_estimation.data_utils.dataset import generate_batch
from pose_estimation.models.mobilenet_pose import MobileNetPose
from pose_estimation.utils import extract_keypoints_from_heatmap


def evaluate(model_path, dataset, vis=False):
    # Load model
    # model = load_model(model_path)

    model = MobileNetPose()
    model.build(input_shape=(None, 256, 192, 3))
    model.load_weights('models/without_skip_new.hdf5')

    # Load validation dataset
    dataset = COCODataset()

    # Iterate over validation dataset
    results = []
    for item in tqdm(dataset.load_val_data_with_annot()):
        joints = np.array(item['joints'], np.float32).reshape(-1, 3)
        bbox = np.array(item['bbox'], np.float32)
        img, crop_info = generate_batch(tf.constant(item['imgpath']),
                                        tf.constant(bbox), tf.constant(joints),
                                        stage='test')

        # Get prediction
        heatmap = model.predict_on_batch(img[None, :])
        x, y, _ = extract_keypoints_from_heatmap(heatmap)

        # Rescale prediction to original image
        x = x / heatmap.shape[2] * (crop_info[2] - crop_info[0]) + crop_info[0]
        y = y / heatmap.shape[1] * (crop_info[3] - crop_info[1]) + crop_info[1]

        # Set visibility to one according to http://cocodataset.org/#format-results
        v = tf.ones_like(x)
        result = tf.concat([x, y, v], 0).numpy().transpose()

        if vis:
            print(result.shape)
            image = plt.imread(item['imgpath'])
            vis_image = dataset.vis_keypoints(image, result.transpose())
            plt.imshow(vis_image)
            plt.show()
            return

        results.append({'image_id': item['image_id'],
                        'category_id': 1,
                        'keypoints': result.reshape(-1).round().tolist(),
                        'score': float(tf.reduce_max(heatmap).numpy())})

    with Path('tmp/out.json').open('w') as file:
        json.dump(results, file)

    cocoGt = COCO(dataset.val_annot_path)
    cocoDt = cocoGt.loadRes('out.json')

    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
