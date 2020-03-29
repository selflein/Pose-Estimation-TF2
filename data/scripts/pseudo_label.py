import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Installation instructions:
# https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--keypoint_threshold', default=0.15)
    parser.add_argument('--detections_threshold', default=0.95)
    args = parser.parse_args()

    keypoint_threshold = args.keypoint_threshold
    detections_threshold = args.detections_threshold
    detectron2.utils.visualizer._KEYPOINT_THRESHOLD = keypoint_threshold

    input_folder = Path(args.input_folder)
    assert input_folder.is_dir(), 'Given input folder does not exist'
    if not args.vis:
        if not args.output_folder:
            raise ValueError('Requires output path argument when vis argument '
                             'not specified!')
        output_folder = Path(args.output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        img_folder = output_folder / 'images'
        img_folder.mkdir()

    cfg = get_cfg()
    model_config = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    predictor = DefaultPredictor(cfg)

    annotations = []
    images = []

    instance_id = 0
    for img_id, image_path in enumerate(tqdm(list(input_folder.iterdir()))):
        try:
            im = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            h, w, c = im.shape
        except Exception as e:
            tqdm.write(str(e))
            continue
        img_name = f'{img_id:06d}.jpg'

        img_dict = {"license": 0, "file_name": img_name, "coco_url": "",
                    "height": h, "width": w, "date_captured": "",
                    "flickr_url": "", "id": img_id}
        images.append(img_dict)

        if c == 1:
            tqdm.write('Got greyscale image. Repeating channel axis.')
            im = im[:, :, np.newaxis].repeat(3, axis=-1)

        outputs = predictor(im)

        if args.vis:
            v = Visualizer(im,
                           MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                           scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.imshow(v.get_image())
            plt.show()
            continue

        cv2.imwrite(str(img_folder / img_name), im)

        fields = outputs['instances'].get_fields()
        for points, score, box, area in zip(fields['pred_keypoints'].cpu().numpy(),
                                            fields['scores'].cpu(),
                                            fields['pred_boxes'].tensor.cpu().numpy(),
                                            fields['pred_boxes'].area()):
            if score < detections_threshold:
                continue

            valid_keypoints = points[:, 2] > keypoint_threshold
            num_keypoints = np.sum(valid_keypoints)
            points[:, 2] = valid_keypoints.astype(np.float32)
            annotation_dict = {
                "segmentation": [],
                "num_keypoints": int(num_keypoints),
                "area": float(area.item()),
                "iscrowd": 0,
                "keypoints": points.reshape(-1).round(2).tolist(),
                "image_id": img_id,
                "bbox": box.round(2).tolist(),
                "category_id": 1,
                "id": instance_id
            }
            annotations.append(annotation_dict)
            instance_id += 1

    output_dict = {
        "info": {"description": "",
                 "url": "",
                 "version": "1.0",
                 "year": 2020,
                 "contributor": "",
                 "date_created": ""},
        "licenses": [{"url": "", "id": 0, "name": ""}, ],
        "images": images,
        "annotations": annotations
    }
    with (output_folder / 'annotations.json').open('w') as out:
        json.dump(output_dict, out)
