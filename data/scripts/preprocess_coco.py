import argparse
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from pycocotools.coco import COCO


def preprocess_coco(coco_folder, output_file):
    data_dir = Path('/storage/remote/atcremers51/w0020/coco/images')
    subset = 'val2017'
    annotations_file = data_dir / 'annotations/person_keypoints_{}.json'.format(subset)
    imgs_path = data_dir / subset

    coco = COCO(annotations_file)
    catIds = coco.getCatIds(supNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)

    with tf.python_io.TFRecordWriter(output_folder) as tfrecord_writer:
        for img_dict in tqdm(imgIds):
            img = (imgs_path / img_dict['file_name']).open('rb').read()
            annIds = coco.getAnnIds(imgIds=img_dict['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            example = coco_to_tf_example(img_dict, img, anns)
            tfrecord_writer.write(example.SerializeToString())


def coco_to_tf_example(img_dict, img, annotations):
    height, width = img_dict['height'], img_dict['width']

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    labels, num_keypoints, keypoints = [], [], []
    for ann in annotations:
        bbox = ann['bbox']
        xmins.append(bbox[0])
        xmaxs.append(bbox[0] + bbox[2])
        ymins.append(bbox[1])
        ymaxs.append(bbox[1] + bbox[3])

        num_keypoints.append(ann['num_keypoints'])
        keypoints.extend(ann['keypoints'])

        labels.append(ann['category_id'])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/keypoints/num_keypoints': tf.train.Feature(int64_list=tf.train.Int64List(valuen=num_keypoints)),
        'image/object/keypoints/keypoints': tf.train.Feature(float_list=tf.train.FloatList(value=keypoints)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
    }))
    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_folder')
    parser.add_argument('--output_folder')
    args = parser.parse_args()

    Path(args.output_folder).mkdir(exist_ok=False, parents=True)
