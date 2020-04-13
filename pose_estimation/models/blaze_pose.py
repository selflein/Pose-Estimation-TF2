""" From https://github.com/dl-maxwang/blazeface-tensorflow """

import tensorflow as tf
from tensorflow.keras import layers


def build_backbone(input_dim: tuple):
    inputs = layers.Input(dtype=tf.float32, shape=input_dim, name='inputs')

    conv1 = layers.Conv2D(24, kernel_size=3, strides=2, padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = tf.nn.relu(conv1)
    bb1 = blaze_block(conv1, filters=24)
    bb1 = blaze_block(bb1, filters=24)
    bb1 = blaze_block(bb1, filters=48, stride=2)
    bb1 = blaze_block(bb1, filters=48)
    bb1 = blaze_block(bb1, filters=48)

    db1 = double_blaze_block(bb1, filters=96, mid_channels=24, stride=2)
    db1 = double_blaze_block(db1, filters=96, mid_channels=24)
    feature32by24 = double_blaze_block(db1, filters=96, mid_channels=24)

    db2 = double_blaze_block(feature32by24, filters=96, mid_channels=24, stride=2)
    db2 = double_blaze_block(db2, filters=96, mid_channels=24)
    db2 = double_blaze_block(db2, filters=96, mid_channels=24)
    feature16by12 = double_blaze_block(db2, filters=96, mid_channels=24)

    db3 = double_blaze_block(feature16by12, filters=96, mid_channels=24, stride=2)
    db3 = double_blaze_block(db3, filters=96, mid_channels=24)
    db3 = double_blaze_block(db3, filters=96, mid_channels=24)
    feature8by6 = double_blaze_block(db3, filters=96, mid_channels=24)

    return inputs, feature32by24, feature16by12, feature8by6


def blaze_block(x: tf.Tensor, filters, stride=1):
    inp_channels = x.get_shape()[3]
    assert stride in [1, 2]
    use_pool = stride > 1

    conv1 = layers.SeparableConv2D(filters=filters, kernel_size=(5, 5), strides=stride, padding='same')(x)
    bn2 = layers.BatchNormalization()(conv1)

    if use_pool:
        channel_padding_dim = filters - inp_channels
        shortcut = layers.MaxPooling2D(pool_size=stride, strides=stride, padding='same')(x)
        x = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [channel_padding_dim, 0]], mode='constant')
    return tf.nn.relu(bn2 + x)


def double_blaze_block(x: tf.Tensor, filters, mid_channels=None, stride=1):
    assert stride in [1, 2]
    inp_channels = x.get_shape()[3]
    mid_channels = mid_channels or inp_channels
    usepool = stride > 1

    conv1 = layers.SeparableConv2D(filters=mid_channels, kernel_size=5, strides=stride, padding='same')(x)
    bn1 = layers.BatchNormalization()(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = layers.SeparableConv2D(filters=filters, kernel_size=5, strides=1, padding='same')(relu1)
    bn2 = layers.BatchNormalization()(conv2)

    if usepool:
        channel_padding_dim = filters - inp_channels
        max_pool1 = layers.MaxPooling2D(pool_size=stride, strides=stride, padding='same')(x)
        x = tf.pad(max_pool1, [[0, 0], [0, 0], [0, 0], [channel_padding_dim, 0]], mode='CONSTANT')
    return tf.nn.relu(bn2 + x)


def build_model(input_shape=(256, 192, 3), num_keypoints=17):
    input, feature0, feature1, feature2 = build_backbone(input_shape)

    up_feat_1 = layers.UpSampling2D()(feature2)
    comb_feat_1 = tf.concat([up_feat_1, feature1], axis=-1)
    comb_feat_1 = layers.Conv2D(96, 3, 1, 'same')(comb_feat_1)
    comb_feat_1 = double_blaze_block(comb_feat_1, filters=96, mid_channels=24)
    comb_feat_1 = double_blaze_block(comb_feat_1, filters=96, mid_channels=24)
    # pred_1 = layers.Conv2D(num_keypoints, 1, 1, 'same')(comb_feat_1)

    up_feat_2 = layers.UpSampling2D()(comb_feat_1)
    comb_feat_2 = tf.concat([up_feat_2, feature0],  axis=-1)
    comb_feat_2 = layers.Conv2D(96, 3, 1, 'same')(comb_feat_2)
    comb_feat_2 = double_blaze_block(comb_feat_2, filters=96, mid_channels=24)
    comb_feat_2 = double_blaze_block(comb_feat_2, filters=96, mid_channels=24)
    pred_2 = layers.Conv2D(num_keypoints, 1, 1, 'same')(comb_feat_2)

    up_feat_3 = layers.UpSampling2D()(comb_feat_2)
    comb_feat_3 = double_blaze_block(up_feat_3, filters=96, mid_channels=24)
    comb_feat_3 = double_blaze_block(comb_feat_3, filters=96, mid_channels=24)
    pred_3 = layers.Conv2D(num_keypoints, 1, 1, 'same')(comb_feat_3)

    model = tf.keras.Model(inputs=input, outputs=(pred_3))
    return model


class BlazePose:

    def __init__(self, input_shape=(256, 192, 3), num_keypoints=17):
        super(BlazePose, self).__init__()
        self.model = build_model(input_shape, num_keypoints)

    def train(self):
        pass

    def predict(self):
        pass

    def load_weights(self):
        pass


if __name__ == '__main__':
    shape = (256, 192, 3)

    m = build_model(shape)
    m.build(input_shape=(None, 256, 192, 3))
    m.summary()
    from tensorflow.keras.utils import plot_model
    plot_model(m, show_shapes=True)
