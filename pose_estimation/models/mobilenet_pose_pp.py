import tensorflow as tf

from tensorflow.keras import layers, Sequential
from tensorflow.keras.utils import plot_model


def upsample(inp, skip, mid_filters=128, out_filters=64):
    up = layers.UpSampling2D()(inp)

    reduced = layers.Conv2D(mid_filters, 1, 1, 'same')(skip)

    concat = layers.concatenate([up, reduced], axis=-1)

    out = layers.SeparableConv2D(out_filters, 3, 1, 'same')(concat)
    out = layers.LeakyReLU(alpha=0.2)(out)
    out = layers.BatchNormalization()(out)
    return out


def build_model(image_shape=(256, 192, 3), out_classes=17, alpha=1.):
    backbone = tf.keras.applications.MobileNetV2(
        alpha=alpha,
        input_shape=image_shape,
        include_top=False,
        weights='imagenet'
    )

    for layer in backbone.layers:
        layer.trainable = False

    backbone_reduced = layers.Conv2D(256, 1, 1, 'same')(backbone.output)

    # * block_13_expand_relu (16, 12, 576)
    out = upsample(backbone_reduced,
                   backbone.get_layer('block_13_expand_relu').output,
                   mid_filters=256,
                   out_filters=256)

    # * block_6_expand_relu (32, 24, 192)
    out = upsample(out,
                   backbone.get_layer('block_6_expand_relu').output,
                   mid_filters=256,
                   out_filters=128)

    # * block_3_expand_relu (64, 48, 144)
    out = upsample(out,
                   backbone.get_layer('block_3_expand_relu').output,
                   mid_filters=128,
                   out_filters=64)

    pred = layers.Conv2D(out_classes, 1, padding='same')(out)

    model = tf.keras.Model(inputs=backbone.input, outputs=pred)

    return model


if __name__ == '__main__':
    m = build_model()
    plot_model(m, '/tmp/model.png')
    print(m.summary())
