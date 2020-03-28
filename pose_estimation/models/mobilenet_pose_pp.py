import tensorflow as tf

from tensorflow.keras import layers, Sequential


def depthwise_separable_conv(filters=256):
    conv_1 = Sequential([
        layers.DepthwiseConv2D(3, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2)
    ])
    conv_2 = Sequential([
        layers.Conv2D(filters, 1),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2)
    ])
    return Sequential([conv_1, conv_2])


def build_model(image_shape=(256, 192, 3)):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=image_shape,
        include_top=False,
        weights='imagenet'
    )

    for layer in backbone.layers[:13]:
        layer.trainable = False

    out = depthwise_separable_conv(1280)(backbone.output)

    # * block_13_expand_relu (16, 12, 576)
    out = layers.Conv2DTranspose(256, 4, 2, padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(0.2)(out)
    out = layers.concatenate([out, backbone.get_layer('block_13_expand_relu').output], axis=-1)
    out = depthwise_separable_conv(256 + 576)(out)

    # * block_6_expand_relu (32, 24, 192)
    out = layers.Conv2DTranspose(128, 4, 2, padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(0.2)(out)
    out = layers.concatenate([out, backbone.get_layer('block_6_expand_relu').output], axis=-1)
    out = depthwise_separable_conv(128 + 192)(out)

    # * block_3_expand_relu (64, 48, 144)
    out = layers.Conv2DTranspose(64, 4, 2, padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(0.2)(out)
    out = layers.concatenate([out, backbone.get_layer('block_3_expand_relu').output], axis=-1)
    out = depthwise_separable_conv(64 + 144)(out)

    out = layers.Conv2D(17, 1, padding='same')(out)

    model = tf.keras.Model(inputs=backbone.input, outputs=out)

    return model
