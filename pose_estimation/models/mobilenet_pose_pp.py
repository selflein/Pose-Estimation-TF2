import tensorflow as tf

from tensorflow.keras import layers


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters=256):
        super(ResidualBlock, self).__init__()
        self.conv_1 = layers.Conv2D(filters, 1, activation='relu')
        self.conv_2 = layers.DepthwiseConv2D(3, padding='same', activation='relu')
        self.conv_3 = layers.Conv2D(filters, 1)
        self.add = layers.Add()

    def call(self, inputs, **kwargs):
        out = self.conv_1(inputs, **kwargs)
        out = self.conv_2(out, **kwargs)
        out = self.conv_3(out, **kwargs)
        out = self.add([out, inputs], **kwargs)
        return out


def build_model(image_shape=(256, 192, 3)):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=image_shape,
        include_top=False,
        weights='imagenet'
    )

    # for layer in backbone.layers[:58]:
    #    layer.trainable = False

    out = ResidualBlock(1280)(backbone.output)

    # * block_13_expand_relu (16, 12, 576)
    out = layers.Conv2DTranspose(256, 4, 2, padding='same', activation='relu')(out)
    out = layers.concatenate([out, backbone.get_layer('block_13_expand_relu').output], axis=-1)
    out = ResidualBlock(256 + 576)(out)

    # * block_6_expand_relu (32, 24, 192)
    out = layers.Conv2DTranspose(128, 4, 2, padding='same', activation='relu')(out)
    out = layers.concatenate([out, backbone.get_layer('block_6_expand_relu').output], axis=-1)
    out = ResidualBlock(128 + 192)(out)

    # * block_3_expand_relu (64, 48, 144)
    out = layers.Conv2DTranspose(64, 4, 2, padding='same', activation='relu')(out)
    out = layers.concatenate([out, backbone.get_layer('block_3_expand_relu').output], axis=-1)
    out = ResidualBlock(64 + 144)(out)

    out = layers.Conv2D(17, 1, padding='same')(out)

    model = tf.keras.Model(inputs=backbone.input, outputs=out)

    return model


class MobileNetPose(tf.keras.Model):
    def __init__(self, image_shape=(256, 192, 3)):
        super(MobileNetPose, self).__init__()
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=image_shape,
            include_top=False,
            weights='imagenet'
        )

        self.skip_1_out = self.base_model.get_layer('block_6_expand_relu').output

        for layer in self.base_model.layers[:58]:
            layer.trainable = False

        self.mid_conv = ResidualBlock(1280)

        self.up_conv1 = layers.Conv2DTranspose(256, 4, 2, padding='same', activation='relu')
        self.concat_1 = layers.Concatenate(axis=-1)
        self.inv_bot_1 = ResidualBlock(256 + 576)

        self.up_conv2 = layers.Conv2DTranspose(128, 4, 2, padding='same', activation='relu')
        self.concat_2 = layers.Concatenate(axis=-1)
        self.inv_bot_2 = ResidualBlock(128 + 192)

        self.up_conv3 = layers.Conv2DTranspose(64, 4, 2, padding='same', activation='relu')
        self.concat_3 = layers.Concatenate(axis=-1)
        self.inv_bot_3 = ResidualBlock(64 + 144)

        self.out_conv = layers.Conv2D(17, 1, padding='same')

    def call(self, inp, **kwargs):
        base_out = self.base_model(inp, **kwargs)
        out = self.mid_conv(base_out, **kwargs)

        # Skip connections:
        # * block_13_expand_relu (16, 12, 576)
        out = self.up_conv1(out, **kwargs)
        out = self.concat_1([out, self.base_model.get_layer('block_13_expand_relu').output])
        out = self.inv_bot_1(out, **kwargs)

        # * block_6_expand_relu (32, 24, 192)
        out = self.up_conv2(out, **kwargs)
        out = self.concat_2([out, self.base_model.get_layer('block_6_expand_relu').output])
        out = self.inv_bot_2(out, **kwargs)

        # * block_3_expand_relu (64, 48, 144)
        out = self.up_conv3(out, **kwargs)
        out = self.concat_3([out, self.base_model.get_layer('block_3_expand_relu').output])
        out = self.inv_bot_3(out, **kwargs)

        out = self.out_conv(out, **kwargs)
        return out
