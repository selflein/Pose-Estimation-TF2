import tensorflow as tf

from tensorflow.keras import layers


class MobileNetPose(tf.keras.Model):
    def __init__(self, image_shape=(256, 192, 3)):
        super(MobileNetPose, self).__init__()
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=image_shape,
            include_top=False,
            weights='imagenet'
        )

        # for layer in self.base_model.layers[:58]:
        #     layer.trainable = False

        self.keypoint_heatmaps = tf.keras.Sequential([
            layers.Conv2DTranspose(256, 4, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(128, 4, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 4, 2, padding='same', activation='relu'),
            layers.Conv2D(17, 1, padding='same')
        ])

    def call(self, inp, **kwargs):
        out = self.base_model(inp, **kwargs)
        out = self.keypoint_heatmaps(out, **kwargs)
        return out

