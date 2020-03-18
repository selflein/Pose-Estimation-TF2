import tensorflow as tf
from tensorflow.keras.losses import MSE
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from pose_estimation.config import cfg
from pose_estimation.data_utils.dataset import get_dataloaders
from pose_estimation.models.mobilenet_pose import MobileNetPose


def train():
    model = MobileNetPose()

    model.compile(optimizer='adam', loss=MSE, metrics=['mse'])

    checkpoint = ModelCheckpoint('models/checkpoints/weights_{epoch:03d}_{val_loss:.5f}.hdf5', save_best_only=True, save_weights_only=False)
    train_data, len_train, val_data, len_val = get_dataloaders(cfg.batch_size, buffer=5, num_workers=cfg.num_thread)
    model.fit(train_data, validation_data=val_data, callbacks=[checkpoint], epochs=20, validation_freq=1, steps_per_epoch=len_train, validation_steps=len_val)


def train_loop():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    model = MobileNetPose()
    for epoch in range(5):
    # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for images, labels in train_ds:
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

        for test_images, test_labels in test_ds:
            # training=False is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)

            test_loss(t_loss)
            test_accuracy(labels, predictions)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))


if __name__ == '__main__':
    train()
