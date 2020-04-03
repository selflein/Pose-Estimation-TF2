import tqdm
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping

from pose_estimation.config import cfg
from pose_estimation.data_utils.dataset import get_dataloaders
from pose_estimation.models.mobilenet_pose import MobileNetPose
# from pose_estimation.models.mobilenet_pose_pp import build_model
from pose_estimation.models.blaze_pose import build_model


def train():
    model_checkpoint_path = cfg.model_dump_dir / 'blaze_pose.hdf5'
    model = build_model(cfg.input_shape)

    if cfg.continue_train:
        model = model.load_weights(str(model_checkpoint_path))
    optim = Adam(cfg.lr, epsilon=cfg.weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])

    checkpoint = ModelCheckpoint(str(model_checkpoint_path), save_best_only=True, save_weights_only=False)
    lr_sched = LearningRateScheduler(cfg.get_lr, verbose=1)
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_on_plateau = ReduceLROnPlateau(patience=3)

    train_data, len_train, val_data, len_val = get_dataloaders('COCO', cfg.batch_size, buffer=5, num_workers=cfg.num_thread, scales=(1, 2, 4))
    model.fit(train_data,
              validation_data=val_data,
              callbacks=[checkpoint, reduce_on_plateau, early_stopping],
              epochs=cfg.end_epoch,
              validation_freq=1,
              steps_per_epoch=len_train,
              validation_steps=len_val)


def train_loop():
    model_checkpoint_path = cfg.model_dump_dir / 'mobilenet_pose_pp.hdf5'
    model = build_model()
    if cfg.continue_train:
        model = model.load_weights(str(model_checkpoint_path))

    optimizer = Adam(cfg.lr, epsilon=cfg.weight_decay)
    criterion = MeanSquaredError()

    train_data, len_train, val_data, len_val = get_dataloaders(cfg.batch_size, buffer=5, num_workers=cfg.num_thread)

    for epoch in range(cfg.end_epoch):
    # Reset the metrics at the start of the next epoch
        pbar = tqdm.tqdm(train_data, total=len_train)
        for images, labels in pbar:
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = model(images, training=True)
                loss = criterion(labels, predictions)
                pbar.set_description(str(loss.numpy()))
                pbar.update()
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == '__main__':
    train()
