import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping

from pose_estimation.config import cfg
# from pose_estimation.models.blaze_pose import build_model
from pose_estimation.data_utils.dataset import get_dataloaders
from pose_estimation.models.mobilenet_pose import MobileNetPose
from pose_estimation.models.mobilenet_pose_pp import build_model


def train():
    model_checkpoint_path = cfg.model_dump_dir / 'mobilenet_pose_pp.hdf5'
    train_data, len_train, val_data, len_val = get_dataloaders('COCO', cfg.batch_size, buffer=10, num_workers=1, scales=(1,))
    min_delta = 1e-4
    checkpoint = ModelCheckpoint(str(model_checkpoint_path), save_best_only=True, save_weights_only=False)
    lr_sched = LearningRateScheduler(cfg.get_lr, verbose=1)
    early_stopping = EarlyStopping(patience=30, restore_best_weights=True, min_delta=min_delta)
    reduce_on_plateau = ReduceLROnPlateau(patience=10, min_delta=min_delta)

    model = build_model(cfg.input_shape, out_classes=17)
    if cfg.continue_train:
        model.load_weights(
            str(cfg.model_dump_dir / 'mobilenet_pose_pp_pretrain_MPII.hdf5'),
            by_name=True,
            skip_mismatch=True
        )

        # Freeze everything except the output layer
        for layer in model.layers:
            if layer.name != 'out':
                layer.trainable = False

    optim = Adam(5e-3, epsilon=cfg.weight_decay)
    model.compile(optimizer=optim, loss='mse')

    model.fit(train_data,
              validation_data=val_data,
              callbacks=[checkpoint, reduce_on_plateau, early_stopping],
              epochs=10,
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

    train_data, len_train, val_data, len_val = get_dataloaders('COCO', cfg.batch_size, buffer=5, num_workers=cfg.num_thread)

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
