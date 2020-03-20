import tqdm
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping

from pose_estimation.config import cfg
from pose_estimation.data_utils.dataset import get_dataloaders
from pose_estimation.models.mobilenet_pose_pp import build_model


def train():
    model_checkpoint_path = cfg.model_dump_dir / 'best.hd5'
    model = build_model()

    if cfg.continue_train:
        model = model.load_weights(str(model_checkpoint_path))
    optim = Adam(cfg.lr, epsilon=cfg.weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])

    checkpoint = ModelCheckpoint(str(model_checkpoint_path), save_best_only=True, save_weights_only=False)
    lr_sched = LearningRateScheduler(cfg.get_lr)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    train_data, len_train, val_data, len_val = get_dataloaders(cfg.batch_size, buffer=5, num_workers=cfg.num_thread)
    model.fit(train_data, validation_data=val_data, callbacks=[checkpoint, lr_sched], epochs=cfg.end_epoch, validation_freq=1, steps_per_epoch=len_train, validation_steps=len_val)


def train_loop():
    model_checkpoint_path = cfg.model_dump_dir / 'best.hd5'
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
