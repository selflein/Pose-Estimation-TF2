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
    model_checkpoint_finetuned_path = cfg.model_dump_dir / 'blaze_pose_fine.hdf5'
    model = build_model(cfg.input_shape)

    model.load_weights(str(model_checkpoint_path))
    optim = Adam(1e-6, epsilon=cfg.weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])

    checkpoint = ModelCheckpoint(str(model_checkpoint_finetuned_path),
                                 save_best_only=False, save_weights_only=False)

    train_data, len_train, val_data, len_val = get_dataloaders('PushUp', cfg.batch_size, buffer=5, num_workers=cfg.num_thread, scales=(1, 2, 4))
    model.fit(train_data,
              validation_data=val_data,
              callbacks=[checkpoint],
              epochs=10,
              validation_freq=1,
              steps_per_epoch=len_train,
              validation_steps=len_val)


if __name__ == '__main__':
    train()
