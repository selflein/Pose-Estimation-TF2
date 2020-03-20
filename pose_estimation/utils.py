import numpy as np
import tensorflow as tf


def extract_keypoints_from_heatmap_numpy(heatmap: tf.Tensor):
    row_idxs = tf.argmax(tf.reduce_max(heatmap, axis=1), axis=0).numpy()
    col_idxs = tf.argmax(tf.reduce_max(heatmap, axis=0), axis=0).numpy()

    heatmap = heatmap.numpy()
    values = heatmap[row_idxs, col_idxs, np.arange(heatmap.shape[-1])]
    keypoints = np.stack([col_idxs, row_idxs, values], axis=1)

    return keypoints


@tf.function
def extract_keypoints_from_heatmap2(heatmap: tf.Tensor):
    row_idxs = tf.argmax(tf.reduce_max(heatmap, axis=1), axis=0)
    col_idxs = tf.argmax(tf.reduce_max(heatmap, axis=0), axis=0)

    idxs = tf.stack([row_idxs, col_idxs, tf.range(heatmap.shape[-1], dtype=tf.int64)], axis=1)
    values = tf.gather_nd(heatmap, idxs)
    keypoints = tf.stack([tf.cast(col_idxs, tf.float32), tf.cast(row_idxs, tf.float32), values], axis=1)
    return keypoints


@tf.function
def extract_keypoints_from_heatmap(heatmap: tf.Tensor):
    b, h, w, c = heatmap.shape
    reshaped_heatmap = tf.reshape(heatmap, [b, h * w, c])
    max_idxs = tf.argmax(reshaped_heatmap, axis=1, output_type=tf.int32)

    y_coords = tf.math.floordiv(max_idxs, w)
    x_coords = tf.math.mod(max_idxs, w)

    idxs = tf.stack([tf.zeros((c,), dtype=tf.int32), y_coords[0], x_coords[0], tf.range(heatmap.shape[-1], dtype=tf.int32)], axis=1)
    confidence = tf.expand_dims(tf.gather_nd(heatmap, idxs), 0)
    return tf.cast(x_coords, tf.float32), tf.cast(y_coords, tf.float32), confidence


if __name__ == '__main__':
    test_tensor = tf.random.normal((64, 48, 17))
    np.testing.assert_array_almost_equal(
        extract_keypoints_from_heatmap2(test_tensor).numpy(),
        extract_keypoints_from_heatmap_numpy(test_tensor)
    )
