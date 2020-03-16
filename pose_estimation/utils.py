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
def extract_keypoints_from_heatmap(heatmap: tf.Tensor):
    row_idxs = tf.argmax(tf.reduce_max(heatmap, axis=1), axis=0)
    col_idxs = tf.argmax(tf.reduce_max(heatmap, axis=0), axis=0)

    idxs = tf.stack([row_idxs, col_idxs, tf.range(heatmap.shape[-1], dtype=tf.int64)], axis=1)
    values = tf.gather_nd(heatmap, idxs)
    keypoints = tf.stack([tf.cast(col_idxs, tf.float32), tf.cast(row_idxs, tf.float32), values], axis=1)
    return keypoints


if __name__ == '__main__':
    test_tensor = tf.random.normal((64, 48, 17))
    np.testing.assert_array_almost_equal(
        extract_keypoints_from_heatmap(test_tensor).numpy(),
        extract_keypoints_from_heatmap_numpy(test_tensor)
    )
