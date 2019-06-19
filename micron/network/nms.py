import numpy as np
import tensorflow as tf


def max_detection(soft_mask, window_size, threshold):
    data_format = "NDHWC"

    w_depth = window_size[1]
    w_height = window_size[2]
    w_width = window_size[3]

    sm_shape = soft_mask.get_shape().as_list()
    sm_depth = sm_shape[1]
    sm_height = sm_shape[2]
    sm_width = sm_shape[3]

    max_pool = tf.nn.max_pool3d(soft_mask, window_size, window_size, padding="SAME", data_format=data_format)

    conv_filter = np.ones([w_depth,w_height,w_width,1,1])

    upsampled = tf.nn.conv3d_transpose(
                            max_pool,
                            conv_filter.astype(np.float32),
                            [1,sm_depth,sm_height,sm_width,1],
                            window_size,
                            padding='SAME',
                            data_format='NDHWC',
                            name="nms_conv_0"
                        )

    
    maxima = tf.equal(upsampled, soft_mask)
    maxima = tf.logical_and(maxima, soft_mask>=threshold)

    # Fix doubles
    # Check the necessary window size and adapt for isotropic vs unisotropic nms:
    nms_dims = np.array(window_size) != 1
    double_suppresion_window = [3**(dim) for dim in nms_dims]

    sm_maxima = tf.add(tf.cast(maxima, tf.float32),soft_mask)
    max_pool = tf.nn.max_pool3d(sm_maxima, double_suppresion_window, [1,1,1,1,1], padding="SAME", data_format=data_format)
    conv_filter = np.ones([1,1,1,1,1])
    upsampled = tf.nn.conv3d_transpose(
                            max_pool,
                            conv_filter.astype(np.float32),
                            [1,sm_depth,sm_height,sm_width,1],
                            [1,1,1,1,1],
                            padding='SAME',
                            data_format=data_format,
                            name="nms_conv_1"
                        )

    reduced_maxima = tf.equal(upsampled, sm_maxima)
    reduced_maxima = tf.logical_and(reduced_maxima, sm_maxima>1)

    return maxima[0,:,:,:,0], reduced_maxima[0,:,:,:,0]
    #return maxima, reduced_maxima
