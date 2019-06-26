from funlib.learn.tensorflow import models
import tensorflow as tf
import os
import json
from micron.network import max_detection
import numpy as np


def create_network(input_shape, 
                   name, 
                   setup,
                   voxel_size=[40,4,4],
                   nms_window=[1,1,10,10,1],
                   nms_threshold=0.5):

    tf.reset_default_graph()

    with tf.variable_scope('setup_{}'.format(setup)):
        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        out, _, _ = models.unet(raw_batched, 12, 5, [[1,3,3],[1,3,3],[1,3,3]])

        lsds_batched, _ = models.conv_pass(
            out,
            kernel_sizes=[1],
            num_fmaps=10,
            activation=None)

        output_shape_batched = lsds_batched.get_shape().as_list()
        output_shape = output_shape_batched[1:] # strip the batch dimension

        lsds = tf.reshape(lsds_batched, output_shape)
        soft_mask = lsds[9,:,:,:]
        soft_mask = tf.clip_by_value(soft_mask, 0, 1.0)
        derivatives = lsds[:9,:,:,:]

        gt_lsds = tf.placeholder(tf.float32, shape=output_shape)
        gt_soft_mask = gt_lsds[9,:,:,:]
        gt_derivatives = gt_lsds[:9,:,:,:]

        print(gt_soft_mask.get_shape().as_list())
        print(soft_mask.get_shape().as_list())
        print(list(output_shape))
        print(list(output_shape_batched))

        gt_maxima, gt_reduced_maxima = max_detection(tf.reshape(gt_soft_mask, [1] + gt_soft_mask.get_shape().as_list() + [1]), 
                                                                nms_window, nms_threshold)
        pred_maxima, pred_reduced_maxima = max_detection(tf.reshape(soft_mask, [1] + gt_soft_mask.get_shape().as_list() + [1]), 
                                                                nms_window, nms_threshold)

        # Soft weights for binary mask
        binary_mask = tf.cast(gt_soft_mask > 0, tf.float32)
        loss_weights_soft_mask = tf.ones(binary_mask.get_shape())
        loss_weights_soft_mask += tf.multiply(binary_mask, tf.reduce_sum(binary_mask))
        loss_weights_soft_mask -= binary_mask

        loss_weights_lsds = tf.stack([loss_weights_soft_mask] * 10)

        loss = tf.losses.mean_squared_error(
                                    lsds,
                                    gt_lsds,
                                    loss_weights_lsds)

        summary = tf.summary.scalar('loss', loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = opt.minimize(loss)

        output_shape = output_shape[1:]
        print("input shape : %s"%(input_shape,))
        print("output shape: %s"%(output_shape,))

        tf.train.export_meta_graph(filename=name + '.meta')

        config = {
            'raw': raw.name,
            'derivatives': derivatives.name,
            'soft_mask': soft_mask.name,
            'gt_lsds': gt_lsds.name,
            'gt_maxima': gt_maxima.name,
            'gt_reduced_maxima': gt_reduced_maxima.name,
            'pred_maxima': pred_maxima.name,
            'pred_reduced_maxima': pred_reduced_maxima.name,
            'loss_weights_lsds': loss_weights_soft_mask.name,
            'loss': loss.name,
            'optimizer': optimizer.name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'summary': summary.name,
        }

        config['outputs'] = {
                'soft_mask':
                {"out_dims": 1,
                    "out_dtype": "uint8"},
                'derivatives':
                {"out_dims": 9,
                    "out_dtype": "uint8"},
                'reduced_maxima':
                    {"out_dims": 1,
                        "out_dtype": "uint8"}
                }

        config['voxel_size'] = voxel_size

        with open(name + '.json', 'w') as f:
            json.dump(config, f)

if __name__ == "__main__":
    # Train:
    create_network((32, 322, 322), 'train_net', 0, [40,4,4], [1,1,10,10,1], 0.5)

    # Predict:
    z = 0 # Enlarge for prediction
    xy = 0 
    create_network((32+z, 322+xy, 322+xy), 'predict_net', 0, [40,4,4], [1,1,10,10,1], 0.5)
