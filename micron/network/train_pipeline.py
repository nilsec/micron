from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
from lsd.gp import AddLocalShapeDescriptor
import os
import math
import json
import tensorflow as tf
import numpy as np
from micron import read_train_config


def train_until(max_iteration,
                training_container,
                raw_dset,
                gt_dset):

    """
    max_iteration [int]: Number of training iterations

    data_dir [string]: Training data base directory

    samples [list of strings]: hdf5 files holding the training data. Each 
                             file is expected to have a dataset called
                             *raw* holding the raw image data and 
                             a dataset called *tracing* holding the microtubule
                             tracings.
    """

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('train_net.json', 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    tracing = ArrayKey('TRACING')
    soft_mask = ArrayKey('SOFT_MASK')
    derivatives = ArrayKey('DERIVATIVES')
    gt_lsds = ArrayKey('GT_LSDS')
    loss_weights_lsds = ArrayKey('LOSS_WEIGHTS_LSDS')
    gt_maxima = ArrayKey('GT_MAXIMA')
    gt_reduced_maxima = ArrayKey('GT_REDUCED_MAXIMA')
    pred_maxima = ArrayKey('PRED_MAXIMA')
    pred_reduced_maxima = ArrayKey('PRED_REDUCED_MAXIMA')

    voxel_size = Coordinate(config['voxel_size'])
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(tracing, output_size)
    request.add(gt_lsds, output_size)
    
    snapshot_request = BatchRequest({
        gt_lsds: request[tracing],
        tracing: request[tracing],
        derivatives: request[tracing],
        soft_mask: request[tracing],
        loss_weights_lsds: request[tracing],
        gt_maxima: request[tracing],
        gt_reduced_maxima: request[tracing],
        pred_maxima: request[tracing],
        pred_reduced_maxima: request[tracing]
        })

    data_sources = tuple(
        Hdf5Source(
            container,
            datasets = {
                raw: raw_dset,
                tracing: gt_dset
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                tracing: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        Pad(tracing, Coordinate((10, 100, 100))) + 
        RandomLocation()
        for container in training_container
    )


    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        AddLocalShapeDescriptor(
            tracing,
            gt_lsds,
            sigma=4.0,
            downsample=1) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'train_net',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['raw']: raw,
                config['gt_lsds']: gt_lsds
            },
            outputs={
                config['soft_mask']: soft_mask,
                config['derivatives']: derivatives,
                config['loss_weights_lsds']: loss_weights_lsds,
                config['gt_maxima']: gt_maxima,
                config['gt_reduced_maxima']: gt_reduced_maxima,
                config['pred_maxima']: pred_maxima,
                config['pred_reduced_maxima']: pred_reduced_maxima
            },
            gradients={},
            summary=config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'raw',
                tracing: 'tracing',
                gt_lsds: 'gt_lsds',
                soft_mask: 'soft_mask',
                derivatives: 'derivatives',
                loss_weights_lsds: 'loss_weights_lsds',
                gt_maxima: 'gt_maxima',
                gt_reduced_maxima: 'gt_reduced_maxima',
                pred_maxima: 'pred_maxima',
                pred_reduced_maxima: 'pred_reduced_maxima'
            },
            dataset_dtypes={
                tracing: np.uint64
            },
            every=1000,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    iteration = int(sys.argv[1])
    train_config = read_train_config("./train_config.ini")
    train_config["max_iteration"] = iteration

    train_until(**train_config)
