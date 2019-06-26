import json
import logging
import numpy as np
import os
import daisy
import sys
import time
import datetime
import pymongo
import configparser
from shutil import copyfile
from micron.network import read_predict_config, read_worker_config, read_data_config
from pysub import run, run_singularity

logger = logging.getLogger(__name__)

def predict_blockwise(
        base_dir,
        experiment,
        train_number,
        predict_number,
        iteration,
        in_container_spec,
        in_container,
        in_dataset,
        in_offset,
        in_size,
        out_container,
        db_name,
        db_host,
        singularity_container,
        num_cpus,
        num_cache_workers,
        num_block_workers,
        queue,
        mount_dirs,
        **kwargs):

    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Args:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):
        raw_dataset (``string``):
        auto_file (``string``):
        auto_dataset (``string``):

            Paths to the input autocontext datasets (affs or lsds). Can be None if not needed.

        out_file (``string``):

            Path to directory where zarr should be stored

        **Note:

            out_dataset no longer needed as input, build out_dataset from config
            outputs dictionary generated in mknet.py

        file_name (``string``):

            Name of output file

        block_size_in_chunks (``tuple`` of ``int``):

            The size of one block in chunks (not voxels!). A chunk corresponds
            to the output size of the network.

        num_workers (``int``):

            How many blocks to run in parallel.

        queue (``string``):

            Name of queue to run inference on (i.e slowpoke, gpu_rtx, gpu_any,
            gpu_tesla, gpu_tesla_large)
    '''


    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "02_predict/train_{}/predict_{}".format(train_number, predict_number))
    train_setup_dir = os.path.join(os.path.join(base_dir, experiment), "01_train/train_{}".format(train_number))

    # from here on, all values are in world units (unless explicitly mentioned)
    # get ROI of source
    source = daisy.open_ds(in_container_spec, in_dataset)
    logger.info('Source dataset has shape %s, ROI %s, voxel size %s'%(source.shape, source.roi, source.voxel_size))

    # Read network config
    predict_net_config = os.path.join(predict_setup_dir, 'predict_net.json')
    with open(predict_net_config) as f:
        logger.info('Reading setup config from {}'.format(predict_net_config))
        net_config = json.load(f)
    outputs = net_config['outputs']

    # get chunk size and context
    net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
    net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size
    context = (net_input_size - net_output_size)/2
    logger.info('Network context: {}'.format(context))

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

    logger.info('Preparing output dataset...')

    for output_name, val in outputs.items():
        out_dims = val['out_dims']
        out_dtype = val['out_dtype']
        out_dataset = 'volumes/%s'%output_name

        ds = daisy.prepare_ds(
            out_container,
            out_dataset,
            output_roi,
            source.voxel_size,
            out_dtype,
            write_roi=block_write_roi,
            num_channels=out_dims,
            compressor={'id': 'gzip', 'level':5}
            )

    logger.info('Starting block-wise processing...')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    if 'blocks_predicted' not in db.list_collection_names():
        blocks_predicted = db['blocks_predicted']
        blocks_predicted.create_index(
            [('block_id', pymongo.ASCENDING)],
            name='block_id')
    else:
        blocks_predicted = db['blocks_predicted']

    # process block-wise
    succeeded = daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: predict_worker(
            train_setup_dir,
            predict_setup_dir,
            predict_number,
            train_number,
            experiment,
            iteration,
            in_container,
            in_dataset,
            out_container,
            db_host,
            db_name,
            queue,
            singularity_container,
            num_cpus,
            num_cache_workers,
            mount_dirs),
        check_function=lambda b: check_block(
            blocks_predicted,
            b),
        num_workers=num_block_workers,
        read_write_conflict=False,
        fit='valid')

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")

def predict_worker(
        train_setup_dir,
        predict_setup_dir,
        predict_number,
        train_number,
        experiment,
        iteration,
        in_container,
        in_dataset,
        out_container,
        db_host,
        db_name,
        queue,
        singularity_container,
        num_cpus,
        num_cache_workers,
        mount_dirs):

    predict_block = os.path.join(predict_setup_dir, 'predict_block.py')

    run_instruction = {
        'queue': queue,
        'num_cpus': num_cpus,
        'num_cache_workers': num_cache_workers,
        'singularity': singularity_container
    }

    worker_instruction = {
        'train_setup_dir': train_setup_dir,
        'iteration': iteration,
        'in_container': in_container,
        'in_dataset': in_dataset,
        'out_container': out_container,
        'db_host': db_host,
        'db_name': db_name,
        'run_instruction': run_instruction
    }

    worker_id = daisy.Context.from_env().worker_id
    worker_dir = os.path.join(predict_setup_dir, "worker_files")
    if not os.path.exists(worker_dir):
        os.makedirs(worker_dir)

    worker_instruction_file = os.path.join(worker_dir, '{}_worker_instruction.json'.format(worker_id))
    log_out = os.path.join(worker_dir, '{}_worker.out'.format(worker_id))
    log_err = os.path.join(worker_dir, '{}_worker.err'.format(worker_id))

    with open(worker_instruction_file, 'w') as f:
        json.dump(worker_instruction, f)

    logger.info('Running block for prediction (e:{}, t:{}, i:{}, p:{}) and worker instruction {}...'.format(experiment, 
                                                                                                            train_number,
                                                                                                            iteration,
                                                                                                            predict_number,
                                                                                                            worker_id))

    base_command = "python -u {} {}".format(predict_block,
                                            worker_instruction_file)
    if queue == "None":
        logger.warning("Running block **locally**, no queue provided.")
        if singularity_container == "None":
            logger.warning("Running block in current environment, no singularity image provided.")
            cmd = base_command
        else:
            cmd = run_singularity(base_command,
                            singularity_container,
                            mount_dirs=mount_dirs,
                            execute=False,
                            expand=False)
    else:
        logger.info("Running block on queue {} and container {}".format(queue,
                                                                        singularity_container))
        cmd = run(command=base_command,
            queue=queue,
            num_gpus=1,
            num_cpus=num_cpus,
            singularity_image=singularity_container,
            mount_dirs=mount_dirs,
            execute=False,
            expand=False)

    daisy.call(cmd, log_out=log_out, log_err=log_err)

    logger.info('Predict worker finished')

def check_block(blocks_predicted, block):
    done = blocks_predicted.count({'block_id': block.block_id}) >= 1
    return done


if __name__ == "__main__":
    predict_config = sys.argv[1]
    worker_config = sys.argv[2]
    data_config = sys.argv[3]

    predict_config_dict = read_predict_config(predict_config)
    worker_config_dict = read_worker_config(worker_config)
    data_config_dict = read_data_config(data_config)

    full_config = predict_config_dict
    full_config.update(worker_config_dict)
    full_config.update(data_config_dict)

    start = time.time()

    predict_blockwise(**full_config)

    end = time.time()
    seconds = end - start

    logger.info('Total time to predict: %f seconds' % seconds)
