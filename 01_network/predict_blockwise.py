import hashlib
import json
import logging
import numpy as np
import os
import daisy
import sys
import time
import datetime
import pymongo

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy').setLevel(logging.DEBUG)

def predict_blockwise(
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        file_name,
        num_workers,
        db_host,
        db_name,
        queue,
        auto_file=None,
        auto_dataset=None):

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

    experiment_dir = '../' + experiment
    data_dir = os.path.join(experiment_dir, '01_data')
    train_dir = os.path.join(experiment_dir, '02_train')
    network_dir = os.path.join(experiment, setup, str(iteration))

    raw_file = os.path.abspath(raw_file)
    out_file = os.path.abspath(os.path.join(out_file, setup, str(iteration), file_name))

    setup = os.path.abspath(os.path.join(train_dir, setup))

    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    try:
        source = daisy.open_ds(raw_file, raw_dataset)
    except:
        raw_dataset = raw_dataset + '/s0'
        source = daisy.open_ds(raw_file, raw_dataset)
    logging.info('Source dataset has shape %s, ROI %s, voxel size %s'%(source.shape, source.roi, source.voxel_size))

    # load config
    with open(os.path.join(setup, 'config.json')) as f:
        logging.info('Reading setup config from %s'%os.path.join(setup, 'config.json'))
        net_config = json.load(f)
    outputs = net_config['outputs']

    # get chunk size and context
    net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
    net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size
    context = (net_input_size - net_output_size)/2
    print('CONTEXT: ', context)

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

    logging.info('Preparing output dataset...')

    for output_name, val in outputs.items():
        out_dims = val['out_dims']
        out_dtype = val['out_dtype']
        out_dataset = 'volumes/%s'%output_name

        ds = daisy.prepare_ds(
            out_file,
            out_dataset,
            output_roi,
            source.voxel_size,
            out_dtype,
            write_roi=block_write_roi,
            num_channels=out_dims,
            compressor={'id': 'gzip', 'level':5}
            )

    logging.info('Starting block-wise processing...')

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
            experiment,
            setup,
            network_dir,
            iteration,
            raw_file,
            raw_dataset,
            auto_file,
            auto_dataset,
            out_file,
            out_dataset,
            db_host,
            db_name,
            queue),
        check_function=lambda b: check_block(
            blocks_predicted,
            b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang')

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")

def predict_worker(
        experiment,
        setup,
        network_dir,
        iteration,
        raw_file,
        raw_dataset,
        auto_file,
        auto_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        queue):

    setup_dir = os.path.join('..', experiment, '02_train', setup)
    predict_script = os.path.abspath(os.path.join(setup_dir, 'predict.py'))

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']

    worker_config = {
        'queue': queue,
        'num_cpus': 2,
        'num_cache_workers': 5,
        'singularity': 'funkey/lsd:v0.8'
    }

    config = {
        'iteration': iteration,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'auto_file': auto_file,
        'auto_dataset': auto_dataset,
        'out_file': out_file,
        'out_dataset': out_dataset,
        'db_host': db_host,
        'db_name': db_name,
        'worker_config': worker_config
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join('.predict_blockwise', network_dir)

    try:
        os.makedirs(output_dir)
    except:
        pass

    config_file = os.path.join(output_dir, '%d.config'%config_hash)

    log_out = os.path.join(output_dir, 'predict_blockwise_%d.out'%worker_id)
    log_err = os.path.join(output_dir, 'predict_blockwise_%d.err'%worker_id)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    command = [
        'run_lsf',
        '-c', str(worker_config['num_cpus']),
        '-g', '1',
        '-q', worker_config['queue']
    ]

    if worker_config['singularity']:
        command += ['-s', worker_config['singularity']]

    command += [
        'python -u %s %s'%(
            predict_script,
            config_file
        )]

    daisy.call(command, log_out=log_out, log_err=log_err)

    logging.info('Predict worker finished')

    # if things went well, remove temporary files
    # os.remove(config_file)
    # os.remove(log_out)
    # os.remove(log_err)

def check_block(blocks_predicted, block):

    done = blocks_predicted.count({'block_id': block.block_id}) >= 1

    return done

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    predict_blockwise(**config)

    end = time.time()

    seconds = end - start
    logging.info('Total time to predict: %f seconds' % seconds)

