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
import configparser
from shutil import copyfile

#logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy').setLevel(logging.DEBUG)

def predict_blockwise(
        base_dir,
        experiment,
        setup_number,
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
        predict_config_hash):

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

    setup = "setup_{}".format(setup_number)
    experiment_dir = os.path.join(base_dir, experiment)
    train_dir = os.path.join(experiment_dir, '01_train', setup)
    predict_dir = os.path.join(experiment_dir, '02_predict', setup)

    # from here on, all values are in world units (unless explicitly mentioned)
    # get ROI of source
    source = daisy.open_ds(in_container_spec, in_dataset)
    logging.info('Source dataset has shape %s, ROI %s, voxel size %s'%(source.shape, source.roi, source.voxel_size))

    # Read network config
    with open(os.path.join(train_dir, 'config.json')) as f:
        logging.info('Reading setup config from %s'%os.path.join(setup, 'config.json'))
        net_config = json.load(f)
    outputs = net_config['outputs']

    # get chunk size and context
    net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
    net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size
    context = (net_input_size - net_output_size)/2
    logging.info('Network context: {}'.format(context))

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
            out_container,
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
            predict_config_hash),
        check_function=lambda b: check_block(
            blocks_predicted,
            b),
        num_workers=num_block_workers,
        read_write_conflict=False,
        fit='overhang')

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")

def predict_worker(
        experiment,
        setup,
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
        predict_config_hash):

    predict_script = 'predict.py'

    worker_instruction = {
        'queue': queue,
        'num_cpus': num_cpus,
        'num_cache_workers': num_cache_workers,
        'singularity': singularity_container
    }

    predict_instruction = {
        'iteration': iteration,
        'in_container': in_container,
        'in_dataset': in_dataset,
        'out_container': out_container,
        'db_host': db_host,
        'db_name': db_name,
        'worker_instruction': worker_instruction
    }

    worker_id = daisy.Context.from_env().worker_id
    output_dir = os.path.join("prediction_" + str(predict_config_hash))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predict_instruction_file = os.path.join(output_dir, 'predict_instruction_{}.json'.format(worker_id))
    log_out = os.path.join(output_dir, 'predict_blockwise_{}.out'.format(worker_id))
    log_err = os.path.join(output_dir, 'predict_blockwise_{}.err'.format(worker_id))

    with open(predict_instruction_file, 'w') as f:
        json.dump(predict_instruction, f)

    logging.info('Running block for cfg {} and instruction {}...'.format(predict_config_hash, predict_instruction_file))

    command = [
        'run_lsf',
        '-s', singularity_container,
        '-c', str(num_cpus),
        '-g', '1',
        '-q', queue
    ]

    command += [
        'python -u %s %s'%(
            predict_script,
            predict_instruction_file
        )]

    daisy.call(command, log_out=log_out, log_err=log_err)

    logging.info('Predict worker finished')

def check_block(blocks_predicted, block):
    done = blocks_predicted.count({'block_id': block.block_id}) >= 1
    return done

def read_config(predict_config):
    config = configparser.ConfigParser()
    config.read(predict_config)

    cfg_dict = {}

    # Predict
    cfg_dict["base_dir"] = config.get("Predict", "base_dir")
    cfg_dict["experiment"] = config.get("Predict", "experiment")
    cfg_dict["setup_number"] = int(config.getint("Predict", "setup_number"))
    cfg_dict["iteration"] = int(config.getint("Predict", "iteration"))

    # Database
    cfg_dict["db_name"] = config.get("Database", "db_name")
    cfg_dict["db_host"] = config.get("Database", "db_host")

    # Worker
    cfg_dict["singularity_container"] = config.get("Worker", "singularity_container")
    cfg_dict["num_cpus"] = int(config.getint("Worker", "num_cpus"))
    cfg_dict["num_block_workers"] = int(config.getint("Worker", "num_block_workers"))
    cfg_dict["num_cache_workers"] = int(config.getint("Worker", "num_cache_workers"))
    cfg_dict["queue"] = config.get("Worker", "queue")

    # Data
    cfg_dict["in_container"] = config.get("Data", "in_container")
    cfg_dict["in_dataset"] = config.get("Data", "in_dataset")
    cfg_dict["in_offset"] = tuple([int(v) for v in np.array(config.get("Data", "in_offset").split(", "), dtype=int)])
    cfg_dict["in_size"] = tuple([int(v) for v in np.array(config.get("Data", "in_size").split(", "), dtype=int)])
    cfg_dict["out_container"] = config.get("Data", "out_container")

    # Create json container spec for in_data:
    in_container_spec = {"container": cfg_dict["in_container"],
                         "offset": cfg_dict["in_offset"],
                         "size": cfg_dict["in_size"]}

    cfg_hash = 0
    in_container_spec_file = "in_container_spec_{}.json".format(cfg_hash)

    with open(in_container_spec_file, "w+") as f:
        json.dump(in_container_spec, f)


    cfg_dict["in_container_spec"] = os.path.abspath(in_container_spec_file)

    # Final hash
    cfg_dict["predict_config_hash"] = cfg_hash

    # Rename config file to correct hash to includes possible modifications:
    copyfile(predict_config, 
             os.path.join(os.path.dirname(predict_config), "predict_config_{}.ini".format(cfg_hash)))

    return cfg_dict


if __name__ == "__main__":
    predict_config_file = sys.argv[1]
    predict_config_dict = read_config(predict_config_file)


    start = time.time()

    predict_blockwise(**predict_config_dict)

    end = time.time()
    seconds = end - start

    logging.info('Total time to predict: %f seconds' % seconds)
