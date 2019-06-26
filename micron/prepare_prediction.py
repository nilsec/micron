import os
import sys
from shutil import copyfile, rmtree
import json
import configargparse
import configparser
from os.path import expanduser
import click

micron_config = expanduser("~/.micron")

p = configargparse.ArgParser(default_config_files=[micron_config])
p.add('-d', '--base_dir', required=False, 
      help='base directory for storing micron experiments, defaults to ``~/micron_experiments``',
      default='~/micron_experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb, defaults to ``base``')
p.add('-t', required=True, help='train setup number to use for this prediction')
p.add('-i', required=True, help='iteration checkpoint number to use for this prediction')
p.add('-p', required=True, help='predict number/id to use for this prediction')
p.add('-c', required=False, action='store_true', help='clean up - remove specified predict setup')
p.add('-o', required=False, action='store_true', help='copy the origional, unmodified' +
                                                      'mknet instead of the one used for training.')
p.add('--db_name', required=False, 
      help='name of the database to write the prediction to, defaults to ``{experiment}_{train-number}_{predict-number}``')
p.add('--db_host', required=False, 
      help='database credential string')
p.add('--mount_dirs', required=False, 
      help='directories to mount in container environment')
p.add('--singularity', required=False,
      help='path singularity image to use')
p.add('--queue', required=False,
      help='cluster queue to submit jobs to')


def set_up_environment(base_dir,
                       experiment,
                       train_number,
                       iteration,
                       predict_number,
                       db_name,
                       db_host,
                       mount_dirs,
                       singularity,
                       queue,
                       copy_original,
                       clean_up):

    input_params = locals()
    train_files = {}

    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "02_predict/train_{}/predict_{}".format(train_number, predict_number))
    train_setup_dir = os.path.join(os.path.join(base_dir, experiment), "01_train/train_{}".format(train_number))
    train_checkpoint = os.path.join(train_setup_dir, "train_net_checkpoint_{}.meta".format(iteration))

    if clean_up:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to remove {} and all its contents?'.format(predict_setup_dir), default=False):
                rmtree(predict_setup_dir)
            else:
                print("Abort clean up")

    if not os.path.exists(train_checkpoint):
        raise ValueError("No checkpoint at {}".format(train_checkpoint))

    if not os.path.exists(predict_setup_dir):
        os.makedirs(predict_setup_dir)
    else:
        if __name__ == "__main__":
            if click.confirm('Predict setup {} exists already, overwrite?'.format(predict_setup_dir), default=False):
                rmtree(predict_setup_dir)
                os.makedirs(predict_setup_dir)
            else:
                print("Abort.")
                return
        else:
            raise ValueError("Predict setup exists already, choose different predict number or clean up.")

    copyfile("network/predict_block.py", os.path.join(predict_setup_dir, "predict_block.py"))
    copyfile("network/predict.py", os.path.join(predict_setup_dir, "predict.py"))
    if copy_original:
        copyfile(os.path.abspath("./network/mknet.py"), os.path.join(predict_setup_dir, "mknet.py"))
    else:
        copyfile(os.path.join(train_setup_dir, "mknet.py"), os.path.join(predict_setup_dir, "mknet.py"))

    predict_config = create_predict_config(**input_params)
    worker_config = create_worker_config(mount_dirs, singularity, queue)
    data_config = create_data_config()


    with open(os.path.join(predict_setup_dir, "predict_config.ini"), "w+") as f:
        predict_config.write(f)
    with open(os.path.join(predict_setup_dir, "worker_config.ini"), "w+") as f:
        worker_config.write(f)
    with open(os.path.join(predict_setup_dir, "data_config.ini"), "w+") as f:
        data_config.write(f)


def create_predict_config(base_dir,
                          experiment,
                          train_number,
                          iteration,
                          predict_number,
                          db_name,
                          db_host,
                          **kwargs):

    config = configparser.ConfigParser()

    config.add_section('Predict')
    config.set('Predict', 'blockwise', str(os.path.abspath('./network/predict_blockwise.py')))
    config.set('Predict', 'base_dir', str(os.path.abspath(base_dir)))
    config.set('Predict', 'experiment', str(experiment))
    config.set('Predict', 'train_number', str(train_number))
    config.set('Predict', 'predict_number', str(predict_number))
    config.set('Predict', 'iteration', str(iteration))

    config.add_section('Database')
    if db_name == "None" or db_name is None:
        db_name = "{}_t{}_i{}_p{}".format(experiment, train_number, iteration, predict_number)
    config.set('Database', 'db_name', str(db_name))
    if db_host == "None" or db_host is None:
        db_host = 'mongodb://localhost'
    config.set('Database', 'db_host', db_host)
    return config

def create_worker_config(mount_dirs,
                         singularity,
                         queue):

    config = configparser.ConfigParser()
    config.add_section('Worker')
    if singularity == None or singularity == "None" or not singularity:
        config.set('Worker', 'singularity_container', str(None))
    else:
        config.set('Worker', 'singularity_container', str(singularity))
    config.set('Worker', 'num_cpus', str(5))
    config.set('Worker', 'num_block_workers', str(1))
    config.set('Worker', 'num_cache_workers', str(5))
    if queue == None or queue == "None" or not queue:
        config.set('Worker', 'queue', str(None))
    else:
        config.set('Worker', 'queue', str(queue))
    if mount_dirs == None or mount_dirs == "None" or not mount_dirs:
        config.set('Worker', 'mount_dirs', "")
    else:
        config.set('Worker', 'mount_dirs', mount_dirs)
    return config

def create_data_config():
    config = configparser.ConfigParser()
    config.add_section('Data')
    config.set('Data', 'in_container', str(None))
    config.set('Data', 'in_dataset', str(None))
    config.set('Data', 'in_offset', str(0) + ", " +\
                                    str(0) + ", " +\
                                    str(0))
    config.set('Data', 'in_size', str(0) + ", " +\
                                  str(0) + ", " +\
                                  str(0))
    config.set('Data', 'out_container', str(None))
    return config


if __name__ == "__main__":
    options = p.parse_args()

    base_dir = options.base_dir
    experiment = options.e
    train_number = int(options.t)
    train_iteration = int(options.i)
    predict_number = int(options.p)
    clean_up = bool(options.c)
    copy_original = bool(options.o)
    db_name = options.db_name
    db_host = options.db_host
    mount_dirs = options.mount_dirs
    singularity = options.singularity
    queue = options.queue


    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       train_iteration,
                       predict_number,
                       db_name,
                       db_host,
                       mount_dirs,
                       singularity,
                       queue,
                       copy_original,
                       clean_up)
