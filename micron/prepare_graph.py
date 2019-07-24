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
p.add('-t', required=True, help='train setup number to use for this graph prep')
p.add('-p', required=True, help='predict number/id to use for this graph prep')
p.add('-g', required=True, help='graph number to use for this graph prep')
p.add('-c', required=False, action='store_true', help='clean up - remove specified graph setup')

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
                       predict_number,
                       graph_number,
                       db_host,
                       mount_dirs,
                       singularity,
                       queue,
                       clean_up):

    input_params = locals()
    train_files = {}

    graph_setup_dir = os.path.join(os.path.join(base_dir, experiment), "03_graph/setup_t{}_p{}_g{}".format(train_number, predict_number, graph_number))
    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "02_predict/setup_t{}_p{}".format(train_number, predict_number))
    train_setup_dir = os.path.join(os.path.join(base_dir, experiment), "01_train/train_{}".format(train_number))

    if clean_up:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to remove {} and all its contents?'.format(graph_setup_dir), default=False):
                rmtree(graph_setup_dir)
            else:
                print("Abort clean up")

    if not os.path.exists(predict_setup_dir):
        raise ValueError("No prediction at {}".format(predict_setup_dir))

    if not os.path.exists(graph_setup_dir):
        os.makedirs(graph_setup_dir)

    else:
        if __name__ == "__main__":
            if click.confirm('Graph setup {} exists already, overwrite?'.format(graph_setup_dir), default=False):
                rmtree(graph_setup_dir)
                os.makedirs(graph_setup_dir)
            else:
                print("Abort.")
                return
        else:
            raise ValueError("Graph setup exists already, choose different graph number or clean up.")

    copyfile(os.path.join(predict_setup_dir, "predict_config.ini"), os.path.join(graph_setup_dir, "predict_config.ini"))
    copyfile(os.path.join(predict_setup_dir, "data_config.ini"), os.path.join(graph_setup_dir, "data_config.ini"))
    copyfile("./graph/graph.py", os.path.join(graph_setup_dir, "graph.py"))

    worker_config = create_worker_config(mount_dirs, singularity, queue)
    graph_config = create_graph_config(graph_number)

    with open(os.path.join(graph_setup_dir, "worker_config.ini"), "w+") as f:
        worker_config.write(f)

    with open(os.path.join(graph_setup_dir, "graph_config.ini"), "w+") as f:
        graph_config.write(f)

def create_graph_config(graph_number):
    config = configparser.ConfigParser()
    config.add_section('Graph')
    config.set('Graph', 'graph_number', str(graph_number))
    config.set('Graph', 'distance_threshold', str(0)) 
    config.set('Graph', 'block_size', "0, 0, 0")
    config.set('Graph', 'build_graph', os.path.abspath('./graph/build_graph.py'))
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


if __name__ == "__main__":
    options = p.parse_args()

    base_dir = options.base_dir
    experiment = options.e
    train_number = int(options.t)
    predict_number = int(options.p)
    graph_number = int(options.g)
    clean_up = bool(options.c)
    db_host = options.db_host
    mount_dirs = options.mount_dirs
    singularity = options.singularity
    queue = options.queue

    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       predict_number,
                       graph_number,
                       db_host,
                       mount_dirs,
                       singularity,
                       queue,
                       clean_up)
