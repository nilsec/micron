import os
import sys
from shutil import copyfile, rmtree
import json
import configargparse
import configparser
from os.path import expanduser
import click
from daisy.persistence import MongoDbGraphProvider
from daisy import Roi
from micron import read_predict_config, read_data_config
from micron.graph.daisy_check_functions import reset_step, reset_solve, attr_exists


p = configargparse.ArgParser()
p.add('-d', '--base_dir', required=True, 
      help='base directory for storing micron experiments, defaults to ``~/micron_experiments``',
      default='~/micron_experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb, defaults to ``base``')
p.add('-t', required=True, help='train setup number to use for this solve run')
p.add('-p', required=True, help='predict number/id to use for this solve run')
p.add('-g', required=True, help='graph number/id to use for this solve run')
p.add('-s', required=True, help='solve number/id to use for this solve run')
p.add('-v', required=True, help='evaluation number/id to use for this evaluation run')


def set_up_environment(base_dir,
                       experiment,
                       train_number,
                       predict_number,
                       graph_number,
                       solve_number,
                       eval_number):

    input_params = locals()
    train_files = {}

    graph_setup_dir = os.path.join(os.path.join(base_dir, experiment), "03_graph/setup_t{}_p{}_g{}".format(train_number, predict_number, graph_number))
    solve_setup_dir = os.path.join(os.path.join(base_dir, experiment), "04_solve/setup_t{}_p{}_g{}_s{}".format(train_number, predict_number, graph_number,solve_number))
    eval_setup_dir = os.path.join(os.path.join(base_dir, experiment), "05_eval/setup_t{}_p{}_g{}_s{}_e{}".format(train_number, predict_number, graph_number,solve_number,eval_number))


    predict_cfg_dict = read_predict_config(os.path.join(graph_setup_dir, "predict_config.ini"))
    data_cfg_dict = read_data_config(os.path.join(graph_setup_dir, "data_config.ini"))
    roi = Roi(data_cfg_dict["in_offset"], data_cfg_dict["in_size"])
    db_name = predict_cfg_dict["db_name"]
    db_host = predict_cfg_dict["db_host"]

    selected_attr = "selected_{}".format(solve_number)
    solved_attr = "solved_{}".format(solve_number)
    edge_collection = "edges_g{}".format(graph_number)
    node_collection = "nodes"

    solved_before = attr_exists(db_name, db_host, edge_collection, solved_attr)
    if not solved_before:
        raise ValueError("Graph not solved, run solve before evaluation.")

    if not os.path.exists(graph_setup_dir):
        raise ValueError("No graph at {}".format(graph_setup_dir))

    if not os.path.exists(solve_setup_dir):
        raise ValueError("No solve setup at {}".format(solve_setup_dir))

    if os.path.exists(eval_setup_dir):
        raise ValueError("Eval setup already exists at {}".format(eval_setup_dir))

    os.makedirs(eval_setup_dir)

    copyfile(os.path.join(graph_setup_dir, "predict_config.ini"), os.path.join(eval_setup_dir, "predict_config.ini"))
    copyfile(os.path.join(graph_setup_dir, "data_config.ini"), os.path.join(eval_setup_dir, "data_config.ini"))
    copyfile(os.path.join(graph_setup_dir, "graph_config.ini"), os.path.join(eval_setup_dir, "graph_config.ini"))
    copyfile(os.path.join(solve_setup_dir, "solve_config.ini"), os.path.join(eval_setup_dir, "solve_config.ini"))

    copyfile(os.path.join(os.path.abspath(os.path.dirname(__file__)), "post/evaluate.py"), os.path.join(eval_setup_dir, "evaluate.py"))

    worker_config = create_worker_config(mount_dirs=None, 
                                         singularity=None, 
                                         queue=None)

    eval_config = create_eval_config(eval_number)

    with open(os.path.join(eval_setup_dir, "worker_config.ini"), "w+") as f:
        worker_config.write(f)
    with open(os.path.join(eval_setup_dir, "eval_config.ini"), "w+") as f:
        eval_config.write(f)

    return eval_setup_dir


def create_eval_config(eval_number):

    config = configparser.ConfigParser()
    config.add_section('Evaluate')
    config.set('Evaluate', 'tracing_file', str(None))
    config.set('Evaluate', 'tracing_offset', "0, 0, 0")
    config.set('Evaluate', 'tracing_size', "0, 0, 0")
    config.set('Evaluate', 'subsample_factor', str(10))
    config.set('Evaluate', 'distance_threshold', str(120))
    config.set('Evaluate', 'max_edges', str(5))
    config.set('Evaluate', 'optimality_gap', str(0.0))
    config.set('Evaluate', 'time_limit', "300")
    config.set('Evaluate', 'evaluation_pipeline', 
                os.path.join(os.path.abspath(os.path.dirname(__file__)), "post/evaluation_pipeline.py"))
    config.set('Evaluate', 'voxel_size', "40, 4, 4")
    config.set('Evaluate', 'eval_number', str(eval_number))

    return config


def create_worker_config(mount_dirs,
                         singularity,
                         queue,
                         num_cpus=5,
                         num_block_workers=1,
                         num_cache_workers=5):

    config = configparser.ConfigParser()
    config.add_section('Worker')
    if singularity == None or singularity == "None" or not singularity:
        config.set('Worker', 'singularity_container', str(None))
    else:
        config.set('Worker', 'singularity_container', str(singularity))
    config.set('Worker', 'num_cpus', str(num_cpus))
    config.set('Worker', 'num_block_workers', str(num_block_workers))
    config.set('Worker', 'num_cache_workers', str(num_cache_workers))
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
    solve_number = int(options.s)
    eval_number = int(options.v)

    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       predict_number,
                       graph_number,
                       solve_number,
                       eval_number)
