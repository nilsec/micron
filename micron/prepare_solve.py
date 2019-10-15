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

micron_config = expanduser("~/.micron")

p = configargparse.ArgParser(default_config_files=[micron_config])
p.add('-d', '--base_dir', required=False, 
      help='base directory for storing micron experiments, defaults to ``~/micron_experiments``',
      default='~/micron_experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb, defaults to ``base``')
p.add('-t', required=True, help='train setup number to use for this solve run')
p.add('-p', required=True, help='predict number/id to use for this solve run')
p.add('-g', required=True, help='graph number/id to use for this solve run')
p.add('-s', required=True, help='solve number/id to use for this solve run')
p.add('-c', required=False, action='store_true', help='clean up - remove specified predict setup')
p.add('-r', required=False, action='store_true', help='reset - reset solve and selected status for resolving')

p.add('--db_host', required=False,
      help='db host credential string')
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
                       solve_number,
                       mount_dirs,
                       singularity,
                       queue,
                       clean_up,
                       reset):

    input_params = locals()
    train_files = {}

    graph_setup_dir = os.path.join(os.path.join(base_dir, experiment), "03_graph/setup_t{}_p{}_g{}".format(train_number, predict_number, graph_number))
    solve_setup_dir = os.path.join(os.path.join(base_dir, experiment), "04_solve/setup_t{}_p{}_g{}_s{}".format(train_number, predict_number, graph_number,solve_number))
    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "02_predict/train_{}/predict_{}".format(train_number, predict_number))
    train_setup_dir = os.path.join(os.path.join(base_dir, experiment), "01_train/train_{}".format(train_number))
    
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
        print("Graph not solved before, build attributes...")
        reset_solve(db_name, db_host, edge_collection, node_collection, selected_attr, solved_attr)

    if clean_up:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to remove {} and all its contents?'.format(solve_setup_dir), default=False):
                rmtree(solve_setup_dir)
            else:
                print("Abort clean up")
        else:
            rmtree(solve_setup_dir)

    if reset:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to reset solve and selected status in {}?'.format(db_name), default=False):
                reset_solve(db_name, db_host, edge_collection, node_collection, selected_attr, solved_attr)
                reset_step("solve_{}".format(solve_number), db_name, db_host)

            else:
                print("Abort reset")
        else:
            reset_solve(db_name, db_host, edge_collection, node_collection, selected_attr, solved_attr)
            reset_step("solve_{}".format(solve_number), db_name, db_host)

    if not os.path.exists(graph_setup_dir):
        raise ValueError("No graph at {}".format(graph_setup_dir))

    if not os.path.exists(solve_setup_dir):
        os.makedirs(solve_setup_dir)

    else:
        if __name__ == "__main__":
            if click.confirm('Solve setup {} exists already, overwrite?'.format(solve_setup_dir), default=False):
                rmtree(solve_setup_dir)
                os.makedirs(solve_setup_dir)
            else:
                print("Abort.")
                return
        else:
            raise ValueError("Solve setup exists already, choose different predict number or clean up.")

    copyfile(os.path.join(graph_setup_dir, "predict_config.ini"), os.path.join(solve_setup_dir, "predict_config.ini"))
    copyfile(os.path.join(graph_setup_dir, "data_config.ini"), os.path.join(solve_setup_dir, "data_config.ini"))
    copyfile(os.path.join(graph_setup_dir, "graph_config.ini"), os.path.join(solve_setup_dir, "graph_config.ini"))
    copyfile(os.path.join(os.path.abspath(os.path.dirname(__file__)), "solve/solve.py"), os.path.join(solve_setup_dir, "solve.py"))

    worker_config = create_worker_config(mount_dirs, singularity, queue)
    solve_config = create_solve_config(solve_number,
                                       selected_attr,
                                       solved_attr)

    with open(os.path.join(solve_setup_dir, "worker_config.ini"), "w+") as f:
        worker_config.write(f)
    with open(os.path.join(solve_setup_dir, "solve_config.ini"), "w+") as f:
        solve_config.write(f)

    return solve_setup_dir


def create_solve_config(solve_number,
                        selected_attr,
                        solved_attr,
                        evidence_factor=12,
                        comb_angle_factor=14,
                        start_edge_prior=180,
                        selection_cost=-80):

    config = configparser.ConfigParser()
    config.add_section('Solve')
    config.set('Solve', 'evidence_factor', str(evidence_factor))
    config.set('Solve', 'comb_angle_factor', str(comb_angle_factor))
    config.set('Solve', 'start_edge_prior', str(start_edge_prior))
    config.set('Solve', 'selection_cost', str(selection_cost))
    config.set('Solve', 'context', "400, 400, 400")
    config.set('Solve', 'daisy_solve', os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                                                    "solve/daisy_solve.py"))
    config.set('Solve', 'solve_block', os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                                                    "solve/solve_block.py"))
    config.set('Solve', 'solve_number', str(solve_number))
    config.set('Solve', 'time_limit', str(120))
    config.set('Solve', 'selected_attr', selected_attr)
    config.set('Solve', 'solved_attr', solved_attr)
    
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
    clean_up = bool(options.c)
    reset = bool(options.r)
    db_host = options.db_host
    mount_dirs = options.mount_dirs
    singularity = options.singularity
    queue = options.queue

    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       predict_number,
                       graph_number,
                       solve_number,
                       mount_dirs,
                       singularity,
                       queue,
                       clean_up,
                       reset)
