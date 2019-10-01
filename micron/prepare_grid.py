import os
import sys
from shutil import copyfile, rmtree
import json
from os.path import expanduser
import itertools
from collections import deque
from micron.prepare_solve import create_solve_config, create_worker_config, set_up_environment


def prepare_grid(base_dir,
                 experiment,
                 train_number,
                 predict_number,
                 graph_number,
                 grid_solve_parameters,
                 mount_dirs,
                 singularity,
                 num_cpus,
                 num_block_workers,
                 queue="normal",
                 num_cache_workers="1",
                 min_solve_number=0):

    grid = deque(dict(zip(grid_solve_parameters, x))\
                 for x in itertools.product(*grid_solve_parameters.values()))


    solve_number = min_solve_number
    while grid:
        # Create solve directory
        solve_setup_dir = set_up_environment(base_dir,
                                             experiment,
                                             train_number,
                                             predict_number,
                                             graph_number,
                                             solve_number,
                                             None,
                                             None,
                                             None,
                                             False,
                                             False)

        # Overwrite default solve config with grid parameters:
        params = grid.pop()
        params["solve_number"] = solve_number
        params["selected_attr"] = "selected_{}".format(solve_number)
        params["solved_attr"] = "solved_{}".format(solve_number)
        grid_solve_config = create_solve_config(**params)
        with open(os.path.join(solve_setup_dir, "solve_config.ini"), "w+") as f:
            grid_solve_config.write(f)


        # Overwrite default worker config:
        grid_worker_config = create_worker_config(mount_dirs,
                                                  singularity,
                                                  queue,
                                                  num_cpus,
                                                  num_block_workers,
                                                  num_cache_workers)


        with open(os.path.join(solve_setup_dir, "worker_config.ini"), "w+") as f:
            grid_worker_config.write(f)

        solve_number += 1
