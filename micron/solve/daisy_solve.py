import daisy
from daisy.persistence import MongoDbGraphProvider
import json
import logging
import numpy as np
import os
import sys
import time
sys.path.append("../")
from micron.graph.daisy_check_functions import check_function, write_done
from solver import Solver
import configparser
from micron import read_solve_config, read_predict_config, read_data_config, read_worker_config, read_graph_config
from funlib.run import run, run_singularity

logger = logging.getLogger(__name__)

def solve(
        predict_config,
        worker_config,
        data_config,
        graph_config,
        solve_config,
        num_block_workers,
        block_size,
        roi_offset,
        roi_size,
        context,
        solve_block,
        base_dir,
        experiment,
        train_number,
        predict_number,
        graph_number,
        solve_number,
        queue,
        singularity_container,
        mount_dirs,
        **kwargs):

    source_roi = daisy.Roi(daisy.Coordinate(roi_offset), daisy.Coordinate(roi_size))

    solve_setup_dir = os.path.join(os.path.join(base_dir, experiment), "04_solve/setup_t{}_p{}_g{}_s{}".format(train_number,
                                                                                                               predict_number,
                                                                                                               graph_number,
                                                                                                               solve_number))

    block_write_roi = daisy.Roi(
        (0, 0, 0),
        block_size)
    block_read_roi = block_write_roi.grow(
        context,
        context)
    total_roi = source_roi.grow(
        context,
        context)

    logger.info("Solving in %s", total_roi)

    daisy.run_blockwise(
        total_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: start_worker(predict_config,
                                              worker_config,
                                              data_config,
                                              graph_config,
                                              solve_config,
                                              queue,
                                              singularity_container,
                                              mount_dirs,
                                              solve_block,
                                              solve_setup_dir),
        num_workers=num_block_workers,
        fit='shrink')

    logger.info("Finished solving, parameters id is %s", solve_number)

def start_worker(predict_config,
                 worker_config,
                 data_config,
                 graph_config,
                 solve_config,
                 queue,
                 singularity_container,
                 mount_dirs,
                 solve_block,
                 solve_setup_dir):

   
    worker_id = daisy.Context.from_env().worker_id

    log_out = os.path.join(solve_setup_dir, '{}_worker.out'.format(worker_id))
    log_err = os.path.join(solve_setup_dir, '{}_worker.err'.format(worker_id))

    base_command = "python -u {} {} {} {} {} {}".format(solve_block,
                                                        predict_config,
                                                        worker_config,
                                                        data_config,
                                                        graph_config,
                                                        solve_config)
    if queue == "None":
        logger.warning("Running block **locally**, no queue provided.")
        if singularity_container == "None":
            logger.warning("Running block in current environment, no singularity image provided.")
            cmd = [base_command]
        else:
            cmd = run_singularity(base_command,
                                  singularity_container,
                                  mount_dirs=mount_dirs,
                                  execute=False,
                                  expand=False,
                                  batch=True)
    else:
        logger.info("Running block on queue {} and container {}".format(queue,
                                                                        singularity_container))
        cmd = run(command=base_command,
                  queue=queue,
                  num_gpus=0,
                  num_cpus=1,
                  singularity_image=singularity_container,
                  mount_dirs=mount_dirs,
                  execute=False,
                  expand=False,
                  batch=True)

    daisy.call(cmd, log_out=log_out, log_err=log_err)

    logger.info('Solve worker finished')

if __name__ == "__main__":
    predict_config = sys.argv[1]
    worker_config = sys.argv[2]
    data_config = sys.argv[3]
    graph_config = sys.argv[4]
    solve_config = sys.argv[5]

    predict_config_dict = read_predict_config(predict_config)
    worker_config_dict = read_worker_config(worker_config)
    data_config_dict = read_data_config(data_config)
    graph_config_dict = read_graph_config(graph_config)
    solve_config_dict = read_solve_config(solve_config)

    print("solve_config_dict", solve_config_dict)

    full_config = predict_config_dict
    full_config.update(worker_config_dict)
    full_config.update(data_config_dict)
    full_config.update(graph_config_dict)
    full_config.update(solve_config_dict)

    full_config["roi_offset"] = full_config["in_offset"]
    full_config["roi_size"] = full_config["in_size"]
    full_config["predict_config"] = predict_config
    full_config["worker_config"] = worker_config
    full_config["data_config"] = data_config
    full_config["graph_config"] = graph_config
    full_config["solve_config"] = solve_config

    start_time = time.time()
    solve(**full_config)
    print("Solving took {} seconds".format(time.time() - start_time))
