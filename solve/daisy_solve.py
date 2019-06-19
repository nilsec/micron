import daisy
from daisy.persistence import MongoDbGraphProvider
import json
import logging
import numpy as np
import os
import sys
import time
sys.path.append("../")
from graph.daisy_check_functions import check_function, write_done
from solve import Solver
import configparser

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
# logging.getLogger(
#         'daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def solve(
        db_host,
        db_name,
        singularity_container,
        num_cpus,
        num_block_workers,
        queue,
        block_size,
        roi_offset,
        roi_size,
        context,
        solve_number,
        evidence_factor,
        comb_angle_factor,
        start_edge_prior,
        selection_cost,
        **kwargs):

    source_roi = daisy.Roi(daisy.Coordinate(roi_offset), daisy.Coordinate(roi_size))

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
        process_function=lambda b: solve_in_block(
            db_host,
            db_name,
            evidence_factor,
            comb_angle_factor,
            start_edge_prior,
            selection_cost,
            b,
            solve_number),
        check_function=lambda b: check_function(
            b,
            'solve_' + str(solve_number),
            db_name,
            db_host),
        num_workers=num_block_workers,
        fit='shrink')

    logger.info("Finished solving, parameters id is %s", solve_number)


def solve_in_block(db_host, 
                   db_name, 
                   evidence_factor, 
                   comb_angle_factor, 
                   start_edge_prior, 
                   selection_cost, 
                   block, 
                   solve_number):

    logger.debug("Solving in block %s", block)


    graph_provider = MongoDbGraphProvider(
        db_name,
        db_host,
        mode='r+',
        position_attribute=['z', 'y', 'x'])

    start_time = time.time()
    graph = graph_provider.get_graph(
            block.read_roi)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    logger.info("Reading graph with %d nodes and %d edges took %s seconds"
                % (num_nodes, num_edges, time.time() - start_time))

    print(num_edges)
    if num_edges == 0:
        logger.info("No edges in roi %s. Skipping"
                    % block.read_roi)
        return 0

    solver = Solver(graph, 
                    evidence_factor, 
                    comb_angle_factor, 
                    start_edge_prior, 
                    selection_cost)

    solver.initialize()
    solver.solve()

    start_time = time.time()
    graph.update_edge_attrs(
            block.write_roi,
            attributes=["selected", "solved"])

    logger.info("Updating attributes %s & %s for %d edges took %s seconds"
                % ("selected",
                   "solved",
                   num_edges,
                   time.time() - start_time))
    write_done(block, 'solve_' + str(solve_number), db_name, db_host)
    return 0


def read_config(solve_config):
    config = configparser.ConfigParser()
    config.read(solve_config)

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
    cfg_dict["queue"] = config.get("Worker", "queue")
    cfg_dict["block_size"] = tuple([int(v) for v in np.array(config.get("Worker", "block_size").split(", "), dtype=int)])

    # Data
    cfg_dict["roi_offset"] = tuple([int(v) for v in np.array(config.get("Roi", "roi_offset").split(", "), dtype=int)])
    cfg_dict["roi_size"] = tuple([int(v) for v in np.array(config.get("Roi", "roi_size").split(", "), dtype=int)])
    cfg_dict["context"] = tuple([int(v) for v in np.array(config.get("Roi", "context").split(", "), dtype=int)])

    # Solve
    cfg_dict["evidence_factor"] = config.getint("Solve", "evidence_factor")
    cfg_dict["comb_angle_factor"] = config.getint("Solve", "comb_angle_factor")
    cfg_dict["start_edge_prior"] = config.getint("Solve", "start_edge_prior")
    cfg_dict["selection_cost"] = config.getint("Solve", "selection_cost")
    cfg_dict["solve_number"] = config.getint("Solve", "solve_number")

    print(cfg_dict)

    return cfg_dict


if __name__ == "__main__":

    config_file = sys.argv[1]
    cfg_dict = read_config(config_file)

    start_time = time.time()
    solve(**cfg_dict)
    print("Solving took {} seconds".format(time.time() - start_time))
