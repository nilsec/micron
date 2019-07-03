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
from solve_block import Solver
import configparser
from micron import read_solve_config, read_predict_config, read_data_config, read_worker_config, read_graph_config

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
        filename='solve.log',
        filemode='w+')
# logging.getLogger(
#         'daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

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
        time_limit,
        selected_attr,
        solved_attr,
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
            time_limit,
            b,
            solve_number,
            selected_attr,
            solved_attr),
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
                   time_limit,
                   block, 
                   solve_number,
                   selected_attr="selected",
                   solved_attr="solved"):

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

    if num_edges == 0:
        logger.info("No edges in roi %s. Skipping"
                    % block.read_roi)
        return 0

    solver = Solver(graph, 
                    evidence_factor, 
                    comb_angle_factor, 
                    start_edge_prior, 
                    selection_cost,
                    time_limit,
                    selected_attr,
                    solved_attr)

    solver.initialize()
    solver.solve()

    start_time = time.time()
    graph.update_edge_attrs(
            block.write_roi,
            attributes=[selected_attr, solved_attr])

    graph.update_node_attrs(
            block.write_roi,
            attributes=[selected_attr, solved_attr])

    logger.info("Updating attributes %s & %s for %d edges took %s seconds"
                % (selected_attr,
                   solved_attr,
                   num_edges,
                   time.time() - start_time))
    write_done(block, 'solve_' + str(solve_number), db_name, db_host)
    return 0


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

    full_config = predict_config_dict
    full_config.update(worker_config_dict)
    full_config.update(data_config_dict)
    full_config.update(graph_config_dict)
    full_config.update(solve_config_dict)

    full_config["roi_offset"] = full_config["in_offset"]
    full_config["roi_size"] = full_config["in_size"]

    start_time = time.time()
    solve(**full_config)
    print("Solving took {} seconds".format(time.time() - start_time))
