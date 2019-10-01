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
import configparser
from micron import read_solve_config, read_predict_config, read_data_config, read_worker_config, read_graph_config
from solver import Solver

logger = logging.getLogger(__name__)

def solve_in_block(db_host, 
                   db_name, 
                   evidence_factor, 
                   comb_angle_factor, 
                   start_edge_prior, 
                   selection_cost, 
                   time_limit,
                   solve_number,
                   graph_number,
                   selected_attr="selected",
                   solved_attr="solved",
                   **kwargs):

    print("Solve in block")

    graph_provider = MongoDbGraphProvider(
        db_name,
        db_host,
        mode='r+',
        position_attribute=['z', 'y', 'x'],
        edges_collection="edges_g{}".format(graph_number))

    client = daisy.Client()

    while True:
        print("Acquire block")
        block = client.acquire_block()

        if not block:
            return 0

        logger.debug("Solving in block %s", block)

        if check_function(block, 'solve_s{}'.format(solve_number), db_name, db_host):
            client.release_block(block, 0)
            continue

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
            write_done(block, 'solve_s{}'.format(solve_number), db_name, db_host)
            client.release_block(block, 0)
            continue

        print("solve")
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

        print("Write done")
        write_done(block, 'solve_s{}'.format(solve_number), db_name, db_host)

        print("Release block")
        client.release_block(block, 0)


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

    start_time = time.time()
    solve_in_block(**full_config)
    print("Solving took {} seconds".format(time.time() - start_time))
