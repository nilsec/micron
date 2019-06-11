from __future__ import absolute_import
from scipy.spatial import KDTree
import daisy
from daisy.persistence import MongoDbGraphProvider
import json
import logging
import numpy as np
import os
import sys
import time
import configparser
from daisy_check_functions import check_function, write_done

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def extract_edges(
        predict_config,
        distance_threshold,
        block_size,
        num_workers):

    # Grab predict parameters:
    config = configparser.ConfigParser()
    config.read(predict_config)
    base_dir = os.path.join(config.get("Predict", "base_dir"), config.get("Predict", "experiment"),
                            "02_predict/setup_{}".format(config.get("Predict", "setup_number")))
    db_host = config.get("Database", "db_host")
    db_name = config.get("Database", "db_name")
    roi_offset = daisy.Coordinate(tuple([int(v) for v in np.array(config.get("Data", "in_offset").split(", "), dtype=int)]))
    roi_size = daisy.Coordinate(tuple([int(v) for v in np.array(config.get("Data", "in_size").split(", "), dtype=int)]))

    # Grab voxel size:
    with open(os.path.join(base_dir, "config.json"), "r") as f:
        net_config = json.load(f)
    voxel_size = daisy.Coordinate(net_config['voxel_size'])

    # Define Rois:
    source_roi = daisy.Roi(roi_offset, roi_size)
    block_write_roi = daisy.Roi(
        (0,) * 3,
        daisy.Coordinate(block_size))

    pos_context = daisy.Coordinate((distance_threshold,)*3)
    neg_context = daisy.Coordinate((distance_threshold,)*3)
    logger.debug("Set pos context to %s", pos_context)
    logger.debug("Set neg context to %s", neg_context)

    input_roi = source_roi.grow(neg_context, pos_context)
    block_read_roi = block_write_roi.grow(neg_context, pos_context)

    print("Following ROIs in world units:")
    print("Input ROI       = %s" % input_roi)
    print("Block read  ROI = %s" % block_read_roi)
    print("Block write ROI = %s" % block_write_roi)
    print("Output ROI      = %s" % source_roi)

    print("Starting block-wise processing...")

    # process block-wise
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(
            db_name,
            db_host,
            distance_threshold,
            b),
        check_function=lambda b: check_function(
            b,
            'extract_edges',
            db_name,
            db_host),
        num_workers=num_workers,
        processes=True,
        read_write_conflict=False,
        fit='shrink')


def extract_edges_in_block(
        db_name,
        db_host,
        distance_threshold,
        block):

    logger.info(
        "Finding edges in %s, reading from %s",
        block.write_roi, block.read_roi)

    start = time.time()

    graph_provider = MongoDbGraphProvider(db_name,
                                          db_host,
                                          mode='r+',
                                          position_attribute=['z', 'y', 'x'],
                                          directed=False)

    graph = graph_provider[block.read_roi]

    if graph.number_of_nodes() == 0:
        logger.info("No nodes in roi %s. Skipping", block.read_roi)
        return

    logger.info(
        "Read %d candidates in %.3fs",
        graph.number_of_nodes(),
        time.time() - start)

    start = time.time()


    candidates = [(candidate_id, 
                   np.array([data[d] for d in ['z', 'y', 'x']])) 
                   for candidate_id, data in graph.nodes(data=True)]

    kdtree = KDTree([candidate[1] for candidate in candidates])

    pairs = kdtree.query_pairs(distance_threshold, p=2.0, eps=0)
    for edge in pairs:
        graph.add_edge(candidates[edge[0]][0],
                       candidates[edge[1]][0])

    """
    graph.add_edge(
        nex_cell_id, pre_cell_id,
        distance=distance,
        prediction_distance=prediction_distance)
    """

    logger.info("Found %d edges", graph.number_of_edges())

    logger.info(
        "Extracted edges in %.3fs",
        time.time() - start)

    start = time.time()

    graph.write_edges(block.write_roi)

    logger.info(
        "Wrote edges in %.3fs",
        time.time() - start)

    write_done(block, 'extract_edges', db_name, db_host)


if __name__ == "__main__":
    predict_config = "/groups/funke/home/ecksteinn/Projects/microtubules/micron_experiments/cremi/02_predict/setup_2/predict_config_0.ini"
    distance_threshold=100
    block_size = (10000,10000,10000)
    num_workers = 1

    start_time = time.time()
    extract_edges(
        predict_config,
        distance_threshold,
        block_size,
        num_workers)
    print(time.time() - start_time)
