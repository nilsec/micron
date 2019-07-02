from __future__ import absolute_import
from scipy.spatial import cKDTree as KDTree
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
from dda3 import DDA3
from micron import read_predict_config, read_worker_config, read_data_config, read_graph_config

logger = logging.getLogger(__name__)

def extract_edges(
        db_host,
        db_name,
        soft_mask_container,
        soft_mask_dataset,
        roi_offset,
        roi_size,
        distance_threshold,
        block_size,
        num_block_workers,
        **kwargs):

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

    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s" % input_roi)
    logger.info("Block read  ROI = %s" % block_read_roi)
    logger.info("Block write ROI = %s" % block_write_roi)
    logger.info("Output ROI      = %s" % source_roi)

    logger.info("Starting block-wise processing...")

    # process block-wise
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(
            db_name,
            db_host,
            soft_mask_container,
            soft_mask_dataset,
            distance_threshold,
            b),
        check_function=lambda b: check_function(
            b,
            'extract_edges',
            db_name,
            db_host),
        num_workers=num_block_workers,
        processes=True,
        read_write_conflict=False,
        fit='shrink')


def extract_edges_in_block(
        db_name,
        db_host,
        soft_mask_container,
        soft_mask_dataset,
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

    soft_mask_array = daisy.open_ds(soft_mask_container,
                                    soft_mask_dataset)

    graph = graph_provider[block.read_roi.intersect(soft_mask_array.roi)]

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
                   for candidate_id, data in graph.nodes(data=True) if 'z' in data]

    kdtree = KDTree([candidate[1] for candidate in candidates])
    pairs = kdtree.query_pairs(distance_threshold, p=2.0, eps=0)

    soft_mask_array = daisy.open_ds(soft_mask_container,
                                    soft_mask_dataset)

    voxel_size = soft_mask_array.voxel_size

    for edge in pairs:
        pos_u_world = np.array(candidates[edge[0]][1])
        pos_v_world = np.array(candidates[edge[1]][1])
        pos_u = np.array(pos_u_world/voxel_size, dtype=int)
        pos_v = np.array(pos_v_world/voxel_size, dtype=int)
        dda3 = DDA3(pos_u, pos_v, scaling=voxel_size)
        line = dda3.draw()
        
        evidence = 0.0
        for p in line:
            p *= voxel_size
            evidence += soft_mask_array[daisy.Coordinate(p)]
        evidence /= (len(line) * 255.)
        graph.add_edge(candidates[edge[0]][0],
                       candidates[edge[1]][0],
                       evidence=evidence,
                       selected=False,
                       solved=False)
 
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
    logger = logging.getLogger("daisy")
    logger.setLevel(logging.DEBUG)

    predict_config = sys.argv[1]
    worker_config = sys.argv[2]
    data_config = sys.argv[3]
    graph_config = sys.argv[4]

    predict_config_dict = read_predict_config(predict_config)
    worker_config_dict = read_worker_config(worker_config)
    data_config_dict = read_data_config(data_config)
    graph_config_dict = read_graph_config(graph_config)

    full_config = predict_config_dict
    full_config.update(worker_config_dict)
    full_config.update(data_config_dict)
    full_config.update(graph_config_dict)

    full_config["soft_mask_container"] = predict_config_dict["out_container"]
    full_config["soft_mask_dataset"] = "/volumes/soft_mask"
    full_config["roi_offset"] = data_config_dict["in_offset"]
    full_config["roi_size"] = data_config_dict["in_size"]

    start_time = time.time()
    extract_edges(**full_config)
    print(time.time() - start_time)
