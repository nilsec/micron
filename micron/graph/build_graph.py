#from __future__ import absolute_import
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
from micron import read_predict_config, read_worker_config, read_data_config, read_graph_config
from ext.cpp_get_evidence import cpp_get_evidence

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
        filename='graph.log',
        filemode='w+')
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

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
        graph_number,
        evidence_threshold=None,
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
            evidence_threshold,
            graph_number,
            b),
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
        evidence_threshold,
        graph_number,
        block):

    if check_function(block, "edges_g{}".format(graph_number), db_name, db_host):
        return 0

    logger.debug(
        "Finding edges in %s, reading from %s",
        block.write_roi, block.read_roi)

    start = time.time()

    graph_provider = MongoDbGraphProvider(db_name,
                                          db_host,
                                          mode='r+',
                                          position_attribute=['z', 'y', 'x'],
                                          directed=False,
                                          edges_collection='edges_g{}'.format(graph_number))

    soft_mask_array = daisy.open_ds(soft_mask_container,
                                    soft_mask_dataset)

    graph = graph_provider[block.read_roi.intersect(soft_mask_array.roi)]

    if graph.number_of_nodes() == 0:
        logger.info("No nodes in roi %s. Skipping", block.read_roi)
        write_done(block, 'edges_g{}'.format(graph_number), db_name, db_host)
        return 0

    logger.debug(
        "Read %d candidates in %.3fs",
        graph.number_of_nodes(),
        time.time() - start)

    start = time.time()

    """
    candidates = [(candidate_id, 
                   np.array([data[d] for d in ['z', 'y', 'x']])) 
                   for candidate_id, data in graph.nodes(data=True) if 'z' in data]
    """
    candidates = np.array([[candidate_id] + [data[d] for d in ['z', 'y', 'x']] 
                            for candidate_id, data in graph.nodes(data=True) if 'z' in data],
                            dtype=np.uint64)


    kdtree_start = time.time()
    kdtree = KDTree([[candidate[1], candidate[2], candidate[3]] for candidate in candidates])
    #kdtree = KDTree(candidates[])
    pairs = kdtree.query_pairs(distance_threshold, p=2.0, eps=0)
    logger.debug(
        "Query pairs in %.3fs",
        time.time() - kdtree_start)


    soft_mask_array = daisy.open_ds(soft_mask_container,
                                    soft_mask_dataset)

    voxel_size = np.array(soft_mask_array.voxel_size, dtype=np.uint32)
    soft_mask_roi = block.read_roi.snap_to_grid(voxel_size=voxel_size).intersect(soft_mask_array.roi)
    soft_mask_array_data = soft_mask_array.to_ndarray(roi=soft_mask_roi) 

    sm_dtype = soft_mask_array_data.dtype
    if sm_dtype == np.uint8: # standard pipeline pm 0-255
        pass
    elif sm_dtype == np.float32 or sm_dtype == np.float64:
        if not (soft_mask_array_data.min() >= 0 and soft_mask_array_data.max() <= 1):
            raise ValueError("Provided soft_mask has dtype float but not in range [0,1], abort")
        else:
            soft_mask_array_data *= 255
    else:
        raise ValueError("Soft mask dtype {} not understood".format(sm_dtype))

    soft_mask_array_data = soft_mask_array_data.astype(np.float64)

    if evidence_threshold is not None:
        soft_mask_array_data = (soft_mask_array_data >= evidence_threshold).astype(np.float64) * 255

    offset = np.array(np.array(soft_mask_roi.get_offset())/voxel_size, dtype=np.uint64)
    evidence_start = time.time()

    if pairs:
        pairs = np.array(list(pairs), dtype=np.uint64)
        evidence_array = cpp_get_evidence(candidates, pairs, soft_mask_array_data, offset, voxel_size)
        graph.add_weighted_edges_from(evidence_array, weight='evidence')

        logger.debug(
            "Accumulate evidence in %.3fs",
            time.time() - evidence_start)

        logger.debug("Found %d edges", graph.number_of_edges())

        logger.debug(
            "Extracted edges in %.3fs",
            time.time() - start)

        start = time.time()

        graph.write_edges(block.write_roi)

        logger.debug(
            "Wrote edges in %.3fs",
            time.time() - start)
    else:
        logger.debug("No pairs in block, skip")

    write_done(block, 'edges_g{}'.format(graph_number), db_name, db_host)
    return 0


if __name__ == "__main__":
    #logger = logging.getLogger("daisy")
    #logger.setLevel(logging.DEBUG)

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

    full_config["soft_mask_container"] = data_config_dict["out_container"]
    full_config["soft_mask_dataset"] = data_config_dict["out_dataset"]
    full_config["roi_offset"] = data_config_dict["in_offset"]
    full_config["roi_size"] = data_config_dict["in_size"]

    start_time = time.time()
    extract_edges(**full_config)
    print(time.time() - start_time)
