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
from dda3 import DDA3

logging.basicConfig(filename="./extract_edges.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def extract_edges(
        db_host,
        db_name,
        soft_mask_container,
        soft_mask_dataset,
        roi_offset,
        roi_size,
        voxel_size,
        distance_threshold,
        block_size,
        num_workers,
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
        num_workers=num_workers,
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
                       evidence=evidence)
 
    """
    for edge in pairs:
        pos_u_world = np.array(candidates[edge[0]][1])
        pos_v_world = np.array(candidates[edge[1]][1])

        pos_u = np.array(pos_u_world/voxel_size, dtype=int)
        pos_v = np.array(pos_v_world/voxel_size, dtype=int)
        dda3 = DDA3(pos_u, pos_v, scaling=voxel_size)
        line = dda3.draw()
        
        evidence = 0.0
        valid_points = 0
        for p in line:
            p *= voxel_size
            if soft_mask_array.roi.contains(daisy.Coordinate(p)):
                evidence += soft_mask_array[daisy.Coordinate(p)]
                valid_points += 1
            else:
                break

        # This takes care of edge effects:
        # Can happen if block is larger than soft_mask
        if valid_points == len(line):
            evidence /= (len(line) * 255.)
            graph.add_edge(candidates[edge[0]][0],
                           candidates[edge[1]][0],
                           evidence=evidence)
        else:
            logging.debug("Skip edge {} because it lies partially outside the soft mask roi".format(edge) +\
                          " and no evidence can be acquired: " +\
                          "u: {}, v: {}, soft_mask roi: {}".format(pos_u_world, pos_v_world, soft_mask_array.roi))
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


def aggregate_config(predict_config):
    # Grab predict parameters:
    config = configparser.ConfigParser()
    config.read(predict_config)

    cfg_dict = {}
    cfg_dict["base_dir"] = os.path.join(config.get("Predict", "base_dir"), config.get("Predict", "experiment"),
                            "02_predict/setup_{}".format(config.get("Predict", "setup_number")))
    cfg_dict["db_host"] = config.get("Database", "db_host")
    cfg_dict["db_name"] = config.get("Database", "db_name")
    cfg_dict["roi_offset"] = daisy.Coordinate(tuple([int(v) for v in np.array(config.get("Data", "in_offset").split(", "), dtype=int)]))
    cfg_dict["roi_size"] = daisy.Coordinate(tuple([int(v) for v in np.array(config.get("Data", "in_size").split(", "), dtype=int)]))

    # Grab voxel size:
    with open(os.path.join(cfg_dict["base_dir"], "config.json"), "r") as f:
        net_config = json.load(f)
    cfg_dict["voxel_size"] = daisy.Coordinate(net_config['voxel_size'])
    cfg_dict["soft_mask_container"] = os.path.join(cfg_dict["base_dir"], config.get("Data", "out_container").split("./")[-1])
    cfg_dict["soft_mask_dataset"] = "/volumes/soft_mask"

    return cfg_dict


if __name__ == "__main__":
    predict_config = "/groups/funke/home/ecksteinn/Projects/microtubules/micron_experiments/cremi/02_predict/setup_2/predict_config_template.ini"
    decrease_roi = np.array([600,600,600])
    distance_threshold=100
    block_size = (1000,1000,1000)
    num_workers = 40

    cfg_dict = aggregate_config(predict_config)
    cfg_dict["distance_threshold"] = 100
    cfg_dict["block_size"] = block_size
    cfg_dict["num_workers"] = num_workers
    cfg_dict["decrease_roi"] = decrease_roi

    start_time = time.time()
    extract_edges(**cfg_dict)
    print(time.time() - start_time)
