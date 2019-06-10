from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
import pymongo
from write_candidates import WriteCandidates

with open('config.json', 'r') as f:
    net_config = json.load(f)

# voxels
input_shape = Coordinate(net_config['input_shape'])
output_shape = Coordinate(net_config['output_shape'])

# nm
voxel_size = Coordinate(net_config['voxel_size'])
input_size = input_shape*voxel_size
output_size = output_shape*voxel_size

def block_done_callback(
        db_host,
        db_name,
        worker_instruction,
        block,
        start,
        duration):

    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_predicted']

    document = dict(worker_instruction)
    document.update({
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    })

    collection.insert(document)

    print("Recorded block-done for %s" % (block,))

def predict(
        iteration,
        in_container,
        in_dataset,
        out_container,
        db_host,
        db_name,
        worker_instruction,
        **kwargs):

    raw = ArrayKey('RAW')
    soft_mask = ArrayKey('SOFT_MASK')
    reduced_maxima = ArrayKey('REDUCED_MAXIMA')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(soft_mask, output_size)
    chunk_request.add(reduced_maxima, output_size)

    pipeline = ZarrSource(
                in_container,
                datasets = {
                    raw: in_dataset
                    },
                array_specs = {
                    raw: ArraySpec(interpolatable=True),
                    }
                )

    pipeline += Pad(raw, None)
    pipeline += Normalize(raw)
    pipeline += IntensityScaleShift(raw, 2,-1)
    
    pipeline += Predict('train_net_checkpoint_%d'%iteration,
                        inputs={
                            net_config['raw']: raw
                        },
                        outputs={
                            net_config['soft_mask']: soft_mask,
                            net_config['pred_reduced_maxima']: reduced_maxima
                        },
                        graph='config.meta',
			max_shared_memory=(2*1024*1024*1024),
                        )
    pipeline += IntensityScaleShift(soft_mask, 255, 0)

    pipeline += ZarrWrite(dataset_names={
                                soft_mask: 'volumes/soft_mask'
                                 },
                          output_filename=out_container
                          )

    """
    pipeline += WriteCandidates(maxima=reduced_maxima,
                                db_host=db_host,
                                db_name=db_name)
    """

    pipeline += PrintProfilingStats(every=10)

    pipeline += DaisyRequestBlocks(
                    chunk_request,
                    roi_map={
                        raw: 'read_roi',
                        soft_mask: 'write_roi',
                        reduced_maxima: 'write_roi'
                        },
                    num_workers=worker_instruction['num_cache_workers'],
                    block_done_callback=lambda b, s, d: block_done_callback(
                        db_host,
                        db_name,
                        worker_instruction,
                        b, s, d))

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('daisy').setLevel(logging.DEBUG)

    worker_instruction = sys.argv[1]

    with open(worker_instruction, 'r') as f:
        worker_instruction = json.load(f)

    predict(**worker_instruction)
