import daisy
from daisy.persistence import MongoDbGraphProvider
import numpy as np

def get_nodes(db_host, 
              db_name,
              roi_offset=(158000, 121800, 448616),
              roi_size=(7600,5200,2200)):

    
    graph = MongoDbGraphProvider(db_name, db_host, directed=False, position_attribute=['z', 'y', 'x'])
    roi = daisy.Roi(roi_offset, roi_size)
    nodes, edges = graph.read_blockwise(roi, block_size=daisy.Coordinate((10000,10000,10000)), num_workers=1)

    return nodes


if __name__ == "__main__":
    get_nodes("mongodb://ecksteinn:ecksteinn@10.150.100.155:27017",
              "calyx_test_small_1")







