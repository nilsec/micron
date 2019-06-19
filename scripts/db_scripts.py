import daisy
from daisy.persistence import MongoDbGraphProvider
import numpy as np

def get_graph(db_host, 
              db_name,
              roi_offset=(158000, 121800, 448616),
              roi_size=(7600,5200,2200),
              selected_only=False):

    
    graph = MongoDbGraphProvider(db_name, db_host, directed=False, position_attribute=['z', 'y', 'x'])
    roi = daisy.Roi(roi_offset, roi_size)
    nodes, edges = graph.read_blockwise(roi, block_size=daisy.Coordinate((10000,10000,10000)), num_workers=1)

    nodes = {node_id: (z,y,x) for z,y,x,node_id in\
             zip(nodes["z"], nodes["y"], nodes["x"], nodes["id"])}

    if selected_only:
        edges = [(u, v, selected, solved) for u,v,selected,solved in zip(edges["u"], edges["v"], edges["selected"], edges["solved"]) if selected]

    else:
        edges = [(u, v, selected, solved) for u,v,selected,solved in zip(edges["u"], edges["v"], edges["selected"], edges["solved"])]

    return nodes, edges


if __name__ == "__main__":
    nodes, edges = get_graph("mongodb://ecksteinn:ecksteinn@10.150.100.155:27017",
              "calyx_test_medium",
              selected_only=True)
