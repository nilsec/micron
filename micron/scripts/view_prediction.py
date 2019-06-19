import daisy
import neuroglancer
import numpy as np
import sys
import os
import configparser
from db_scripts import get_graph
from funlib.show.neuroglancer import add_layer, ScalePyramid


predict_config = "/groups/funke/home/ecksteinn/Projects/microtubules/micron_experiments/cremi/02_predict/setup_2/predict_config_template.ini"
dsets = ["volumes/soft_mask"]
selected_only=True

config = configparser.ConfigParser()
config.read(predict_config)

base_dir = os.path.join(config.get("Predict", "base_dir"), config.get("Predict", "experiment"), 
                        "02_predict/setup_{}".format(config.get("Predict", "setup_number")))

f_raw = config.get("Data", "in_container")
f_prediction = os.path.join(base_dir, config.get("Data", "out_container").split("./")[-1])
db_host = config.get("Database", "db_host")
db_name = config.get("Database", "db_name")
roi_offset = tuple([int(v) for v in np.array(config.get("Data", "in_offset").split(", "), dtype=int)])
roi_size = tuple([int(v) for v in np.array(config.get("Data", "in_size").split(", "), dtype=int)])

raw = [
    daisy.open_ds(f_raw, 'volumes/raw/s%d'%s)
    for s in range(17)
]

view_dsets = {}
print("View", f_prediction)
for dset in dsets:
    view_dsets[dset] = daisy.open_ds(f_prediction, dset)

nodes, edges = get_graph(db_host,
                         db_name,
                         roi_offset,
                         roi_size,
                         selected_only)
if edges:
    nodes_in_edges = []
    k = 0
    edge_connectors = []
    for u, v, selected, solved in edges:
        try:
            pos_u = (nodes[u][2], nodes[u][1], nodes[u][0])
            pos_v = (nodes[v][2], nodes[v][1], nodes[v][0])
            nodes_in_edges.append(u)
            nodes_in_edges.append(v)

            edge_connectors.append(neuroglancer.LineAnnotation(point_a=pos_u,
                                                               point_b=pos_v,
                                                               id=k,
                                                               segments=None)
                                  )
            k += 1
        except KeyError:
            pass

if nodes:
    maxima = []
    maxima_dict = {}
    if not selected_only:
        for node_id,z,y,x,selected,solved in nodes:
            maxima_dict[node_id] = (x,y,z)
            maxima.append(neuroglancer.EllipsoidAnnotation(center=(x,y,z+1), 
                                                           radii=(tuple([10] * 3)),
                                                           id=node_id,
                                                           segments=None
                                                           )
                         )

    else:
        for node_id in nodes_in_edges:
            x = nodes[node_id][2]
            y = nodes[node_id][1]
            z = nodes[node_id][0]
            maxima_dict[node_id] = (x,y,z)
            maxima.append(neuroglancer.EllipsoidAnnotation(center=(x,y,z+1), 
                                                           radii=(tuple([10] * 3)),
                                                           id=node_id,
                                                           segments=None
                                                           )
                         )


viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    for dset, dset_data in view_dsets.items():
        add_layer(s, dset_data, dset)
    add_layer(s, raw, 'raw')

    s.layers['maxima'] = neuroglancer.AnnotationLayer(voxel_size=(1,1,1),
                                                      filter_by_segmentation=False,
                                                      annotation_color='#add8e6',
                                                      annotations=maxima)

    s.layers['conectors'] = neuroglancer.AnnotationLayer(voxel_size=(1,1,1),
                                                         filter_by_segmentation=False,
                                                         annotation_color='#00ff00',
                                                         annotations=edge_connectors)

print(viewer)
