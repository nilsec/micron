import daisy
import neuroglancer
import numpy as np
import sys
import os
import configparser
from db_scripts import get_graph
from funlib.show.neuroglancer import add_layer, ScalePyramid


predict_config = "/groups/funke/home/ecksteinn/Projects/microtubules/micron_experiments/cremi/02_predict/setup_2/predict_config_0.ini"
dsets = ["volumes/soft_mask"]

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
                         roi_size)

if nodes:
    maxima = []
    maxima_dict = {}
    for z,y,x,node_id in zip(nodes["z"], nodes["y"], nodes["x"], nodes["id"]):
        maxima_dict[node_id] = (x,y,z)
        maxima.append(neuroglancer.EllipsoidAnnotation(center=(x,y,z+1), 
                                                       radii=(tuple([10] * 3)),
                                                       id=node_id,
                                                       segments=None
                                                       )
                     )

if edges:
    k = 0
    edge_connectors = []
    for u, v in zip(edges["u"], edges["v"]):
        try:
            pos_u = maxima_dict[u]
            pos_v = maxima_dict[v]

            edge_connectors.append(neuroglancer.LineAnnotation(point_a=pos_u,
                                                               point_b=pos_v,
                                                               id=k,
                                                               segments=None)
                                  )
            k += 1
        except KeyError:
            pass

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
