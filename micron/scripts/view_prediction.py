import daisy
import neuroglancer
import numpy as np
import sys
import os
import configparser
from db_scripts import get_graph
from funlib.show.neuroglancer import add_layer, ScalePyramid
import configargparse

p = configargparse.ArgParser()


p.add('-d', required=True,
      help='base directory for storing micron experiments, defaults to ``~/micron_experiments``',
      default='~/micron_experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb')
p.add('-t', required=True, help='train number')
p.add('-p', required=True, help='Predictions to visualize, e.g. "1, 2, 3"')
p.add('-s', action='store_true', required=False, 
      help='Show selected nodes and edges only',
      default=False)

options = p.parse_args()

base_dir = options.d
experiment = options.e
train_number = int(options.t)
predictions = tuple([int(p) for p in options.p.split(", ")])
selected_only = bool(options.s)

predict_setup_dirs = [os.path.join(os.path.join(base_dir, experiment), "02_predict/train_{}/predict_{}".format(train_number, p)) for p in predictions] 
predict_configs = [os.path.join(base, "predict_config.ini") for base in predict_setup_dirs]
dsets = ["volumes/soft_mask"]

print(predict_setup_dirs)

viewer = neuroglancer.Viewer()
prediction_views = []
for k, base_dir in enumerate(predict_setup_dirs):
    data_k = {}
    predict_config_file = os.path.join(base_dir, "predict_config.ini")
    data_config_file = os.path.join(base_dir, "data_config.ini")
    predict_config = configparser.ConfigParser()
    predict_config.read(predict_config_file)
    data_config = configparser.ConfigParser()
    data_config.read(data_config_file)

    f_raw = data_config.get("Data", "in_container")
    f_prediction = os.path.join(base_dir, data_config.get("Data", "out_container").split("./")[-1])
    print(f_prediction)
    db_host = predict_config.get("Database", "db_host")
    db_name = predict_config.get("Database", "db_name")
    roi_offset = tuple([int(v) for v in np.array(data_config.get("Data", "in_offset").split(", "), dtype=int)])
    roi_size = tuple([int(v) for v in np.array(data_config.get("Data", "in_size").split(", "), dtype=int)])

    print(roi_offset)
    print(roi_size)

    raw = [
        daisy.open_ds(f_raw, 'volumes/raw/s%d'%s)
        for s in range(17)
    ]

    data_k["raw"] = raw

    view_dsets = {}
    print("View", f_prediction)
    for dset in dsets:
        view_dsets[dset] = daisy.open_ds(f_prediction, dset)
    
    data_k["view_dsets"] = view_dsets
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
        data_k["edge_connectors"] = edge_connectors

    if nodes:
        maxima = []
        maxima_dict = {}
        if not selected_only:
            for node_id, node_data in nodes.items():
                x = node_data[2]
                y = node_data[1]
                z = node_data[0]
                maxima_dict[node_id] = (x,y,z)
                maxima.append(neuroglancer.EllipsoidAnnotation(center=(x,y,z+1), 
                                                               radii=(tuple([10] * 3)),
                                                               id=node_id,
                                                               segments=None
                                                               )
                             )
            data_k["maxima"] = maxima

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
            data_k["maxima"] = maxima

    prediction_views.append(data_k)


with viewer.txn() as s:
    for k, view in enumerate(prediction_views):
        for dset, dset_data in view["view_dsets"].items():
            add_layer(s, dset_data, str(k) + "_" + dset)

        s.layers['{}_maxima'.format(k)] = neuroglancer.AnnotationLayer(voxel_size=(1,1,1),
                                                          filter_by_segmentation=False,
                                                          annotation_color='#add8e6',
                                                          annotations=view["maxima"])

        try:
            s.layers['{}_connectors'.format(k)] = neuroglancer.AnnotationLayer(voxel_size=(1,1,1),
                                                                 filter_by_segmentation=False,
                                                                 annotation_color='#00ff00',
                                                                 annotations=view["edge_connectors"])
        except KeyError:
            print("No edges in prediction")
    add_layer(s, prediction_views[0]["raw"], 'raw')

print(viewer)
