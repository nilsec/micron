import daisy
import neuroglancer
import numpy as np
import sys
import os
from funlib.show.neuroglancer import add_layer, ScalePyramid

datasets=["raw", "soft_mask", "tracing"]
base_dir = os.path.abspath(sys.argv[1])
experiment = sys.argv[2]

setup_number = int(sys.argv[3])
snapshot = int(sys.argv[4])
snapshot_path = os.path.join(base_dir, experiment, "01_train/setup_{}/snapshots/batch_{}.hdf".format(setup_number, snapshot))

view_dsets = {}
for dset in datasets:
    view_dsets[dset] = daisy.open_ds(snapshot_path, dset)

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    for dset, data in view_dsets.items():
        add_layer(s, data, dset)

print(viewer)

    
