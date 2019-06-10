import daisy
import neuroglancer
import numpy as np
import sys
from funlib.show.neuroglancer import add_layer, ScalePyramid

f='/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(17)
]

f_prediction = "/groups/funke/home/ecksteinn/Projects/microtubules/micron_experiments/cremi/02_predict/setup_2/test_calix_small.zarr"
dset = 'volumes/soft_mask'

soft_mask = daisy.open_ds(f_prediction, dset)

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, soft_mask, 'soft_mask')
    add_layer(s, raw, 'raw')
#    add_layer(s, lsds, 'mean', shader='rgb')
#    add_layer(s, seg, 'seg')
print(viewer)
