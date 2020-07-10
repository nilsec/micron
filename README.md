# Library for automatic tracking of microtubules in large scale EM datasets
![](calyx.gif)
Rendering of automatically reconstructed microtubules in selected, automatically segmented neurons in the Calyx, a 76 x 52 x 65 micron region of the Drosophila Melanogaster brain. Microtubules of the same color belong to the same neuron.

## Prerequisites

1. Install and start a mongodb instance:

- Download and install [mongodb](https://www.mongodb.com/)
- Start a mongodb server on your local machine or a server of your choice via
```
sudo mongod --config <path_to_config>
```

2. For usage in a container environment a [Gurobi floating licencse](https://www.gurobi.com/documentation/8.1/quickstart_mac/setting_up_and_using_a_flo.html) is required.
   If that is not available a free academic license can be obtained [here](https://www.gurobi.com/downloads/end-user-license-agreement-academic/). In the latter case
   task 4 (solving the constrained optimization problem) does not support usage of the provided singularity container.


3. Install [Singularity](https://singularity.lbl.gov/docs-installation)

## Installation
1. Clone the repository
```
git clone https://github.com/nilsec/micron.git
```
2. Install provided conda environment
```
cd micron
conda env create -f micron.yml
```
3. Install micron and build singularity image
```
conda activate micron
make
make singularity
```

## Usage
Reconstructing microtubules in any given EM dataset consists of the following 4 steps:

##### 1. Training a network:

```
cd micron/micron
python prepare_training.py -d <base_dir> -e <experiment_name> -t <id_of_training_run>
```

This will create a directory at 
```
<base_dir>/<experiment_name>/01_train/setup_t<id_of_training_run> 
```
with all the necessary 
files to train a network that can detect microtubules in EM data.

In order to train a network on your data you need to provide ground truth skeletons and the corresponding raw data.
The paths to the data need to be specified in the provided ```train_config.ini```. Ground truth skeletons should be given
as volumetric data where each skeleton is represented by a corresponding id in the ground truth volume. Raw 
data and ground truth should have the same shape, background should be labeled as zero.

Our training data traced on the 3 CREMI test cubes and raw tracings (Knossos skeletons)
is available [here](https://github.com/nilsec/micron_data.git) and 
can be used for microtubule prediction on FAFB. If you want to train on your own data this can be used as an example
of how to format your data for training. 

An example train_config.ini:
```
training_container = ~/micron_data/a+_master.h5, ~/micron_data/b+_master.h5, ~/micron_data/c+_master.h5
raw_dset = raw
gt_dset = tracing
```
Once the appropriate changes have been made to the train config, network training can be started
via: 
```
python train.py <num_iterations>
```
which will train the network for num_iterations (e.g. 300000) iterations on the provided data and
training checkpoints will be saved every 1000 iterations.

##### 2. Predicting microtubule candidates:

```
cd micron/micron
python prepare_prediction -d <base_dir> -e <experiment_name> -t <id_of_training_run> -i <checkpoint/iteration> -p <id_of_prediction>
```

This will create a directory at 
```
<base_dir>/<experiment_name>/02_predict/setup_t<id_of_training_run>_<id_of_prediction>
```
 with all the
necessary files to predict a region of interest with an already trained network as specified by the -t and -i flags.

In particular the directory will hold 3 config files that specify parameters for the given predict run:

1. data_config.ini
    Specifies the paths and region of interests for the prediction run. Offset and size 
    should be given in world coordinates. An example config for fafb prediction looks like
    the following:
    
```
[Data]
in_container = ./fafb.n5
in_dataset = /volumes/raw/s0
in_offset = 158000, 121800, 403560
in_size = 76000, 52000, 64000
out_container = ./softmask.zarr
```

2. ```predict_config.ini```
	Holds paths to necessary scripts and ids as specified. Furthermore it
    contains information about the database to write the predictions to.
    The db_host entry should be adjusted to point to the mongodb 
    instance that was set up earlier. All other settings are fixed 
    and should not be modified.


3. ```worker_config.ini```
    Holds information about how many workers (and thus GPUs) to use
    for the prediction. Furthermore a singularity container
    to run the prediction in can be specified as well as
    the name of any job queue that might be available on a cluster.
    If ```None``` is given the prediction will be run locally.

If the necessary adjustments have been made a prediction can be started via
```
python predict.py 
```

Once started the predict script writes microtubule candidates to the specified database and 
keeps track of which blocks have been predicted. Restarting the prediction will skip already 
processed blocks. Logs for each worker are written to
 ``` 
./worker_files/<worker_id>_worker.out
```

The final two steps follow the same exact pattern and each generate one additional config file that should be 
edited to need.
##### 3. Constructing the microtubule graph:


```
cd micron/micron
python prepare_graph.py -d <base_dir> -e <experiment_name> -t <id_of_training_run> -p <id_of_prediction> -g <id_of_graph>
```
Go to the newly created directory, edit config files to need.
```
python graph.py
```

##### 4. Solving the constrained optimization problem to extract final microtubule trajectories:
```
cd micron/micron
python prepare_solve.py -d <base_dir> -e <experiment_name> -t <id_of_training_run> -p <id_of_prediction> -g <id_of_graph> -s <id_of_solve_run>
```
Go to the newly created directory, edit config files to need.
```
python solve.py
```





