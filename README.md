# Library for automatic tracking of microtubules in large scale EM datasets


## Prerequisites

1. Install and start a mongodb instance:

- Download and install mongodb https://www.mongodb.com/
- Start a mongodb server on your local machine or a server of your choice via
```
sudo mongod --config /etc/mongod.conf 
```

2. Gurobi setup (path in Singularity)


3. Install Singularity and build image


## Usage
Reconstructing microtubules in any given EM dataset consists of the following 4 steps:

##### 1. Training a network:

```
cd micron/micron
python prepare_training.py -d <base_dir> -e <experiment_name> -t <id_of_training_run>
```

This will create a directory at <base_dir>/<experiment_name>/01_train/setup_t<id_of_training_run> with all the necessary 
files to train a network that can detect microtubules in EM data.

In order to train a network on your data you need to provide ground truth skeletons and the corresponding raw data.
The paths to the data need to be specified in the provided train_config.ini. Ground truth skeletons should be given
as volumetric data where each skeleton is represented by a corresponding id in the ground truth volume. Raw 
data and ground truth should have the same shape, background should be labeled as zero.

An example train_config.ini:
```
training_container = ../00_data/a+_master.h5, ../00_data/b+_master.h5, ../00_data/c+_master.h5
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

This will create a directory at <base_dir>/<experiment_name>/02_predict/setup_t<id_of_training_run>_<id_of_prediction> with all the
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

2. predict_config.ini
	Holds paths to necessary scripts and ids as specified. Furthermore it
    contains information about the database to write the predictions to.
    The db_host entry should be adjusted to point to the mongodb 
    instance that was set up earlier. All other settings are fixed 
    and should not be modified.


3. worker_config.ini
    Holds information about how many workers (and thus GPUs) to use
    for the prediction. Furthermore a singularity container
    to run the prediction in can be specified as well as
    the name of any job queue that might be available on a cluster.
    If ```None``` is given the prediction will be run locally.

If the necessary adjustmants have been made a prediction can be started via
```
python predict.py 
```

Once started the predict script writes microtubule candidates to the specified database and 
keeps track of which blocks have been predicted. Restarting the prediction will skip already 
processed blocks. Logs for each worker are written to ./worker_files/<worker_id>_worker.out.

##### 3. Constructing the microtubule graph:
```
cd micron/micron
python prepare_graph.py -d <base_dir> -e <experiment_name> -t <id_of_training_run> -p <id_of_prediction> -g <id_of_graph>
```

##### 4. Solving the constrained optimization problem to extract final microtubule trajectories:
```
cd micron/micron
python prepare_solve.py -d <base_dir> -e <experiment_name> -t <id_of_training_run> -p <id_of_prediction> -g <id_of_graph>
```





