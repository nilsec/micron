import configparser
import os
import numpy as np
import json


def read_predict_config(predict_config):
    config = configparser.ConfigParser()
    config.read(predict_config)

    cfg_dict = {}

    # Predict
    cfg_dict["blockwise"] = config.get("Predict", "blockwise")
    cfg_dict["base_dir"] = config.get("Predict", "base_dir")
    cfg_dict["experiment"] = config.get("Predict", "experiment")
    cfg_dict["train_number"] = int(config.getint("Predict", "train_number"))
    cfg_dict["predict_number"] = int(config.getint("Predict", "predict_number"))
    cfg_dict["iteration"] = int(config.getint("Predict", "iteration"))

    # Database
    cfg_dict["db_name"] = config.get("Database", "db_name")
    cfg_dict["db_host"] = config.get("Database", "db_host")
    
    return cfg_dict

def read_worker_config(worker_config):
    config = configparser.ConfigParser()
    config.read(worker_config)

    cfg_dict = {}

    # Worker
    cfg_dict["singularity_container"] = config.get("Worker", "singularity_container")
    cfg_dict["num_cpus"] = int(config.getint("Worker", "num_cpus"))
    cfg_dict["num_block_workers"] = int(config.getint("Worker", "num_block_workers"))
    cfg_dict["num_cache_workers"] = int(config.getint("Worker", "num_cache_workers"))
    cfg_dict["queue"] = config.get("Worker", "queue")
    cfg_dict["mount_dirs"] = tuple([v for v in config.get("Worker", "mount_dirs").split(", ")])

    return cfg_dict

def read_data_config(data_config):
    config = configparser.ConfigParser()
    config.read(data_config)

    cfg_dict = {}

    # Data
    cfg_dict["in_container"] = config.get("Data", "in_container")
    cfg_dict["in_dataset"] = config.get("Data", "in_dataset")
    cfg_dict["in_offset"] = tuple([int(v) for v in np.array(config.get("Data", "in_offset").split(", "), dtype=int)])
    cfg_dict["in_size"] = tuple([int(v) for v in np.array(config.get("Data", "in_size").split(", "), dtype=int)])
    cfg_dict["out_container"] = config.get("Data", "out_container")

    # Create json container spec for in_data:
    in_container_spec = {"container": cfg_dict["in_container"],
                         "offset": cfg_dict["in_offset"],
                         "size": cfg_dict["in_size"]}

    in_container_spec_file = os.path.join(os.path.dirname(data_config), "in_container_spec.json")

    with open(in_container_spec_file, "w+") as f:
        json.dump(in_container_spec, f)

    cfg_dict["in_container_spec"] = os.path.abspath(in_container_spec_file)

    return cfg_dict
