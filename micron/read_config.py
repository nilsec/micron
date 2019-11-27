import configparser
import os
import numpy as np
import json

def read_eval_config(eval_config):
    config = configparser.ConfigParser()
    config.read(eval_config)

    cfg_dict = {}
    cfg_dict["tracing_file"] = config.get("Evaluate", "tracing_file")
    cfg_dict["tracing_offset"] = tuple([int(v) for v in config.get("Evaluate", "tracing_offset").split(", ")])
    cfg_dict["tracing_size"] = tuple([int(v) for v in config.get("Evaluate", "tracing_size").split(", ")])
    cfg_dict["subsample_factor"] = config.getint("Evaluate", "subsample_factor")
    cfg_dict["distance_threshold"] = config.getint("Evaluate", "distance_threshold")
    cfg_dict["max_edges"] = config.getint("Evaluate", "max_edges")
    cfg_dict["optimality_gap"] = float(config.get("Evaluate", "optimality_gap"))
    cfg_dict["time_limit"] = config.getint("Evaluate", "time_limit")
    cfg_dict["evaluation_pipeline"] = config.get("Evaluate", "evaluation_pipeline")
    cfg_dict["voxel_size"] = tuple([int(v) for v in config.get("Evaluate", "voxel_size").split(", ")])
    cfg_dict["eval_number"] = config.getint("Evaluate", "eval_number")

    return cfg_dict

def read_train_config(train_config):
    config = configparser.ConfigParser()
    config.read(train_config)

    cfg_dict = {}
    cfg_dict["training_container"] = tuple([v for v in config.get("Training", "training_container").split(", ")])
    cfg_dict["raw_dset"] = config.get("Training", "raw_dset")
    cfg_dict["gt_dset"] = config.get("Training", "gt_dset")

    return cfg_dict

def read_graph_config(graph_config):
    config = configparser.ConfigParser()
    config.read(graph_config)

    cfg_dict = {}
    cfg_dict["graph_number"] = int(config.get("Graph", "graph_number"))
    cfg_dict["distance_threshold"] = int(config.get("Graph", "distance_threshold"))
    cfg_dict["block_size"] = tuple([int(v) for v in np.array(config.get("Graph", "block_size").split(", "), dtype=int)])
    cfg_dict["build_graph"] = config.get("Graph", "build_graph")
    try:
        tmp = config.get("Graph", "evidence_threshold")
        if tmp == "None":
            tmp = None
        else:
            tmp = float(tmp)

        cfg_dict["evidence_threshold"] = tmp
    except:
        cfg_dict["evidence_threshold"] = None
        pass

    return cfg_dict

def read_solve_config(solve_config):
    config = configparser.ConfigParser()
    config.read(solve_config)

    cfg_dict = {}
    
    # Solve
    cfg_dict["evidence_factor"] = int(config.get("Solve", "evidence_factor"))
    cfg_dict["comb_angle_factor"] = int(config.get("Solve", "comb_angle_factor"))
    cfg_dict["start_edge_prior"] = int(config.get("Solve", "start_edge_prior"))
    cfg_dict["selection_cost"] = int(config.get("Solve", "selection_cost"))
    cfg_dict["context"] = tuple([int(v) for v in np.array(config.get("Solve", "context").split(", "), dtype=int)])
    cfg_dict["daisy_solve"] = config.get("Solve", "daisy_solve")
    cfg_dict["solve_block"] = config.get("Solve", "solve_block")
    cfg_dict["solve_number"] = int(config.get("Solve", "solve_number"))
    time_limit = config.get("Solve", "time_limit")
    if time_limit == "None":
        cfg_dict["time_limit"] = None
    else:
        cfg_dict["time_limit"] = int(time_limit)
    cfg_dict["selected_attr"] = config.get("Solve", "selected_attr")
    cfg_dict["solved_attr"] = config.get("Solve", "solved_attr")

    return cfg_dict

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
    cfg_dict["out_dataset"] = config.get("Data", "out_dataset")


    # Create json container spec for in_data:
    in_container_spec = {"container": cfg_dict["in_container"],
                         "offset": cfg_dict["in_offset"],
                         "size": cfg_dict["in_size"]}

    in_container_spec_file = os.path.join(os.path.dirname(data_config), "in_container_spec.json")

    with open(in_container_spec_file, "w+") as f:
        json.dump(in_container_spec, f)

    cfg_dict["in_container_spec"] = os.path.abspath(in_container_spec_file)

    return cfg_dict
