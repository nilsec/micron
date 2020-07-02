from micron.post.analyse_graph import construct_matching_graph, evaluate
from pymongo import MongoClient
from micron import read_predict_config, read_graph_config, read_data_config, read_worker_config, read_eval_config, read_solve_config
import sys
import time
import os


def evaluation_pipeline(db_host,
                        db_name,
                        graph_number,
                        solve_number,
                        in_size,
                        in_offset,
                        tracing_file,
                        tracing_offset,
                        tracing_size,
                        voxel_size,
                        subsample_factor,
                        distance_threshold,
                        max_edges,
                        optimality_gap,
                        time_limit,
                        eval_collection_name,
                        **kwargs):

    matching_graph, gt_graph, rec_graph, gt_component_map, rec_component_map =\
        construct_matching_graph(db_host,
                                 db_name,
                                 graph_number,
                                 solve_number,
                                 in_offset,
                                 in_size,
                                 tracing_file,
                                 tracing_offset,
                                 tracing_size,
                                 voxel_size,
                                 subsample_factor,
                                 distance_threshold)

    node_errors, topological_errors = evaluate(matching_graph, max_edges, optimality_gap, time_limit)


    write_evaluation(db_host, 
                     db_name,
                     eval_collection_name,
                     node_errors,
                     topological_errors)


def write_evaluation(db_host,
                     db_name,
                     collection_name,
                     node_errors,
                     topological_errors):

    client = MongoClient(db_host, connect=False)
    db = client[db_name]
    collection = db[collection_name]

    db.drop_collection(collection)

    node_errors["type"] = "nodes"
    collection.insert_one(node_errors)
    topological_errors["type"] = "topological"
    collection.insert_one(topological_errors)


if __name__ == "__main__":
    predict_config = sys.argv[1]
    worker_config = sys.argv[2]
    data_config = sys.argv[3]
    graph_config = sys.argv[4]
    solve_config = sys.argv[5]
    eval_config = sys.argv[6]

    predict_config_dict = read_predict_config(predict_config)
    worker_config_dict = read_worker_config(worker_config)
    data_config_dict = read_data_config(data_config)
    graph_config_dict = read_graph_config(graph_config)
    eval_config_dict = read_eval_config(eval_config)
    solve_config_dict = read_solve_config(solve_config)

    full_config = predict_config_dict
    full_config.update(worker_config_dict)
    full_config.update(data_config_dict)
    full_config.update(graph_config_dict)
    full_config.update(eval_config_dict)
    full_config.update(solve_config_dict)
    eval_collection_name = "evaluation_g{}_s{}_e{}".format(full_config["graph_number"], 
                                                           full_config["solve_number"], 
                                                           full_config["eval_number"])
    full_config.update({"eval_collection_name": eval_collection_name})

    start_time = time.time()
    # Check if setup dir exists:
    setup_directory = os.path.join(full_config["base_dir"], 
                                   full_config["experiment"])
    setup_directory = os.path.join(setup_directory, "05_eval/setup_t{}_p{}_g{}_s{}_e{}".format(full_config["train_number"],
                                                                                               full_config["predict_number"],
                                                                                               full_config["graph_number"],
                                                                                               full_config["solve_number"],
                                                                                               full_config["eval_number"]))
    if not os.path.exists(setup_directory):
        raise ValueError("No setup directory at {}".format(setup_directory))

    evaluation_pipeline(**full_config)
    print("Evaluation took {} seconds".format(time.time() - start_time))
