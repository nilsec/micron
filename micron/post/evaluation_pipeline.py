from micron.post.analyse_graph import construct_matching_graph, evaluate
from pymongo import MongoClient
from micron import read_predict_config, read_graph_config, read_data_config, read_worker_config, read_eval_config, read_solve_config
import sys
import time
import os


def evaluation_pipeline(db_host,
                        db_name,
                        base_dir,
                        experiment,
                        train_number,
                        predict_number,
                        graph_number,
                        solve_number,
                        eval_number,
                        tracing_file,
                        tracing_offset,
                        tracing_size,
                        voxel_size,
                        subsample_factor,
                        distance_threshold,
                        max_edges,
                        optimality_gap,
                        time_limit,
                        **kwargs):


    setup_directory = os.path.abspath(".")

    matching_graph, gt_graph, rec_graph, gt_component_map, rec_component_map =\
        construct_matching_graph(setup_directory,
                                 graph_number,
                                 solve_number,
                                 tracing_file,
                                 tracing_offset,
                                 tracing_size,
                                 voxel_size,
                                 subsample_factor,
                                 distance_threshold)

    node_errors, topological_errors = evaluate(matching_graph, max_edges, optimality_gap, time_limit)

    collection_name = "evaluation_g{}_s{}_e{}".format(graph_number, solve_number, eval_number)

    write_evaluation(db_host, 
                     db_name,
                     collection_name,
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

    start_time = time.time()
    evaluation_pipeline(**full_config)
    print("Evaluation took {} seconds".format(time.time() - start_time))
