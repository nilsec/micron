import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
import scipy.interpolate.interpnd # Workaround for cython __reduce_cython__ error
from micron import read_predict_config, read_worker_config, read_data_config, read_graph_config, read_solve_config

directory = os.path.dirname(os.path.abspath(__file__))
predict_config_path = os.path.join(directory, "predict_config.ini") 
worker_config_path = os.path.join(directory, "worker_config.ini")
data_config_path = os.path.join(directory,"data_config.ini")
graph_config_path = os.path.join(directory, "graph_config.ini")
solve_config_path = os.path.join(directory, "solve_config.ini")

solve_config = read_solve_config(solve_config_path)

base_cmd = "python {} {} {} {} {} {}".format(solve_config["daisy_solve"],
                                             predict_config_path,
                                             worker_config_path,
                                             data_config_path,
                                             graph_config_path,
                                             solve_config_path)

check_call(base_cmd, shell=True)
