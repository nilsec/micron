import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
import scipy.interpolate.interpnd # Workaround for cython __reduce_cython__ error
from micron import read_predict_config, read_worker_config, read_data_config, read_graph_config, read_solve_config

predict_config = read_predict_config("predict_config.ini")
worker_config = read_worker_config("worker_config.ini")
data_config = read_data_config("data_config.ini")
graph_config = read_graph_config("graph_config.ini")
solve_config = read_solve_config("solve_config.ini")

base_cmd = "python {} {} {} {} {} {}".format(solve_config["daisy_solve"],
                                             os.path.abspath("predict_config.ini"),
                                             os.path.abspath("worker_config.ini"),
                                             os.path.abspath("data_config.ini"),
                                             os.path.abspath("graph_config.ini"),
                                             os.path.abspath("solve_config.ini"))

check_call(base_cmd, shell=True)
