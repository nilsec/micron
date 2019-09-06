import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
from micron import read_predict_config, read_worker_config, read_data_config, read_graph_config



predict_config = read_predict_config("predict_config.ini")
worker_config = read_worker_config("worker_config.ini")
data_config = read_data_config("data_config.ini")
graph_config = read_graph_config("graph_config.ini")

base_cmd = "python {} {} {} {} {}".format(graph_config["build_graph"],
                                          os.path.abspath("predict_config.ini"),
                                          os.path.abspath("worker_config.ini"),
                                          os.path.abspath("data_config.ini"),
                                          os.path.abspath("graph_config.ini")) 

if worker_config["singularity_container"] != "None" and worker_config["queue"] == "None":
    run_singularity(base_cmd,
                    singularity_image=worker_config["singularity_container"],
                    mount_dirs=worker_config["mount_dirs"],
                    execute=True)

elif worker_config["singularity_container"] != "None" and worker_config["queue"] != "None":
    run(base_cmd, 
        singularity_image=worker_config["singularity_container"],
        mount_dirs=worker_config["mount_dirs"],
        queue=worker_config["queue"],
        num_cpus=worker_config["num_cpus"],
        num_gpus=0,
        batch=True,
        execute=True)

else:
    assert(worker_config["singularity_container"] == "None")
    check_call(base_cmd, shell=True)
