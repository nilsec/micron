import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
from micron import read_worker_config, read_eval_config

worker_config = read_worker_config("worker_config.ini")
eval_config = read_eval_config("eval_config.ini")

base_cmd = "python {} {} {} {} {} {} {}".format(eval_config["evaluation_pipeline"],
                                          os.path.abspath("predict_config.ini"),
                                          os.path.abspath("worker_config.ini"),
                                          os.path.abspath("data_config.ini"),
                                          os.path.abspath("graph_config.ini"),
                                          os.path.abspath("solve_config.ini"),
                                          os.path.abspath("eval_config.ini"))

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

