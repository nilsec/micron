import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
from micron import read_worker_config, read_train_config

iteration = int(sys.argv[1])
worker_config = read_worker_config("worker_config.ini")

base_cmd = "python {} {}".format("train_pipeline.py", iteration)
					  
if worker_config["singularity_container"] != "None":
    run_singularity("python mknet.py",
                    singularity_image=worker_config["singularity_container"],
                    mount_dirs=["/groups", "/nrs", "/scratch", "/misc"],
                    execute=True)

else:
    check_call("python mknet.py",
               shell=True)


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
        num_cpus=num_cpus,
        num_gpus=1,
        batch=False,
        shell=True)

else:
    assert(worker_config["singularity_container"] == "None")
    check_call(base_cmd, shell=True)
