import os
from subprocess import check_call
from pysub import run, run_singularity
import logging
from micron import read_predict_config, read_worker_config, read_data_config
import time

predict_config = read_predict_config("predict_config.ini")
worker_config = read_worker_config("worker_config.ini")
data_config = read_data_config("data_config.ini")

start = time.time()

if worker_config["singularity_container"] != "None":
    run_singularity("python mknet.py",
                    singularity_image=worker_config["singularity_container"],
                    mount_dirs=["/groups", "/nrs", "/scratch", "/misc"],
                    execute=True)

else:
    check_call("python mknet.py",
               shell=True)


check_call("python {} {} {} {}".format(predict_config["blockwise"],
                                       os.path.abspath("predict_config.ini"),
                                       os.path.abspath("worker_config.ini"),
                                       os.path.abspath("data_config.ini")),
                                       shell=True)


end = time.time()
with open("./time_prediction.json", "w") as f:
    json.dump({"t_predict": end - start}, f)    
