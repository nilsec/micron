import os
import sys
from shutil import copyfile
import json
import ConfigParser


def set_up_environment(base_dir,
                       experiment,
                       setup_number,
                       iteration,
                       db_credentials=None):

    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "02_predict/setup_{}".format(setup_number))
    train_setup_dir = os.path.join(os.path.join(base_dir, experiment), "01_train/setup_{}".format(setup_number))
    train_checkpoint = os.path.join(train_setup_dir, "train_net_checkpoint_{}.meta".format(iteration)) 

    if not os.path.exists(train_checkpoint):
        raise ValueError("No checkpoint at {}".format(train_checkpoint))

    if not os.path.exists(predict_setup_dir):
        os.makedirs(predict_setup_dir)

    copyfile("01_network/predict.py", os.path.join(predict_setup_dir, "predict.py"))
    copyfile("01_network/predict_blockwise.py", os.path.join(predict_setup_dir, "predict_blockwise.py"))
    copyfile("01_network/write_candidates.py", os.path.join(predict_setup_dir, "write_candidates.py"))
    for sufix in ["json", "meta"]:
        copyfile(os.path.join(train_setup_dir, "config.{}".format(sufix)), 
                 os.path.join(predict_setup_dir, "config.{}".format(sufix)))
    for sufix in ["index", "meta", "data-00000-of-00001"]:
        copyfile(os.path.join(train_setup_dir, "train_net_checkpoint_{}.{}".format(iteration, sufix)), 
                 os.path.join(predict_setup_dir, "train_net_checkpoint_{}.{}".format(iteration, sufix)))

    conf = create_predict_configs(base_dir,
                                  experiment,
                                  setup_number,
                                  iteration,
                                  db_credentials)

    with open(os.path.join(predict_setup_dir, "config_{}.json".format(iteration)), "w+") as f:
        json.dump(conf, f)


def create_predict_configs(base_dir,
                           experiment,
                           setup_number,
                           iteration,
                           db_credentials=None):

    """
    Create a default predict config file used
    by predict and predict_blockwise downstream.
    Needs to be adjusted after creation.
    """

    if db_credentials is not None:
        with open(db_credentials) as fp:
            config = ConfigParser.ConfigParser()
            config.readfp(fp)
            credentials = {}
            credentials["user"] = config.get("Credentials", "user")
            credentials["password"] = config.get("Credentials", "password")
            credentials["host"] = config.get("Credentials", "host")
            credentials["port"] = config.get("Credentials", "port")

        auth_string = 'mongodb://{}:{}@{}:{}'.format(credentials["user"],
                                                     credentials["password"],
                                                     credentials["host"],
                                                     credentials["port"])

    else:
        auth_string = 'mongodb://localhost'

    conf = {"experiment": experiment,
            "setup": "setup_{}".format(setup_number),
            "iteration": iteration,
            "in_data_config": "path_to_data_config",
            "out_file": "path_to_out_file",
            "num_workers": 1,
            "db_host": auth_string,
            "db_name": "db_name",
            "queue": "slowpoke"}

    return conf


if __name__ == "__main__":
    base_dir = sys.argv[1]
    experiment = sys.argv[2]
    setup_number = int(sys.argv[3])
    iteration = int(sys.argv[4])
    try:
        db_credentials = sys.argv[5]
    except IndexError:
        db_credentials = None
        print("No db credentials provided, standard settings used.")

    set_up_environment(base_dir,
                       experiment,
                       setup_number,
                       iteration,
                       db_credentials)
