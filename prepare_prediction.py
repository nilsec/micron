import os
import sys
from shutil import copyfile
import json
import ConfigParser


def set_up_environment(base_dir,
                       experiment,
                       setup_number,
                       iteration,
                       db_name=None,
                       db_credentials=None,
                       singularity_container=None,
                       queue=None,
                       num_cpus=None,
                       num_cache_workers=None,
                       num_block_workers=None,
                       in_container=None,
                       in_dataset=None,
                       in_offset=(None,None,None),
                       in_size=(None,None,None),
                       out_container=None):

    input_params = locals()

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

    predict_config = create_predict_config(**input_params)

    with open(os.path.join(predict_setup_dir, "predict_config_template.ini"), "w+") as f:
        predict_config.write(f)


def create_predict_config(base_dir,
                          experiment,
                          setup_number,
                          iteration,
                          db_name,
                          db_credentials,
                          singularity_container,
                          queue,
                          num_cpus,
                          num_cache_workers,
                          num_block_workers,
                          in_container,
                          in_dataset,
                          in_offset,
                          in_size,
                          out_container):

    config = ConfigParser.ConfigParser()

    config.add_section('Predict')
    config.set('Predict', 'base_dir', str(os.path.abspath(base_dir)))
    config.set('Predict', 'experiment', str(experiment))
    config.set('Predict', 'setup_number', str(setup_number))
    config.set('Predict', 'iteration', str(iteration))

    config.add_section('Database')
    config.set('Database', 'db_name', str(db_name))
    if db_credentials is not None:
        with open(db_credentials) as fp:
            config_db = ConfigParser.ConfigParser()
            config_db.readfp(fp)
            credentials = {}
            credentials["user"] = config_db.get("Credentials", "user")
            credentials["password"] = config_db.get("Credentials", "password")
            credentials["host"] = config_db.get("Credentials", "host")
            credentials["port"] = config_db.get("Credentials", "port")

        auth_string = 'mongodb://{}:{}@{}:{}'.format(credentials["user"],
                                                     credentials["password"],
                                                     credentials["host"],
                                                     credentials["port"])

    else:
        auth_string = 'mongodb://localhost'
    config.set('Database', 'db_host', auth_string)

    config.add_section('Worker')
    config.set('Worker', 'singularity_container', str(singularity_container))
    config.set('Worker', 'num_cpus', str(num_cpus))
    config.set('Worker', 'num_block_workers', str(num_block_workers))
    config.set('Worker', 'num_cache_workers', str(num_cache_workers))
    config.set('Worker', 'queue', str(queue))

    config.add_section('Data')
    config.set('Data', 'in_container', str(in_container))
    config.set('Data', 'in_dataset', str(in_dataset))
    config.set('Data', 'in_offset', str(in_offset[0]) + ", " +\
                                    str(in_offset[1]) + ", " +\
                                    str(in_offset[2]))
    config.set('Data', 'in_size', str(in_size[0]) + ", " +\
                                  str(in_size[1]) + ", " +\
                                  str(in_size[2]))
    config.set('Data', 'out_container', str(out_container))

    return config


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
                       db_credentials=db_credentials)
