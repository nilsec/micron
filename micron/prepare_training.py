import os
import sys
from shutil import copyfile, rmtree
import configargparse
import configparser
import click

p = configargparse.ArgParser()
p.add('-d', '--base_dir', required=False, 
      help='base directory for storing micron experiments, defaults to ``~/micron_experiments``', 
      default='~/micron_experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb')
p.add('-t', required=True, help='train number/id for this particular run')
p.add('-c', required=False, action='store_true', help='clean up - remove specified train setup')

def set_up_environment(base_dir,
                       experiment,
                       train_number,
                       clean_up=False):
    ''' Sets up the directory structure and config file for 
        training a network for microtubule prediction.

    Args:

        base_dir (``string``):

            The base directory for storing all micron related experiments and data.

        experiment (``string``):

            The name of the experiment this training run belongs to.

        train_number (``int``):

            The number/id of the training run.

        clean_up (``bool``):

            If true removes the specified train directory
    '''


    base_dir = os.path.expanduser(base_dir)
    setup_dir = os.path.join(base_dir, experiment, "01_train/setup_t{}".format(train_number))

    if clean_up:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to remove {} and all its contents?'.format(setup_dir), default=False):
                rmtree(setup_dir)
            else:
                print("Abort clean up.")
                return
        else:
            rmtree(setup_dir)
    else:
        if not (os.path.exists(setup_dir)):
            try:
                os.makedirs(setup_dir)
            except:
                raise ValueError("Cannot create setup {}, path invalid".format(setup_dir))
        else:
            raise ValueError("Cannot create setup {}, setup exists already.".format(setup_dir))

        this_dir = os.path.dirname(__file__)
        copyfile(os.path.join(this_dir, "network/mknet.py"), os.path.join(setup_dir, "mknet.py"))
        copyfile(os.path.join(this_dir, "network/train_pipeline.py"), os.path.join(setup_dir, "train_pipeline.py"))
        copyfile(os.path.join(this_dir, "network/train.py"), os.path.join(setup_dir, "train.py"))
        
        train_config = create_train_config(training_container=["None", "None", "None"],
                                           raw_dset=None,
                                           gt_dset=None)

        worker_config = create_worker_config(mount_dirs=None,
                                             singularity=os.path.abspath("../singularity/micron.img"),
                                             queue=None)

        with open(os.path.join(setup_dir, "train_config.ini"), "w+") as f:
            train_config.write(f)

        with open(os.path.join(setup_dir, "worker_config.ini"), "w+") as f:
            worker_config.write(f)



def create_train_config(training_container,
                        raw_dset,
                        gt_dset):

    config = configparser.ConfigParser()

    config.add_section('Training')
    container_str = ""
    for container in training_container:
        container_str += "{}, ".format(container)
    container_str = container_str[:-2]
    config.set('Training', 'training_container', container_str)
    config.set('Training', 'raw_dset', str(raw_dset))
    config.set('Training', 'gt_dset', str(gt_dset))

    return config


def create_worker_config(mount_dirs,
                         singularity,
                         queue):

    config = configparser.ConfigParser()
    config.add_section('Worker')
    if singularity == None or singularity == "None" or not singularity:
        config.set('Worker', 'singularity_container', str(None))
    else:
        config.set('Worker', 'singularity_container', str(singularity))
    config.set('Worker', 'num_cpus', str(5))
    config.set('Worker', 'num_block_workers', str(1))
    config.set('Worker', 'num_cache_workers', str(5))
    if queue == None or queue == "None" or not queue:
        config.set('Worker', 'queue', str(None))
    else:
        config.set('Worker', 'queue', str(queue))
    if mount_dirs == None or mount_dirs == "None" or not mount_dirs:
        config.set('Worker', 'mount_dirs', "")
    else:
        config.set('Worker', 'mount_dirs', mount_dirs)
    return config



if __name__ == "__main__":
    options = p.parse_args()

    base_dir = options.base_dir
    experiment = options.e
    train_number = int(options.t)
    clean_up = bool(options.c)
    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       clean_up)
