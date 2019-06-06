import os
import sys
from shutil import copyfile

def set_up_environment(base_dir,
                       setup_number):

    setup_dir = os.path.join(base_dir, "setup_{}".format(setup_number))

    if not os.path.exists(setup_dir):
        os.makedirs(setup_dir)

    copyfile("01_network/mknet.py", os.path.join(setup_dir, "mknet.py"))
    copyfile("01_network/train.py", os.path.join(setup_dir, "train.py"))
    copyfile("01_network/nms.py", os.path.join(setup_dir, "nms.py"))

if __name__ == "__main__":
    base_dir = sys.argv[1]
    setup_number = int(sys.argv[2])
    set_up_environment(base_dir,
                       setup_number)
