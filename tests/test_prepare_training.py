import unittest
import os
from micron.prepare_training import set_up_environment


class TestPrepareTrain(unittest.TestCase):
    def setUp(self):
        self.base_dir = "."
        self.experiment = "tests/data"
        self.train_number = 0

    def tearDown(self):
        set_up_environment(self.base_dir,
                           self.experiment,
                           self.train_number,
                           True)

    def test_set_up_env(self):
        set_up_environment(self.base_dir,
                           self.experiment,
                           self.train_number)

        setup_dir = os.path.join(self.base_dir,
                                 self.experiment,
                                 "01_train",
                                 "train_{}".format(self.train_number))
        mknet = os.path.join(setup_dir, "mknet.py")
        train = os.path.join(setup_dir, "train.py")
        self.assertTrue(os.path.exists(mknet))
        self.assertTrue(os.path.exists(train))

