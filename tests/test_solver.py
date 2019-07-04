import unittest
import os
from micron.solve.solve_block import Solver
import daisy
from daisy.persistence import MongoDbGraphProvider
import pymongo
import configparser
import os


class TestIsolatedVertexSelectedNotSolved(unittest.TestCase):
    def setUp(self):
        # y: 1|   0(s)--1(s)--2(s)
        #    2|
        #    3|   3(s)--4(ns)--5(ns) 
        #     |--------------------->
        # x:     1     2      3
        #
        # s = selected
        # ns = not selected
        self.nodes = [
                {'id': 0, 'z': 1, 'y': 1, 'x': 1, 'selected': True, 'solved': True},
                {'id': 1, 'z': 1, 'y': 1, 'x': 2, 'selected': True, 'solved': True},
                {'id': 2, 'z': 1, 'y': 1, 'x': 3, 'selected': True, 'solved': True},
                {'id': 3, 'z': 1, 'y': 3, 'x': 1, 'selected': True, 'solved': True},
                {'id': 4, 'z': 1, 'y': 3, 'x': 2, 'selected': False, 'solved': False},
                {'id': 5, 'z': 1, 'y': 3, 'x': 3, 'selected': False, 'solved': False}
                ]

        self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 1, 'v': 2, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 3, 'v': 4, 'evidence': 0.5, 'selected': False, 'solved': False},
                      {'u': 4, 'v': 5, 'evidence': 0.5, 'selected': False, 'solved': False}]

        self.db_name = 'micron_test_solver'
        config = configparser.ConfigParser()
        config.read(os.path.expanduser("../mongo.ini"))
        self.db_host = "mongodb://{}:{}@{}:{}".format(config.get("Credentials", "user"),
                                                      config.get("Credentials", "password"),
                                                      config.get("Credentials", "host"),
                                                      config.get("Credentials", "port"))

        self.graph_provider = MongoDbGraphProvider(self.db_name,
                                                   self.db_host,
                                                   mode='w',
                                                   position_attribute=['z', 'y', 'x'])
        self.roi = daisy.Roi((0,0,0), (4,4,4))
        self.graph = self.graph_provider[self.roi]
        self.graph.add_nodes_from([(node['id'], node) for node in self.nodes])
        self.graph.add_edges_from([(edge['u'], edge['v'], edge) for edge in self.edges])
        
        self.solve_params = {"graph": self.graph, 
                             "evidence_factor": 12,
                             "comb_angle_factor": 14,
                             "start_edge_prior": 180,
                             "selection_cost": -80}

    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)


    def test_constraints(self):
        solver = Solver(**self.solve_params)
        solver.initialize()

        # Only 3 is isolated selected:
        v_selected = [solver.t_to_center[t] for s in solver.v_selected for t in s] 
        self.assertTrue(len(v_selected) == 1)
        self.assertTrue(v_selected[0] == 3)

        # No isolated edges:
        self.assertTrue(len(solver.e_selected) == 0)

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [e for e in graph.edges]
        selected_nodes = [v for v in graph.nodes]
        self.assertTrue(len(selected_edges) == len(self.edges))
        self.assertTrue(len(selected_nodes) == len(self.nodes))


class TestFullyIsolatedVertex(unittest.TestCase):
    def setUp(self):
        # y: 1|   0(s)--1(s)--2(s)
        #    2|
        #    3|   3(s) 
        #     |--------------------->
        # x:     1     2      3
        #
        # s = selected
        # ns = not selected
        self.nodes = [
                {'id': 0, 'z': 1, 'y': 1, 'x': 1, 'selected': True, 'solved': True},
                {'id': 1, 'z': 1, 'y': 1, 'x': 2, 'selected': True, 'solved': True},
                {'id': 2, 'z': 1, 'y': 1, 'x': 3, 'selected': True, 'solved': True},
                {'id': 3, 'z': 1, 'y': 3, 'x': 1, 'selected': True, 'solved': True},
                ]

        self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 1, 'v': 2, 'evidence': 0.5, 'selected': True, 'solved': True},
                      ]

        #self.nodes = self.nodes[:3]

        self.db_name = 'micron_test_solver'
        config = configparser.ConfigParser()
        config.read(os.path.expanduser("../mongo.ini"))
        self.db_host = "mongodb://{}:{}@{}:{}".format(config.get("Credentials", "user"),
                                                      config.get("Credentials", "password"),
                                                      config.get("Credentials", "host"),
                                                      config.get("Credentials", "port"))

        self.graph_provider = MongoDbGraphProvider(self.db_name,
                                                   self.db_host,
                                                   mode='w',
                                                   position_attribute=['z', 'y', 'x'])
        self.roi = daisy.Roi((0,0,0), (4,4,4))
        self.graph = self.graph_provider[self.roi]
        self.graph.add_nodes_from([(node['id'], node) for node in self.nodes])
        self.graph.add_edges_from([(edge['u'], edge['v'], edge) for edge in self.edges])
        
        self.solve_params = {"graph": self.graph, 
                             "evidence_factor": 12,
                             "comb_angle_factor": 14,
                             "start_edge_prior": 180,
                             "selection_cost": -80}


    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)

    def test_constraints(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        # There is no option to continue and
        # form a triplet, ignoring the isolated vertex,
        # Thus v_selected only holds one set which is empty:
        self.assertTrue(len(solver.v_selected) == 1)
        self.assertFalse(solver.v_selected[0])
        # No isolated edges:
        self.assertTrue(len(solver.e_selected) == 0)

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [e for e in graph.edges]
        selected_nodes = [v for v in graph.nodes]

        self.assertTrue(len(selected_edges) == len(self.edges))
        self.assertTrue(len(selected_nodes) == len(self.nodes))


class TestEdgeIsolated(unittest.TestCase):
    def setUp(self):
        # y: 1|   0(s)--1(s)--2(s)
        #    2|
        #    3|   3(s)--4(s)
        #     |--------------------->
        # x:     1     2      3
        #
        # s = selected
        # ns = not selected
        self.nodes = [
                {'id': 0, 'z': 1, 'y': 1, 'x': 1, 'selected': True, 'solved': True},
                {'id': 1, 'z': 1, 'y': 1, 'x': 2, 'selected': True, 'solved': True},
                {'id': 2, 'z': 1, 'y': 1, 'x': 3, 'selected': True, 'solved': True},
                {'id': 3, 'z': 1, 'y': 3, 'x': 1, 'selected': True, 'solved': True},
                {'id': 4, 'z': 1, 'y': 4, 'x': 1, 'selected': True, 'solved': True}

                ]

        self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 1, 'v': 2, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 3, 'v': 4, 'evidence': 0.5, 'selected': True, 'solved': True}
                      ]

        #self.nodes = self.nodes[:3]

        self.db_name = 'micron_test_solver'
        config = configparser.ConfigParser()
        config.read(os.path.expanduser("../mongo.ini"))
        self.db_host = "mongodb://{}:{}@{}:{}".format(config.get("Credentials", "user"),
                                                      config.get("Credentials", "password"),
                                                      config.get("Credentials", "host"),
                                                      config.get("Credentials", "port"))

        self.graph_provider = MongoDbGraphProvider(self.db_name,
                                                   self.db_host,
                                                   mode='w',
                                                   position_attribute=['z', 'y', 'x'])
        self.roi = daisy.Roi((0,0,0), (4,4,4))
        self.graph = self.graph_provider[self.roi]
        self.graph.add_nodes_from([(node['id'], node) for node in self.nodes])
        self.graph.add_edges_from([(edge['u'], edge['v'], edge) for edge in self.edges])
        
        self.solve_params = {"graph": self.graph, 
                             "evidence_factor": 12,
                             "comb_angle_factor": 14,
                             "start_edge_prior": 180,
                             "selection_cost": -80}


    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)

    def test_constraints(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        # There is no option to continue and
        # form a triplet, ignoring the isolated vertex,
        # Thus v_selected only holds one set which is empty:
        self.assertTrue(len(solver.v_selected) == 0)
        # 1 isolated edge:
        print("E SELECTED: ", solver.e_selected)
        print("TRIPLETS: ", solver.triplets)

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [e for e in graph.edges]
        selected_nodes = [v for v in graph.nodes]

        print(selected_edges)
        print(selected_nodes)

        self.assertTrue(len(selected_edges) == len(self.edges))
        self.assertTrue(len(selected_nodes) == len(self.nodes))



if __name__ == "__main__":
    unittest.main()
