import unittest
import os
from micron.solve.solver import Solver
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
        print("solver_v_selected", solver.v_selected)
        v_selected = list(set([solver.t_to_center[t] for s in solver.v_selected for t in s]))
        print("V_selected", v_selected)
        self.assertTrue(len(v_selected) == 1)
        self.assertTrue(v_selected[0] == 3)

        # No isolated edges:
        self.assertTrue(len(solver.e_selected) == 0)

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
        selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

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
        self.assertTrue(len(solver.v_selected) == 1)
        self.assertTrue(len(solver.e_selected) == 0)

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
        selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

        print(selected_nodes)
        print(selected_edges)

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
        self.assertTrue(len(solver.v_selected) == 0)

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
        selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

        print(selected_edges)
        print(selected_nodes)

        self.assertTrue(len(selected_edges) == len(self.edges))
        self.assertTrue(len(selected_nodes) == len(self.nodes))

class TestTrappedVertex(unittest.TestCase):
    """
    This situation leads to an infeasible model
    if edges 4-9 and 7-9 are not solved and 
    thus constraints are not dropped. Needs special 
    handling:


    Solution: Detect situations in which 
    a selected vertex has no incoming selected
    edges and none of them has an unsolved
    target vertex and drop the must pick constraint
    on these.
    """


    def setUp(self):
        # y: 1|   0(s)------1(s)------2(s)
        #    2|
        #    3|   3(s)                6(s)
        #    4|    |                  |
        #    5|   4(s)-(ns)-9(s)-(ns)-7(s)
        #    6|    |                  |
        #    7|   5(s)                8(s)
        #     |-------------------------------->
        # x:       1         2         3
        #
        # s = selected
        # ns = not selected
        self.nodes = [
                {'id': 0, 'z': 1, 'y': 1, 'x': 1, 'selected': True, 'solved': True},
                {'id': 1, 'z': 1, 'y': 1, 'x': 2, 'selected': True, 'solved': True},
                {'id': 2, 'z': 1, 'y': 1, 'x': 3, 'selected': True, 'solved': True},
                {'id': 3, 'z': 1, 'y': 3, 'x': 1, 'selected': True, 'solved': True},
                {'id': 4, 'z': 1, 'y': 5, 'x': 1, 'selected': True, 'solved': True},
                {'id': 5, 'z': 1, 'y': 7, 'x': 1, 'selected': True, 'solved': True},
                {'id': 6, 'z': 1, 'y': 3, 'x': 3, 'selected': True, 'solved': True},
                {'id': 7, 'z': 1, 'y': 5, 'x': 3, 'selected': True, 'solved': True},
                {'id': 8, 'z': 1, 'y': 7, 'x': 3, 'selected': True, 'solved': True},
                {'id': 9, 'z': 1, 'y': 5, 'x': 2, 'selected': True, 'solved': True}
                ]

        self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 1, 'v': 2, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 3, 'v': 4, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 4, 'v': 5, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 6, 'v': 7, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 7, 'v': 8, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 4, 'v': 9, 'evidence': 0.5, 'selected': False, 'solved': False},
                      {'u': 7, 'v': 9, 'evidence': 0.5, 'selected': False, 'solved': False}
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

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
        selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]
        solved_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["solved"]]
 

        self.assertTrue(len(selected_nodes) == len(self.nodes))
        self.assertTrue(set(selected_nodes) == set([v for v in range(10)]))

        self.assertTrue(len(selected_edges) == len(self.edges) - 2)
        self.assertTrue(set(selected_edges) == set([(0,1), (1,2), (3,4), (4,5), (6,7), (7,8)]))

        self.assertTrue(len(solved_edges) == len(self.edges))
        self.assertTrue(set(solved_edges) == set([(0,1), (1,2), (3,4), (4,5), (6,7), (7,8), (7,9), (4,9)]))


        print(selected_edges)
        print(selected_nodes)
        print("Solved", solved_edges)



class TestFullyIsolatedVertexNotSolved(unittest.TestCase):
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
                {'id': 3, 'z': 1, 'y': 3, 'x': 1, 'selected': False, 'solved': False},
                ]

        self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': True, 'solved': True},
                      {'u': 1, 'v': 2, 'evidence': 0.5, 'selected': True, 'solved': True},
                      ]


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

    def test_solve(self):
        print("ISOLATEDNOTSOLVED")
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
        selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

        solved_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["solved"]]
        solved_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["solved"]]

        print("Solved:")
        print(solved_nodes)
        print(solved_edges)

        print("Selected:")
        print(selected_nodes)
        print(selected_edges)

class TestAllTriplets(unittest.TestCase):
    def test_solve(self):
        # y: 1|   0--1--2
        # x:      1  2 3
        # Test all combinations of
        # s/ns

        combinations = [(False, False), (False, True), (True, True)]

        for v0 in combinations:
            for v1 in combinations:
                for v2 in combinations:
                    for e0 in combinations:
                        for e1 in combinations:
                            self.nodes = [
                                    {'id': 0, 'z': 1, 'y': 1, 'x': 1, 'selected': v0[0], 'solved': v0[1]},
                                    {'id': 1, 'z': 1, 'y': 1, 'x': 2, 'selected': v1[0], 'solved': v1[1]},
                                    {'id': 2, 'z': 1, 'y': 1, 'x': 3, 'selected': v2[0], 'solved': v2[1]},
                                    ]

                            self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': e0[0], 'solved': e0[1]},
                                          {'u': 1, 'v': 2, 'evidence': 0.5, 'selected': e1[0], 'solved': e1[1]}]

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

                            solver = Solver(**self.solve_params)
                            solver.initialize()
                            solver.solve()
                            graph = self.solve_params["graph"]
                            selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
                            selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

                            client = pymongo.MongoClient(self.db_host)
                            client.drop_database(self.db_name)

    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)


class TestAllDuplets(unittest.TestCase):
    def test_solve(self):
        # y: 1|   0--1
        # x:      1  2
        # Test all combinations of
        # s/ns

        combinations = [(False, False), (False, True), (True, True)]

        for v0 in combinations:
            for v1 in combinations:
                for e0 in combinations:
                    self.nodes = [
                            {'id': 0, 'z': 1, 'y': 1, 'x': 1, 'selected': v0[0], 'solved': v0[1]},
                            {'id': 1, 'z': 1, 'y': 1, 'x': 2, 'selected': v1[0], 'solved': v1[1]}
                            ]

                    self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': e0[0], 'solved': e0[1]}]

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

                    solver = Solver(**self.solve_params)
                    solver.initialize()
                    solver.solve()
                    graph = self.solve_params["graph"]
                    selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
                    selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

                    client = pymongo.MongoClient(self.db_host)
                    client.drop_database(self.db_name)

    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)


class TestAllSingletons(unittest.TestCase):
    def test_solve(self):
        # y: 1|   0
        # x:      1
        # Test all combinations of
        # s/ns

        combinations = [(False, False), (False, True), (True, True)]

        for v0 in combinations:
            self.nodes = [
                    {'id': 0, 'z': 1, 'y': 1, 'x': 1, 'selected': v0[0], 'solved': v0[1]}
                    ]

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
            
            self.solve_params = {"graph": self.graph, 
                                 "evidence_factor": 12,
                                 "comb_angle_factor": 14,
                                 "start_edge_prior": 180,
                                 "selection_cost": -80}

            solver = Solver(**self.solve_params)
            solver.initialize()
            solver.solve()
            graph = self.solve_params["graph"]
            selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
            selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

            client = pymongo.MongoClient(self.db_host)
            client.drop_database(self.db_name)

    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)



class TestAllTripletsConstrained(unittest.TestCase):
    def test_solve(self):
        # y: 1|  4s
        #    2|  |ns
        #    3|  0--1--2
        #    4|  |s  
        #    5|  3s
        #
        # x:      1  2 3
        # Test all combinations of
        # s/ns

        combinations = [(False, False), (False, True), (True, True)]

        for v0 in combinations:
            for v1 in combinations:
                for v2 in combinations:
                    for e0 in combinations:
                        for e1 in combinations:
                            self.nodes = [
                                    {'id': 0, 'z': 1, 'y': 3, 'x': 1, 'selected': v0[0], 'solved': v0[1]},
                                    {'id': 1, 'z': 1, 'y': 3, 'x': 2, 'selected': v1[0], 'solved': v1[1]},
                                    {'id': 2, 'z': 1, 'y': 3, 'x': 3, 'selected': v2[0], 'solved': v2[1]},
                                    {'id': 3, 'z': 1, 'y': 3, 'x': 1, 'selected': True, 'solved': True},
                                    {'id': 4, 'z': 1, 'y': 1, 'x': 1, 'selected': True, 'solved': True}

                                    ]

                            self.edges = [{'u': 0, 'v': 1, 'evidence': 0.5, 'selected': e0[0], 'solved': e0[1]},
                                          {'u': 1, 'v': 2, 'evidence': 0.5, 'selected': e1[0], 'solved': e1[1]},
                                          {'u': 0, 'v': 3, 'evidence': 0.5, 'selected': True, 'solved': True},
                                          {'u': 0, 'v': 4, 'evidence': 0.5, 'selected': False, 'solved': False}]

                            print("Nodes")
                            for n in self.nodes:
                                print(n)

                            print("Edges")
                            for e in self.edges:
                                print(e)

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

                            solver = Solver(**self.solve_params)
                            solver.initialize()
                            solver.solve()
                            graph = self.solve_params["graph"]
                            selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
                            selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

                            client = pymongo.MongoClient(self.db_host)
                            client.drop_database(self.db_name)

    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)


class TestTwoFullyIsolatedVertices(unittest.TestCase):
    def setUp(self):
        # y: 1|   0(s)--1(s)--2(s)
        #    2|
        #    3|   3(s)  4(s)
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
                {'id': 4, 'z': 1, 'y': 3, 'x': 2, 'selected': True, 'solved': True}
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

    def test_solve(self):
        solver = Solver(**self.solve_params)
        solver.initialize()
        solver.solve()
        graph = self.solve_params["graph"]
        selected_edges = [(e[0], e[1]) for e in graph.edges(data=True) if e[2]["selected"]]
        selected_nodes = [v[0] for v in graph.nodes(data=True) if v[1]["selected"]]

        print(selected_nodes)
        print(selected_edges)

        print(solver.e_to_t)
        print(solver.t_to_e)
        print(solver.center_to_t)
        print(solver.t_to_center)

        self.assertTrue(len(selected_edges) == len(self.edges))
        self.assertTrue(len(selected_nodes) == len(self.nodes))



if __name__ == "__main__":
    unittest.main()
