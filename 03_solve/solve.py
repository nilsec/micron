import logging
import numpy as np

import pylp
import daisy
from daisy.persistence import MongoDbGraphProvider

logger = logging.getLogger(__name__)

START_EDGE = (-1, -1, {})

class Solver(object):
    def __init__(self, graph):
        self.graph = graph
        self.nodes = {v: v_data for v, v_data in self.graph.nodes(data=True) if 'z' in v_data}
        # Only consider edges that are fully contained in context - 
        # this does not affect anything just that potentially a larger context has to be chosen
        # compared to the case where edges are grabbed that go over the boarder.
        # This implementation now is in analogy to mtrack original
        self.edges_fully_contained = [tuple(sorted(e)) for e in self.graph.edges if ((e[0] in self.nodes.keys()) and (e[1] in self.nodes.keys()))]
        self.edge_id = {e: k for k, e in enumerate(self.edges_fully_contained)}
        self.edge_id[(START_EDGE[0], START_EDGE[1])] = -1
        self.n_triplets = 0


    def create_indicators(self):
        v_selected = []
        e_selected = []
        t_selected = []
        t_solved = []
        t_center_conflicts = []
        triplets = {}
        center_to_t = {v: [] for v in self.nodes.keys()}
        t_to_center = {}
        t_to_e = {}
        e_to_t = {e: [] for e in self.edges_fully_contained + [(-1,-1)]}

        for v, v_data in self.nodes.items():
            v_incident = [e for e in self.graph.edges(v, data=True) if tuple(sorted((e[0], e[1]))) in self.edges_fully_contained]
            t_conflict = []

            # Detect if vertex is isolated selected:
            if v_data["selected"]:
                v_incident_selected = [e for e in v_incident if e[2]["selected"]]
                v_isolated = False
                if not v_incident_selected:
                    v_selected.append([v, v_data])
                    v_isolated = True

            for e1 in v_incident:
                # Detect if edge is isolated selected:
                e1_sorted = tuple(sorted((e1[0], e1[1])))
                e_isolated = False

                if e1[2]["selected"]:
                    e1_u_selected = [e for e in self.graph.edges(e1[0], data=True) if e[2]["selected"] and e1_sorted in self.edges_fully_contained]
                    e1_v_selected = [e for e in self.graph.edges(e1[1], data=True) if e[2]["selected"] and e1_sorted in self.edges_fully_contained]

                    if len(e1_u_selected) == 1 and len(e1_v_selected) == 1:
                        e_selected.append(e1)
                        e_isolated = True

                # Build triplets:
                for e2 in list(v_incident) + [START_EDGE]:
                    # Avoid doubles:
                    e2_sorted = tuple(sorted((e2[0], e2[1])))

                    e_id_1 = self.edge_id[e1_sorted]
                    e_id_2 = self.edge_id[e2_sorted]
                    e_id_start = self.edge_id[tuple(sorted((START_EDGE[0], START_EDGE[1])))]
 
                    if e_id_1 >= e_id_2 and\
                       e_id_2 != e_id_start:
                        continue

                    # Create triplet:
                    t = self.n_triplets
                    self.n_triplets += 1


                    # Create maps
                    triplets[t] = [e1, e2, v]
                    center_to_t[v].append(t)
                    t_to_center[t] = v
                    t_to_e[t] = (e1_sorted, e2_sorted)
                    e_to_t[e1_sorted].append(t)
                    e_to_t[e2_sorted].append(t)

                    # All triplets centered around v are in conflict:
                    t_conflict.append(t)

                    if not e_isolated:
                        # Detect if the entire triplet is selected/solved if no start edges present:
                        if e1 != START_EDGE and e2 != START_EDGE:
                            vertices_in_t = [e1[0], e1[1], e2[0], e2[1]]
                            vertices_data_in_t = [self.nodes[v] for v in vertices_in_t if v in self.nodes.keys()]
                            vertices_in_t = [v for v in vertices_in_t if v in self.nodes.keys()]

                            v12_solved = [v["solved"] for v in vertices_data_in_t]

                            e1_solved = e1[2]["solved"]
                            e1_selected = e1[2]["selected"]
                            
                            e2_solved = e2[2]["solved"]
                            e2_selected = e2[2]["selected"]


                            if e1_selected and e2_selected:
                                t_selected.append(t)

                            if e1_solved and e2_solved and np.all(v12_solved):
                                t_solved.append(t)

                        elif e1 != START_EDGE and e2 == START_EDGE:
                            e1_solved = e1[2]["solved"]
                            
                            vertices_in_e1 = [e1[0], e1[1]]
                            vertices_data_in_e1 = [self.nodes[v] for v in vertices_in_e1 if v in self.nodes.keys()]
                            assert(len(vertices_data_in_e1) == 2)

                            v12_solved = [v["solved"] for v in vertices_data_in_e1]

                            if e1_solved and np.all(v12_solved):
                                t_solved.append(t)


                        elif e1 == START_EDGE and e2 != START_EDGE:
                            e2_solved = e2[2]["solved"]
                            
                            vertices_in_e2 = [e2[0], e2[1]]
                            vertices_data_in_e2 = [self.nodes[v] for v in vertices_in_e2 if v in self.nodes.keys()]
                            assert(len(vertices_data_in_e2) == 2)

                            v12_solved = [v["solved"] for v in vertices_data_in_e2]

                            if e2_solved and np.all(v12_solved):
                                t_solved.append(t)

            t_center_conflicts.append(t_conflict)


        if v_selected:
            v_selected_tmp = [set(center_to_t[v]) for v in v_selected]
            v_selected = []
            for subset in v_selected_tmp:
                v_selected.append(set([v for v in subset if (START_EDGE[0], START_EDGE[1]) in t_to_e[v]]))


        if e_selected:
            e_selected = [set(e_to_t[e]) for e in e_selected]

        
        maps = {"center_to_t": center_to_t,
                "t_to_center": t_to_center,
                "t_to_e": t_to_e,
                "e_to_t": e_to_t}

        return triplets, t_center_conflicts, t_selected, t_solved, v_selected, e_selected, maps


    def get_continuation_constraints(self, e_to_t, t_to_center):
        continuation_constraints = []

        for e in self.edges_fully_contained:
            e_triplets = e_to_t[e]
            
            v_l = e[0]
            v_r = e[1]

            t_l = [t for t in e_triplets if t_to_center[t] == v_l]
            t_r = [t for t in e_triplets if t_to_center[t] == v_r]

            assert(len(t_l) + len(t_r) == len(e_triplets))
            t_rl = {"t_l": t_l, "t_r": t_r}
            continuation_constraints.append(t_rl)

        return continuation_constraints


    def get_cost(self, t, t_to_e, t_to_center, evidence_factor, comb_angle_factor, start_edge_prior, selection_cost):
        e_in_t = t_to_e[t]
        e_data_in_t = [self.graph.get_edge_data(e[0], e[1]) for e in e_in_t]
        assert(len(e_data_in_t) == 2)

        if (START_EDGE[0], START_EDGE[1]) in e_in_t:
            e_data_in_t.remove(None)
            assert(len(e_data_in_t) == 1)
            edge_cost = evidence_factor * e_data_in_t[0]["evidence"] + 2.0 * start_edge_prior
            comb_edge_cost = 0.0

        else:
            # Edge Cost:
            e_evidence_in_t = [e["evidence"] for e in e_data_in_t]
            edge_cost = np.sum(e_evidence_in_t) * evidence_factor

            # Edge Combination cost:
            v_center = t_to_center[t]
            v_data_in_t = [(v, self.nodes[v]) for v in [e_in_t[0][0], e_in_t[0][1], e_in_t[1][0], e_in_t[1][1]]]
            angle = self.__get_angle(e_in_t[0], e_in_t[1], v_center)

            comb_edge_cost = (angle * comb_angle_factor)**2
            
        return (1./2.) * edge_cost + comb_edge_cost + selection_cost


    def __get_angle(self, e1, e2, v_center):
        v_center_data = self.nodes[v_center]
        v_center_zyx = np.array([v_center_data["z"], v_center_data["y"], v_center_data["x"]])

        v_1 = [e1[0], e1[1]]
        v_2 = [e2[0], e2[1]]
        v_1.remove(v_center)
        v_2.remove(v_center)

        assert(len(v_1) == len(v_2) == 1)
        v_1 = v_1[0]
        v_2 = v_2[0]

        v_1_data = self.nodes[v_1]
        v_2_data = self.nodes[v_2]

        v_1_zyx = np.array([v_1_data["z"], v_1_data["y"], v_1_data["x"]])
        v_2_zyx = np.array([v_2_data["z"], v_2_data["y"], v_2_data["x"]])

        vec_1c = v_1_zyx - v_center_zyx
        vec_2c = v_2_zyx - v_center_zyx
        norm_1c = np.linalg.norm(vec_1c)
        norm_2c = np.linalg.norm(vec_2c)

        u_1c = vec_1c/np.clip(norm_1c, a_min=10**(-8), a_max=None)
        u_2c = vec_2c/np.clip(norm_2c, a_min=10**(-8), a_max=None)

        angle = np.arccos(np.clip(np.dot(u_1c, u_2c), -1.0, 1.0))
        angle = np.pi - angle

        return angle


def get_graph(db_host="mongodb://ecksteinn:ecksteinn@10.150.100.155:27017", 
              db_name="calyx_test_medium",
              roi=daisy.Roi((160000, 123000, 449500), (1000,1000,1000))): 

        graph_provider = MongoDbGraphProvider(db_name,
                                              db_host,
                                              mode='r',
                                              position_attribute=['z', 'y', 'x'])

        graph = graph_provider.get_graph(roi)
        return graph






if __name__ == "__main__":
    graph = get_graph()
    solver = Solver(graph)
    triplets, t_center_conflicts, t_selected, t_solved, v_selected, e_selected, maps = solver.create_indicators()
    continuation_constraints = solver.get_continuation_constraints(maps["e_to_t"], maps["t_to_center"])
    print(solver.get_cost(10, maps["t_to_e"], maps["t_to_center"], 1.0, 1.0, 1.0, 1.0))

    #print(continuation_constraints)
