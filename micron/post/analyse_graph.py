import daisy
from daisy.persistence import MongoDbGraphProvider
import numpy as np
from funlib.segment.graphs import find_connected_components 
import operator
import networkx as nx
import copy
from scipy.spatial import KDTree
import os

from micron import read_data_config, read_solve_config, read_predict_config
from micron.post.nml_io import parse_nml
from comatch import match_components


def evaluate(matching_graph,
             max_edges=1,
             optimality_gap=0.0,
             time_limit=None,
             n_gts=-1,
             n_recs=-1):


    
    nodes_gt, nodes_rec, labels_gt, labels_rec, edges_gt_rec, edge_conflicts = matching_graph.export()

    label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec,
                                                                                             edges_gt_rec, labels_gt, labels_rec,
                                                                                             edge_conflicts=edge_conflicts,
                                                                                             max_edges=max_edges,
                                                                                             optimality_gap=optimality_gap,
                                                                                             time_limit=time_limit)

    topological_errors = {"n_gt": n_gts, "n_rec": n_recs, "splits": num_splits, "merges": num_merges, "fps": num_fps, "fns": num_fns}

    matching_graph.import_node_matches(node_matches)

    node_errors = matching_graph.get_stats()

    return node_errors, topological_errors


def construct_matching_graph(setup_directory,
                             graph_number,
                             solve_number,
                             tracing_file,
                             tracing_offset,
                             tracing_size,
                             voxel_size,
                             subsample_factor,
                             distance_threshold):

    voxel_size = np.array(voxel_size)

    edge_collection = "edges_g{}".format(graph_number)

    if not os.path.exists(setup_directory):
        raise ValueError("No setup directory at {}".format(setup_directory))


    data_config = read_data_config(os.path.join(setup_directory, "data_config.ini"))
    solve_config = read_solve_config(os.path.join(setup_directory, "solve_config.ini"))
    predict_config = read_predict_config(os.path.join(setup_directory, "predict_config.ini"))

    solve_roi = daisy.Roi(data_config["in_offset"], data_config["in_size"])
    tracing_roi = daisy.Roi(tracing_offset, tracing_size)

  
    if not solve_roi.intersects(tracing_roi):
        raise ValueError("No overlap between solve region and tracing")

    shared_roi = solve_roi.intersect(tracing_roi)

    if not shared_roi.get_shape() == tracing_roi.get_shape():
        raise Warning("Solve roi only partially overlaps with tracing")


    # Get Graph:    
    rec_graph = get_graph(predict_config["db_host"], 
                          predict_config["db_name"],
                          shared_roi.get_offset(),
                          shared_roi.get_shape(),
                          solve_number,
                          edge_collection)

    #Validate Graph:
    for v in rec_graph.nodes():
        nbs = [v for v in rec_graph.neighbors(v)]
        if len(nbs)>2:
            raise ValueError("Branching in graph, abort.")


    # Label connected components:
    rec_graph = label_connected_components(rec_graph,
                                           solve_number)


    # Get voxel lines:
    rec_lines = []
    component_ids = set()
    rec_component_map = {}
    for node, data in rec_graph.nodes(data=True):
        component_id = data["cc_{}".format(solve_number)]
        component_ids.add(component_id)

    component_ids = list(component_ids)
    component_ids.sort()

    n = 0
    for component_id in component_ids:
        start_vertex_id, end_vertex_id = get_start_end_vertex_id(rec_graph, solve_number, component_id)
        if start_vertex_id == end_vertex_id == None:
            # Loop
            continue

        interpolated_cc = interpolate_cc(rec_graph, start_vertex_id, end_vertex_id, voxel_size)

        if interpolated_cc:
            rec_lines.append(interpolated_cc)
            rec_component_map[n] = component_id
            n += 1

        else:
            assert(start_vertex_id == end_vertex_id)

    # Subsample:
    rec_subsampled_lines = []
    for line in rec_lines:
        rec_subsampled_lines.append(subsample(line, subsample_factor))

    # Get gt graph:
    node_dic, edge_list = parse_nml(tracing_file)
   
    gt_graph = nx.Graph()
    nodes = [(id_, {"z": pos[0] * voxel_size[0], "y": pos[1] * voxel_size[1], "x": pos[2] * voxel_size[2]}) for id_, pos in node_dic.items()]
    gt_graph.add_nodes_from(nodes)
    gt_graph.add_edges_from(edge_list)

    # Label connected components:
    gt_graph = label_connected_components(gt_graph,
                                          solve_number)

    # Get voxel lines:
    gt_lines = []
    component_ids = set()
    gt_component_map = {}
    for node, data in gt_graph.nodes(data=True):
        component_id = data["cc_{}".format(solve_number)] 
        component_ids.add(component_id)
        
    component_ids = list(component_ids)
    component_ids.sort()

    for component_id in component_ids:
        start_vertex_id, end_vertex_id = get_start_end_vertex_id(gt_graph, solve_number, component_id)
        if start_vertex_id == end_vertex_id == None:
            # Loop
            continue

        interpolated_cc = interpolate_cc(gt_graph, start_vertex_id, end_vertex_id, voxel_size)

        if interpolated_cc:
            gt_lines.append(interpolated_cc)
            gt_component_map[n] = component_id
            n += 1

        else:
            assert(start_vertex_id == end_vertex_id)

    # Subsample:
    gt_subsampled_lines = []
    for line in gt_lines:
        gt_subsampled_lines.append(subsample(line, subsample_factor))


    matching_graph = MatchingGraph(gt_subsampled_lines,
                                   rec_subsampled_lines,
                                   list(gt_component_map.keys()),
                                   list(rec_component_map.keys()),
                                   distance_threshold,
                                   voxel_size)

    return matching_graph, gt_graph, rec_graph, gt_component_map, rec_component_map



def get_graph(db_host,
              db_name,
              roi_offset,
              roi_size,
              solve_number,
              edge_collection):

    """
    Get selected subgraph containing no isolated
    selected edges. (All edges with not 
    2 selected vertices are filtered out)
    """

    selected_attr = "selected_{}".format(solve_number)
    solved_attr = "solved_{}".format(solve_number)

    graph_provider = MongoDbGraphProvider(db_name, db_host, directed=False, position_attribute=['z', 'y', 'x'], edges_collection=edge_collection)
    roi = daisy.Roi(roi_offset, roi_size)

    nx_graph = graph_provider.get_graph(roi, nodes_filter={selected_attr: True}, edges_filter={selected_attr: True}, 
                                        node_attrs=[selected_attr, solved_attr, "x", "y", "z"], 
                                        edge_attrs=[selected_attr, solved_attr, "x", "y", "z"])

    # NOTE: We filter edges that do not have 
    # two selected vertices.
    edges_to_remove = set()
    nodes_to_remove = set()
    for e in nx_graph.edges():
        u = nx_graph.nodes()[e[0]]
        v = nx_graph.nodes()[e[1]] 

        if not u:
            edges_to_remove.add((e[0], e[1]))
            nodes_to_remove.add(e[0])
        if not v:
            edges_to_remove.add((e[0], e[1]))
            nodes_to_remove.add(e[1])


    nx_graph.remove_edges_from(edges_to_remove)
    nx_graph.remove_nodes_from(nodes_to_remove)

    return nx_graph


def label_connected_components(nx_graph,
                               solve_number):
    find_connected_components(nx_graph, node_component_attribute="cc_{}".format(solve_number), return_lut=False)
    return nx_graph


def overlay_segmentation(db_host,
                         db_name,
                         roi_offset,
                         roi_size,
                         selected_attr,
                         solved_attr,
                         edge_collection,
                         segmentation_container,
                         segmentation_dataset,
                         segmentation_number,
                         voxel_size=(40,4,4)):


    graph_provider = MongoDbGraphProvider(db_name, db_host, directed=False, position_attribute=['z', 'y', 'x'], edges_collection=edge_collection)
    graph_roi = daisy.Roi(roi_offset, roi_size)

    segmentation = daisy.open_ds(segmentation_container, segmentation_dataset)
    intersection_roi = segmentation.roi.intersect(graph_roi).snap_to_grid(voxel_size)

    nx_graph = graph_provider.get_graph(intersection_roi, nodes_filter={selected_attr: True}, edges_filter={selected_attr: True})

    for node_id, data in nx_graph.nodes(data=True):
        node_position = daisy.Coordinate((data["z"], data["y"]), data["x"])
        nx_graph.nodes[node_id]["segmentation_{}".format(segmentation_number)] = segmentation[node_position]

    graph_provider.write_nodes(intersection_roi)


def interpolate_on_grid(z0, y0, x0, z1, y1, x1, voxel_size):
    """
    Interpolate a line on a 3D voxel grid.

    x0, y0, ... (int, physical space e.g. nm):

    returns: list of voxels forming a line from p0 to p1
    """
    def dda_round(x):
        return (x + 0.5).astype(int)


    start = np.array([z0, y0, x0], dtype=float)
    end = np.array([z1, y1, x1], dtype=float)

    voxel_size = np.array(voxel_size)
    if np.any(start % voxel_size) or np.any(end % voxel_size):
        print(start%voxel_size, end%voxel_size)
        raise ValueError("Start end end position must be multiples of voxel size")

    line = [dda_round(start / voxel_size)]

    if not np.any(start - end):
        return line

    max_direction, max_length = max(enumerate(abs(end - start)),
                                              key=operator.itemgetter(1))

    dv = (end - start) / max_length

    # We interpolate in physical space to find the shortest distance
    # linear interpolation but the path is represented in voxels
    for step in range(int(max_length)):
        step_point_rescaled = np.array(dda_round(dda_round((step + 1) * dv + start)/voxel_size))
        if not np.all(step_point_rescaled == line[-1]):
            line.append(step_point_rescaled)

    assert(np.all(line[-1] == dda_round(end/voxel_size)))
    return line


def subsample(voxel_line, factor):
    if factor <= 0:
        raise ValueError("Factor must be larger 1")
    if not isinstance(factor, int):
        raise ValueError("Factor must be integer type")

    subsampled_line = []

    assert(len(voxel_line)>1)

    for i in range(len(voxel_line)):
        if i % factor == 0:
            subsampled_line.append(voxel_line[i])

    # Line always contains start and end
    if not np.all(voxel_line[i] == voxel_line[-1]):
        subsampled_line.append(voxel_line[-1])
    return subsampled_line


def get_start_end_vertex_id(nx_graph, solve_number, component_number):
    cc_attribute = "cc_{}".format(solve_number)

    nodes_in_cc = [node_id for node_id, data in nx_graph.nodes(data=True) if data[cc_attribute] == component_number]
    if not nodes_in_cc:
        raise ValueError("No nodes with component number {} in graph".format(component_number))

    if len(nodes_in_cc) > 1:
        start_end_vertex = [node_id for node_id in nodes_in_cc if len([v for v in nx_graph.neighbors(node_id)]) == 1]
        if len(start_end_vertex) == 0:
            # Loop - filter out
            start = None
            end = None
        else:
            assert(len(start_end_vertex) == 2)
            start = start_end_vertex[0]
            end = start_end_vertex[1]

    else:
        start = nodes_in_cc[0]
        end = nodes_in_cc[0]

    return start, end


def interpolate_cc(nx_graph, start_vertex_id, end_vertex_id, voxel_size):
    cc_line = []

    successors = nx.dfs_successors(nx_graph, source=start_vertex_id)
    if successors:
        for current, next_ in successors.items():
            assert(len(next_) == 1)
            next_ = next_[0]
            x_current = nx_graph.nodes()[current]["x"]
            y_current = nx_graph.nodes()[current]["y"]
            z_current = nx_graph.nodes()[current]["z"]

            
            x_next = nx_graph.nodes()[next_]["x"]
            y_next = nx_graph.nodes()[next_]["y"]
            z_next = nx_graph.nodes()[next_]["z"]

            edge_line = interpolate_on_grid(z_current, y_current, x_current, 
                                            z_next, y_next, x_next, voxel_size)

            cc_line.extend(edge_line)

        assert(next_ == end_vertex_id)

    return cc_line


class MatchingGraph(nx.Graph):
    def __init__(self, gt_lines, rec_lines, 
                       gt_lines_cc_ids,
                       rec_lines_cc_ids,
                       distance_threshold, voxel_size,
                       gt_lines_attributes=None, 
                       rec_lines_attributes=None, 
                       graph_data=None):

        """
        As always z,y,x convention.

        distance threshold in world units (nm)

        lines in voxel units as they are expected
        to come from interpolation on a voxel grid.
        """

        super(MatchingGraph, self).__init__(incoming_graph_data=graph_data)
        self.max_id = 0
        self.voxel_size = np.array(voxel_size)
        self.distance_threshold = int(distance_threshold)
        if self.distance_threshold != distance_threshold:
            raise TypeError("Distance threshold needs to be convertible to an integer")

        if set(gt_lines_cc_ids) & set(rec_lines_cc_ids):
            raise ValueError("cc ids have to be unique in rec and gt")

        self.__add_lines(gt_lines, "gt", gt_lines_cc_ids, lines_attributes=gt_lines_attributes)
        self.__add_lines(rec_lines, "rec", rec_lines_cc_ids, lines_attributes=rec_lines_attributes)

        self.__add_matching_edges(self.distance_threshold, self.voxel_size)

        self.matched = False


    def __add_lines(self, lines, line_type, lines_cc_ids, lines_attributes=None):
        """
        lines (list of voxel lines)

        line_type: gt or rec

        lines_cc_ids: connected component ids of each line

        line_attributes: (optional) additional custom attributes for each line.
        """

        nodes = []
        edges = []

        if not line_type in ["gt", "rec"]:
            raise ValueError("Line type must be 'gt' or 'rec'")

        assert(len(lines) == len(lines_cc_ids))

        if lines_attributes is not None:
            assert(len(lines) == len(lines_attributes))
        else:
            lines_attributes = [{} for i in range(len(lines))]

        i = self.max_id
        for j in range(len(lines)):
            for k in range(len(lines[j])):
                node_attribute = {"z": lines[j][k][0],
                                  "y": lines[j][k][1],
                                  "x": lines[j][k][2],
                                  "line_type": line_type}

                line_attribute = lines_attributes[j]
                line_cc_id = lines_cc_ids[j]

                nodes.append((i, {**node_attribute,
                                  **{"cc_id": line_cc_id},
                                  **line_attribute}))
                
                if k < (len(lines[j]) - 1):
                    edges.append((i, i+1))

                i += 1

        self.max_id = i

        self.add_nodes_from(nodes)
        self.add_edges_from(edges, edge_type="structural")


    def __add_matching_edges(self, 
                             distance_threshold,
                             voxel_size):

        assert(len(voxel_size) == 3)
        assert(isinstance(voxel_size, np.ndarray))
        assert(isinstance(distance_threshold, int))

        gt_nodes = {}
        rec_nodes = {}
        for v, data in self.nodes(data=True):
            if data["line_type"] == 'gt':
                gt_nodes[v] = np.array([data["z"], data["y"], data["x"]])
            elif data["line_type"] == 'rec':
                rec_nodes[v] = np.array([data["z"], data["y"], data["x"]])
            else:
                raise ValueError("Node with line type {} in db. Abort.".format(data["line_type"]))

        gt_positions = np.array(list(gt_nodes.values()))
        gt_node_ids = list(gt_nodes.keys())

        rec_positions = np.array(list(rec_nodes.values()))
        rec_node_ids = list(rec_nodes.keys())

        gt_tree = KDTree(gt_positions * voxel_size)
        rec_tree = KDTree(rec_positions * voxel_size)

        """
        From the docs:
        KDTree.query_ball_tree(other, r, p=2.0, eps=0)
        For each element self.data[i] of this tree,
        results[i] is a list of the indices of its neighbors in other.data.
        """

        results = gt_tree.query_ball_tree(rec_tree, r=distance_threshold)

        for gt_idx in range(len(results)):
            gt_node_id = gt_node_ids[gt_idx]

            for rec_idx in results[gt_idx]:
                rec_node_id = rec_node_ids[rec_idx]
                self.add_edge(gt_node_id, rec_node_id, edge_type="matching")


    def get_edge_conflicts(self):
        edge_conflicts = set()
        nodes = self.nodes(data=True)
        nodes_dict = {node[0]: node[1] for node in nodes}

        for v in nodes:
            # Retrieve all neighbors on matching edges subgraph:
            nbs = [n for n in self.neighbors(v[0]) if nodes_dict[n]["line_type"] != v[1]["line_type"]]
            incident_edges = self.edges(v[0])

            if v[1]["line_type"] == "rec":
                incident_edges = [[e[1], e[0]] for e in incident_edges]

            for i in range(len(nbs)):
                for j in range(i + 1, len(nbs)):
                    v_i = nbs[i]
                    v_j = nbs[j]

                    if nodes_dict[v_i]["cc_id"] != nodes_dict[v_j]["cc_id"]:
                        pairwise_edge_conflicts = [tuple(e) for e in incident_edges if v_i in e or v_j in e]
                        assert(len(pairwise_edge_conflicts) == 2)
                        edge_conflicts.add(tuple(pairwise_edge_conflicts))

        return edge_conflicts


    def export(self):
        nodes_gt = []
        labels_gt = {}
        nodes_rec = []
        labels_rec = {}
        edges_gt_rec = []


        node_dict = {v[0]: v[1] for v in self.nodes(data=True)}

        for v, data in self.nodes(data=True):
            if data["line_type"] == "gt":
                nodes_gt.append(v)
                labels_gt[v] = data["cc_id"]

            elif data["line_type"] == "rec":
                nodes_rec.append(v)
                labels_rec[v] = data["cc_id"]
            else:
                raise ValueError("Line type not understood, abort...")

        for e in self.edges(data=True):
            if e[2]["edge_type"] == "matching":
                if node_dict[e[0]]["line_type"] == "gt":
                    edges_gt_rec.append((e[0], e[1]))
                else:
                    edges_gt_rec.append((e[1], e[0]))


        edge_conflicts = self.get_edge_conflicts()

        return nodes_gt, nodes_rec, labels_gt, labels_rec, edges_gt_rec, edge_conflicts


    def import_node_matches(self, node_matches):
        for v in self.nodes(data=True):
            v[1]["matched"] = False

        for match in node_matches:
            self.edges[match[0], match[1]]["matched"] = True
            self.nodes[match[0]]["matched"] = True
            self.nodes[match[1]]["matched"] = True


        self.matched = True
        self.__mark_truefalse_nodes()
        self.__add_matched_labels()
        self.__mark_splitmerge_edges()

    def get_stats(self):
        stats = {"vertices": None,
                 "edges": None,
                 "tps_rec": None,
                 "tps_gt": None,
                 "fps": None,
                 "fns": None,
                 "merges": None,
                 "splits": None}

        nodes = self.nodes(data=True)
        edges = self.edges(data=True)

        stats["vertices"] = len(nodes)
        stats["edges"] = len([e for e in edges if e[2]["edge_type"] == "structural"])
        stats["tps_rec"] = len([v for v in nodes if (v[1]["line_type"] == "rec" and v[1]["match"] == "tp")])
        stats["tps_gt"] = len([v for v in nodes if (v[1]["line_type"] == "gt" and v[1]["match"] == "tp")])
        stats["fps"] = len([v for v in nodes if v[1]["match"] == "fp"])
        stats["fns"] = len([v for v in nodes if v[1]["match"] == "fn"])

        tags = set()
        for v in self.nodes(data=True):
            tags.add(v[1]["match"])

        merges = 0
        splits = 0
        for e in edges:
            try:
                if e[2]["match"] == "merge":
                    merges += 1
                elif e[2]["match"] == "split":
                    splits += 1
                else:
                    pass
            except KeyError:
                pass

        stats["merges"] = merges
        stats["splits"] = splits

        return stats

    def __mark_truefalse_nodes(self):
        if not self.matched:
            raise ValueError("Graph not matched yet, import matching before evaluation")

        for v in self.nodes(data=True):
            incident_edges = self.edges(v[0], data=True)
            incident_matched = []
            for e in incident_edges:
                try:
                    if e[2]["matched"]:
                        incident_matched.append(e)
                except:
                    pass

            if not incident_matched or not v[1]["matched"]:
                if v[1]["line_type"] == "gt":
                    v[1]["match"] = "fn"
                elif v[1]["line_type"] == "rec":
                    v[1]["match"] = "fp"
                else:
                    raise TypeError("Line type not understood")

            else:
                v[1]["match"] = "tp"

    def __add_matched_labels(self):
        nodes = self.nodes(data=True)
        nodes_dict = {node[0]: node[1] for node in nodes}

        for v in nodes:
            incident_edges = self.edges(v[0], data=True)
            incident_matched = []
            for e in incident_edges:
                try:
                    if e[2]["matched"]:
                        incident_matched.append(e)
                except:
                    pass

            nbs = [e[1] for e in incident_matched]

            if nbs:
                matched_label = set([nodes_dict[n]["cc_id"] for n in nbs])
                assert(len(matched_label) == 1)
                matched_label = list(matched_label)[0]
                v[1]["matched_label"] = matched_label


    def __mark_splitmerge_edges(self):
        nodes = self.nodes(data=True)
        nodes_dict = {node[0]: node[1] for node in nodes}

        for e in self.edges(data=True):
            if e[2]["edge_type"] == "structural":
                v0 = e[0]
                v1 = e[1]

                try:
                    if nodes_dict[v0]["matched_label"] != nodes_dict[v1]["matched_label"]:
                        if nodes_dict[v0]["line_type"] == "gt":
                            tag = "split"
                        else:
                            tag = "merge"

                        e[2]["match"] = tag
                except KeyError:
                    pass
