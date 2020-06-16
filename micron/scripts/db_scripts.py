import daisy
from daisy.persistence import MongoDbGraphProvider
import numpy as np
from xml.dom import minidom
from funlib.segment.graphs import find_connected_components


def build_attributes(xml_elem, attributes):
    for attr in attributes:
        try:
            xml_elem.setAttribute(attr[0], str(attr[1]))
        except UnicodeEncodeError:
            xml_elem.setAttribute(attr[0], str(attr[1].encode('ascii', 'replace')))
    return xml_elem

def nx_to_nml(nodes,
              edges,
              output_file,
              voxel_size=(40,4,4)):



    #nodes = {v[0]: (v[1]["z"], v[1]["y"], v[1]["x"]) for v in nx_graph.nodes(data=True)}
    #edges = [(e[0], e[1]) for e in nx_graph.edges()]

    doc = minidom.Document()
    annotations_elem = doc.createElement("things")
    doc.appendChild(annotations_elem)

    annotation_elem = doc.createElement("thing")
    build_attributes(annotation_elem, [["id", 3]])

    nodes_elem = doc.createElement("nodes")
    edges_elem = doc.createElement("edges")

    for node_id, position in nodes.items():
        node_elem = doc.createElement("node")
        #position = (np.array(position) - np.array(roi_offset))/voxel_size 
        position = np.array(position)/voxel_size
        position = np.rint(position).astype(int) 
        identifier = node_id

        build_attributes(node_elem, [["x", position[2]],
                                     ["y", position[1]],
                                     ["z", position[0]],
                                     ["id", node_id]
                                    ])

        nodes_elem.appendChild(node_elem)

    for e in edges:
        source_id = e[0]
        target_id = e[1]

        edge_elem = doc.createElement("edge")

        build_attributes(edge_elem, [["source", source_id],
                                     ["target", target_id]
                                    ])

        edges_elem.appendChild(edge_elem)

    annotation_elem.appendChild(nodes_elem)
    annotation_elem.appendChild(edges_elem)

    annotations_elem.appendChild(annotation_elem)

    doc = doc.toprettyxml()

    with open(output_file, "w+") as f:
        f.write(doc)



def graph_to_nml(output_file,
                 db_host, 
                 db_name,
                 voxel_size=(40,4,4),
                 roi_offset=(158000, 121800, 448616),
                 roi_size=(7600,5200,2200),
                 selected_only=False,
                 selected_attr="selected",
                 solved_attr="solved",
                 edge_collection="edges"):

    nodes, edges = get_graph(db_host, 
                              db_name,
                              roi_offset,
                              roi_size,
                              selected_only,
                              selected_attr,
                              solved_attr,
                              edge_collection)

    doc = minidom.Document()
    annotations_elem = doc.createElement("things")
    doc.appendChild(annotations_elem)

    annotation_elem = doc.createElement("thing")
    build_attributes(annotation_elem, [["id", 3]])

    nodes_elem = doc.createElement("nodes")
    edges_elem = doc.createElement("edges")

    for node_id, position in nodes.items():
        node_elem = doc.createElement("node")
        #position = (np.array(position) - np.array(roi_offset))/voxel_size 
        position = np.array(position)/voxel_size
        position = np.rint(position).astype(int) 
        identifier = node_id

        build_attributes(node_elem, [["x", position[2]],
                                     ["y", position[1]],
                                     ["z", position[0]],
                                     ["id", node_id]
                                    ])

        nodes_elem.appendChild(node_elem)

    for e in edges:
        source_id = e[0]
        target_id = e[1]

        edge_elem = doc.createElement("edge")

        build_attributes(edge_elem, [["source", source_id],
                                     ["target", target_id]
                                    ])

        edges_elem.appendChild(edge_elem)

    annotation_elem.appendChild(nodes_elem)
    annotation_elem.appendChild(edges_elem)

    annotations_elem.appendChild(annotation_elem)

    doc = doc.toprettyxml()

    with open(output_file, "w+") as f:
        f.write(doc)


def get_graph(db_host, 
              db_name,
              roi_offset=(158000, 121800, 448616),
              roi_size=(7600,5200,2200),
              selected_only=False,
              selected_attr="selected",
              solved_attr="solved",
              edge_collection="edges"):

    
    graph = MongoDbGraphProvider(db_name, db_host, directed=False, position_attribute=['z', 'y', 'x'], edges_collection=edge_collection)
    roi = daisy.Roi(roi_offset, roi_size)
    nodes, edges = graph.read_blockwise(roi, block_size=daisy.Coordinate((10000,10000,10000)), num_workers=40)

    if len(edges["u"]) != 0:
        if selected_only:
            edges = [(u, v, selected, solved) for u,v,selected,solved in zip(edges["u"], edges["v"], edges[selected_attr], edges[solved_attr]) if selected]
            nodes = {node_id: (z,y,x) for z,y,x,node_id, selected in\
                               zip(nodes["z"], nodes["y"], nodes["x"], nodes["id"], nodes[selected_attr]) if selected}

        else:
            edges = [(u, v, selected, solved) for u,v,selected,solved in zip(edges["u"], edges["v"], edges[selected_attr], edges[solved_attr])]
    else:
        edges = []

    return nodes, edges

def label_connected_components(db_host,
                               db_name,
                               roi,
                               selected_attr,
                               solved_attr,
                               edge_collection,
                               label_attribute="label"):

    graph_provider = MongoDbGraphProvider(db_name, db_host, directed=False, position_attribute=['z', 'y', 'x'], edges_collection=edge_collection)
    graph = graph_provider.get_graph(roi, nodes_filter={selected_attr: True}, edges_filter={selected_attr: True})

    lut = find_connected_components(graph, node_component_attribute=label_attribute, return_lut=True)

    return graph, lut
