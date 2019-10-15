from xml.dom import minidom
import numpy as np

def parse_nml(filename, edge_attribute=None):
    doc = minidom.parse(filename)
    annotations = doc.getElementsByTagName("thing")

    node_dic = {}
    edge_list = []
    for annotation in annotations:
        nodes = annotation.getElementsByTagName("node")
        for node in nodes:
            node_position, node_id = parse_node(node) 
            node_dic[node_id] = node_position

        edges = annotation.getElementsByTagName("edge")
        for edge in edges:
            (source_id, target_id) = parse_attributes(edge, [["source", int], ["target", int]])
            edge_list.append((source_id, target_id))

    return node_dic, edge_list


def parse_node(node):
    [x, y, z, id_] =\
        parse_attributes(node,
                        [
                            ["x", float],
                            ["y", float],
                            ["z", float],
                            ["id", int],
                        ]
                        )

    point = np.array([z, y, x])

    return point, id_

def parse_edge(edge):
    [source, target] =\
        parse_attributes(edge,
                         [
                             ["source", int],
                             ["target", int]
                             
                         ]
                         )

    return source, target


def parse_attributes(xml_elem, parse_input):
    parse_output = []
    attributes = xml_elem.attributes
    for x in parse_input:
        try:
            parse_output.append(x[1](attributes[x[0]].value))
        except KeyError:
            parse_output.append(None)
    return parse_output


