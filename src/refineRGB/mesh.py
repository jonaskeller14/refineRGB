import numpy as np
import meshio


def label3(nodes: np.ndarray, elements: np.ndarray, marked_elements: np.ndarray):
    """
    Reorder marked-elements so that the first and second node is the longest edge.
    :param nodes: list of nodes with x,y,z coordinates
    :param elements:
    :param marked_elements:
    :return: updated elements list
    """
    # Compute edge lengths.
    edges = np.concatenate([
        elements[np.ix_(marked_elements, [0, 1])],
        elements[np.ix_(marked_elements, [0, 2])],
        elements[np.ix_(marked_elements, [0, 3])],
        elements[np.ix_(marked_elements, [1, 2])],
        elements[np.ix_(marked_elements, [1, 3])],
        elements[np.ix_(marked_elements, [2, 3])],
    ], axis=0)
    edge_lengths = np.linalg.norm(nodes[edges[:,0],:] - nodes[edges[:,1],:], axis=1)
    elem_edge_lengths = np.reshape(edge_lengths, [6, len(marked_elements)]).T
    idx = np.argmax(elem_edge_lengths, axis=1)
    # Reorder the boundary flags
    for i_me, id in enumerate(idx):
        if id == 1:
            elements[marked_elements[i_me], :] = elements[marked_elements[i_me], [2, 0, 1, 3]]
        elif id == 2:
            elements[marked_elements[i_me], :] = elements[marked_elements[i_me], [0, 3, 1, 2]]
        elif id == 3:
            elements[marked_elements[i_me], :] = elements[marked_elements[i_me], [1, 2, 0 , 3]]
        elif id == 4:
            elements[marked_elements[i_me], :] = elements[marked_elements[i_me], [1, 3, 2, 0]]
        elif id == 5:
            elements[marked_elements[i_me], :] = elements[marked_elements[i_me], [3, 2, 1, 0]]
    return elements


def fix_order3(nodes: np.ndarray, elements: np.ndarray):
    """
    Fix element order so that the volume is positive.
    :param nodes:
    :param elements:
    :return:
    """
    volume = tetrahedron_volume(nodes, elements, return_pos=False)
    idx = np.where(volume < 0)[0]
    elements[np.ix_(idx, [1,2])] = elements[np.ix_(idx, [2,1])]
    return elements


def tetrahedron_volume(nodes: np.ndarray, elements: np.ndarray, return_pos=True):
    """
    Computes volume of a tetrahedron.
    :param nodes: list of nodes with x,y,z-coordinates
    :param elements: list of elements with 4-node-indices
    :param return_pos: return positive value if true
    :return: volume
    """
    # Computes volume
    d12 = nodes[elements[:, 1], :] - nodes[elements[:, 0], :]
    d13 = nodes[elements[:, 2], :] - nodes[elements[:, 0], :]
    d14 = nodes[elements[:, 3], :] - nodes[elements[:, 0], :]
    volume = np.sum(np.cross(d12, d13, axis=1)*d14, axis=1)/6
    if return_pos:
        idx = np.where(volume < 0)
        print(idx)
        volume[idx] = -volume[idx]
    return volume


def get_element_midpoints(nodes: np.ndarray, elements: np.ndarray):
    """
    Computes midpoint-coordinates for each element in elements.
    :param nodes: x,y,z-coordinates
    :param elements:
    :return: x,y,z-coordinates of midpoint
    """
    midpoints = np.zeros([len(elements),3])
    for idx,element in enumerate(elements):
        midpoints[idx,0] = np.mean([nodes[j, 0] for j in element])
        midpoints[idx,1] = np.mean([nodes[j, 1] for j in element])
        midpoints[idx,2] = np.mean([nodes[j, 2] for j in element])
    return midpoints


def get_neighbours(elements: np.ndarray, marked_elements: np.ndarray, common_nodes: int = 1):
    """
    Extends selection with all elements which share min. 1 (default) node.
    :param common_nodes:
    :param elements:
    :param marked_elements:
    :return:
    """
    nodes = np.zeros(np.max(elements)+1, dtype="int8")
    nodes[elements[marked_elements]] = 1
    new_marked_elements = np.nonzero(np.sum(nodes[elements], axis=1) >= common_nodes)[0]
    return new_marked_elements


def are_face_neighbors(element1: np.ndarray, element2: np.ndarray):
    """
    3 vertices of element1 must be contained also by element2. Does not recognize false elements.
    :param element1:
    :param element2:
    :return:
    """
    return np.sum(np.array([element1]) == np.transpose([element2])) == 3


def get_boundary_elements(elements: np.ndarray):
    """
    Returns a list of boundary elements.
    :param elements:
    :return:
    """
    nt = len(elements)
    faces = np.concatenate([
        elements[:, [0,1,2]],
        elements[:, [0,1,3]],
        elements[:, [0,2,3]],
        elements[:, [1,2,3]]
    ], axis=0)
    unique_faces, idx, inv, counts \
        = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_inverse=True, return_counts=True)
    is_boundary_face = (counts == 1)[inv]
    return np.unique(is_boundary_face.nonzero()[0] % nt)


def get_boundary_nodes(elements: np.ndarray):
    """
    Returns a list of boundary nodes.
    :param elements:
    :return:
    """
    faces = np.concatenate([
        elements[:, [0,1,2]],
        elements[:, [0,1,3]],
        elements[:, [0,2,3]],
        elements[:, [1,2,3]]
    ], axis=0)
    unique_faces, face_counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    boundary_faces = unique_faces[face_counts == 1]
    return np.unique(boundary_faces)


def is_boundary_element(elements: np.ndarray, check_elements: np.ndarray):
    """
    Check whether elements are boundary elements, returns True if so.
    Boundary elements are elements with a face which is not also used by another element.
    :param nodes:
    :param elements:
    :param check_elements:
    :return: logical array
    """
    boundary_elements = get_boundary_elements(elements)
    is_boundary_ele = np.zeros(len(elements), dtype=bool)
    is_boundary_ele[boundary_elements] = True
    return is_boundary_ele[check_elements]


def is_boundary_node(nodes: np.ndarray, elements: np.ndarray, check_nodes: np.ndarray):
    boundary_nodes = get_boundary_nodes(elements)
    is_boundary_nod = np.zeros(len(nodes), dtype=bool)
    is_boundary_nod[boundary_nodes] = True
    return is_boundary_nod[check_nodes]


def filter_surface_elements(elements: np.ndarray, check_elements: np.ndarray):
    return check_elements[is_boundary_element(elements, check_elements)]


def points2elements(nodes: np.ndarray, elements: np.ndarray, points: np.ndarray):
    marked_elements = np.array([], dtype="int32")
    for point in points:
        for id_ele, element in enumerate(elements):
            v1 = nodes[element[0]]
            v2 = nodes[element[1]]
            v3 = nodes[element[2]]
            v4 = nodes[element[3]]
            v12 = v2 - v1
            v13 = v3 - v1
            v14 = v4 - v1
            mat = np.array((v12, v13, v14)).T
            mat_inv = np.linalg.inv(mat)
            newp = mat_inv.dot(point - v1)
            if np.all(newp >= 0) and np.all(newp <= 1) and np.sum(newp) <= 1:
                marked_elements = np.append(marked_elements, id_ele)
    return np.unique(marked_elements)


def conservative_intersection(
        tet_nodes: np.ndarray,
        tet_elements: np.ndarray,
        tri_nodes: np.ndarray,
        tri_elements: np.ndarray
):
    """
    Approximate intersection of triangular- and tetrahedral-mesh by creating a box around each element.
    Only the intersection of boxes are detected.
    This leads to more intersections than computing the exact solution.
    :param tet_nodes:
    :param tet_elements:
    :param tri_nodes:
    :param tri_elements:
    :return:
    """
    marked_elements_bool = np.zeros(len(tet_elements), dtype=bool)
    min_tet_coord = np.min(tet_nodes[tet_elements], axis=1)
    max_tet_coord = np.max(tet_nodes[tet_elements], axis=1)
    for tri_element in tri_elements:
        min_coord = np.min(tri_nodes[tri_element, :], axis=0)
        max_coord = np.max(tri_nodes[tri_element, :], axis=0)

        # case 1: intersection of boxes
        max_min_coord = (min_coord > min_tet_coord)*min_coord + (min_coord <= min_tet_coord)*min_tet_coord
        min_max_coord = (max_coord < max_tet_coord)*max_coord + (max_coord >= max_tet_coord)*max_tet_coord
        marked_elements_bool += np.all((min_max_coord - max_min_coord) >= 0, axis=1)

        # case 2: tri-box in tet-box
        marked_elements_bool += np.all(min_coord >= min_tet_coord, axis=1) * np.all(max_coord <= max_tet_coord, axis=1)

        # case 3: tet-box in tri-box
        temp_nodes = np.all(tet_nodes <= max_coord, axis=1) * np.all(tet_nodes >= min_coord, axis=1)
        marked_elements_bool += np.any(temp_nodes[tet_elements], axis=1)
    return marked_elements_bool.nonzero()[0]


def first_to_second_order(nodes: np.ndarray, elements: np.ndarray, node_sets: dict = None, element_sets: dict = None):
    """
    C3D4 -> C3D10
    Conversion of tetra4-elements (first order) to tetra10-elements (second order).
    Node order: https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html
    1 - 2 - 3 - 4 - 12 - 23 - 13 - 14 - 24 - 34
    :param element_sets:
    :param node_sets:
    :param nodes:
    :param elements:
    :return:
    """
    edges = np.concatenate([
        elements[:, [0, 1]],
        elements[:, [1, 2]],
        elements[:, [0, 2]],
        elements[:, [0, 3]],
        elements[:, [1, 3]],
        elements[:, [2, 3]],
    ], axis=0)
    edges = np.sort(edges, axis=1)
    edges, idx, j = np.unique(edges, axis=0, return_index=True, return_inverse=True)
    elem2edge = np.reshape(j, [6, len(elements)]).T
    elements = np.concatenate([elements, np.max(elements) + 1 + elem2edge], axis=1)
    nodes = np.append(nodes, (nodes[edges[:, 0], :] + nodes[edges[:, 1], :]) / 2, axis=0)

    output = nodes, elements
    if node_sets is not None:
        # TODO: hier alle zwischennodes hinzufügen wenn beide parents dabei sind
        for key, node_set in node_sets.items():
            pass
        output += (node_sets,)
    if element_sets is not None:
        output += (element_sets,)
    return output


def second_to_first_order(nodes: np.ndarray, elements: np.ndarray, node_sets: dict = None, element_sets: dict = None):
    """
    C3D10 -> C3D4
    Conversion of tetra10-elements (second order)to tetra4-elements (first order).
    Only works, if elements are ordered! This is the case when creating a mesh in Abaqus.
    :param element_sets:
    :param node_sets:
    :param nodes:
    :param elements:
    :return:
    """
    # TODO: node sets testen
    all_nodes = np.zeros(len(nodes), dtype=bool)
    elements = elements[:, :4]
    unique_nodes = np.unique(elements)
    all_nodes[unique_nodes] = True
    nodes = nodes[unique_nodes, :]

    output = nodes, elements
    if node_sets is not None:
        for key, node_set in node_sets.items():
            node_sets[key] = node_set[all_nodes[node_set]]
        output += (node_sets,)
    if element_sets is not None:
        output += (element_sets,)
    return output


def triangular_to_thin_tetrahedral(nodes: np.ndarray, elements: np.ndarray):
    """
    Conversion of triangular mesh to super thin tetrahedral mesh.
    :param nodes:
    :param elements:
    :return:
    """
    nt = len(elements)
    n = len(nodes)
    np.random.seed(0)
    # new node, addition of random noise, so that volume is not zero
    new_nodes = np.mean(nodes[elements], axis=1) + np.random.uniform(low=-1, high=1) * 1e-6
    nodes = np.append(nodes, new_nodes, axis=0)
    elements = np.append(elements, np.transpose([n + np.arange(nt)]), axis=1)
    return nodes, elements


def sets_to_data(mesh: meshio.Mesh):
    """
    Add point/cell-sets to point/cell-data with value one.
    Useful for visualization of sets for example in Paraview.
    :param mesh: input mesh
    :return: updated mesh
    """
    nodes = mesh.points
    elements = mesh.cells[0][1]

    # point sets --> point data
    point_sets = mesh.point_sets
    for key, value in point_sets.items():
        point_data = np.zeros(len(nodes))
        point_data[value] = 1
        mesh.point_data[key] = point_data

    # cell sets --> cell_data
    cell_sets = mesh.cell_sets
    for key, value in cell_sets.items():
        cell_data = np.zeros(len(elements))
        cell_data[value[0]] = 1
        mesh.cell_data[key] = cell_data
    return mesh

