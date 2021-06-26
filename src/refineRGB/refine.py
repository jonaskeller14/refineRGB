import numpy as np
from scipy.sparse import csr_matrix
from help_functions import n_unique_per_row
from mesh import *


def refine_red_uniform(nodes: np.ndarray, elements: np.ndarray):
    # Construct data structure
    edges = np.concatenate([
        elements[:, [0, 1]],
        elements[:, [0, 2]],
        elements[:, [0, 3]],
        elements[:, [1, 2]],
        elements[:, [1, 3]],
        elements[:, [2, 3]],
    ], axis=0)
    edges = np.sort(edges, axis=1)
    edges, idx, j = np.unique(edges, axis=0, return_index=True, return_inverse=True)
    elem2edge = np.reshape(j, [6, len(elements)]).T
    elem2dof = np.concatenate([elements, np.max(elements) + 1 + elem2edge], axis=1)

    nel = len(elements)

    # Add new nodes
    nodes = np.append(nodes, (nodes[edges[:, 0], :] + nodes[edges[:, 1], :]) / 2, axis=0)

    # Refine each tetrahedron into 8 tetrahedrons
    t = np.arange(nel)
    # print(t)
    p = elem2dof
    elements = np.append(elements, np.zeros([7 * nel, 4], dtype="int32"), axis=0)

    elements[t, :] = np.array([p[t, 0], p[t, 4], p[t, 5], p[t, 6]]).T  # 1
    elements[1 * nel:2 * nel, :] = np.array([p[t, 4], p[t, 1], p[t, 7], p[t, 8]]).T  # 2
    elements[2 * nel:3 * nel, :] = np.array([p[t, 5], p[t, 7], p[t, 2], p[t, 9]]).T  # 3
    elements[3 * nel:4 * nel, :] = np.array([p[t, 6], p[t, 8], p[t, 9], p[t, 3]]).T  # 4
    # always use diagonal edge 5-8 (-> included in all inner tetrahedrons)
    elements[4 * nel:5 * nel, :] = np.array([p[t, 4], p[t, 5], p[t, 6], p[t, 8]]).T  # 5
    elements[5 * nel:6 * nel, :] = np.array([p[t, 4], p[t, 5], p[t, 7], p[t, 8]]).T  # 6
    elements[6 * nel:7 * nel, :] = np.array([p[t, 5], p[t, 6], p[t, 8], p[t, 9]]).T  # 7
    elements[7 * nel:8 * nel, :] = np.array([p[t, 5], p[t, 7], p[t, 8], p[t, 9]]).T  # 8
    return nodes, elements


def refine_red(nodes: np.ndarray, elements: np.ndarray, marked_elements: np.ndarray, return_cut_edge=False):
    # TODO: add sets, nicht schlimm, wenn sie nicht funktionieren
    """
    Regular red-refinement of tetrahedral mesh.
    Only marked-for-refinement elements are refined.
    Hanging nodes are created!
    :param nodes:
    :param elements:
    :param marked_elements:
    :param return_cut_edge:
    :return:
    """
    # Data Structure
    n = len(nodes)
    nt = len(elements)
    nm = len(marked_elements)
    edges = np.concatenate([
        elements[np.ix_(marked_elements, [0, 1])],
        elements[np.ix_(marked_elements, [0, 2])],
        elements[np.ix_(marked_elements, [0, 3])],
        elements[np.ix_(marked_elements, [1, 2])],
        elements[np.ix_(marked_elements, [1, 3])],
        elements[np.ix_(marked_elements, [2, 3])],
    ], axis=0)
    edges = np.sort(edges, axis=1)
    edges, _, j = np.unique(edges, axis=0, return_index=True, return_inverse=True)
    cut_edge = np.array([edges[:, 0], edges[:, 1], n + np.arange(len(edges))]).T
    elem2edge = np.reshape(j, [6, nm]).T
    p = np.concatenate([elements[marked_elements], n + elem2edge], axis=1)
    # Add new nodes
    nodes = np.append(nodes, (nodes[edges[:, 0], :] + nodes[edges[:, 1], :]) / 2, axis=0)

    # Refine each tetrahedron into 8 tetrahedrons
    elements = np.append(elements, np.zeros([7 * len(marked_elements), 4], dtype="int32"), axis=0)  # Preallocation
    idx = np.arange(len(marked_elements))
    elements[marked_elements] = np.transpose([p[idx, 0], p[idx, 4], p[idx, 5], p[idx, 6]])  # 1 # overwrite old elements
    elements[nt + 0 * nm + idx] = np.transpose([p[idx, 4], p[idx, 1], p[idx, 7], p[idx, 8]])  # 2
    elements[nt + 1 * nm + idx] = np.transpose([p[idx, 5], p[idx, 7], p[idx, 2], p[idx, 9]])  # 3
    elements[nt + 2 * nm + idx] = np.transpose([p[idx, 6], p[idx, 8], p[idx, 9], p[idx, 3]])  # 4
    elements[nt + 3 * nm + idx] = np.transpose([p[idx, 4], p[idx, 5], p[idx, 6], p[idx, 8]])  # 5
    elements[nt + 4 * nm + idx] = np.transpose([p[idx, 4], p[idx, 5], p[idx, 7], p[idx, 8]])  # 6
    elements[nt + 5 * nm + idx] = np.transpose([p[idx, 5], p[idx, 6], p[idx, 8], p[idx, 9]])  # 7
    elements[nt + 6 * nm + idx] = np.transpose([p[idx, 5], p[idx, 7], p[idx, 8], p[idx, 9]])  # 8

    # Output
    output = nodes, elements

    # Return cut_edge
    if return_cut_edge:
        output += (cut_edge,)

    return output


def refine_bisect(
        nodes: np.ndarray,
        elements: np.ndarray,
        marked_elements: np.ndarray,
        node_sets: dict = None,
        element_sets: dict = None,
        surface_element_set_keyword="surface"
):
    """
    Local Refinement via the longest edge bisection.
    :param surface_element_set_keyword:
    :param element_sets:
    :param node_sets:
    :param nodes: nx3 np.ndarray with x,y,z-coordinates
    :param elements: nx4 np.ndarray with 4 node-indices for each tetrahedron
    :param marked_elements: marked-for-refinement element indices
    :return: nodes, elements np.ndarrays
    """
    n = len(nodes)
    nt = len(elements)

    # Pre-allocation
    nodes = np.append(nodes, np.zeros([8 * n, 3]), axis=0)
    elements = np.append(elements, np.zeros([3 * nt, 4], dtype="int32"), axis=0)
    generation = np.zeros(n + 6 * nt, dtype="int8")
    cut_edge = np.zeros([8 * n, 3], dtype="int32")
    non_conforming = np.ones(8 * n, dtype="int8")
    n_cut = 0

    while len(marked_elements) != 0:
        # print(f"Marked Elements: {marked_elements}")
        # switch element nodes such that elem(t,0:1) is the longest edge of t
        elements = label3(nodes, elements, marked_elements)
        p1 = elements[marked_elements, 0]
        p2 = elements[marked_elements, 1]
        p3 = elements[marked_elements, 2]
        p4 = elements[marked_elements, 3]

        # Find new cut edges and new nodes
        n_marked = len(marked_elements)
        p12 = np.zeros(n_marked, dtype="int32")

        if n_cut == 0:
            idx = np.arange(n_marked)
        else:
            nc_edge = non_conforming[:n_cut].nonzero()[0]
            nv2v = csr_matrix(
                (np.ones(2 * len(cut_edge[nc_edge, 2]), dtype=bool), (
                    np.concatenate([cut_edge[nc_edge, 2], cut_edge[nc_edge, 2]], axis=0),
                    np.concatenate([cut_edge[nc_edge, 0], cut_edge[nc_edge, 1]], axis=0)
                )),
                shape=(n, n)
            )
            i, j = nv2v[:, p1].multiply(nv2v[:, p2]).nonzero()
            p12[j] = i  # existing nodes
            idx = (p12 == 0).nonzero()[0]  # not existing nodes

        if len(idx) != 0:
            elem_cut_edge = np.sort(np.transpose([p1[idx], p2[idx]]), axis=1)  # add new cut edges
            i, j = csr_matrix(
                (np.ones(len(elem_cut_edge), dtype=bool), (
                    elem_cut_edge[:, 0],
                    elem_cut_edge[:, 1]
                )), shape=(n, n)
            ).nonzero()
            # Add new cut edges to cut_edge and new add middle points to nodes
            n_new = len(i)
            new_cut_edge_ids = np.arange(n_cut, n_cut + n_new)
            cut_edge[new_cut_edge_ids, 0] = i  # 1/2 parent node
            cut_edge[new_cut_edge_ids, 1] = j  # 2/2 parent node
            cut_edge[new_cut_edge_ids, 2] = np.arange(n, n + n_new)  # new node between parent nodes
            nodes[n:n + n_new, :] = (nodes[i, :] + nodes[j, :]) / 2  # new node between parent nodes
            n_cut += n_new  # update number of cut edges
            n += n_new  # update number of nodes

            # incidence matrix of new vertices and old vertices
            nv2v = csr_matrix(
                (np.ones(2 * len(cut_edge[new_cut_edge_ids, 2]), dtype=bool), (
                    np.concatenate([cut_edge[new_cut_edge_ids, 2], cut_edge[new_cut_edge_ids, 2]], axis=0),
                    np.concatenate([cut_edge[new_cut_edge_ids, 0], cut_edge[new_cut_edge_ids, 1]], axis=0)
                )),
                shape=(n, n)
            )  # if nv2v[m,i] = 1 and nv2v[m,i] = 1, then m is the middle point of i,j
            i, j = nv2v[:, p1].multiply(nv2v[:, p2]).nonzero()
            p12[j] = i  # middle point which is already added

        # Bisect marked elements
        idx = np.nonzero(generation[p12] == 0)[0]
        if len(idx) == 1:
            elem_generation = np.max(generation[elements[marked_elements[idx], :]].T)
        else:
            elem_generation = np.max(generation[elements[marked_elements[idx], :]], axis=1)
        generation[p12[idx]] = elem_generation + 1
        elements[marked_elements, :] = np.array([p4, p1, p3, p12]).T  # overwrite old elements
        elements[nt:nt + n_marked, :] = np.array([p3, p2, p4, p12]).T  # append second element to elements

        # add to element sets (new bisect)
        if element_sets is not None:
            for key, element_set in element_sets.items():
                element_list = np.zeros(len(elements), dtype=bool)
                element_list[element_set[0]] = True
                idx_me = element_list[marked_elements].nonzero()[0]
                element_sets[key] = [np.sort(np.append(element_set[0], nt + idx_me))]
        nt += n_marked

        # Find non-conforming elements
        nc_edge = non_conforming[:n_cut].nonzero()[0]
        is_check_node = np.zeros(n, dtype="int8")
        is_check_node[cut_edge[nc_edge, 0]] = 1  # parent 1
        is_check_node[cut_edge[nc_edge, 1]] = 1  # parent 2
        is_check_elem = np.any(is_check_node[elements[:nt, :]], axis=1).astype(int)
        check_elem = np.nonzero(is_check_elem)[0]
        t2v = csr_matrix(
            (np.ones(len(check_elem) * 4, dtype=bool), (
                np.tile(check_elem, 4),
                elements[check_elem, :].flatten("F")
            )),
            shape=(nt, n)
        )
        i, j = t2v[:, cut_edge[nc_edge, 0]].multiply(t2v[:, cut_edge[nc_edge, 1]]).nonzero()
        marked_elements = np.unique(i)
        non_conforming[nc_edge] = 0
        non_conforming[nc_edge[j]] = 1
    # Output
    nodes = nodes[:n, :]
    elements = elements[:nt, :]
    output = nodes, elements

    # Update + return node_sets
    if node_sets is not None:
        for key, node_set in node_sets.items():
            node_list = np.zeros(n, dtype=bool)
            node_list[node_set] = True
            parents_in_set = node_list[cut_edge[:, 0]] * node_list[cut_edge[:, 1]]
            node_sets[key] = np.sort(np.append(node_sets[key], cut_edge[parents_in_set, 2]))
        output += (node_sets,)

    # Filter for surface element sets and return element sets
    if element_sets is not None:
        for key, element_set in element_sets.items():
            if surface_element_set_keyword in key:
                element_sets[key] = [filter_surface_elements(elements, element_set[0])]
        output += (element_sets,)

    return output


def refine_red_green_bisect(
        nodes: np.ndarray,
        elements: np.ndarray,
        marked_elements: np.ndarray,
        node_sets: dict = None,
        element_sets: dict = None,
        surface_element_set_keyword="surface"
):
    """
    Red-Refinement of marked-for-refinement elements + Green- and Bisect-refinement of neighbours.
    No hanging nodes.
    Only for tetrahedral meshes.
    :param surface_element_set_keyword: (optional)
    :param element_sets: optional dictionary of element sets {"set-1": [array]}.
    Each element set contains list with a single array.
    :param node_sets: optional dictionary of node sets {"set-1": array}
    :param nodes:
    :param elements:
    :param marked_elements:
    :return: tuple of nodes, elements, ()
    """
    # RED refinement of marked + non-conforming elements with 5/6 hanging nodes
    red_marked_elements = marked_elements
    new_red_56_marked_elements = n_cut = n = nt = nodes_temp = elements_temp = cut_edge = new_red_4_marked_elements = None
    temp_element_sets = {}
    iter = 1
    while new_red_56_marked_elements is None \
            or len(new_red_56_marked_elements) != 0 \
            or len(new_red_4_marked_elements) != 0:
        print(f"\tRed-refinement iteration: \t{iter:02}\tred-marked-elements: \t{len(red_marked_elements)}")
        nt_old = len(elements)
        n_new = len(red_marked_elements)
        nodes_temp, elements_temp, cut_edge = refine_red(nodes, elements, red_marked_elements, return_cut_edge=True)
        n, nt, n_cut = len(nodes_temp), len(elements_temp), len(cut_edge)

        # add to element sets (new red)
        if element_sets is not None:
            for key, element_set in element_sets.items():
                element_list = np.zeros(nt, dtype=bool)
                element_list[element_set[0]] = True
                idx_me = element_list[red_marked_elements].nonzero()[0]  # red_marked_elements which are in set
                temp_element_sets[key] = [
                    np.sort(np.append(element_set[0], nt_old + np.array([
                        0 * n_new + idx_me,
                        1 * n_new + idx_me,
                        2 * n_new + idx_me,
                        3 * n_new + idx_me,
                        4 * n_new + idx_me,
                        5 * n_new + idx_me,
                        6 * n_new + idx_me,
                    ])))]

        # Find new non-conforming elements
        nc_edge = np.arange(n_cut)
        is_check_node = np.zeros(n, dtype="int8")
        is_check_node[cut_edge[nc_edge, 0]] = 1  # parent 1
        is_check_node[cut_edge[nc_edge, 1]] = 1  # parent 2
        is_check_elem = np.any(is_check_node[elements_temp[:nt, :]], axis=1).astype(int)
        check_elem = np.nonzero(is_check_elem)[0]  # element indices which contain min. 1 parent
        t2v = csr_matrix(
            (np.ones(len(check_elem) * 4, dtype=bool), (
                np.tile(check_elem, 4),
                elements_temp[check_elem, :].flatten("F")
            )),
            shape=(nt, n)
        )
        i, j = t2v[:, cut_edge[nc_edge, 0]].multiply(t2v[:, cut_edge[nc_edge, 1]]).nonzero()
        new_marked_elements, unique_inv, n_nc_edges = np.unique(i, return_inverse=True, return_counts=True)

        # Filter non-conforming elements for elements with 5/6 hanging nodes
        new_red_56_marked_elements = new_marked_elements[n_nc_edges >= 5]

        # Filter non-conforming elements for elements with 4 hanging nodes and 1 red face
        possible_red_4_marked_elements = new_marked_elements[n_nc_edges == 4]  # first criterion
        cut_edge_idx = j[(n_nc_edges == 4)[unique_inv]]
        parents = cut_edge[np.ix_(cut_edge_idx, [0, 1])].reshape(len(possible_red_4_marked_elements), 8)
        n_parents, counts_two = n_unique_per_row(parents, return_row_counts_equals_2=True)
        new_red_4_marked_elements = possible_red_4_marked_elements[counts_two == False]  # second criterion

        # append new elements
        red_marked_elements = np.unique(
            np.concatenate([red_marked_elements, new_red_56_marked_elements, new_red_4_marked_elements], axis=0))
        iter += 1

    # assign temp variables
    if element_sets is not None:
        element_sets = temp_element_sets
    nodes, elements = nodes_temp, elements_temp

    # Preallocation
    elements = np.append(elements, np.zeros([5 * nt, 4], dtype="int32"), axis=0)
    nodes = np.append(nodes, np.zeros([4 * nt, 3]), axis=0)

    # GREEN refinement of non-conforming elements
    # find remaining non-conforming elements
    nc_edge = np.arange(n_cut)
    is_check_node = np.zeros(n, dtype="int8")
    is_check_node[cut_edge[nc_edge, 0]] = 1  # parent 1
    is_check_node[cut_edge[nc_edge, 1]] = 1  # parent 2
    is_check_elem = np.any(is_check_node[elements[:nt, :]], axis=1).astype(int)
    check_elem = np.nonzero(is_check_elem)[0]
    t2v = csr_matrix(
        (np.ones(len(check_elem) * 4, dtype=bool), (
            np.tile(check_elem, 4),
            elements[check_elem, :].flatten("F")
        )),
        shape=(nt, n)
    )
    i, j = t2v[:, cut_edge[nc_edge, 0]].multiply(t2v[:, cut_edge[nc_edge, 1]]).nonzero()
    marked_elements, unique_inv, n_nc_edges = np.unique(i, return_inverse=True, return_counts=True)

    # Find green_marked_elements -> elements with (1) 3 hanging nodes (2) 1 red face
    possible_green_marked_elements = marked_elements[n_nc_edges == 3]  # first criterion
    print(f"\tpossible_green_marked_elements:\t{len(possible_green_marked_elements)}")
    cut_edge_idx = j[(n_nc_edges == 3)[unique_inv]]
    parents = cut_edge[np.ix_(cut_edge_idx, [0, 1])].reshape(len(possible_green_marked_elements), 6)
    n_parents = n_unique_per_row(parents)
    green_marked_elements = possible_green_marked_elements[n_parents == 3]  # second criterion
    n_new = len(green_marked_elements)
    print(f"\tgreen_marked_elements:\t\t\t{n_new}")

    # refine 1. case elements (3 hanging nodes, 1 red face)
    green_cut_edges = cut_edge[cut_edge_idx, :].reshape(len(possible_green_marked_elements), 3, 3)[n_parents == 3]
    p1 = green_cut_edges[:, 0, 0]
    p2 = green_cut_edges[:, 0, 1]
    p12 = green_cut_edges[:, 0, 2]
    p3_idx = (green_cut_edges[:, 1, :-1] != np.transpose([p1])) * (green_cut_edges[:, 1, :-1] != np.transpose([p2]))
    p3 = green_cut_edges[:, 1, :-1][p3_idx]
    p13_idx = np.any(green_cut_edges[:, 1:, :-1] == np.transpose([[p1]], (2, 0, 1)), axis=2)
    p13 = green_cut_edges[:, 1:, 2][p13_idx]
    p23_idx = np.any(green_cut_edges[:, 1:, :-1] == np.transpose([[p2]], (2, 0, 1)), axis=2)
    p23 = green_cut_edges[:, 1:, 2][p23_idx]
    p4_idx = (elements[green_marked_elements, :] != np.transpose([p1])) \
             * (elements[green_marked_elements, :] != np.transpose([p2])) \
             * (elements[green_marked_elements, :] != np.transpose([p3]))
    p4 = elements[green_marked_elements, :][p4_idx]

    elements[green_marked_elements, :] = np.transpose([p4, p12, p13, p23])
    elements[nt:nt + n_new, :] = np.transpose([p4, p12, p13, p1])
    elements[nt + n_new:nt + 2 * n_new, :] = np.transpose([p4, p12, p2, p23])
    elements[nt + 2 * n_new:nt + 3 * n_new, :] = np.transpose([p4, p3, p13, p23])

    # add to element sets (new green)
    if element_sets is not None:
        for key, element_set in element_sets.items():
            element_list = np.zeros(len(elements), dtype=bool)
            element_list[element_set[0]] = True
            idx_me = element_list[green_marked_elements].nonzero()[0]
            element_sets[key] = [
                np.sort(np.append(element_set[0], nt + np.array([idx_me, n_new + idx_me, 2 * n_new + idx_me])))]

    nt += n_new * 3

    # BISECTION of remaining non-conforming elements
    bisect_marked_elements = None
    iter = 1
    while bisect_marked_elements is None or len(bisect_marked_elements) != 0:
        # Sort cut_edge -> longest non conforming edge cutted first
        sort_idx = np.argsort(np.linalg.norm(nodes[cut_edge[:n_cut, 0]] - nodes[cut_edge[:n_cut, 1]], axis=1))
        cut_edge[:n_cut, :] = cut_edge[sort_idx, :]
        # Find non-conforming elements
        nc_edge = np.arange(n_cut)
        is_check_node = np.zeros(n, dtype="int8")
        is_check_node[cut_edge[nc_edge, 0]] = 1  # parent 1
        is_check_node[cut_edge[nc_edge, 1]] = 1  # parent 2
        is_check_elem = np.any(is_check_node[elements[:nt, :]], axis=1).astype(int)
        check_elem = np.nonzero(is_check_elem)[0]
        t2v = csr_matrix(
            (np.ones(len(check_elem) * 4, dtype=bool), (
                np.tile(check_elem, 4),
                elements[check_elem, :].flatten("F")
            )),
            shape=(nt, n)
        )
        i, j = t2v[:, cut_edge[nc_edge, 0]].multiply(
            t2v[:, cut_edge[nc_edge, 1]]).nonzero()  # i: element, j: cut_edge-ids
        bisect_marked_elements, unique_idx = np.unique(i, return_index=True)
        print(f"\tBisection-refinement iteration: \t{iter:02}\tbisect_marked_elements: \t{len(bisect_marked_elements)}")

        # actual bisection of non-conforming elements
        n_new = len(bisect_marked_elements)
        bisect_elements = elements[bisect_marked_elements, :]

        p1 = cut_edge[j[unique_idx], 0]
        p2 = cut_edge[j[unique_idx], 1]
        p12 = cut_edge[j[unique_idx], 2]
        p3p4 = bisect_elements[
            (bisect_elements != np.transpose([p1])) * (bisect_elements != np.transpose([p2]))
            ].reshape(n_new, 2)
        p3 = p3p4[:, 0]
        p4 = p3p4[:, 1]

        elements[bisect_marked_elements, :] = np.transpose([p1, p12, p3, p4])
        elements[nt:nt + n_new, :] = np.transpose([p2, p12, p3, p4])

        # add to element sets (new bisect)
        if element_sets is not None:
            for key, element_set in element_sets.items():
                element_list = np.zeros(len(elements), dtype=bool)
                element_list[element_set[0]] = True
                idx_me = element_list[bisect_marked_elements].nonzero()[0]
                element_sets[key] = [np.sort(np.append(element_set[0], nt + idx_me))]

        nt += n_new
        iter += 1
    # Output
    nodes = nodes[:n, :]
    elements = elements[:nt, :]
    output = nodes, elements

    # Update + return node_sets
    if node_sets is not None:
        for key, node_set in node_sets.items():
            node_list = np.zeros(n, dtype=bool)
            node_list[node_set] = True
            parents_in_set = node_list[cut_edge[:, 0]] * node_list[cut_edge[:, 1]]
            node_sets[key] = np.sort(np.append(node_sets[key], cut_edge[parents_in_set, 2]))
        output += (node_sets,)

    # Filter for surface element sets and return element sets
    if element_sets is not None:
        for key, element_set in element_sets.items():
            if surface_element_set_keyword in key:
                element_sets[key] = [filter_surface_elements(elements, element_set[0])]
        output += (element_sets,)

    return output