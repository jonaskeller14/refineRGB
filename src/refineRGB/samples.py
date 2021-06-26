import gmsh
import sys
import meshio
import numpy as np


def create_cube(path: str, r: float = 1, mesh_size: float = 1e-1, format: str="vtk"): #tODO: implement format
    """
    Creates cube geometry and mesh,
    saves mesh as .vtk as/in path. Reads mesh via meshio.
    :param r: Radius of cube, or half of the length
    :param mesh_size: global mesh size
    :param path: folder + filename of mesh-file location
    :return: tuple of nodes, elements
    """
    gmsh.initialize(sys.argv)
    gmsh.model.add("cuboid")
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Format", 16)  # 16 = .vtk
    gmsh.model.occ.addBox(-r, -r, -r, 2 * r, 2 * r, 2 * r, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    if path is not None:
        gmsh.write(path + ".vtk")
    # Open GUI
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()
    gmsh.finalize()
    mesh = meshio.read(path + ".vtk")
    nodes = mesh.points
    elements = mesh.cells[3][1]
    meshio.write_points_cells(path + "." + format, points=nodes, cells=[("tetra", elements),])
    return nodes, elements


def marked_elements_sphere_intersection(nodes: np.ndarray, elements: np.ndarray, r_min: float = 0, r_max: float = 1):
    """
    Computes intersection of given mesh (nodes, elements) with a (hollow) sphere.
    :param nodes: list of nodes x,y,z-coordinates as np.ndarray
    :param elements: list of elements nodes-ids as np.ndarray
    :param r_min: minimum or inner radius of sphere
    :param r_max: maximum or outer radius of sphere
    :return: marked_elements as np.ndarray
    """
    norm = np.linalg.norm(nodes, axis=1)**2
    marked_nodes = (norm >= r_min)*(norm <= r_max)
    marked_elements = np.nonzero(np.any(marked_nodes[elements], axis=1))[0]
    return marked_elements


def marked_elements_ball(nodes: np.ndarray, elements: np.ndarray, r_max=0.6):
    norm = np.linalg.norm(nodes, axis=1) ** 2
    marked_nodes = (norm <= r_max) * np.logical_or((nodes[:, 1] >= 2*nodes[:, 0]), (nodes[:, 1] >= -2*nodes[:, 0]))
    marked_elements = np.nonzero(np.any(marked_nodes[elements], axis=1))[0]
    return marked_elements


def marked_elements_random(elements: np.ndarray, size: float = 0.2, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    nt = len(elements)
    return np.random.rand(nt).argsort()[:int(nt*size)]



