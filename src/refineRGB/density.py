import numpy as np
import meshio
from scipy import interpolate
from .mesh import get_element_midpoints


def add_cell_data_to_mesh(mesh: meshio.Mesh, path: str):
    density = np.zeros(len(mesh.cells[0][1]))
    with open(path, "r") as file:
        lines = file.read().splitlines()
        i = 0
        while i < len(lines):
            if "*DISTRIBUTION, DESIGN VARIABLE, LOCATION=ELEMENT, NAME=__SMATso_Distribution_Mass_1" in lines[i]:
                i += 2
                j = 0
                while i < len(lines):
                    density[j] = float(lines[i].split(",")[1])
                    i += 1
                    j += 1
            else:
                i += 1
    mesh.cell_data["density"] = [density]
    return mesh


def get_ids_by_density(mesh: meshio.Mesh, min=0):
    density = mesh.cell_data["density"][0]
    ids = np.nonzero(density >= min)[0]
    return ids


def interpolate_density(old_mesh: meshio.Mesh, new_nodes: np.ndarray, new_elements: np.ndarray, method="nearest"):
    old_density = old_mesh.cell_data["density"][0]
    old_nodes = old_mesh.points
    old_elements = old_mesh.cells[0][1]
    old_points = get_element_midpoints(old_nodes, old_elements)
    new_points = get_element_midpoints(new_nodes, new_elements)
    new_density = interpolate.griddata(old_points, old_density, new_points, method=method)
    new_mesh = meshio.Mesh(
        points=new_nodes,
        cells=[("tetra", new_elements),],
        cell_data={"density": [new_density]}
    )
    return new_mesh
