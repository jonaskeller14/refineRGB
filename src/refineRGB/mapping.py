import pymapping
import meshio
import numpy as np
from mesh import triangular_to_thin_tetrahedral


def stl_to_marked_elements(target_nodes: np.ndarray, target_elements: np.ndarray, files: list):
    """
    Mapping of 2D-.stl-meshes to 3D-tetrahedral-mesh.
    :param target_nodes:
    :param target_elements:
    :param files: list of .stl-file-names (paths)
    :return:
    """
    # target mesh
    target_mesh = meshio.Mesh(
        points=target_nodes.astype("float64"),
        cells=[("tetra", target_elements), ]
    )
    marked_elements = np.zeros(len(target_elements), dtype="float64")

    for file in files:
        # source mesh: convert triangular to tetrahedral mesh
        source_stl_mesh = meshio.read(file)
        stl_nodes = source_stl_mesh.points
        stl_elements = source_stl_mesh.cells[0][1]
        source_nodes, source_elements = triangular_to_thin_tetrahedral(stl_nodes, stl_elements)
        source_mesh = meshio.Mesh(
            points=source_nodes.astype("float64"),
            cells=[("tetra", source_elements), ],
            cell_data={"marked_elements": [np.ones(len(source_elements), dtype="float64")]}
        )

        # mapping
        mapper = pymapping.Mapper()
        mapper.prepare(source_mesh, target_mesh, method="P0P0", intersection_type="Triangulation")
        res = mapper.transfer("marked_elements", default_value=0.0)
        mapping_res = pymapping.MappingResult(res.field_target, res.mesh_target)
        mapping_mesh = mapping_res.mesh_meshio()

        # update marked_elements
        marked_elements += mapping_mesh.cell_data["marked_elements"][0]
    return marked_elements.nonzero()[0]