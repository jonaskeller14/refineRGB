from src.refineRGB import *

dir = "example02_cantilever/"

import meshio
mesh1 = meshio.read(dir + "sample07_cantilever_beam.inp")
mesh.sets_to_data(mesh1).write(dir + "sample07_cantilever_beam.vtk")

# interface.refine_by_surfaces(
#     mesh_file=dir + "sample07_cantilever_beam.inp",
#     source_files=[
#         dir + "sample07_iso07.stl",
#         dir + "sample07_iso01.stl"
#     ],
#     method="RGB_T",
#     iterations=2,
#     save=False,
#     debug=True
# )
#
# interface.refine_by_surfaces(
#     mesh_file=dir + "sample07_cantilever_beam.inp",
#     source_files=[
#         dir + "sample07_iso07.stl",
#         dir + "sample07_iso01.stl"
#     ],
#     method="B",
#     iterations=4,
#     save=False,
#     debug=True
# )