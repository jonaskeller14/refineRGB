import meshio
from src.refineRGB import *
import time

dir = "example03_topo/"


# interface.refine_by_surfaces(
#     dir + "topo_opt_job.vtk",
#     [dir + "smooth_033.stl", dir + "smooth_032_offset_1.stl", dir + "smooth_032_offset_2.stl"],
#     iterations=1,
#     method="RGB_T",
#     save=True,
#     debug=True
# )
#
# interface.refine_by_surfaces(
#     dir + "topo_opt_job.vtk",
#     [dir + "smooth_033.stl", dir + "smooth_032_offset_1.stl", dir + "smooth_032_offset_2.stl"],
#     iterations=2,
#     method="B",
#     save=True,
#     debug=True
# )
mesh1 = meshio.read(dir + "topo_opt_job.inp")
mesh1 = mesh.second_to_first_order_mesh(mesh1)
mesh1.write(dir + "topo_opt_job_C3D4.vtk")