from src.refineRGB import *


dir = "example01_pipes/"

# samples.create_cube(r=1, mesh_size=0.1, path=dir + "cube")
interface.refine(
    dir + "cube.vtk",
    [dir + "cone.stl", dir + "cylinder.stl"],
    iterations=3,
    method="B",
    save=False,
    debug=True
)