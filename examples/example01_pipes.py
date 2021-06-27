from src.refineRGB import *

dir = "example01_pipes/"

samples.create_cube(r=1, mesh_size=0.5, path=dir + "cube")

interface.refine_by_stl(
    dir + "cube.vtk",
    [dir + "x_pipe.stl", dir + "y_pipe.stl"],
    iterations=1,
    method="B",
    transition=False,
    save=True
)
