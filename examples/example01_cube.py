from src.refineRGB import *


# directory path
dir = "example01_cube/"

# create initial mesh "cube.vtk"
# samples.create_cube(r=1, mesh_size=0.1, path=dir + "cube")

# RGB_T1-refinement
refined_mesh = interface.refine(
        input_mesh=dir + "cube.vtk",
        mapping_meshes=[dir + "cone.stl", dir + "cylinder.stl"],
        iterations=2,
        method="RGB_T1",
        save=True,  # save final mesh in directory
        debug=True  # records runtime, saves additional meshes for inspection --> significant increase in runtime
)

# B-refinement
refined_mesh = interface.refine(
        input_mesh=dir + "cube.vtk",
        mapping_meshes=[dir + "cone.stl", dir + "cylinder.stl"],
        iterations=6,
        method="B",
        save=True,  # save final mesh in directory
        debug=True  # records runtime, saves additional meshes for inspection --> significant increase in runtime
)