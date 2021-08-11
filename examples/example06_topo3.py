from src.refineRGB import *


# directory path
dir = "example06_topo3/"

# RGB_T1-refinement
refined_mesh = interface.refine(
        input_mesh=dir + "topo_opt_job3.inp",
        mapping_meshes=[dir + "smooth_033.stl"],
        iterations=3,
        method="RGB_T1",
        save=True,  # save final mesh in directory
        debug=True  # records runtime, saves additional meshes for inspection --> significant increase in runtime
)

# B-refinement
refined_mesh = interface.refine(
        input_mesh=dir + "topo_opt_job3.inp",
        mapping_meshes=[dir + "smooth_033.stl"],
        iterations=6,
        method="B",
        save=True,  # save final mesh in directory
        debug=True  # records runtime, saves additional meshes for inspection --> significant increase in runtime
)