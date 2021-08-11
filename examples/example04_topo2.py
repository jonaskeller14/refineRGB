from src.refineRGB import *

for i in [1,2,3]:
    interface.refine(
        input_mesh="example04_topo2/topo_opt_job2.inp",
        mapping_meshes=[
            "example04_topo2/smooth_033.stl"
        ],
        iterations=i,
        method="RGB_T",
        save=False,
        save_format="vtk",
        debug=True,
    )
