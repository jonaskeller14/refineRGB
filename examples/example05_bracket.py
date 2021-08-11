from src.refineRGB import *

interface.refine(
    input_mesh="example05_bracket/bracket_job.vtk",
    mapping_meshes=[
        "example05_bracket/solution02.stl"
    ],
    iterations=5,
    method="B",
    save=False,
    save_format="vtk",
    debug=True,
)
for i in [1,2,3]:
    interface.refine(
        input_mesh="example05_bracket/bracket_job.vtk",
        mapping_meshes=[
            "example05_bracket/solution02.stl"
        ],
        iterations=i,
        method="RGB_T",
        save=False,
        save_format="vtk",
        debug=True,
    )