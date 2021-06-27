from src.refineRGB import *

dir = "example02_cantilever/"

interface.refine_by_density(
    "example02_cantilever/sample03_3D_C3D4.inp",
    "example02_cantilever/sample03_3D_C3D4_tosca_distribution.inp",
    min_density=0.8,
    method="RGB",
    save=True
)
