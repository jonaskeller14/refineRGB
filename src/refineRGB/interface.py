import meshio


# TODO: add exampke files, die das hier aufrufen. mach aus allem ein python package

def refine_stl(mesh_file: str, stl_files: list, iterations: int = 1, method: str = "red_green_bisect", save: bool = True, transition: bool = True):
    """

    :param save:
    :param transition:
    :param mesh_file:
    :param stl_files:
    :param iterations:
    :param method:
    :return:
    """
    assert method in ["red_green_bisect", "bisect", "red"]
    assert iterations >= 0

    initial_mesh = meshio.read(mesh_file)

    for iter in range(iterations):
        marked_elements = pass
        if transition:
            for tran in range(iterations-iter): # FIXME: anpassen, wenn eine iteration -> keine transition
                pass

    # TODO: Funktion, für jede refinement iteration > 1, nachbarelemente mitauswählen
    # TODO: maplib integrieren
    # TODO: create prefix "00_" and save all iterations ? same location as input mesh
    # TODO: unterscheide ob mesh tetra oder tretra 10 und wenn nötig convertieren
    # Todo negative volume fix
    return refined_mesh
