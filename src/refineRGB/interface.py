import meshio
from .mesh import get_neighbours, fix_order3, second_to_first_order_mesh, first_to_second_order_mesh
from .mapping import mapping_to_marked_elements
from .refine import *
from .density import *
import pandas as pd
import time


def refine(
        mesh_file: str,
        source_files: list,
        iterations: int = 1,
        method: str = "RGB_T",
        save: bool = True,
        save_format: str = "vtk",
        max_elements: int = -1,
        debug: bool = False,
):
    """
    :param debug: if True, all node/element sets + + marked_elements + green closures are written to node/element data. Each iteration is saved.
    :param save_format:
    :param max_elements:
    :param save:
    :param transition:
    :param mesh_file:
    :param source_files:
    :param iterations:
    :param method:
    :return:
    """
    assert method in ["RGB", "RGB_T", "B", "R"]
    assert iterations >= 0

    acc_tic = time.time()

    mesh_path = mesh_file.split(".")[0]
    mesh = meshio.read(mesh_file)

    is_tetra10 = mesh.cells[0][1].shape[1] == 10
    if is_tetra10:
        mesh = second_to_first_order_mesh(mesh)

    if debug:
        mesh.write(f"{mesh_path}_{method}_00.{save_format}")
        # read or create runtime data
        try:
            df = pd.read_excel(mesh_path + "_runtime.xlsx")
        except OSError:
            df = pd.DataFrame(columns=[
                "time",
                "method",
                "iter",
                "iterations",
                "before n_nodes",
                "before n_elements",
                "n_marked_elements",
                "mapping runtime",
                "after n_nodes",
                "after n_elements",
                "refinement runtime",
                "iter runtime",
                "accumulated runtime"
            ])
            print("Created new DataFrame")

    for iter in range(1,iterations+1):
        iter_tic = time.time()
        # REFINEMENT
        nodes = mesh.points
        elements = mesh.cells[0][1]
        node_sets = mesh.point_sets
        element_sets = mesh.cell_sets

        # mapping
        mapping_tic = time.time()
        marked_elements = mapping_to_marked_elements(nodes, elements, source_files=source_files)
        mapping_runtime = time.time() - mapping_tic
        if debug:
            element_sets["marked_elements"] = [marked_elements]
            old_elements = elements

        # transition
        if "_T" in method and iter != iterations:  #
            marked_elements = get_neighbours(elements, marked_elements, common_nodes=1)
            if debug:
                element_sets["marked_elements_transition"] = [marked_elements]

        # runtime data
        if debug:
            runtime_data = {
                "time": [time.ctime()],
                "method": [method],
                "iter": [iter],
                "iterations": [iterations],
                "before n_nodes": [len(nodes)],
                "before n_elements": [len(elements)],
                "n_marked_elements": [len(marked_elements)],
                "mapping runtime": [mapping_runtime]
            }

        # refinement
        refine_tic = time.time()
        if method == "R":
            nodes, elements, node_sets, element_sets = refine_red(nodes, elements, marked_elements, node_sets, element_sets)
        elif method == "B":
            nodes, elements, node_sets, element_sets = refine_bisect(nodes, elements, marked_elements, node_sets, element_sets)
        elif method == "RGB" or method == "RGB_T":
            nodes, elements, node_sets, element_sets = refine_red_green_bisect(nodes, elements, marked_elements, node_sets, element_sets)
        if len(elements) > max_elements and max_elements != -1:
            break
        elements = fix_order3(nodes, elements)

        if debug:
            # runtime data
            runtime_data["after n_nodes"] = [len(nodes)]
            runtime_data["after n_elements"] = [len(elements)]
            runtime_data["refinement runtime"] = [time.time() - refine_tic]

            # refinement regions
            if "_T" in method and iter != iterations:
                element_sets["green_closures"] = [get_green_elements(old_elements, elements, element_sets, marked_elements_key="marked_elements_transition")]
            else:
                element_sets["green_closures"] = [get_green_elements(old_elements, elements, element_sets, marked_elements_key="marked_elements")]

        # create new mesh
        mesh = meshio.Mesh(
            points=nodes,
            cells=[("tetra", elements),],
            point_sets=node_sets,
            cell_sets=element_sets
        )
        if debug:
            # refinement regions
            mesh = sets_to_data(mesh)
            if "_T" in method and iter != iterations:
                kwargs = {
                    "marked_elements_transition": 2,
                    "marked_elements": 3,
                    "green_closures": 1
                }
            else:
                kwargs = {
                    "marked_elements": 3,
                    "green_closures": 1
                }
            mesh = consolidate_data(mesh, **kwargs)
            mesh = add_ref_data(mesh, mesh_file)
            mesh.write(f"{mesh_path}_{method}_{iter:02}.{save_format}")

            # append new runtime data
            runtime_data["iter runtime"] = [time.time() - iter_tic]
            runtime_data["accumulated runtime"] = [time.time() - acc_tic]
            df = df.append(pd.DataFrame.from_dict(runtime_data))
    # final conversion and export
    if is_tetra10:
        mesh = first_to_second_order_mesh(mesh)
    if save:
        mesh.write(f"{mesh_path}_{method}_refined.{save_format}")
    if debug:
        # write .xlsx data
        # df.to_excel(mesh_path + "_runtime.xlsx", index=False, header=True)
        writer = pd.ExcelWriter(mesh_path + "_runtime.xlsx")
        df.to_excel(writer, sheet_name='runtime', index=False, na_rep='NaN')
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets['runtime'].set_column(col_idx, col_idx, column_length)
        writer.save()
        writer.close()
    return mesh