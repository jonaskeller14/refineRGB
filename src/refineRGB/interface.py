import meshio
from .mesh import get_neighbours, fix_order3, second_to_first_order_mesh, first_to_second_order_mesh
from .mapping import mapping_to_marked_elements
from .refine import *
from .density import *
import pandas as pd
import time


def refine(
        input_mesh: str,
        mapping_meshes: list,
        iterations: int = 1,
        method: str = "RGB_T1",
        save: bool = True,
        save_format: str = "vtk",
        max_elements: int = -1,
        debug: bool = False,
):
    """
    interface function for mesh refinement.

    :param input_mesh: unrefined mesh
    :param mapping_meshes: list of mapping-meshes
    :param iterations: number of refinement iterations
    :param method: refinement method e.g. "RGB_T1"
    :param save: saves refined mesh if set to True
    :param save_format: only if save=True, file format e.g. ".vtk"
    :param max_elements: upper boundary for number of elements in refined mesh
    :param debug: records runtime, saves additional meshes for inspection --> significant increase in runtime
    :return: refined mesh

    Documentation:
    - Keller, Jonas (2021). Implementation and analysis of a local refinement method for tetrahedral meshes
    """
    assert method in ["RGB", "RGB_T1", "RGB_T2", "RGB_T3", "B", "R"]
    assert iterations >= 0

    acc_tic = time.clock()

    # debug preferences
    debug_record_runtime = True
    debug_export_meshes = True

    mesh_path = input_mesh.split(".")[0]
    mesh = meshio.read(input_mesh)

    is_tetra10 = mesh.cells[0][1].shape[1] == 10
    if is_tetra10:
        mesh = second_to_first_order_mesh(mesh)

    if debug and debug_record_runtime:
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
        iter_tic = time.clock()
        # REFINEMENT
        nodes = mesh.points
        elements = mesh.cells[0][1]
        node_sets = mesh.point_sets
        element_sets = mesh.cell_sets

        # mapping
        mapping_tic = time.clock()
        marked_elements = mapping_to_marked_elements(nodes, elements, source_files=mapping_meshes)
        mapping_runtime = time.clock() - mapping_tic
        if debug and debug_export_meshes:
            element_sets["marked_elements"] = [marked_elements]
            old_elements = elements

        # transition
        if "_T" in method and iter != iterations:
            marked_elements = get_neighbours(elements, marked_elements, common_nodes=int(method[-1]))
            if debug and debug_export_meshes:
                element_sets["marked_elements_transition"] = [marked_elements]

        # runtime data
        if debug and debug_record_runtime:
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
        refine_tic = time.clock()
        if method == "R":
            nodes, elements = refine_red(nodes, elements, marked_elements)
            node_sets, element_sets = {}, {}
        elif method == "B":
            nodes, elements, node_sets, element_sets = refine_bisect(nodes, elements, marked_elements, node_sets, element_sets)
        elif method in ["RGB", "RGB_T1", "RGB_T2", "RGB_T3"]:
            nodes, elements, node_sets, element_sets = refine_red_green_bisect(nodes, elements, marked_elements, node_sets, element_sets)
        if len(elements) > max_elements and max_elements != -1:
            break
        elements = fix_order3(nodes, elements)

        if debug and debug_record_runtime:
            # runtime data
            runtime_data["after n_nodes"] = [len(nodes)]
            runtime_data["after n_elements"] = [len(elements)]
            runtime_data["refinement runtime"] = [time.clock() - refine_tic]

        if debug and debug_export_meshes:
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
        if debug and debug_export_meshes:
            # refinement regions
            mesh.cell_sets["mapping_again"] = [mapping_to_marked_elements(nodes, elements, source_files=mapping_meshes)]  # optional
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
            mesh = merge_data(mesh, **kwargs)
            mesh = add_ref_data(mesh, input_mesh)
            mesh.write(f"{mesh_path}_{method}_{iter:02}_{iterations:02}.{save_format}")

        if debug and debug_record_runtime:
            # append new runtime data
            runtime_data["iter runtime"] = [time.clock() - iter_tic]
            runtime_data["accumulated runtime"] = [time.clock() - acc_tic]
            df = df.append(pd.DataFrame.from_dict(runtime_data))
    # final conversion and export
    if is_tetra10:
        mesh = first_to_second_order_mesh(mesh)
    if save:
        mesh.write(f"{mesh_path}_{method}_refined.{save_format}")
    if debug and debug_record_runtime:
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