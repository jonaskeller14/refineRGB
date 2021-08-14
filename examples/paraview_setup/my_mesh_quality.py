import vtk
import numpy as np


def main():
    m = [
        ["Radius Ratio", 2],
        ["Relative Size Squared", 12],
        ["Minimum Dihedral Angle", 6],
        ["Volume", 19]
    ]
    thresholds = [
        [3, "upper"],
        [0.3, "lower"],
        [10, "lower"],
        [1, "upper"]
    ]
    fd_labels = ["Mean", "Min", "Max", "std", "var", "Q1", "Q2", "Q3", "Sum", "n_bad", "p_bad"]
    fd_values = np.zeros((len(m), len(fd_labels)))

    inp = self.GetInputDataObject(0, 0)
    outp = self.GetOutputDataObject(0)

    for idx in range(len(m)):
        # cell data
        mesh_quality = vtk.vtkMeshQuality()
        mesh_quality.SetInputData(inp)
        mesh_quality.SetTetQualityMeasure(m[idx][1])
        mesh_quality.Update()

        vtk_array = mesh_quality.GetOutput().GetCellData().GetVectors("Quality")
        vtk_array.SetName(m[idx][0] + "_" + str(idx))
        array = np.array(vtk_array)
        outp.GetCellData().AddArray(vtk_array)

        if thresholds[idx][1] == "upper":
            n_bad = np.sum(array > thresholds[idx][0])
        elif thresholds[idx][1] == "lower":
            n_bad = np.sum(array < thresholds[idx][0])
        # field data
        fd_values[idx, :] = np.array([
            np.mean(array),
            np.min(array),
            np.max(array),
            np.std(array),
            np.var(array),
            np.percentile(array, 25),
            np.percentile(array, 50),
            np.percentile(array, 75),
            np.sum(array),
            n_bad,
            n_bad / len(array)
        ])

    for fd_lab, fd_val_col in zip(fd_labels, fd_values.T):
        fd = vtk.vtkDoubleArray()
        fd.SetName(fd_lab)
        for val in fd_val_col:
            fd.InsertNextValue(val)
        outp.GetFieldData().AddArray(fd)


if __name__ == '__main__':
    main()