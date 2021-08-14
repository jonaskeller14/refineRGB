import vtk
import numpy as np


def main():
    nbins = 10
    density = True
    ranges = np.array([
        [1, 5],  # Radius Ratio
        [0, 1],  # Relative Size Squard
        [0, 100],  # Minimum Dihedral Angle
        [0, 1],   # Volume
    ])
    input = inputs[0]
    for idx,cd in enumerate(input.CellData):
        cd_name = cd.GetName()
        hist, bins = np.histogram(cd, range=ranges[idx], bins=nbins)
        l_outlier = np.sum(cd < ranges[idx, 0])
        hist[0] += l_outlier
        u_outlier = np.sum(cd > ranges[idx, 1])
        hist[-1] += u_outlier
        if density:
            hist = hist / np.sum(hist)
        output.RowData.append(hist, cd_name)
        bins = (bins[:-1] + bins[1:]) / 2
        output.RowData.append(bins, cd_name + "_extents")


if __name__ == '__main__':
    main()