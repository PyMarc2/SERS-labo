from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import glob
import numpy as np


def set_ticks(axes, xMajor, xMinor, xFormat, yMajor, yMinor, yFormat):
    # FAMOUS TICKS
    axes.xaxis.set_major_locator(MultipleLocator(xMajor))
    axes.xaxis.set_major_formatter(FormatStrFormatter(xFormat))
    axes.xaxis.set_minor_locator(MultipleLocator(xMinor))

    axes.yaxis.set_major_locator(MultipleLocator(yMajor))
    axes.yaxis.set_major_formatter(FormatStrFormatter(yFormat))
    axes.yaxis.set_minor_locator(MultipleLocator(yMinor))


def get_data(path, delimiter):
    with open(path, 'r') as file:
        allData = []
        data = np.loadtxt(file, delimiter=delimiter)
        allData.append(data)
        allData = np.asarray(allData)
        return allData
