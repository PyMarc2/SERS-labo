from __future__ import annotations
import numpy as np
import cv2
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import AutoMinorLocator
from tkinter.filedialog import asksaveasfilename
from scipy.signal import order_filter
from scipy import signal
from spectrumuncurver import SpectrumUncurver
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
np.seterr(all='warn')
import warnings
warnings.filterwarnings('error')


class RamanSpectrumImage:

    def __init__(self, imageLoadingInfo=None):
        self.dataType = np.uint32
        self.imageArray = None
        
        if imageLoadingInfo is not None:
            self.load(imageLoadingInfo)
        else:
            self.imageArray = np.zeros((400, 1340), dtype=self.dataType)

        self.metadata = {"name": "", "expositionTime": 0, "sampleId": 0, "xunits": "pixels"}

    @property
    def shape(self):
        return self.imageArray.shape[1], self.imageArray.shape[0]

    @property
    def width(self):
        return self.imageArray.shape[1]

    @property
    def height(self):
        return self.imageArray.shape[0]

    def __call__(self):
        return self.imageArray

    def __getitem__(self, position):
        return self.imageArray[position[1], position[0]]

    def __setitem__(self, position, value):
        self.imageArray[position[1], position[0]] = value

    def __iter__(self):
        longIterrator = []
        for x in range(self.width):
            for y in range(self.height):
                longIterrator.append(self[x, y])
        return longIterrator.__iter__()

    def __sub__(self, other: RamanSpectrumImage):
        # print("MAX OF ARRAY:", max(other))
        if max(other) <= 2**64:
            for x in range(self.width):
                for y in range(self.height):
                    try:
                        sub = self[x, y] - other[x, y]
                        self[x, y] = sub
                    except Warning as e:
                        self[x, y] = 0
            return self
        else:
            print("subtraction is garanteed to hit 0. You cannot multiply the subtrator by a higher number.")

    def __mul__(self, mul):
        prod = max(self) * mul
        if prod <= 2**64:
            for x in range(self.width):
                for y in range(self.height):
                    self[x, y] *= mul
            return self
        else:
            raise ValueError("Product exceeds 64bits. Can't proceed.")

    def load(self, imageInstance):
        # print(type(imageInstance))
        if isinstance(imageInstance, str):
            try:
                imagePath = imageInstance
                self.imageArray = np.array(Image.open(imagePath)).astype(self.dataType)
            except Exception as e:
                print("\nERROR: You must enter a valid path\n", e)

        elif isinstance(imageInstance, np.ndarray):
            try:
                if imageInstance.ndim != 2:
                    raise TypeError("Image must be grayscale (2d).")

                self.imageArray = imageInstance.astype(self.dataType)
            except Exception as e:
                print("\nERROR: You must input a valid array\n", e)

        elif isinstance(imageInstance, (type(Image), TiffImageFile)):
            try:
                tempImageArray = np.array(imageInstance)
                if tempImageArray.ndim != 2:
                    raise TypeError("Image must be grayscale (2d).")

                self.imageArray = np.array(imageInstance).astype(self.dataType)
            except Exception as e:
                print("\nERROR: You must input a valid PIL Image\n", e)


class RamanGrapher:

    def __init__(self, image=None, dataUnit="pixel"):
        self.initialImage = None
        self.outputImage = None

        self.plotData = [[], []]
        self.figure = None
        self.ax = None
        self.xlabel = "pixel"
        self.ylabel = "relative intensity [a. u.]"

        self.dataUnit = dataUnit
        self.calibrationUnit = None
        self.calibrationEquation = None
        self.curvatureEquation = None

        self.excitationWavelenght = 785
        self.amountOfPlots = 0
        self.dataType = np.uint32

        if image is not None:
            self.load_ramanSpectrum_image(image, dataUnit)

    def reset_grapher(self):
        self.initialImage = None
        self.outputImage = None
        self.plotData = [[], []]
        self.figure = None
        self.ax = None
        self.calibrationEquation = None
        self.uncurveEquation = None
        self.dataUnit = "pixel"
        self.calibrationUnit = None
        self.xlabel = "pixel"
        self.ylabel = "relative intensity [a. u.]"
        self.amountOfPlots = 0
        self.excitationWavelenght = 785

    def partially_reset_grapher(self):
        self.initialImage = None
        self.outputImage = None
        self.plotData = [[], []]

    def load_ramanSpectrum_image(self, image, dataUnit="pixel"):
        self.partially_reset_grapher()
        self.initialImage = image
        self.outputImage = image
        self.dataUnit = dataUnit

    def reset_output_image(self):
        self.outputImage = self.initialImage

    def save_plot_dialog(self):
        try:
            path = asksaveasfilename()
            self.figure.savefig(path, dpi=600)
        except Exception as e:
            print(e)

    def find_calibration_from_points(self, pixelData, nmData):
        polynomial = np.polyfit(pixelData, nmData, deg=3)
        return polynomial

    def load_calibration_polynomial(self, *args, unit='nm'):
        self.calibrationEquation = np.poly1d(args)
        print(self.calibrationEquation)
        self.calibrationUnit = unit

    def load_curvature_polynomial(self, *args, type='poly'):
        if type == 'poly':
            self.curvatureEquation = np.poly1d(args)

    def modify_image_calibration(self):
        if self.calibrationEquation is not None and self.dataUnit == "pixel":
            self.plotData[0] = self.calibrationEquation(np.linspace(0, self.outputImage.width-1, self.outputImage.width))
            self.dataUnit = self.calibrationUnit
            # print("\nLENGHT OF XAXIS FROM CALIBRATION:\n", len(self.calibratedXAxis))
        else:
            pass

    def modify_curvature(self):
        for ypos, i in enumerate(range(self.outputImage.height)):
            corr = -int(self.curvatureEquation(i))
            if corr >= 1:
                self.outputImage[corr:, ypos] = self.outputImage[0:-corr, ypos]
            elif corr < 0:
                self.outputImage[0:corr-1, ypos] = self.outputImage[-corr:-1, ypos]
            else:
                self.outputImage[:, ypos] = self.outputImage[:, ypos]

    def modify_image_to_summed_plot(self):
        self.plotData[1] = [sum(self.outputImage[_, :]) for _ in range(self.outputImage.width)]
        self.plotData[0] = np.linspace(0, self.outputImage.width-1, self.outputImage.width)
        print(self.plotData[0])

    def modify_switch_units(self, units):

        if self.dataUnit == units or self.dataUnit == "pixel":
            pass
        elif units == 'cm-1':
            output = np.divide((1*10**7), self.excitationWavelenght) - np.divide((1*10**7), self.plotData[0])
            self.plotData[0] = output
            self.dataUnit = 'cm-1'

        elif units == 'nm':
            pass

    def modify_smoothen(self, order, fc, btype="lowpass"):
        b, a = signal.butter(order, fc, btype=btype)
        output = signal.filtfilt(b, a, self.plotData[1])
        self.plotData[1] = output

    def modify_normalize_plot(self):
        self.plotData[1] /= max(self.plotData[1])

    def prepare_plot_change_xlabel(self):
        if self.dataUnit == 'nm':
            self.xlabel = "wavelenght [nm]"
        elif self.dataUnit == 'cm-1':
            self.xlabel = "wavenumber [cm$^{-1}$]"
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def prepare_plot_reformat_subplots(self):
        n = len(self.figure.axes)
        for i in range(n):
            self.figure.axes[i].change_geometry(n + 1, 1, i + 1)

        self.ax = self.figure.add_subplot(n + 1, 1, n + 1)

    def add_plot(self, figsize=(9, 8), label="", normalized=True, uncurve=True,  xlimits=(0, 1340), xunit='nm', subTicks=True, majorTicks=100):
        if self.figure is None:
            self.figure = plt.figure(figsize=figsize)
        if uncurve:
            self.modify_curvature()
        self.modify_image_to_summed_plot()
        self.modify_image_calibration()
        self.modify_smoothen(2, 0.2)
        self.modify_switch_units(xunit)

        if normalized:
            self.modify_normalize_plot()

        self.prepare_plot_reformat_subplots()
        self.prepare_plot_change_xlabel()

        self.ax.plot(self.plotData[0][xlimits[0]:xlimits[1]], self.plotData[1][xlimits[0]:xlimits[1]], label=label, linewidth=1.2)
        
        if subTicks:
            self.ax.xaxis.set_minor_locator(AutoMinorLocator())

        self.ax.xaxis.set_major_locator(MultipleLocator(majorTicks))

        self.ax.legend()
        plt.tight_layout()

        if xunit == 'both':
            self.ax2 = self.ax.twiny()
            self.modify_switch_units('cm-1')
            self.ax2.plot(self.plotData[0], np.ones(len(self.plotData[0])))

    def add_peaks(self, distance=4, height=0.2, threshold=0, prominence=0.1, width=2):
        peaks, _ = signal.find_peaks(self.plotData[1], distance=distance, height=height, threshold=threshold, prominence=prominence, width=width)
        print(peaks)
        if peaks.any():
            self.ax.plot(self.plotData[0][peaks], self.plotData[1][peaks], 'o')
        for x, y in zip(self.plotData[0][peaks], self.plotData[1][peaks]):
            label = "{}".format(int(x))
            self.ax.annotate(label, (x, y), textcoords="offset points", xytext=(-10, -10), ha="center", rotation=90, color='k')

    @staticmethod
    def show_plot():
        plt.tight_layout()
        plt.show()

    def show_image(self, figsize=(12, 5)):
        if self.figure is None:
            self.figure = plt.figure(figsize=figsize)
        if self.curvatureEquation is not None:
            self.modify_curvature()
        plt.imshow(self.outputImage.imageArray.astype(np.float32))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # R6G 20mg/ml(saturated) sur Thorlabs paper
    # rmg1 = RamanGrapher()
    # rmg1.load_calibration_polynomial(1.67 * 10 ** -8, -4.89 * 10 ** -5, 0.164, 789, unit="nm")
    #
    # measure = RamanSpectrumImage("data/07-08-2020/measure_KimwipePaper_R6Gsaturated_300s_62mW_#4.tif")
    # background = RamanSpectrumImage("data/07-08-2020/ref_KimwipePaper_300s_62mW_#4.tif")
    #
    # measureSERS = RamanSpectrumImage("data/02-08-2020/measure_OOSERSAu_R6G_5min-dry_10s_31p3_relcm.TIF")
    # refSERS = RamanSpectrumImage("data/02-08-2020/ref_OOSERSAu_empty_noDescription_10s_31p3_nm.TIF")
    #
    # rmg1.load_ramanSpectrum_image(measure-background)
    # rmg1.add_plot(xunit='cm-1', normalized=False, label="07-08-2020, R6G, C=saturated, kimwipePaper, 300s")
    # rmg1.show_plot()

    rmg1 = RamanGrapher()
    rmg1.load_calibration_polynomial(6.31*10**-9, -3.33*10**-5, 0.145, 900, unit="nm")
    rmg1.load_curvature_polynomial(0.0002, -0.08, +8)
    # x = [22, 186, 380, 477, 543, 552, 582, 658, 719, 766]
    # y = [903.4, 926.0, 950.8, 962.3, 970.4, 971.1, 974.0, 982.9, 989.5, 994.4]
    # rmg1.find_calibration_from_points(x, y)

    # R6G 20mg/ml(saturated) sur Thorlabs paper
    measure = RamanSpectrumImage("C:\\Users\\marc-\\Documents\\Github\\SERS-labo\\data\\10-08-2020\\measure_Butter_300s_62mW_#5.tif")
    rmg1.load_ramanSpectrum_image(measure)
    rmg1.add_plot(figsize=(6, 4), xunit='cm-1', normalized=True, label="Butter, C=sat, glass slide, 300s, 62mW", xlimits=(750, 1200), majorTicks=50)
    rmg1.show_plot()
