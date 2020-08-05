import PIL
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import AutoMinorLocator
from tkinter.filedialog import asksaveasfilename
from scipy.signal import order_filter
from scipy import signal
from spectrumuncurver import SpectrumUncurver


class RamanGrapher:

    def __init__(self, figsize=(9, 8)):
        self.spectrumImagePath = None

        self.initialImage = None
        self.intermediateImage = None
        self.outputImage = None

        self.initialPlot = None
        self.initialPlotXData = []
        self.initialPlotYData = []

        self.intermediatePlot = None
        self.intermediatePlotXData = []
        self.intermediatePlotYData = []

        self.outputPlot = None
        self.outputPlotXData = []
        self.outputPlotYData = []

        self.figure = plt.figure(figsize=figsize)
        self.ax = None
        self.canvas = FigureCanvas(self.figure)
        self.xlabel = "wavelenght [nm]"
        self.ylabel = "relative intensity [a. u.]"

        self.calibratedXAxis = []

        # Experiment parameters
        self.units = "nm"
        self.excitationWavelenght = 785

        self.amountOfPlots = 0
        self.imageType = np.int16
        self.uncurver = SpectrumUncurver()

    def load_image(self, imagePath: str):
        self.spectrumImagePath = imagePath
        self.initialImage = np.array(Image.open(self.spectrumImagePath))
        # self.initialImage.astype(np.int32)
        self.intermediateImage = self.initialImage

    def reset_image(self):
        self.intermediateImage = self.initialImage

    def save_image_dialog(self):
        try:
            path = asksaveasfilename()
            self.figure.savefig(path, dpi=600)
        except Exception as e:
            print(e)

    def modify_calibration_polynomial(self, *args, unit='nm'):
        calibrationEquation = np.poly1d(args)
        self.calibratedXAxis = calibrationEquation(np.linspace(0, self.intermediateImage.shape[1], self.intermediateImage.shape[1]))
        print("\nLENGHT OF XAXIS FROM CALIBRATION:\n", len(self.calibratedXAxis))
        self.intermediatePlotXData = self.calibratedXAxis
        self.units = unit

    def modify_curvature(self, xlim, ylim, method='gaussian'):
        self.uncurver.load_array_image(imageArray=self.intermediateImage)
        output = self.uncurver.uncurve_spectrum_image(xlim=xlim, ylim=ylim, method=method)
        self.intermediateImage = output

    def modify_image_to_summed_plot(self):
        print("\nIMAGE TO PLOT VALUES:\n", self.intermediateImage)
        self.intermediatePlotYData = [sum(self.intermediateImage[:, _]) for _ in range(self.intermediateImage.shape[1])]
        self.intermediatePlotXData = np.linspace(0, self.intermediateImage.shape[1], self.intermediateImage.shape[1])

    def modify_subtract_ref_image(self, refImagePath):
        self.refImage = np.array(Image.open(refImagePath))
        # self.refImage.asType(self.imageType)
        print("\nFIRST IMAGE:\n", self.intermediateImage)
        print("\nSECOND IMAGE:\n", self.refImage)
        output = cv2.subtract(self.intermediateImage, self.refImage)
        self.intermediateImage = output.astype(np.int8)
        print("\nSUBSTRACTION OF REFERENCE:\n", self.intermediateImage)

    def modify_subtract_data_from(self, secondImagePath):
        self.secondImage = np.array(Image.open(secondImagePath))
        self.intermediateImage = cv2.subtract(self.secondImage, self.intermediateImage)
        print("\nSUBSTRACTION OF DATA FROM IMAGE:\n", self.intermediateImage)

    def modify_switch_units(self, units):
        if units == 'both':
            pass
        elif self.units == units:
            pass
        elif units == 'cm-1':
            output = np.divide((1*10**7), self.excitationWavelenght) - np.divide((1*10**7), self.intermediatePlotXData)
            self.intermediatePlotXData = output
            self.units = 'cm-1'

        elif units == 'nm':
            pass

    def modify_smoothen(self, order, fc, btype="lowpass"):
        b, a = signal.butter(order, fc, btype=btype)
        output = signal.filtfilt(b, a, self.intermediatePlotYData)
        self.intermediatePlotYData = output

    def add_plot(self, label="", normalized=True, xlimits=(0, 1340), xunit='nm', subTicks=True, peakfind=(False, 0, 0, 0, 0, 0, 0, 0, 0)):

        if normalized:
            self.prepare_normalize_plot()

        if peakfind[0]:
            print("should find peaks and output x_index")
            peaks = []

        self.modify_switch_units(xunit)

        self.prepare_make_output_data()
        self.prepare_plot_reformat_subplots()
        self.ax.plot(self.outputPlotXData[xlimits[0]:xlimits[1]], self.outputPlotYData[xlimits[0]:xlimits[1]], label=label, linewidth=1.2)
        if subTicks:
            self.ax.xaxis.set_minor_locator(AutoMinorLocator())

        self.prepare_plot_change_xlabel()
        self.ax.legend()
        plt.tight_layout()

        if xunit == 'both':
            self.ax2 = self.ax.twiny()
            self.modify_switch_units('cm-1')
            self.prepare_make_output_data()
            self.ax2.plot(self.outputPlotXData, np.ones(len(self.outputPlotXData)))

    def prepare_plot_change_xlabel(self):
        if self.units == 'nm':
            self.xlabel = "wavelenght [nm]"
        elif self. units == 'cm-1':
            self.xlabel = "wavenumber [cm$^{-1}$]"
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def prepare_plot_reformat_subplots(self):
        n = len(self.figure.axes)
        for i in range(n):
            self.figure.axes[i].change_geometry(n + 1, 1, i + 1)

        self.ax = self.figure.add_subplot(n + 1, 1, n + 1)

    def prepare_normalize_plot(self):
        self.intermediatePlotYData /= max(self.intermediatePlotYData)

    def prepare_make_output_data(self):
        self.outputPlotYData = self.intermediatePlotYData
        self.outputPlotXData = self.intermediatePlotXData

    def add_peaks(self, distance=4, height=0.2, threshold=0, prominence=0.1, width=2):
        peaks, _ = signal.find_peaks(self.outputPlotYData, distance=distance, height=height, threshold=threshold, prominence=prominence, width=width)
        print(peaks)
        if peaks.any():
            self.ax.plot(self.outputPlotXData[peaks], self.outputPlotYData[peaks], 'o')
        for x, y in zip(self.outputPlotXData[peaks], self.outputPlotYData[peaks]):
            label = "{}".format(int(x))
            self.ax.annotate(label, (x, y), textcoords="offset points", xytext=(-10, -10), ha="center", rotation=90, color='k')

    @staticmethod
    def show_plot():
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    rmg1 = RamanGrapher(figsize=(9, 8))
    rmg1.load_image("data/04-08-2020/measure_OOSERSAu_R6G[10-4]_15min_30s_31p3.TIF")
    rmg1.modify_subtract_ref_image("data/04-08-2020/ref_OOSERSAu_empty_forR6G10-4M_30s_31p3.TIF")
    rmg1.modify_image_to_summed_plot()
    rmg1.modify_calibration_polynomial(1.67*10**-8, -4.89*10**-5, 0.164, 789)
    rmg1.modify_smoothen(2, 0.2)
    rmg1.add_plot(xunit='cm-1', label="R6G 10$^{-4}$[M] on 100nm AuNP")
    rmg1.add_peaks(distance=4, height=0.2, threshold=0, prominence=0.08, width=1)

    rmg1.load_image("data/02-08-2020/measure_OOSERSAu_R6G_5min-dry_10s_31p3_relcm.TIF")
    rmg1.modify_subtract_ref_image("data/02-08-2020/ref_OOSERSAu_empty_noDescription_10s_31p3_nm.TIF")
    rmg1.modify_image_to_summed_plot()
    rmg1.modify_calibration_polynomial(1.67 * 10 ** -8, -4.89 * 10 ** -5, 0.164, 789)
    rmg1.modify_smoothen(2, 0.2)
    rmg1.add_plot(xunit='cm-1', label="R6G unkown concentration on 100nmAuNP")
    rmg1.add_peaks(distance=4, height=0.2, threshold=0, prominence=0.1, width=1)


    # rmg1.save_image_dialog()
    rmg1.show_plot()
    rmg1 = RamanGrapher()
    rmg1.load_image("data/measure_OOSERSAu_R6G_5min-dry_10s_31p3_relcm.TIF")
    rmg1.modify_subtract_ref_image("data/ref_OOSERSAu_empty_noDescription_10s_31p3_nm.TIF")
    rmg1.modify_image_to_summed_plot()
    rmg1.modify_calibration_polynomial(1.67*10**-8, -4.89*10**-5, 0.164, 789)
    rmg1.modify_smoothen(2, 0.2)
    rmg1.add_plot(xunit='cm-1', label="R6G w/ 100nm AuNP")
    rmg1.add_peaks()

    rmg1.load_image("data/measure_OOSERSAu_R6G_5min-dry_10s_31p3_relcm.TIF")
    rmg1.modify_subtract_ref_image("data/04-08-2020/measure_OOSERSAu_R6G_noDescription_10s_31p3_04_08_2020.TIF")
    rmg1.modify_image_to_summed_plot()
    rmg1.modify_calibration_polynomial(1.67 * 10 ** -8, -4.89 * 10 ** -5, 0.164, 789)
    rmg1.modify_smoothen(2, 0.2)
    rmg1.add_plot(xunit='cm-1', label="R6G w/ 100nm AuNP")
    rmg1.add_peaks()

    rmg1.show_plot()
