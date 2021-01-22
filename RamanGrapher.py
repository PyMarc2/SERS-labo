from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import asksaveasfilename
from scipy import signal
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, AutoLocator)
import warnings
np.seterr(all='warn')
warnings.filterwarnings('error')


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
        self.isCalibrated = False
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
        if self.calibrationEquation is not None and self.dataUnit == "pixel" and not self.isCalibrated:
            self.plotData[0] = self.calibrationEquation(np.linspace(0, self.outputImage.width-1, self.outputImage.width))
            self.dataUnit = self.calibrationUnit
            self.isCalibrated = True
            # print("\nLENGHT OF XAXIS FROM CALIBRATION:\n", len(self.calibratedXAxis))
        else:
            pass

    def modify_curvature(self):
        if not self.outputImage.isUncurved:
            for ypos, i in enumerate(range(self.outputImage.height)):
                corr = -int(self.curvatureEquation(i))
                if corr >= 1:
                    self.outputImage[corr:, ypos] = self.initialImage[0:-corr, ypos]
                elif corr < 0:
                    self.outputImage[0:corr-1, ypos] = self.initialImage[-corr:-1, ypos]
                else:
                    self.outputImage[:, ypos] = self.initialImage[:, ypos]
            self.outputImage.isUncurved = True

    def modify_image_to_summed_plot(self, a=0, b=400):
        self.plotData[1] = [sum(self.outputImage[_, a:b]) for _ in range(self.outputImage.width)]
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

    def baseline_subtraction(self, sp, lam=1e4, p=0.001, niter=10):
        """
        Applies an asymmetric least squares baseline correction to the spectra.
        (Updated April 2020: improved computing speed)

        Parameters:
            sp : array
                Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.
            lam : integer or float,  default = 1e4
                ALS 2nd derivative constraint that defines the smoothing degree of the baseline correction.
            p : int or float, default=0.001
                ALS positive residue weighting that defines the asymmetry of the baseline correction.
            niter : int, default=10
                Maximum number of iterations to optimize the baseline.

        Returns:
            array
                Baseline substracted spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra
                and (n_pixels,) for a single spectrum.
            array
                Baseline signal(s), array shape = (n_spectra, n_pixels) for multiple spectra
                and (n_pixels,) for a single spectrum.
        """
        # sp is forced to be a two-dimensional array
        sp = np.array(sp, ndmin=2)
        # initialization and space allocation
        baseline = np.zeros(sp.shape)  # baseline signal array
        sp_length = sp.shape[1]  # length of a spectrum
        diag = sparse.diags([1, -2, 1], [0, -1, -2], shape=(sp_length, sp_length - 2))
        diag = lam * diag.dot(diag.transpose())
        w = np.ones(sp_length)
        w_matrix = sparse.spdiags(w, 0, sp_length, sp_length)

        for n in range(0, len(sp)):
            for i in range(niter):
                w_matrix.setdiag(w)
                z = w_matrix + diag
                baseline[n] = spsolve(z, w * sp[n])
                w = p * (sp[n] > baseline[n]) + (1 - p) * (sp[n] < baseline[n])  # w is updated according to baseline
        return sp - baseline, baseline

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

    def add_plot(self, figsize=(9, 8), label="", normalized=False, uncurved=False, ylimits=(0, 400), xlimits=(0, 1340), xunit='nm', minorTicks="auto", majorTicks="auto"):
        if self.figure is None:
            self.figure = plt.figure(figsize=figsize)
        if uncurved:
            self.modify_curvature()
        self.modify_image_to_summed_plot(a=ylimits[0], b=ylimits[1])
        if xunit != "pixel":
            self.modify_image_calibration()
        self.modify_smoothen(2, 0.2)
        self.modify_switch_units(xunit)

        if normalized:
            self.modify_normalize_plot()

        self.prepare_plot_reformat_subplots()
        self.prepare_plot_change_xlabel()

        self.ax.plot(self.plotData[0][xlimits[0]:xlimits[1]], self.plotData[1][xlimits[0]:xlimits[1]], label=label, linewidth=1.2)
        
        if minorTicks != "auto":
            self.ax.xaxis.set_minor_locator(MultipleLocator(minorTicks))
        else:
            self.ax.xaxis.set_minor_locator(AutoMinorLocator())

        if majorTicks != "auto":
            self.ax.xaxis.set_major_locator(MultipleLocator(majorTicks))
        else:
            self.ax.xaxis.set_major_locator(AutoLocator())

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

    def add_image(self, figsize=(12, 5), uncurved=False, xlimits=(0, 1340)):
        if self.figure is None:
            self.figure = plt.figure(figsize=figsize)

        self.prepare_plot_reformat_subplots()

        if uncurved:
            self.modify_curvature()

        self.ax.imshow(self.outputImage.imageArray.astype(np.float32)[xlimits[0]:xlimits[1]])

    @staticmethod
    def show():
        plt.tight_layout()
        plt.show()
