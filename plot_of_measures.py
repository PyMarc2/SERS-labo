from RamanGrapher import RamanGrapher
from RamanSpectrumImage import RamanSpectrumImage

# PI ACTON SPECTROMETER FROM 10-08-2020
rmg1 = RamanGrapher()
rmg1.load_calibration_polynomial(6.31*10**-9, -3.33*10**-5, 0.145, 900, unit="nm")
rmg1.load_curvature_polynomial(0.000125, -0.05, +5)


# Beurre sur glass slide.
measure = RamanSpectrumImage(".\\data\\10-08-2020\\measure_Butter_300s_62mW_#5.tif")
rmg1.load_ramanSpectrum_image(measure)
rmg1.add_plot(figsize=(6, 4), xunit='cm-1', normalized=True, uncurved=True, label="Butter, C=sat, glass slide, 300s, 62mW", majorTicks="auto", minorTicks="auto")
rmg1.add_image(uncurved=True)
rmg1.show()

