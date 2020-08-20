from RamanGrapher import RamanGrapher
from RamanSpectrumImage import RamanSpectrumImage

# PI ACTON SPECTROMETER FROM 10-08-2020
rmg1 = RamanGrapher()
rmg1.load_calibration_polynomial(8.66*10**-9, -3.51*10**-5, 0.156, 799, unit="nm")
rmg1.load_curvature_polynomial(0.000125, -0.05, +5)


# Beurre sur glass slide.
measure = RamanSpectrumImage(".\\data\\13-08-2020\\SERS_R6G_saturated.tif")
Background = RamanSpectrumImage(".\\data\\13-08-2020\\SERS_rincé.tif")
measure2 = RamanSpectrumImage(".\\data\\13-08-2020\\SERS_glycerole.tif")
measure3 = RamanSpectrumImage(".\\data\\13-08-2020\\Glycerol_volumetric.tif")
measure4 = RamanSpectrumImage(".\\data\\20-08-2020\\Sucrose30min.tif")
Vodka = RamanSpectrumImage(".\\data\\20-08-2020\\Vodka.tif")
#rmg1.load_ramanSpectrum_image(Background-measure)
#rmg1.add_plot(figsize=(6, 4), xunit='cm-1', normalized=False, uncurved=True, label="Butter, C=sat, glass slide, 300s, 62mW", majorTicks="auto", minorTicks="auto")
#rmg1.add_image(uncurved=True)
rmg1.load_ramanSpectrum_image(Vodka)
rmg1.add_plot(figsize=(6, 4), xunit='cm-1', xlimits=(2, 1340), ylimits=(20,400),  normalized=False, uncurved=True, label=", Sucrose Saturated, 30min, 83mW", majorTicks="auto", minorTicks="auto")

rmg1.show()

