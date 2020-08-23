from RamanGrapher import RamanGrapher
from RamanSpectrumImage import RamanSpectrumImage

# PI ACTON SPECTROMETER FROM 10-08-2020
rmg1 = RamanGrapher()
rmg1.load_calibration_polynomial(8.66*10**-9, -3.51*10**-5, 0.156, 799, unit="nm")
rmg1.load_curvature_polynomial(0.000125, -0.05, +5)


# Beurre sur glass slide.
# measure = RamanSpectrumImage(".\\data\\13-08-2020\\SERS_R6G_saturated.tif")
# Background = RamanSpectrumImage(".\\data\\13-08-2020\\SERS_rincé.tif")
# measure2 = RamanSpectrumImage(".\\data\\13-08-2020\\SERS_glycerole.tif")
# measure3 = RamanSpectrumImage(".\\data\\13-08-2020\\Glycerol_volumetric.tif")
# measure4 = RamanSpectrumImage(".\\data\\20-08-2020\\Sucrose30min.tif")
Vodka = RamanSpectrumImage(".\\data\\20-08-2020\\Vodka.tif")
# Guru = RamanSpectrumImage(".\\data\\20-08-2020\\Guru.tif")
# SersBackground = RamanSpectrumImage(".\\data\\20-08-2020\\SERSseul.tif")
# SucroseSERS = RamanSpectrumImage(".\\data\\20-08-2020\\SERSsucrose.tif")
# vodkaSERS = RamanSpectrumImage(".\\data\\20-08-2020\\SERSvodka.tif")
# SERS10 = RamanSpectrumImage(".\\data\\20-08-2020\\SERS#10_alone300s_31mW.tif")
# SERS10_vodka = RamanSpectrumImage(".\\data\\20-08-2020\\SERS10_vodka.tif")
SERSno9_Ethanol = RamanSpectrumImage(".\\data\\21-08-2020\\measure_OOSERS#9_MiliQ+ethanol_spoonFed_300s_31mW_5x5MF.tif")
SERSno9_MiliQ = RamanSpectrumImage(".\\data\\21-08-2020\\ref_OOSERS#9_MiliQ_300s_31mW_5x5MF.tif")
#rmg1.load_ramanSpectrum_image(Background-measure)
#rmg1.add_plot(figsize=(6, 4), xunit='cm-1', normalized=False, uncurved=True, label="Butter, C=sat, glass slide, 300s, 62mW", majorTicks="auto", minorTicks="auto")
#rmg1.add_image(uncurved=True)
rmg1.load_ramanSpectrum_image(SERSno9_Ethanol-SERSno9_MiliQ*1)

#rmg1.load_ramanSpectrum_image(Vodka)
rmg1.add_plot(figsize=(6, 4), xunit='cm-1', xlimits=(2, 1340), ylimits=(20,400),  normalized=False, uncurved=True, label=", Vodka [40% eth] on SERS , 5min, 31mW", majorTicks="auto", minorTicks="auto")
rmg1.load_ramanSpectrum_image(Vodka)
rmg1.add_plot(figsize=(6, 4), xunit='pixel', xlimits=(2, 1340), ylimits=(20,400),  normalized=False, uncurved=True, label=", Vodka [40% eth] on SERS , 5min, 31mW", majorTicks="auto", minorTicks="auto")
rmg1.show()

