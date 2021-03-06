from RamanGrapher import RamanGrapher
from RamanSpectrumImage import RamanSpectrumImage

# PI ACTON SPECTROMETER FROM 10-08-2020
rmg1 = RamanGrapher()
rmg1.load_calibration_polynomial(8.66*10**-9, -3.51*10**-5, 0.156, 799, unit="nm")
rmg1.load_curvature_polynomial(0.000125, -0.05, +5)

# 1- Create Images Objects
test = RamanSpectrumImage(".\\data\\volumique\\23-01-2021\\glutamate1.tif")

# 2- Make operation on Image Objects
#result = test-test2

# 3- Load Images Object inside RamanGrapher (rmg1)
rmg1.load_ramanSpectrum_image(test)

# 4- Add_plot or/and add_image and then show()
# rmg1.add_image()
rmg1.add_plot(figsize=(6, 4), xunit='cm-1', xlimits=(2, 940), ylimits=(20, 400),  normalized=False, uncurved=True, label="Polystyrene Raman\nSpectrum", majorTicks=250, minorTicks="auto")

rmg1.show()


'''TODO: normalization when zero crash'''