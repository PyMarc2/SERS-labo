from ramanGrapher import *


grapher = RamanGrapher()
image1 = RamanSpectrumImage("./test/test_gray32768_image.tif")
image2 = RamanSpectrumImage("./test/test_gray32768_image.tif")

grapher.load_ramanSpectrum_image(image1-image2*4)
grapher.add_plot(xunit='nm', normalized=False, label="02-08-2020, R6G, C=?, 100nmAuNP, 10s")
grapher.show_plot()
