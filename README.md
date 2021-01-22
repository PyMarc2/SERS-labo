# SERS-labo
### Goals
The SERS-labo objects and methods will allow you to 
- Load a spectral image regardless of resolution / size
- Apply correction on the images you load, such as un-curving, image subtraction, vertical sum of pixels, etc.
- Add calibration data to map pixels to nm or to cm-1
- Apply baseline correction
- Save as high def pdf spectrum.

### Quick Guide
**RamanGrapher** : this object will will represent a grapher object for a particular setup or experiment.
Curvature and calibration are usually experiment redundant, so you only have to pass the parameters once. Then, 
the RamanGrapher allows you to load in RamanSpectrumImage objects to apply different
modifications to the image

**RamanSpectrumImage**: this object is only a wrapper object for an image.

### Example
```
from RamanGrapher import RamanGrapher
from RamanSpectrumImage import RamanSpectrumImage

# PI ACTON SPECTROMETER FROM 10-08-2020
rmg1 = RamanGrapher()
rmg1.load_calibration_polynomial(8.66*10**-9, -3.51*10**-5, 0.156, 799, unit="nm")
rmg1.load_curvature_polynomial(0.000125, -0.05, +5)

# 1- Create Images Objects
test = RamanSpectrumImage(".\\data\\25-09-2020\\test.tif")
test2 = RamanSpectrumImage(".\\data\\25-09-2020\\test2.tif")

# 2- Make operation on Image Objects
result = test-test2

# 3- Load Images Object inside RamanGrapher (rmg1)
rmg1.load_ramanSpectrum_image(result)

# 4- `add_plot` or/and `add_image` and then `show()`
# Will add a plot or an image to the output. You can add multiple plots or images.
rmg1.add_plot(figsize=(6, 4), xunit='cm-1', xlimits=(2, 1340), ylimits=(20, 400),  normalized=False, uncurved=True, label=", Vodka [40% eth] on SERS , 5min, 31mW", majorTicks="auto", minorTicks="auto")
rmg1.show()
```

### More
- `find_calibration_from_points(pixelList, nmList)`:  en ter the corespondent pixel to the wavelenght,
`pixelList =  [2, 356, 1288]`, `nmList = [802, 890, 1018]`
Will return a polynomial, which you then save an enter into `load_curvature_polynomial()`