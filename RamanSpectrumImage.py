from __future__ import annotations
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile
import warnings
np.seterr(all='warn')
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
        self.isUncurved = False

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
        if max(other) <= 2** 64:
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
        if prod <= 2 ** 64:
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
