from unittest import TestCase
from PIL import Image
import numpy as np
from RamanSpectrumImage import RamanSpectrumImage


class TestRamanSpectrumImage(TestCase):

    def setUp(self) -> None:
        self.testImage = RamanSpectrumImage()
        print("setup")

    def tearDown(self) -> None:
        del self.testImage

    def test_getitem(self):
        self.testImage.load("test_gray32768_image.tif")

        with self.subTest("load an element"):
            self.assertEqual(self.testImage[0, 0], 32768)
            self.tearDown()

        with self.subTest("load an column"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            columnArray = np.ones((400, 1340))[:, 0]
            print("COLUML", columnArray)
            print("COLUM GETTERL", self.testImage[0, :])
            self.assertEqual(self.testImage[0, :].shape, columnArray.shape)
            self.tearDown()

        with self.subTest("load an row"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            rowArray = np.ones((400, 1340))[0, :]
            self.assertEqual(self.testImage[:, 0].shape, rowArray.shape)
            self.tearDown()

        with self.subTest("load part of a row"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            rowArray = np.ones((400, 1340))[0, 10:100]
            self.assertEqual(self.testImage[10:100, 0].shape, rowArray.shape)

    def test_setitem(self):
        with self.subTest("set an element"):
            self.testImage.load("test_gray32768_image.tif")
            self.testImage[0, 0] = 500
            self.assertEqual(self.testImage[0, 0], 500)
            self.tearDown()

        with self.subTest("set a column"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            columnArray = np.ones((400, 1340))[:, 0]
            self.testImage[0, :] = columnArray
            self.assertCountEqual(self.testImage[0, :], columnArray)
            self.tearDown()

        with self.subTest("set a row"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            rowArray = np.ones((400, 1340))[0, :]
            self.testImage[:, 0] = rowArray
            self.assertCountEqual(self.testImage[:, 0], rowArray)

    def test_shape(self):

        with self.subTest("Shape returns correct"):
            self.testImage.load("test_gray32768_image.tif")
            self.assertEqual(self.testImage.shape, (1340, 400))
            self.tearDown()

        with self.subTest("Width returns correct"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            self.assertEqual(self.testImage.width, 1340)
            self.tearDown()

        with self.subTest("Height returns correct"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            self.assertEqual(self.testImage.height, 400)

    def test_subtract(self):
        self.testImage.load("test_gray32768_image.tif")

        with self.subTest("Test without limit"):
            print(self.testImage)
            iamge2 = self.testImage
            self.testImage - iamge2
            print(self.testImage)
            self.assertEqual(self.testImage[0, 0], 0)

    def test_add(self):
        pass

    def test_mul(self):
        with self.subTest("multi < than 2**64"):
            pass

    def test_load_meta(self):
        pass

    def test_load(self):
        with self.subTest("load from path"):
            self.testImage.load("test_gray32768_image.tif")
            self.assertIsNotNone(self.testImage.imageArray)

        with self.subTest("load from nparray"):
            self.setUp()
            testGrayImageArray = Image.open("test_gray32768_image.tif")
            self.testImage.load(testGrayImageArray)
            self.assertIsNotNone(self.testImage.imageArray)
            self.tearDown()

        with self.subTest("load from PIL Image"):
            self.setUp()
            testGrayImage = Image.open("test_gray32768_image.tif")
            self.testImage.load(testGrayImage)
            self.assertIsNotNone(self.testImage.imageArray)
            self.tearDown()

        with self.subTest("loaded image is array"):
            self.setUp()
            testGrayImageArray = Image.open("test_gray32768_image.tif")
            self.testImage.load(testGrayImageArray)
            self.assertIsInstance(self.testImage.imageArray, np.ndarray)

        with self.subTest("Image.open() keeps uint16 format"):
            self.setUp()
            testGrayImageArray = np.array(Image.open("test_gray32768_image.tif"))
            self.assertEqual(testGrayImageArray.dtype, np.uint16)
            self.assertEqual(testGrayImageArray[0,0], 32768)

        with self.subTest("loaded image converted to uint32"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            self.assertEqual(self.testImage.imageArray.dtype, np.uint32)
            self.tearDown()

        with self.subTest("image value not scaledUp uint16 -> uint32"):
            self.setUp()
            self.testImage.load("test_gray32768_image.tif")
            self.assertEqual(self.testImage.imageArray[0, 0], 32768)
