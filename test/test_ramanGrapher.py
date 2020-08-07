from unittest import TestCase
from unittest.mock import Mock, patch
import gc
import numpy as np
from PIL import Image
from ramanGrapher import *


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


class TestRamanGrapher(TestCase):

    def setUp(self) -> None:
        self.ramanGrapher = RamanGrapher()
        self.test_create_grayscaleImage()

    def tearDown(self) -> None:
        del self.ramanGrapher
        gc.collect()

    def test_load_image(self):
        pass

    def test_create_grayscaleImage(self):
        self.grayscale = np.ones((400, 1340), np.uint16)
        self.grayscale *= 32768
        print(self.grayscale[0, 0])
        self.assertEqual(self.grayscale[0, 0], 32768)
        self.grayscale = Image.fromarray(self.grayscale)
        print(self.grayscale.getpixel((0, 0)))
        #self.grayscale.save("test_gray32768_image.tif")

    def test_load_image_scaleUp(self):
        self.ramanGrapher.load_image_from_PIL(self.grayscale)
        print(self.ramanGrapher.initialImage[0, 0])
        print(self.ramanGrapher.initialImage.dtype)
        self.assertEqual(self.ramanGrapher.initialImage[0, 0], 32768)
        self.assertEqual(self.ramanGrapher.initialImage.dtype, np.uint32)

    def test_reset_image(self):
        for i in range(10**6):
            print(i)

    def test_save_image_dialog(self):
        pass

    def test_modify_calibration_polynomial(self):
        pass

    def test_modify_curvature(self):
        pass

    def test_modify_image_to_summed_plot(self):
        pass

    def test_modify_subtract_ref_image(self):
        pass

    def test_modify_subtract_data_from(self):
        pass

    def test_modify_switch_units(self):
        pass

    def test_modify_smoothen(self):
        pass

    def test_add_plot(self):
        pass

    def test_prepare_plot_change_xlabel(self):
        pass

    def test_prepare_plot_reformat_subplots(self):
        pass

    def test_prepare_normalize_plot(self):
        pass

    def test_prepare_make_output_data(self):
        pass

    def test_add_peaks(self):
        pass

    def test_show_plot(self):
        pass
