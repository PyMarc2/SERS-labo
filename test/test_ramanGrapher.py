from unittest import TestCase
import gc
from PIL import Image
import numpy as np
from RamanGrapher import RamanGrapher


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
        pass

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
