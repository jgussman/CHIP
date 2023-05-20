import unittest
from src.CHIP import CHIP
import numpy as np

class TestCHIP(unittest.TestCase):

    def setUp(self):
        self.chip = CHIP()

    def test_sigma_calculation(self):
        filename = "test_spectrum.fits"
        star_ID = "test_star"
        self.chip.spectraDic[filename] = 10.0 # set a test value for spectraDic
        self.chip.sigma_calculation(filename, star_ID)
        self.assertIsNotNone(self.chip.ivarDic[filename])

    def test_sigma_calculation_with_nan(self):
        filename = "test_spectrum.fits"
        star_ID = "test_star"
        self.chip.spectraDic[filename] = np.nan # set a test value for spectraDic
        self.chip.sigma_calculation(filename, star_ID)
        self.assertNotIn(filename, self.chip.ivarDic)
        self.assertIn(star_ID, self.chip.removed_stars["Nan in IVAR"])

    def test_sigma_calculation_with_zero_gain(self):
        filename = "test_spectrum.fits"
        star_ID = "test_star"
        self.chip.spectraDic[filename] = 10.0 # set a test value for spectraDic
        self.chip.sigma_calculation(filename, star_ID)
        self.assertIsNotNone(self.chip.ivarDic[filename])
        self.chip.spectraDic[filename] = 0.0 # set a test value for spectraDic
        self.chip.sigma_calculation(filename, star_ID)
        self.assertNotIn(filename, self.chip.ivarDic)
        self.assertIn(star_ID, self.chip.removed_stars["Nan in IVAR"])



if __name__ == '__main__':
    unittest.main()