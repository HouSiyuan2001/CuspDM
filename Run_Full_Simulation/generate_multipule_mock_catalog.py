#!/usr/bin/env python3
import sys

sys.path.append("lib")

from multipoleprior import generate_mock_kwargs_from_fits, save_samples_to_fits

import numpy as np
from astropy.io import fits


file_path = "Data/lensed_qso_mock.fits"
output_file = "demo/Data/lensed_qso_mock_multipole_temp.fits"

with fits.open(file_path) as hdul:
    data = hdul[1].data
    n_rows = len(data)


samples = generate_mock_kwargs_from_fits(
    data,
    seed=42
)

save_samples_to_fits(samples, output_file)