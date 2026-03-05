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

rng = np.random.default_rng(123)
n_pick = min(1000, n_rows)

row_indices = rng.choice(n_rows, size=n_pick, replace=False)

data_subset = data[row_indices]

samples = generate_mock_kwargs_from_fits(
    data_subset,
    seed=42
)

save_samples_to_fits(samples, output_file)