# oiginal author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

import numpy as np
from scipy import sparse

def library_size_normalize(data, verbose=False):
    """
    Normalize the library size of the given count matrix.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The count matrix to normalize.
    verbose : bool, optional
        If True, print messages about the normalization process.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
        The library size normalized count matrix.
    """

    # Calculate the library size factors
    lib_size = data.sum(axis=1)
    lib_size_factors = lib_size / np.mean(lib_size)

    # Apply the normalization
    if sparse.issparse(data):
        norm_data = data.multiply(lib_size_factors[:, np.newaxis])
    else:
        norm_data = data / lib_size_factors[:, np.newaxis]

    if verbose:
        print("Normalized library size of count matrix.")

    return norm_data
