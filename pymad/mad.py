"""
Calculate MAD values.

Version 0.2

author: Johannes Zschocke johannes.zschocke@uk-halle.de or
                          JohannesZschocke@t-online.de

modified by: Yaopeng Ma yaopeng.ma@biu.ac.il
                        yaopeng.ma97@gmail.com

date: 09/2022
"""

import numpy as np
import cupy as cp
import time
from numba.pycc import CC

cc = CC('mad_cc')


def cal_mad_cuda(data, points):
    """
    Calculate MAD values by using cuda. (by Yaopeng Ma)

    This routine calculates the MAD values based on the number of points
    (given by points) on the given 3D data.

    Parameters
    ----------
    data : 3D cupy array
        contains 3D data.
    points : integer
        number of datapoints to use for mad calculation.

    Returns
    -------
    mad : cupy array
        contains the calculated mad values.

    """
    _, length = data.shape
    N_windows = length // points
    mod_points = length % points
    r = cp.sqrt(cp.sum(data ** 2, axis=0))
    if mod_points > 0:
        r = r[: - mod_points]
    r = r.reshape((N_windows, points))
    mad = cp.abs(r - cp.nanmean(r, axis=-1, keepdims=True)).mean(axis=1)
    return mad


@cc.export('cal_mad_jit', 'f8[:](f8[:,:], i4)')
@cc.export('cal_mad_jit', 'f4[:](f4[:,:], i4)')
def cal_mad_jit(data, points):
    """
    Calculate MAD values. (by Johannes Zschocke)

    This routine calculates the MAD values based on the number of points
    (given by points) on the given 3D data.

    Parameters
    ----------
    data : 3D numpy array
        contains 3D data.
    points : integer
        number of datapoints to use for mad calculation.

    Returns
    -------
    mad : numpy array
        contains the calculated mad values.

    """
    # 'allocate' arrays and variables
    # the following variables are used as described and introduced by VahaYpya
    # 2015 DOI: 10.1111/cpf.12127

    # array for mad values
    mad = np.zeros(int(len(data[0]) / points))

    # array for r_i values
    r_i_array = np.empty(points)
    r_i_array[:] = np.nan

    # R_ave value
    R_ave = 0
    i_mad = 0

    # iterate over all values in data
    i = 0
    for (x, y, z) in zip(data[0], data[1], data[2]):
        r_i = np.sqrt(x**2 + y**2 + z**2)
        r_i_array[i] = r_i
        i += 1
        if (i == points):
            R_ave = np.nanmean(r_i_array)
            s = 0
            for ri in r_i_array:
                s += np.abs(ri - R_ave)

            s = s / points
            mad[i_mad] = s
            i_mad += 1
            r_i_array[:] = np.nan
            i = 0

    return mad


if __name__ == "__main__":
    cc.compile()
    xyz = np.random.rand(3, 128 * 3600 * 8).astype(np.float32)
    xyz_cuda = cp.asarray(xyz)
    mad = cal_mad_cuda(xyz_cuda, 128)
    start = time.time()
    mad1 = cal_mad_cuda(xyz_cuda, 128)
    print('cuda: ', time.time() - start)

    mad = cal_mad_jit(xyz, 128)
    start = time.time()
    mad2 = cal_mad_jit(xyz, 128)
    print('jit: ', time.time() - start)
    print(np.allclose(mad1.get(), mad2))
