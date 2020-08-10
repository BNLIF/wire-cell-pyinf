'''
read 2D numpy from file, send out pybytes
'''
import numpy as np


def read_npz(path, key, col, dtype='float32'):
    data = np.load(path)
    arr = data[key].astype(dtype)
    return arr[:,col].tobytes()

if __name__ == '__main__':
    dtype = 'float32'

    # simulation from nue-6972-54-2707.root
    path = '/lbne/u/hyu/lbne/uboone/wire-cell-pydata/scn_vtx/nuecc-sample.npz'
    x = read_npz(path, 'coords', 0, dtype=dtype)
    y = read_npz(path, 'coords', 1, dtype=dtype)
    z = read_npz(path, 'coords', 2, dtype=dtype)
    q = read_npz(path, 'ft', 0, dtype=dtype)

    q = np.frombuffer(z, dtype=dtype)