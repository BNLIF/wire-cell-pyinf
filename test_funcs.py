'''
functions, import for testing
'''

import numpy as np
import torch

def multiply(a=0, b=0):
    c = a * b
    print('result of {} * {} is {}'.format(a,b,c))
    return c

def conv(bytes):
	arr = np.frombuffer(bytes, dtype=np.intc)
	print("python: ", arr)
	return arr.tobytes()
