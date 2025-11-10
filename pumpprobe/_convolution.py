# This module provides convolution functions, originally part of the mistofrutta library,
# now integrated into pumpprobe to reduce external dependencies.

import numpy as np
# We need the integral function from the _integration module we just created
from ._integration import integral

def convolution1(A, B, t, dt, k=8):
    """
    Computes a single point of the convolution of A and B up to index t.
    """
    # Ensure we don't go out of bounds
    t = min(t, len(A) - 1, len(B) - 1)
    
    # The convolution integral is sum(A(tau) * B(t - tau)) d(tau)
    # For discrete signals, this is a sum.
    # We need to reverse the first signal A up to the point t.
    y = integral(A[:t+1][::-1] * B[:t+1], dx=dt, k=k)
    return y

def convolution(A, B, dt, k=8):
    """
    Computes the full convolution of signals A and B.
    """
    n = len(A)
    y = np.zeros_like(A, dtype=float) # Ensure the output array is float
    for i in range(n):
        y[i] = convolution1(A, B, i, dt, k)
    return y

def slice_test():
    """
    A simple test function.
    """
    print("Slice test from _convolution module.")

