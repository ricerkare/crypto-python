# Copyright 2025-06-26 Valdimar Sigurdsson
# Released under the GPL-3 license.

# This implementation of SHA3 will be much faster than the barebones
# "academic" version I wrote ("sha3_slow.py").

# IMPORTANT: String bytes of the input need to be processed in LITTLE
# ENDIAN order for all bit operations.

# For SHA-3, w = 64, b = 1600 and l = 6.

# TODO: Cleanup.
# TODO: Convert arrays to uint64


import numpy
import random
from conversion_functions import int_to_array, array_to_int, \
    hexlify_array, str_to_array, array_to_str, flip_bits_per_byte, \
    roll_uint64
np = numpy
u8 = np.uint64

RHO_OFFSETS = dict()
RHO_OFFSETS[(u8(0), u8(0))] = u8(0)
x = u8(1)
y = u8(0)
for t in range(24):
    RHO_OFFSETS[(x, y)] = u8(((t+1)*(t+2) // 2) % 64)
    x_prime = x
    x = u8(y)
    y = u8((2*x_prime + 3*y) % 5)

    

# xor multiple bits together
def xor(array):
    a = array[0]
    for b in array[1:]:
        a ^= b
    return a

# xor multiple arrays together
# TODO: Consider removing this since I haven't actually used it.
def xor_arrays(arrays):
    B = arrays[0]
    for A in arrays[1:]:
        B ^= A
    return B

def gen_state_array(S):
    """Generate a 5x5 array of lanes, each of which is represented as a
     numpy.uint64.
        S: bytes object
    """
    A = numpy.zeros((5, 5), dtype="<u8")
    for y in range(5):
        for x in range(5):
            A[x][y] = numpy.frombuffer(\
                S[8*(5*y+x) : 8*(5*y+x+1)], dtype="<u8")[0]
    return A

def flatten_state_array(A):
    # TODO: Make sure A is 5x5.
    return b"".join([[A[x][y].tobytes() for y in range(5)] for x in
                     range(5)])


# theta: xor each bit A[x][y][z] with the parities of 2 columns. 
def theta(A):
    # If you look at the FIPS 202 spec, the format is somewhat
    # different - the columns are xor summed before rolling. But a)
    # xor is commutative & associative, and b) rolling is linear. So
    # the rearranged operations below are equivalent to the spec.
    D = numpy.zeros((5), dtype="<u8")
    for x in range(5):
        D[x] = xor([A[(x-1) % 5][y] ^ roll_uint64(A[(x+1) % 5][y], u8(1)) \
                    for y in range(5)])
    for y in range(5):
        A[:, y] = A[:, y] ^ D

# rho: rotate bits of a state array by an offset.
def rho(A):
    x = 1
    y = 0
    for x in range(5):
        for y in range(5):
            A[x][y] = roll_uint64(A[x][y], RHO_OFFSETS[(x, y)])

