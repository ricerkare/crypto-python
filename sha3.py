# Copyright 2025-06-26 Valdimar Sigurdsson
# Released under the GPL-3 license.

# This implementation of SHA3 will be much faster than the barebones
# "academic" version I wrote ("sha3_slow.py").

# IMPORTANT: String bytes of the input need to be processed in LITTLE
# ENDIAN order for all bit operations.

# For SHA-3, w = 64, b = 1600 and l = 6.

import numpy
import random
from conversion_functions import *
np = numpy
u8 = np.uint64

RHO_OFFSETS = {
    (0, 0): u8(0),
    (1, 0): u8(1),
    (0, 2): u8(3),
    (2, 1): u8(6),
    (1, 2): u8(10),
    (2, 3): u8(15),
    (3, 3): u8(21),
    (3, 0): u8(28),
    (0, 1): u8(36),
    (1, 3): u8(45),
    (3, 1): u8(55),
    (1, 4): u8(2),
    (4, 4): u8(14),
    (4, 0): u8(27),
    (0, 3): u8(41),
    (3, 4): u8(56),
    (4, 3): u8(8),
    (3, 2): u8(25),
    (2, 2): u8(43),
    (2, 0): u8(62),
    (0, 4): u8(18),
    (4, 2): u8(39),
    (2, 4): u8(61),
    (4, 1): u8(20),
    (1, 1): u8(44)
}

# Predefine round constants for the iota function.
ROUND_CONSTANTS = np.array([
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008
], dtype=u8)

def gen_state_array(S):
    """Generate a 5x5 array of lanes, each of which is represented as a
     numpy.uint64.
        S: bytes object
    """
    A = np.zeros((5, 5), dtype="<u8")
    for x in range(5):
        for y in range(5):
            A[x][y] = np.frombuffer(\
                S[8*(5*y+x) : 8*(5*y+x+1)], dtype="<u8")[0]
    return A

def flatten_state_array(A):
    return b"".join([b"".join([A[x][y].tobytes() for x in range(5)]) for y in
                     range(5)])

# All the 5 functions called inside keccak_f modify the state array
# in-place and do not return anything.

# theta: xor each bit A[x][y][z] with the parities of 2 columns. 
def theta(A):
    # If you look at the FIPS 202 spec, the format is somewhat
    # different - the columns are xor summed before rolling. But a)
    # xor is commutative & associative, and b) rolling is linear. So
    # the rearranged operations below are equivalent to the spec.
    D = np.zeros((5), dtype="<u8")
    for x in range(5):
        D[x] = xor([A[(x-1) % 5][y] ^ roll_uint64(A[(x+1) % 5][y], u8(1)) \
                    for y in range(5)])
    for y in range(5):
        A[:, y] = A[:, y] ^ D

# rho: roll bits of a state array by an offset.
def rho(A):
    x = 1
    y = 0
    for x in range(5):
        for y in range(5):
            A[x][y] = roll_uint64(A[x][y], RHO_OFFSETS[(x, y)])

# pi: rearrange positions of lanes of a state array.
def pi(A):
    A_original = A.copy()
    for x in range(5):
        for y in range(5):
            A[x][y] = A_original[(x + 3*y) % 5][x]

# chi: apply a nonlinear function to each bit of a state array using
# the two adjacent bits in its row.
def chi(A):
    A_original = A.copy()
    for x in range(5):
        for y in range(5):
            A[x][y] = A_original[x][y] \
                ^ ((A_original[(x+1) % 5][y] ^ u8(0xffffffffffffffff))\
                   & A_original[(x+2) % 5][y])

# Modify A[0][0] of state array A by adding round constants to every
# 2^j-th element of A[0][0] for j from 0 to 7.
def iota(A, round_index):
    A[0][0] ^= ROUND_CONSTANTS[round_index]

# Keccak-f is simply 24 rounds of the composition of five (Greek
# letter) functions defined above.
def keccak_f(S):
    A = gen_state_array(S)
    for round_index in range(24):
        theta(A)
        rho(A)
        pi(A)
        chi(A)
        iota(A, round_index)
    S = flatten_state_array(A)
    return S

def pad101(rate, message_len):
    # `rate` and `message_len` represent # of bytes, not bits.
    # In this function we are including the SHA3 domain separator in
    # the padding. So the padding becomes
    # 00000110 000000000 ... 10000000
    n_zeros = rate - ((message_len + 2) % rate)
    return b"\x06" + n_zeros*b"\x00" + b"\x80"

def keccak(N, d, c):
    # Inputs: bytes object N, output length (in BITS) d, capacity (in
    # BITS) c.
    rate = 200 - c//8
    P = N + pad101(rate, len(N))
    n = len(P) // rate
    P_blocks = np.array(
        [P[i * rate : (i+1) * rate] for i in range(n)]
    )
    S = b"\x00"*200
    for i in range(n):
        P_block_filled = P_blocks[i] + b"\x00" * (c//8)
        S = keccak_f(xor_bytes(S, P_block_filled))
    Z = S[:rate]
    while len(Z) < d // 8:
        S = keccak_f(S)
        Z += S[:rate]
    return Z[: d//8]

# For all the following functions we still take byte objects.

def sha3_224(M):
    K = keccak(M, 224, 448)
    return K.hex()

def sha3_256(M):
    K = keccak(M, 256, 512)
    return K.hex()

def sha3_384(M):
    K = keccak(M, 384, 768)
    return K.hex()

def sha3_512(M):
    K = keccak(M, 512, 1024)
    return K.hex()
