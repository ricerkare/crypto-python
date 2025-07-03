# First try implementing SHA-3 from scratch.
# Copyright 2025-06-22 Valdimar Sigurdsson
# Released under the GPL-3 license.

# NOTE: This is a barebones implementation of the SHA-3 algorithm. All
# the internal functions use (numpy) arrays of integers.

# IMPORTANT: String bytes of the input need to be processed in LITTLE
# ENDIAN order for all bit operations.

# For SHA-3, w = 64, b = 1600 and l = 6.

# TODO: Cleanup.

import numpy
import random
from conversion_functions import int_to_array, array_to_int, \
    hexlify_array, str_to_array, array_to_str, flip_bits_per_byte 

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

# Generate a 5x5x64 state array from an array of 1600 bits
# (represented as integers).
def gen_state_array(S):
    A = numpy.zeros((5, 5, 64), dtype=int)
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A[x][y][z] = S[int(64*(5*y + x) + z)]
    return A

def flatten_state_array(A):
    assert A.shape == (5, 5, 64)
    S = numpy.zeros((1600), dtype=int)
    for y in range(5):
        for x in range(5):
            for z in range(64):
                S[64*(x + 5*y) + z] = A[x][y][z]
    return S

# TODO: define plane and lane if necessary.


# TODO: implement theta.

# theta: xor each bit A[x][y][z] with the parities of 2 columns -
# A[(x-1) mod 5][:][z] and A[(x+1) mod 5][:][(z-1) % 64].
def theta(A):
    C = numpy.zeros((5, 64), dtype=int)
    D = numpy.zeros((5, 64), dtype=int)
    A_new = numpy.zeros((5, 5, 64), dtype=int)
    
    for x in range(5):
        for z in range(64):
            C[x][z] = xor([A[x][y][z] for y in range(5)])
    for x in range(5):
        for z in range(64):
            D[x][z] = C[(x - 1) % 5][z] ^ C[(x + 1) % 5][(z - 1) % 64]
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A_new[x][y][z] = A[x][y][z] ^ D[x][z]
    return A_new

# rho: rotate bits of a state array by an offset.
def rho(A):
    A_new = numpy.zeros((5, 5, 64), dtype=int)
    A_new[0][0] = A[0][0]
    x = 1
    y = 0
    for t in range(24):
        for z in range(64):
            A_new[x][y][z] = A[x][y][(z - (t+1)*(t+2)//2) % 64]
        x_prime = x
        x = y
        y = (2*x_prime + 3*y) % 5
    return A_new

# pi: rearrange positions of lanes of a state array.
def pi(A):
    A_new = numpy.zeros((5, 5, 64), dtype=int)
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A_new[x][y][z] = A[(x + 3*y) % 5][x][z]
    return A_new

# chi: apply a nonlinear function to each bit of a state array using
# the two adjacent bits in its row.
def chi(A):
    A_new = numpy.zeros((5, 5, 64), dtype=int)
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A_new[x][y][z] = A[x][y][z] ^ \
                    ((A[(x+1) % 5][y][z] ^ 1) & A[(x+2) % 5][y][z])
    return A_new

# rc: round constants
def rc(t):
    if t % 255 == 0:
        return 1
    R = [1, 0, 0, 0, 0, 0, 0, 0]
    for i in range(t % 255):
        R = [0] + R
        R[0] = R[0] ^ R[8]
        R[4] = R[4] ^ R[8]
        R[5] = R[5] ^ R[8]
        R[6] = R[6] ^ R[8]
        R = R[:8]
    return R[0]

# Modify A[0][0] of state array A by adding round constants to every
# 2^j-th element of A[0][0] for j from 0 to 7.
def iota(A, round_index):
    A_new = A.copy()
    RC = numpy.array([0]*64, dtype=int)
    for j in range(7):
        RC[2**j - 1] =  rc(j + 7*round_index)
    A_new[0][0] ^= RC
    return A_new

# Keccak-f is simply 24 rounds of the composition of five (Greek
# letter) functions defined above.
def keccak_f(S):
    A = gen_state_array(S)
    for round_index in range(24):
        A = iota(chi(pi(rho(theta(A)))), round_index)
    S = flatten_state_array(A)
    return S

# pad101: padding of message of length m with 10*1 (regex notation)
# where 0s are added until total length is a multiple of x.
def pad101(x, m):
    offset = x - (m + 2) % x
    return numpy.array([1] + [0]*offset + [1], dtype=int)

# Inputs: (flat) array of ints N; output length d; capacity c.
# d will be in {224, 256, 384, 512}. 
def keccak(N, d, c):
    P = numpy.append(N, pad101(1600 - c, len(N)))
    n = len(P) // (1600 - c)
    P_blocks = numpy.array( \
                        [P[i * (1600-c) : (i+1) * (1600-c)] for i \
                         in range(n)], dtype=int)
    S = numpy.zeros((1600), dtype=int)
    for i in range(n):
        S = keccak_f(S ^ numpy.append(P_blocks[i], \
                                      numpy.zeros((c), dtype=int)))
        
    Z = S[:1600 - c]
    while len(Z) < d:
        S = keccak_f(S)
        Z = numpy.append(Z, S[:1600-c])
    return Z[:d]


def sha3_224(M):
    A = str_to_array(M)
    K = keccak(numpy.append(A, [0, 1]), 224, 448)
    sha3_string = hexlify_array(K)
    return sha3_string

def sha3_256(M):
    A = str_to_array(M)
    A = numpy.append(A, numpy.array([0, 1], dtype=int))
    K = keccak(A, 256, 512)
    sha3_string = hexlify_array(K)
    return sha3_string

def sha3_384(M):
    A = str_to_array(M)
    K = keccak(numpy.append(A, [0, 1]), 384, 768)
    sha3_string = hexlify_array(K)
    return sha3_string

def sha3_512(M):
    A = str_to_array(M)
    K = keccak(numpy.append(A, [0, 1]), 512, 1024)
    sha3_string = hexlify_array(K)
    return sha3_string

# TESTING INDIVIDUAL ROUNDS

def hexlify_lanes(A):
    return [[hexlify_array(y, all_backwards=True) for y in x] for x in A]

INPUT_STRING = "\x00" * 1600
S = str_to_array(INPUT_STRING)
A = gen_state_array(S)
numpy.set_printoptions(threshold=numpy.inf)
print("A\n\n")
print(A)
this_round = A.copy()

for n_round in range(3):
    A = iota(chi(pi(rho(theta(A)))), n_round)
    A_hex = hexlify_lanes(A)
    print("ROUND", n_round, "\n\n", A_hex)

A = iota(chi(pi(rho(theta(A)))), n_round)
A_hex = hexlify_lanes(A)
print("ROUND", n_round, "\n\n", A_hex)
    
# TESTING SHA3-224

with numpy.printoptions(threshold=numpy.inf):
    INPUT_STRING = ""
    print("\nORIGINAL STRING:\n(Empty string)")
    print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))

    INPUT_STRING = "testing!"*25
    print("\nORIGINAL STRING:\n", INPUT_STRING)
    print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))

    INPUT_STRING = "abc"
    print("\nORIGINAL STRING:\n", INPUT_STRING)
    print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))

    INPUT_STRING = "".join([chr(random.randint(97, 97+25)) for _ in range(1000)])
    print("\nORIGINAL STRING:\n", INPUT_STRING)
    print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))
