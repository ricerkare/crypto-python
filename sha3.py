# First try implementing SHA-3 from scratch.
# Copyright 2025-06-22 Valdimar Sigurdsson
# Released under the GPL-3 license.


# For SHA-3, w = 64, b = 1600 and l = log_2(w) = 6.

import numpy
import random

def xor(array):
    a = array[0]
    for b in array:
        a ^= b
    return a

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

# TODO: define plane and lane if necessary.


# TODO: implement theta.

# theta: xor each bit A[x][y][z] with the parities of 2 columns -
# A[(x-1) mod 5][:][z] and A[(x+1) mod 5][:][z].
def theta(A):
    C = numpy.zeros((5, 64), dtype=int)
    D = numpy.zeros((5, 64), dtype=int)
    A_new = numpy.zeros((5, 5, 64), dtype=int)
    for x in range(5):
        for z in range(64):
            C[x][z] = xor([A[x][y][z] for y in range(5)])
            D[x][z] = C[x - 1][z] ^ C[(x + 1) % 5][z - 1]
            for y in range(5):
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

def keccak_f(S):
    A = gen_state_array(S)
    for round_index in range(0, 24):
        A = iota(chi(pi(rho(pi(A)))), round_index)
    S = numpy.zeros((1600), dtype=int)
    for y in range(5):
        for x in range(5):
            for z in range(64):
                S[64*(x + 5*y) + z] = A[x][y][z]
    return S


# TESTING.
S = numpy.array([random.randint(0, 1) for _ in range(1600)])
A = gen_state_array(S)

with numpy.printoptions(threshold=numpy.inf):
    print("  ORIGINAL STRING\n\n")
    print(S)
    print("  KECCAK'D STRING\n\n")
    print(keccak_f(S))


# TODO: sponge
# def sponge(f, pd, r
