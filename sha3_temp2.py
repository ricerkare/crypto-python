# Copyright 2025-06-26 Valdimar Sigurdsson
# Released under the GPL-3 license.

# This implementation of SHA3 will be much faster than the barebones
# "academic" version I wrote ("sha3_slow.py").

# IMPORTANT: String bytes of the input need to be processed in LITTLE
# ENDIAN order for all bit operations.

# For SHA-3, w = 64, b = 1600 and l = 6.

# TODO: Cleanup.
# TODO: Convert arrays to uint64
# TODO: shorten "numpy" to "np"


import numpy
import random
from conversion_functions import int_to_array, array_to_int, \
    hexlify_array, str_to_array, array_to_str, flip_bits_per_byte, \
    roll_uint64
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
    for x in range(5):
        for y in range(5):
            A[x][y] = numpy.frombuffer(\
                S[8*(5*y+x) : 8*(5*y+x+1)], dtype="<u8")[0]
    return A

def flatten_state_array(A):
    return b"".join([b"".join([A[x][y].tobytes() for x in range(5)]) for y in
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


# TESTING KECCAK_F
S = b"\x00"*200
S = keccak_f(S)
print("keccak_f of all 0 bytes:\n", S.hex())
S = keccak_f(S)
print("keccak_f of previous input:\n", S.hex())

def pad101(rate, message_len):
    # `rate` and `message_len` represent # of bytes, not bits.
    n_zeros = x - ((m + 2) % x)
    return b"\x80" + n_zeros*b"\x00" + b"\x01"


# def keccak(N, d, c):
#     # Inputs: bytes object N, output length (in BITS) d, capacity (in
#     # BITS) c.
#     P = N + pad101(200 - c//8, len(N) // 8)
#     n = len(P) // (200 - c//8)
#     P_blocks = 


# # Inputs: (flat) array of ints N; output length d; capacity c.
# # d will be in {224, 256, 384, 512}. 
# def keccakOLD(N, d, c):
#     P = numpy.append(N, pad101(1600 - c, len(N)))
#     n = len(P) // (1600 - c)
#     P_blocks = numpy.array( \
#                         [P[i * (1600-c) : (i+1) * (1600-c)] for i \
#                          in range(n)], dtype=int)
#     S = numpy.zeros((1600), dtype=int)
#     for i in range(n):
#         S = keccak_f(S ^ numpy.append(P_blocks[i], \
#                                       numpy.zeros((c), dtype=int)))
        
#     Z = S[:1600 - c]
#     while len(Z) < d:
#         S = keccak_f(S)
#         Z = numpy.append(Z, S[:1600-c])
#     return Z[:d]


# def sha3_224(M):
#     A = str_to_array(M)
#     K = keccak(numpy.append(A, [0, 1]), 224, 448)
#     sha3_string = hexlify_array(K)
#     return sha3_string

# def sha3_256(M):
#     A = str_to_array(M)
#     A = numpy.append(A, numpy.array([0, 1], dtype=int))
#     K = keccak(A, 256, 512)
#     sha3_string = hexlify_array(K)
#     return sha3_string

# def sha3_384(M):
#     A = str_to_array(M)
#     K = keccak(numpy.append(A, [0, 1]), 384, 768)
#     sha3_string = hexlify_array(K)
#     return sha3_string

# def sha3_512(M):
#     A = str_to_array(M)
#     K = keccak(numpy.append(A, [0, 1]), 512, 1024)
#     sha3_string = hexlify_array(K)
#     return sha3_string

# # TESTING INDIVIDUAL ROUNDS

# def hexlify_lanes(A):
#     return [[hexlify_array(y, all_backwards=True) for y in x] for x in A]

# INPUT_STRING = "\x00" * 1600
# S = str_to_array(INPUT_STRING)
# A = gen_state_array(S)
# numpy.set_printoptions(threshold=numpy.inf)
# print("A\n\n")
# print(A)
# this_round = A.copy()

# for n_round in range(3):
#     A = iota(chi(pi(rho(theta(A)))), n_round)
#     A_hex = hexlify_lanes(A)
#     print("ROUND", n_round, "\n\n", A_hex)

# A = iota(chi(pi(rho(theta(A)))), n_round)
# A_hex = hexlify_lanes(A)
# print("ROUND", n_round, "\n\n", A_hex)
    
# # TESTING SHA3-224

# with numpy.printoptions(threshold=numpy.inf):
#     INPUT_STRING = ""
#     print("\nORIGINAL STRING:\n(Empty string)")
#     print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))

#     INPUT_STRING = "testing!"*25
#     print("\nORIGINAL STRING:\n", INPUT_STRING)
#     print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))

#     INPUT_STRING = "abc"
#     print("\nORIGINAL STRING:\n", INPUT_STRING)
#     print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))

#     INPUT_STRING = "".join([chr(random.randint(97, 97+25)) for _ in range(1000)])
#     print("\nORIGINAL STRING:\n", INPUT_STRING)
#     print("\nSHA3-256 OUTPUT\n", sha3_256(INPUT_STRING))


# # This test input is in the form of 25 64-bit words. This is the
# # initial state array.
# TEST_INPUT = """F1258F7940E1DDE7 84D5CCF933C0478A D598261EA65AA9EE
# BD1547306F80494D 8B284E056253D057 FF97A42D7F8E6FD4 90FEE5A0A44647C4
# 8C5BDA0CD6192E76 AD30A6F71B19059C 30935AB7D08FFC64 EB5AA93F2317D635
# A9A6E6260D712103 81A57C16DBCF555F 43B831CD0347C826 01F22F1A11A5569F
# 05E5635A21D9AE61 64BEFEF28CC970F2 613670957BC46611 B87C5A554FD00ECB
# 8C3EE88A1CCF32C8 940C7922AE3A2614 1841F924A2C509E4 16F53526E70465C2
# 75F644E97F30A13B EAF1FF7B5CECA249"""

# # Separate into 16-character strings.
# TEST_INPUT = TEST_INPUT.split()

# A = numpy.zeros((5, 5), dtype="<u8")

# for x in range(5):
#     for y in range(5):
#         A[x][y] = int(TEST_INPUT[5*y + x], 16)

# print([[hex(y) for y in x] for x in A])
# print([[hex(y) for y in x] for x in theta(A)])
