import numpy

# Convert integer from 0 to 255 to an array of its binary
# representation - LSB first.
def int_to_array(n):
    arr = numpy.array([], dtype=int)
    foo = n
    for _ in range(8):
        arr = numpy.append(arr, [foo & 1])
        foo >>= 1
    return arr

# Binary array to int - LSB first.
def array_to_int(A):
    return sum([A[i] << i for i in range(8)])

# hexlify an array A of binary digits whose length is a multiple of
# 8. By default it reads bits LSB-first on a per-byte basis. If
# all_backwards is set to True, then the entire array is read backwards.
def hexlify_array(A, all_backwards=False):
    assert len(A) % 8 == 0
    sums = [sum(A[i] << (i%8) for i in range(8*j, 8*(j+1))) \
            for j in range(len(A) // 8)]
    if all_backwards:
        return bytearray(sums[::-1]).hex()
    return bytearray(sums).hex()

# Convert a string to an array of its binary representation.
def str_to_array(S):
    A = numpy.array([], dtype=int)
    for c in S:
        A = numpy.append(A, int_to_array(ord(c)))
    return A

# Convert a binary array into a string.
def array_to_str(A):
    S = ""
    for block in range(len(A) // 8):
        S += chr(array_to_int(A[8*block : 8*(block+1)]))
    return S

def flip_bits_per_byte(bits):
    assert len(bits) % 8 == 0
    flipped = numpy.zeros_like(bits)
    for i in range(0, len(bits), 8):
        flipped[i:i+8] = bits[i:i+8][::-1]
    return flipped
