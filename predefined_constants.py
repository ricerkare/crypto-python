# I ran this to find the rotation offsets for rho as well as the round
# constants for iota.

import numpy
np = numpy
u8 = np.uint64

# Find rho offsets.
RHO_OFFSETS = dict()
RHO_OFFSETS[(u8(0), u8(0))] = u8(0)
x = u8(1)
y = u8(0)
for t in range(24):
    RHO_OFFSETS[(x, y)] = u8(((t+1)*(t+2) // 2) % 64)
    x_prime = x
    x = u8(y)
    y = u8((2*x_prime + 3*y) % 5)

print("\nRHO OFFSETS:\n")
print(RHO_OFFSETS)


ROUND_CONSTANTS = np.zeros((24), dtype=u8)

def array_to_uint64(A):
    return sum([A[i] << u8(i) for i in range(len(A))])

# Find round constants for the function iota.
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

def find_rc(round_index):
    RC = numpy.array([0]*64, dtype=u8)
    for j in range(7):
        RC[2**j - 1] =rc(j + 7*round_index)
    return array_to_uint64(RC)

print("\nROUND CONSTANTS FOR IOTA\n")
print("[", end="")
for i in range(24):
    ROUND_CONSTANTS[i] = find_rc(i)
    print("0x" + format(ROUND_CONSTANTS[i], "016x"), end="")
    if i != 23:
        print(",")
print("]")
    

