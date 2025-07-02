import numpy, random
from conversion_functions import roll_uint64
from sha3_temp import theta, rho 

# This test input is in the form of 25 64-bit words. This is the
# initial state array.
TEST_INPUT = """F1258F7940E1DDE7 84D5CCF933C0478A D598261EA65AA9EE
BD1547306F80494D 8B284E056253D057 FF97A42D7F8E6FD4 90FEE5A0A44647C4
8C5BDA0CD6192E76 AD30A6F71B19059C 30935AB7D08FFC64 EB5AA93F2317D635
A9A6E6260D712103 81A57C16DBCF555F 43B831CD0347C826 01F22F1A11A5569F
05E5635A21D9AE61 64BEFEF28CC970F2 613670957BC46611 B87C5A554FD00ECB
8C3EE88A1CCF32C8 940C7922AE3A2614 1841F924A2C509E4 16F53526E70465C2
75F644E97F30A13B EAF1FF7B5CECA249"""

# Separate into 16-character strings.
TEST_INPUT = TEST_INPUT.split()

A = numpy.zeros((5, 5), dtype="<u8")

for x in range(5):
    for y in range(5):
        A[x][y] = numpy.uint64(int(TEST_INPUT[5*y + x], 16))

print("\nORIGINAL STATE ARRAY\n")
for x in A:
    print([hex(y) for y in x])

theta(A)
print("\nAFTER THETA\n")

for x in A:
    print([hex(y) for y in x])

rho(A)
print("\nAFTER RHO\n")

for x in A:
    print([hex(y) for y in x])
