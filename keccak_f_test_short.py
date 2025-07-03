from sha3 import *

# TESTING KECCAK_F
S = b"\x00"*200
S = keccak_f(S)
print("keccak_f of all 0 bytes:\n", S.hex())
S = keccak_f(S)
print("keccak_f of previous input:\n", S.hex())
